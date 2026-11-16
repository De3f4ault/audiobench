"""Transcript refinement using LLM post-processing.

Two modes:
- ``refine_segments(texts)``: batch-numbered-lines protocol — used by both the
  background auto-clean thread and the ``audiobench clean`` CLI command.
  Timestamps are never touched; only segment texts change.
- ``refine(raw_text)``: legacy single-pass full-text mode (kept for potential
  non-segment use cases but not called by the main pipeline).

Designed for models accessible via OllamaClient, including cloud variants
(e.g. ``qwen4-next:110b-cloud``).
"""

from __future__ import annotations

from collections.abc import Callable

from audiobench.core.logger_factory import get_logger

logger = get_logger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────

REFINE_SYSTEM_PROMPT = """You are a transcript auto-corrector. Your ONLY job is to:
1. Fix spacing errors (e.g. "a like" → "alike", "to morrow" → "tomorrow")
2. Fix punctuation (add periods, commas, question marks where appropriate)
3. Fix capitalization (sentence starts, proper nouns)
4. Fix obvious homophones from context (e.g. "their" vs "there")

STRICT RULES:
- Do NOT add, remove, summarize, or rephrase any information
- Do NOT change the meaning or intent of any sentence
- Do NOT add commentary or explanations
- Output ONLY the corrected transcript text, nothing else
"""

SEGMENT_SYSTEM_PROMPT = """You are a careful transcript editor for spoken audio recordings.

Your task is to lightly clean each numbered speech segment:
- Fix obvious word-boundary errors ("builda" → "build a", "toa" → "to a", "isa" → "is a")
- Fix clear capitalization ("jesus" → "Jesus", "god" → "God", "lord" → "Lord")
- Add natural punctuation (periods, commas, question marks) where clearly missing
- Fix obvious homophones only when the context makes the correction unambiguous

What you must NOT do:
- Do not rewrite, paraphrase, or summarize any content
- Do not add words that were not spoken
- Do not remove words or merge sentences
- Do not change proper nouns, names, places, or religious terms you are unsure about
- Do not "correct" code-switching, accents, or non-standard phrasing that is intentional

Return the SAME number of numbered lines as the input — one line per input segment.
Each line must start with its original number and a period (e.g. "1. text here").
"""


class TranscriptRefiner:
    """Segment-aware transcript refinement using an LLM."""

    def __init__(self, client, model: str | None = None):
        """Initialize refiner.

        Args:
            client: OllamaClient instance.
            model: Model to use for refinement (defaults to client's model).
        """
        self._client = client
        self._model = model

    # ── Segment-batch mode (primary path) ─────────────────────────────────

    def refine_segments(
        self,
        segment_texts: list[str],
        batch_size: int = 20,
        on_progress: Callable[[int, int], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[list[str], bool]:
        """Clean segment texts in numbered batches via streaming.

        Sends batches of up to ``batch_size`` segments as numbered lines to the
        LLM using a streaming call (avoids read-timeout on large cloud models).
        The LLM must return the same number of numbered lines.  If it doesn't
        (sanity check), the original batch is kept unchanged.

        Args:
            segment_texts: Raw segment texts in order.
            batch_size: Max segments per LLM call (default 20).
            on_progress: Optional callback(batch_num, total_batches) called
                after each batch completes.
            on_token: Optional callback(token_str) called for each streaming
                token received. Useful for showing live activity dots.

        Returns:
            Tuple of (cleaned_texts, has_failures)
        """
        if not segment_texts:
            return segment_texts, False

        result: list[str] = []
        has_failures = False
        total = len(segment_texts)
        total_batches = (total + batch_size - 1) // batch_size
        batch_num = 0

        for batch_start in range(0, total, batch_size):
            batch = segment_texts[batch_start : batch_start + batch_size]
            cleaned, batch_failed = self._refine_batch(batch, batch_start, on_token=on_token)
            if batch_failed:
                has_failures = True
            result.extend(cleaned)
            batch_num += 1
            if on_progress:
                on_progress(batch_num, total_batches)

        return result, has_failures

    def _refine_batch(
        self,
        batch: list[str],
        offset: int,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[list[str], bool]:
        """Send one numbered batch to the LLM via streaming and return cleaned texts.

        Uses chat_stream() with think=True so Qwen4 reasons before answering.
        The on_token callback fires for each content token received, enabling
        real-time activity display in the CLI.
        On any failure or line-count mismatch, returns the original batch.
        """
        numbered_input = "\n".join(f"{i + 1}. {text.strip()}" for i, text in enumerate(batch))

        try:
            # think=True: Qwen4 uses its reasoning chain first then returns clean segments
            # chat_stream: 300s read timeout (vs 120s for non-streaming)
            chunks = self._client.chat_stream(
                messages=[
                    {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": numbered_input},
                ],
                model=self._model,
                temperature=0.1,
                think=True,
            )
            raw_output = ""
            for chunk in chunks:
                token = chunk.get("content", "")
                if token:
                    raw_output += token
                    if on_token:
                        on_token(token)
            raw_output = raw_output.strip()
            # If Qwen's thinking logic leaked into the raw stream content, strip it out.
            if "<think>" in raw_output:
                import re
                raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

            parsed = self._parse_numbered_lines(raw_output)

            if len(parsed) != len(batch):
                logger.warning(
                    "Segment refiner: batch at offset %d — line count mismatch "
                    "(%d returned vs %d sent). Keeping originals.",
                    offset,
                    len(parsed),
                    len(batch),
                )
                return batch, True

            return parsed, False

        except Exception as e:
            logger.warning(
                "Segment refiner: batch at offset %d failed (%s). Keeping originals.", offset, e
            )
            return batch, True

    @staticmethod
    def _parse_numbered_lines(text: str) -> list[str]:
        """Extract text content from numbered lines (e.g. '1. text' → 'text')."""
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading "N." or "N) " prefix
            if line and line[0].isdigit():
                dot_pos = line.find(".")
                paren_pos = line.find(")")
                sep = (
                    min(p for p in (dot_pos, paren_pos) if p > 0)
                    if any(p > 0 for p in (dot_pos, paren_pos))
                    else -1
                )
                if sep > 0:
                    lines.append(line[sep + 1 :].strip())
                    continue
            lines.append(line)
        return lines

    # ── Legacy full-text mode ──────────────────────────────────────────────

    def refine(self, raw_text: str) -> str:
        """Send raw transcript to LLM for cleanup (single-pass full-text mode).

        Args:
            raw_text: Original Whisper transcription output.

        Returns:
            Refined transcript text, or raw_text on failure.
        """
        if not raw_text or len(raw_text.strip()) < 20:
            return raw_text  # Too short to bother refining

        try:
            result = self._client.chat(
                messages=[
                    {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                model=self._model,
                temperature=0.1,
                think=False,
            )
            refined = result.get("content", "").strip()
            if refined and len(refined) > len(raw_text) * 0.5:
                return refined
            logger.warning(
                "Refinement produced suspicious output (len %d vs raw %d), keeping raw text",
                len(refined),
                len(raw_text),
            )
            return raw_text
        except Exception as e:
            logger.warning("Transcript refinement failed: %s", e)
            return raw_text
