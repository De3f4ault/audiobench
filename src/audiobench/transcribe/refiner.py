"""Transcript refinement using LLM post-processing.

Uses a strict auto-corrector prompt to clean up raw Whisper output
without altering content meaning. Designed to run in the background
after initial transcription completes.
"""

from __future__ import annotations

from audiobench.core.logger_factory import get_logger

logger = get_logger(__name__)

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


class TranscriptRefiner:
    """Background transcript refinement using an LLM."""

    def __init__(self, client, model: str | None = None):
        """Initialize refiner.

        Args:
            client: OllamaClient instance.
            model: Model to use for refinement (defaults to client's model).
        """
        self._client = client
        self._model = model

    def refine(self, raw_text: str) -> str:
        """Send raw transcript to LLM for cleanup.

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
                temperature=0.1,  # Low creativity for faithful correction
                think=False,  # No thinking needed for correction
            )
            refined = result.get("content", "").strip()
            if refined and len(refined) > len(raw_text) * 0.5:
                # Sanity check: refined text shouldn't be drastically shorter
                return refined
            logger.warning(
                "Refinement produced suspicious output (len %d vs raw %d), "
                "keeping raw text",
                len(refined),
                len(raw_text),
            )
            return raw_text
        except Exception as e:
            logger.warning("Transcript refinement failed: %s", e)
            return raw_text
