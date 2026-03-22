"""Google Gemini transcription engine — cloud-based audio understanding.

Implements the TranscriptionEngine protocol using Google's Gemini API.
Requires: pip install google-genai

Usage:
    audiobench transcribe meeting.m4a --engine gemini
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from audiobench.core.error_types import EngineError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.transcription_result import Segment, Transcript, Word

logger = get_logger("engines.gemini")

# Prompt that instructs Gemini to return structured transcription JSON.
_TRANSCRIPTION_PROMPT = """\
Transcribe the following audio accurately and completely.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Return raw JSON only. No explanation, no markdown fences.
"""

_DIARIZATION_PROMPT = """\
Transcribe the following audio accurately and completely, identifying each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "speaker": "Speaker 1",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Return raw JSON only. No explanation, no markdown fences.
"""

_TRANSLATE_PROMPT = """\
Transcribe the following audio and translate everything to English.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "words": []
    }
  ]
}

Rules:
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- Return raw JSON only. No explanation, no markdown fences.
"""

_DIARIZATION_TRANSLATE_PROMPT = """\
Transcribe the following audio, translate everything to English, and identify each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "speaker": "Speaker 1",
      "words": []
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- Return raw JSON only. No explanation, no markdown fences.
"""

# Maximum inline upload size (20 MB). Larger files use the Files API.
_INLINE_MAX_BYTES = 20 * 1024 * 1024

# ── Chunking constants ──────────────────────────────────────
_CHUNK_DURATION = 15 * 60      # 15 minutes per chunk (seconds)
_CHUNK_OVERLAP = 30            # 30 seconds overlap between chunks
_CHUNK_THRESHOLD = 20 * 60     # Only chunk files longer than 20 minutes

# Map common audio extensions to MIME types.
_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
}


def _get_mime(path: Path) -> str:
    """Resolve MIME type from file extension."""
    return _MIME_MAP.get(path.suffix.lower(), "audio/mpeg")


class GeminiEngine(TranscriptionEngine):
    """Transcription engine backed by Google Gemini API."""

    def __init__(self) -> None:
        self._model_name: str = "gemini-2.5-pro"
        self._client = None
        self._is_loaded = False

    # ── Protocol Implementation ─────────────────────────────

    def load_model(
        self,
        model_name: str = "gemini-2.5-pro",
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ) -> None:
        """Initialize the Gemini client.

        `device` and `compute_type` are ignored (cloud engine).
        `model_name` selects the Gemini model variant.
        """
        try:
            from google import genai
        except ImportError:
            raise EngineError(
                message="Google GenAI SDK not installed",
                details="Install with: pip install google-genai",
            ) from None

        from audiobench.core.settings import get_settings

        settings = get_settings()
        api_key = settings.gemini_api_key

        if not api_key:
            raise EngineError(
                message="Gemini API key not configured",
                details=(
                    "Set AUDIOBENCH_GEMINI_API_KEY in your .env file "
                    "or environment. Get a free key at https://aistudio.google.com/apikey"
                ),
            )

        self._model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._is_loaded = True
        logger.info("Gemini engine ready: model=%s", self._model_name)

    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        beam_size: int = 5,
        on_phase: object | None = None,
        **kwargs,
    ) -> Transcript:
        """Send audio to Gemini and return a Transcript.

        For files longer than _CHUNK_THRESHOLD (20 min), the audio is
        automatically split into overlapping chunks, each transcribed
        independently, and the results are stitched back together.
        """
        if not self._is_loaded or self._client is None:
            raise EngineError(
                message="Engine not loaded",
                details="Call load_model() before transcribe()",
            )

        # Handle numpy arrays by writing to a temp WAV file.
        if isinstance(audio, np.ndarray):
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, samplerate=16000)
                audio_path = Path(tmp.name)
        else:
            audio_path = Path(audio)

        if not audio_path.exists():
            raise EngineError(
                message=f"Audio file not found: {audio_path}",
            )

        # Choose prompt based on task and diarization.
        diarize = kwargs.get("diarize", False)
        if diarize and task == "translate":
            prompt = _DIARIZATION_TRANSLATE_PROMPT
        elif diarize:
            prompt = _DIARIZATION_PROMPT
        elif task == "translate":
            prompt = _TRANSLATE_PROMPT
        else:
            prompt = _TRANSCRIPTION_PROMPT

        if language and task != "translate":
            prompt += f"\nThe audio is spoken in language code: {language}\n"

        # ── Check if chunking is needed ─────────────────────
        from audiobench.transcribe.audio_converter import probe

        try:
            info = probe(audio_path)
            duration = info.duration
        except Exception:
            duration = 0.0

        if duration > _CHUNK_THRESHOLD:
            return self._transcribe_chunked(
                audio_path, prompt, on_phase, duration,
            )

        return self._transcribe_single(audio_path, prompt, on_phase)

    def _transcribe_single(
        self,
        audio_path: Path,
        prompt: str,
        on_phase: object | None = None,
    ) -> Transcript:
        """Transcribe a single (non-chunked) audio file."""
        from google.genai import types

        file_size = audio_path.stat().st_size
        mime = _get_mime(audio_path)

        logger.info(
            "Sending to Gemini: file=%s size=%d mime=%s model=%s",
            audio_path.name,
            file_size,
            mime,
            self._model_name,
        )

        try:
            if file_size > _INLINE_MAX_BYTES and on_phase and callable(on_phase):
                size_mb = file_size / (1024 * 1024)
                on_phase(
                    "uploading",
                    f"Uploading {size_mb:.0f} MB to Gemini...",
                    None,
                )

            audio_bytes = audio_path.read_bytes()
            logger.info("Sending %d bytes inline to Gemini", len(audio_bytes))

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ],
            )

        except Exception as e:
            raise EngineError(
                message="Gemini API call failed",
                details=str(e),
            ) from e

        return self._parse_response(response, audio_path)

    def _transcribe_chunked(
        self,
        audio_path: Path,
        prompt: str,
        on_phase: object | None = None,
        total_duration: float = 0.0,
    ) -> Transcript:
        """Split long audio into chunks, transcribe each, and stitch."""
        import shutil

        from audiobench.transcribe.audio_converter import split_audio

        chunks = split_audio(
            audio_path,
            chunk_duration=_CHUNK_DURATION,
            overlap=_CHUNK_OVERLAP,
        )

        logger.info(
            "Chunked %s into %d parts (%.0f min total)",
            audio_path.name,
            len(chunks),
            total_duration / 60,
        )

        chunk_results: list[tuple[Transcript, float]] = []
        chunk_dir = chunks[0][0].parent if chunks else None

        try:
            for i, (chunk_path, time_offset) in enumerate(chunks):
                if on_phase and callable(on_phase):
                    on_phase(
                        "transcribing",
                        f"Transcribing chunk {i + 1}/{len(chunks)}...",
                        i / len(chunks),
                    )

                logger.info(
                    "Transcribing chunk %d/%d (offset=%.0fs): %s",
                    i + 1, len(chunks), time_offset, chunk_path.name,
                )

                try:
                    # Don't forward on_phase to individual chunks —
                    # _transcribe_chunked already emits chunk-level progress.
                    transcript = self._transcribe_single(
                        chunk_path, prompt, None,
                    )
                    chunk_results.append((transcript, time_offset))
                except EngineError as e:
                    logger.warning(
                        "Chunk %d/%d failed (skipping): %s",
                        i + 1, len(chunks), e,
                    )
        finally:
            # Clean up chunk temp files (but not the original).
            if chunk_dir and chunk_dir != audio_path.parent:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        if not chunk_results:
            raise EngineError(
                message="All chunks failed",
                details="Every chunk transcription attempt failed.",
            )

        return self._stitch_transcripts(chunk_results, audio_path)

    @staticmethod
    def _stitch_transcripts(
        chunk_results: list[tuple[Transcript, float]],
        audio_path: Path,
    ) -> Transcript:
        """Merge chunked transcripts into one, offset-adjusting timestamps.

        For overlapping regions, segments from the later chunk whose start
        time (after offset) falls within the previous chunk's last segment
        end time are dropped to avoid duplication.
        """
        all_segments: list[Segment] = []
        language = "en"
        last_end = 0.0  # tracks the end time of the last accepted segment

        for transcript, time_offset in chunk_results:
            language = transcript.language  # use last non-empty

            for seg in transcript.segments:
                # Offset timestamps to the original audio timeline.
                adjusted_start = seg.start + time_offset
                adjusted_end = seg.end + time_offset

                # Overlap dedup: skip segments that start before the
                # end of the last accepted segment (they are from the
                # overlapping region and already covered).
                if adjusted_start < last_end - 1.0:  # 1s tolerance
                    continue

                # Enforce monotonically increasing timestamps —
                # Gemini's chunk-relative times can be imprecise,
                # causing backwards jumps after offset.
                if adjusted_start < last_end:
                    adjusted_start = last_end
                if adjusted_end <= adjusted_start:
                    adjusted_end = adjusted_start + (seg.end - seg.start)

                # Offset word timestamps too.
                adjusted_words = [
                    Word(
                        word=w.word,
                        start=max(w.start + time_offset, adjusted_start),
                        end=max(w.end + time_offset, adjusted_start),
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

                new_seg = Segment(
                    id=len(all_segments),
                    start=adjusted_start,
                    end=adjusted_end,
                    text=seg.text,
                    words=adjusted_words,
                    speaker=seg.speaker,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                )
                all_segments.append(new_seg)
                last_end = adjusted_end

        duration = all_segments[-1].end if all_segments else 0.0
        total_words = sum(len(s.text.split()) for s in all_segments)

        logger.info(
            "Stitched %d chunks → %d segments, %d words, %.0fs",
            len(chunk_results), len(all_segments), total_words, duration,
        )

        return Transcript(
            text=" ".join(s.text.strip() for s in all_segments),
            segments=all_segments,
            language=language,
            language_probability=0.99,
            duration_seconds=duration,
            word_count=total_words,
            file_name=audio_path.name,
            file_hash="",
        )

    def get_supported_languages(self) -> list[str]:
        """Gemini supports 70+ languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "ru",
            "zh",
            "ja",
            "ko",
            "ar",
            "hi",
            "sw",
            "pl",
            "tr",
            "vi",
            "th",
            "id",
            "cs",
            "ro",
            "hu",
            "el",
            "da",
            "fi",
            "no",
            "sv",
            "he",
            "uk",
            "bg",
        ]

    def get_model_info(self) -> dict:
        return {
            "name": self._model_name,
            "engine": "gemini",
            "device": "cloud",
            "compute_type": "n/a",
            "size_mb": 0,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def engine_name(self) -> str:
        return "gemini"

    # ── Response Parsing ────────────────────────────────────

    @staticmethod
    def _repair_truncated_json(raw: str) -> dict | None:
        """Try to salvage a truncated JSON response.

        When Gemini hits its output-token limit the JSON is cut off
        mid-object.  We find the last *complete* segment object (the
        last ``}`` that closes a segment before the break), trim
        everything after it, and close the ``]}`` to make the JSON
        valid again.

        Returns the parsed dict, or None if repair fails.
        """
        # Strategy: find the last `}` that is followed by either `,`
        # or `]` (i.e. a properly closed segment boundary), discard
        # everything after it, and close the structure.

        # Locate the "segments" array opening.
        seg_match = re.search(r'"segments"\s*:\s*\[', raw)
        if not seg_match:
            return None

        # Walk backwards from the end to find the last `}` that ends
        # a complete segment.  A complete segment ends with `}` and
        # the next non-whitespace character (before truncation) would
        # be `,` or `]`.
        search_region = raw[seg_match.end():]

        # Find all closing braces that are followed by a comma or by
        # another opening brace (next segment) — these mark complete
        # segment boundaries.
        last_good = -1
        depth = 0
        i = 0
        while i < len(search_region):
            ch = search_region[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # This `}` closes a top-level object in the array.
                    last_good = seg_match.end() + i
            elif ch == '"':
                # Skip over strings to avoid counting braces inside them.
                i += 1
                while i < len(search_region) and search_region[i] != '"':
                    if search_region[i] == '\\':
                        i += 1  # skip escaped char
                    i += 1
            i += 1

        if last_good == -1:
            return None

        # Rebuild: everything up to and including the last good `}`,
        # then close the array and outer object.
        repaired = raw[:last_good + 1] + "\n  ]\n}"

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # ── Brute-force fallback ─────────────────────────────
        # If the structured repair still fails (e.g. truncation inside
        # an escaped string that unbalances quotes), try progressively
        # trimming from the last good position backwards until we find
        # a parseable prefix.
        for trim_pos in range(last_good, seg_match.end(), -1):
            if raw[trim_pos] == '}':
                candidate = raw[:trim_pos + 1] + "\n  ]\n}"
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        return None

    def _parse_response(self, response, audio_path: Path) -> Transcript:
        """Parse Gemini's JSON response into a Transcript."""
        raw_text = response.text.strip()

        # Strip markdown fences if Gemini wraps the JSON.
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```\s*$", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(
                "Gemini response is not valid JSON (likely truncated at "
                "output-token limit). Attempting repair…"
            )
            data = self._repair_truncated_json(raw_text)
            if data is None:
                logger.error("JSON repair failed. First 500 chars: %s", raw_text[:500])
                raise EngineError(
                    message="Failed to parse Gemini transcription response",
                    details=f"Invalid JSON: {e}. Raw response: {raw_text[:200]}...",
                ) from e
            logger.warning(
                "Repair succeeded — recovered %d segments (tail may be missing)",
                len(data.get("segments", [])),
            )

        language = data.get("language", "en")
        raw_segments = data.get("segments", [])

        segments: list[Segment] = []
        full_text_parts: list[str] = []
        total_words = 0

        for seg_data in raw_segments:
            seg_text = seg_data.get("text", "").strip()
            if not seg_text:
                continue

            words: list[Word] = []
            for w in seg_data.get("words", []):
                words.append(
                    Word(
                        word=w.get("word", ""),
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        probability=1.0,
                    )
                )

            segment = Segment(
                id=seg_data.get("id", len(segments)),
                start=float(seg_data.get("start", 0.0)),
                end=float(seg_data.get("end", 0.0)),
                text=seg_text,
                words=words,
                speaker=seg_data.get("speaker"),
                avg_logprob=0.0,
                no_speech_prob=0.0,
            )
            segments.append(segment)
            full_text_parts.append(seg_text)
            total_words += len(seg_text.split())

        # Compute duration from last segment end, or 0 if empty.
        duration = segments[-1].end if segments else 0.0

        return Transcript(
            text=" ".join(full_text_parts),
            segments=segments,
            language=language,
            language_probability=0.99,
            duration_seconds=duration,
            word_count=total_words,
            file_name=audio_path.name,
            file_hash="",
        )
