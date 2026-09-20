"""Google Gemini transcription engine — cloud-based audio understanding.

Implements the TranscriptionEngine protocol using Google's Gemini API.
Requires: pip install google-genai

Usage:
    audiobench transcribe meeting.m4a --engine gemini
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from audiobench.core.error_types import EngineError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.transcription_result import Segment, Transcript, Word

logger = get_logger("engines.gemini")

from audiobench.core.prompts import (
    GEMINI_DIARIZATION_PROMPT,
    GEMINI_DIARIZATION_TEXT_ONLY_PROMPT,
    GEMINI_DIARIZATION_TRANSLATE_PROMPT,
    GEMINI_TRANSCRIPTION_PROMPT,
    GEMINI_TRANSCRIPTION_TEXT_ONLY_PROMPT,
    GEMINI_TRANSLATE_PROMPT,
)

# Default inline upload threshold (100 MB).
# Overridden at runtime from settings.gemini_inline_max_mb.
# Google increased the inline payload limit from 20 MB to 100 MB in early 2026.
_INLINE_MAX_BYTES_DEFAULT = 100 * 1024 * 1024

# ── Chunking constants ──────────────────────────────────────
_CHUNK_DURATION = int(2.5 * 60)  # 2.5 minutes per chunk (seconds)
_CHUNK_OVERLAP = 10  # 10 seconds overlap between chunks
_CHUNK_DURATION_ALIGN = 15 * 60  # 15 minutes per chunk for alignment
# With the Files API handling up to 2 GB per file, we relax the threshold
# to 45 minutes (from 20 min). Long files still benefit from chunking due
# to the model's practical output-token limit per request.
_CHUNK_THRESHOLD_DEFAULT = 45 * 60  # 45 minutes (seconds)

# ── Ephemeral chunk cache ────────────────────────────────────
# Stores successfully transcribed chunks as JSON so a failed run
# can be resumed without re-sending already-completed chunks.
def _get_cache_dir() -> Path:
    from audiobench.core.settings import get_settings
    return get_settings().data_dir / "cache" / "chunks"

from audiobench.core.constants import get_mime


class GeminiEngine(TranscriptionEngine):
    """Transcription engine backed by Google Gemini API.

    Upload strategy:
      - Files ≤ gemini_inline_max_mb  → inline Part.from_bytes (fast, no round-trip)
      - Files  > gemini_inline_max_mb  → Gemini Files API upload + generate_content(file_ref)

    Rate-limit resilience:
      - generate_content is wrapped with tenacity exponential-backoff
        retrying on google.genai.errors.APIError (HTTP 429).
      - On full retry exhaustion, falls back to gemini_upload_fallback_model.
    """

    def __init__(self) -> None:
        self._model_name: str = "gemini-2.5-pro"
        self._fallback_model: str = "gemini-2.0-flash"
        self._inline_max_bytes: int = _INLINE_MAX_BYTES_DEFAULT
        self._chunk_threshold: int = _CHUNK_THRESHOLD_DEFAULT
        self._max_retries: int = 6
        self._client: Any = None
        self._is_loaded = False

    # ── Protocol Implementation ─────────────────────────────

    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        device_index: int | list[int] = 0,
        **kwargs,
    ) -> None:
        """Initialize the Gemini client.

        `device` and `compute_type` are ignored (cloud engine).
        `model_name` selects the Gemini model variant.
        Settings-derived fields (inline threshold, chunk threshold, retries,
        fallback model) are applied here so they reflect the current .env.
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
        self._fallback_model = settings.gemini_upload_fallback_model
        self._inline_max_bytes = settings.gemini_inline_max_mb * 1024 * 1024
        self._chunk_threshold = settings.gemini_chunk_threshold_min * 60
        self._max_retries = settings.gemini_max_retries
        self._client = genai.Client(api_key=api_key)
        self._is_loaded = True
        logger.info(
            "Gemini engine ready: model=%s fallback=%s inline_max=%dMB chunk_threshold=%dmin retries=%d",
            self._model_name,
            self._fallback_model,
            settings.gemini_inline_max_mb,
            settings.gemini_chunk_threshold_min,
            self._max_retries,
        )

    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        beam_size: int = 5,
        on_phase: object | None = None,
        on_segment: object | None = None,
        align: bool = False,
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

        # ── Auto-trigger logic for alignment ────────────────
        from audiobench.core.settings import get_settings
        settings = get_settings()
        align_threshold = getattr(settings, "align_threshold_min", 0.5) * 60

        from audiobench.transcribe.audio_converter import probe

        try:
            info = probe(audio_path)
            duration = info.duration
        except Exception:
            duration = 0.0

        if not align and "align" not in kwargs:
            if duration > align_threshold:
                logger.info(f"File duration {duration/60:.1f}m > {align_threshold/60:.1f}m. Auto-triggering alignment.")
                align = True

        if align:
            prompt = GEMINI_DIARIZATION_TEXT_ONLY_PROMPT if diarize else GEMINI_TRANSCRIPTION_TEXT_ONLY_PROMPT
            if language and task != "translate":
                prompt += f"\nThe audio is spoken in language code: {language}\n"
            return self._transcribe_chunked_text_only(
                audio_path, prompt, on_phase, duration, on_segment, language
            )

        if diarize and task == "translate":
            prompt = GEMINI_DIARIZATION_TRANSLATE_PROMPT
        elif diarize:
            prompt = GEMINI_DIARIZATION_PROMPT
        elif task == "translate":
            prompt = GEMINI_TRANSLATE_PROMPT
        else:
            prompt = GEMINI_TRANSCRIPTION_PROMPT

        if language and task != "translate":
            prompt += f"\nThe audio is spoken in language code: {language}\n"

        if duration > self._chunk_threshold:
            return self._transcribe_chunked(
                audio_path,
                prompt,
                on_phase,
                duration,
                on_segment,
            )

        return self._transcribe_single(audio_path, prompt, on_phase, on_segment)

    # ── Chunk cache helpers ─────────────────────────────────

    @staticmethod
    def _file_identity_prefix(audio_path: Path) -> str:
        """16-char prefix that identifies this audio file at this point in time.

        Incorporates path, file size, and mtime so the prefix changes if the
        file is edited, renamed, or replaced with a different recording.
        Used to group and scan all cached chunks belonging to one file.
        """
        stat = audio_path.stat()
        raw = f"{audio_path.resolve()}\0{stat.st_size}\0{stat.st_mtime:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _chunk_cache_key(
        audio_path: Path,
        chunk_index: int,
        time_offset: float,
        model_name: str,
        prompt: str,
        language: str | None,
        task: str,
        diarize: bool,
    ) -> str:
        """Full deterministic cache key for one specific chunk."""
        prefix = GeminiEngine._file_identity_prefix(audio_path)
        chunk_raw = (
            f"{prefix}\0{chunk_index}\0{time_offset:.3f}\0"
            f"{model_name}\0{prompt}\0{language}\0{task}\0{diarize}"
        )
        chunk_hash = hashlib.sha256(chunk_raw.encode()).hexdigest()[:32]
        return f"{prefix}_{chunk_hash}"

    @staticmethod
    def _load_chunk_from_cache(key: str) -> Transcript | None:
        """Return a cached Transcript, or None on miss or any read error."""
        path = _get_cache_dir() / f"{key}.json"
        if not path.exists():
            return None
        try:
            return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Chunk cache read failed for %s (ignoring): %s", key, exc)
            return None

    @staticmethod
    def _save_chunk_to_cache(key: str, transcript: Transcript) -> None:
        """Persist a chunk transcript to disk atomically (write-then-rename)."""
        d = _get_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(transcript.model_dump_json(), encoding="utf-8")
            tmp.rename(path)
        except Exception as exc:
            logger.warning("Chunk cache write failed for %s (non-fatal): %s", key, exc)
            tmp.unlink(missing_ok=True)

    @staticmethod
    def count_cached_chunks(audio_path: Path) -> int:
        """Return how many chunk cache files exist for this audio file."""
        d = _get_cache_dir()
        if not d.exists():
            return 0
        prefix = GeminiEngine._file_identity_prefix(audio_path)
        return sum(1 for _ in d.glob(f"{prefix}_*.json"))

    @staticmethod
    def clear_chunk_cache(audio_path: Path) -> int:
        """Delete all chunk cache files for this audio file. Returns count deleted."""
        d = _get_cache_dir()
        if not d.exists():
            return 0
        prefix = GeminiEngine._file_identity_prefix(audio_path)
        count = 0
        for f in d.glob(f"{prefix}_*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count

    # ── Retry-wrapped generate call ─────────────────────────

    def _make_generate_caller(self):
        """Build a tenacity-wrapped generate_content function bound to current retry settings.

        We construct the decorator dynamically so that `_max_retries` (set in
        `load_model` from settings) is respected at call time rather than at
        class-definition time.
        """

        def is_retryable(exc):
            import socket

            from google.genai.errors import APIError

            if isinstance(exc, APIError):
                return exc.code == 429 or exc.code >= 500
            if isinstance(exc, (socket.gaierror, ConnectionError, TimeoutError, OSError)):
                return True
            return False

        @retry(
            retry=retry_if_exception(is_retryable),
            wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
            stop=stop_after_attempt(self._max_retries),
            reraise=True,
        )
        def _generate_with_retry(model_name: str, contents: list) -> Any:
            return self._client.models.generate_content(
                model=model_name,
                contents=contents,
            )

        return _generate_with_retry

    def _call_generate(self, contents: list) -> Any:
        """Call generate_content with exponential-backoff retry on 429s.

        Falls back to `_fallback_model` if the primary model exhausts all
        retry attempts (quota fully depleted for the request window).
        """
        try:
            _generate = self._make_generate_caller()
            return _generate(self._model_name, contents)
        except Exception as exc:
            # Check if it's a quota error (429) or overloaded (503) and we have a fallback.
            try:
                from google.genai.errors import APIError

                should_fallback = isinstance(exc, APIError) and exc.code in (429, 503)
            except ImportError:
                exc_str = str(exc).lower()
                should_fallback = "quota" in exc_str or "429" in exc_str or "503" in exc_str

            if should_fallback and self._fallback_model != self._model_name:
                logger.warning(
                    "Primary model %s quota exhausted after %d retries — "
                    "switching to fallback model %s",
                    self._model_name,
                    self._max_retries,
                    self._fallback_model,
                )
                _generate = self._make_generate_caller()
                return _generate(self._fallback_model, contents)

            raise EngineError(
                message="Gemini API call failed",
                details=str(exc),
            ) from exc

    # ── Upload via Files API ────────────────────────────────

    def _upload_via_files_api(
        self,
        audio_path: Path,
        mime: str,
        on_phase: object | None = None,
    ) -> tuple[str, str]:
        """Upload audio to the Gemini Files API and wait for it to become ACTIVE.

        Returns:
            (file_uri, file_name) — URI for use in generate_content,
            name for deletion after transcription.
        """
        from google.genai import types as gtypes

        size_mb = audio_path.stat().st_size / (1024 * 1024)

        if on_phase and callable(on_phase):
            on_phase(
                "uploading",
                f"Uploading {size_mb:.1f} MB to Gemini Files API...",
                None,
            )

        logger.info(
            "Uploading %s (%.1f MB) to Gemini Files API (mime=%s)",
            audio_path.name,
            size_mb,
            mime,
        )

        try:
            @retry(
                retry=retry_if_exception(lambda exc: True),  # Retry any upload failure
                wait=wait_exponential_jitter(initial=2, max=30, jitter=5),
                stop=stop_after_attempt(self._max_retries),
                reraise=True,
            )
            def _upload():
                return self._client.files.upload(
                    file=str(audio_path),
                    config=gtypes.UploadFileConfig(
                        mime_type=mime,
                        display_name=audio_path.name,
                    ),
                )
            uploaded = _upload()
        except Exception as e:
            raise EngineError(
                message="Gemini Files API upload failed",
                details=str(e),
            ) from e

        logger.info("Upload complete: file_name=%s uri=%s", uploaded.name, uploaded.uri)

        # Poll until the file reaches ACTIVE state (server-side processing).
        if on_phase and callable(on_phase):
            on_phase("processing", "Gemini is processing the audio file...", None)

        max_polls = 30  # up to ~60 seconds
        poll_interval = 2.0  # seconds between polls

        for attempt in range(max_polls):
            try:
                file_meta = self._client.files.get(name=uploaded.name)
            except Exception as e:
                logger.warning("Files API status poll failed (attempt %d): %s", attempt + 1, e)
                time.sleep(poll_interval)
                continue

            state = getattr(file_meta, "state", None)
            state_name = state.name if hasattr(state, "name") else str(state)

            if state_name == "ACTIVE":
                logger.info(
                    "File %s is ACTIVE after %d poll(s)",
                    uploaded.name,
                    attempt + 1,
                )
                return uploaded.uri, uploaded.name

            if state_name == "FAILED":
                raise EngineError(
                    message="Gemini file processing failed",
                    details=f"File {uploaded.name} entered FAILED state: {file_meta}",
                )

            logger.debug(
                "File %s state=%s — polling again in %.0fs",
                uploaded.name,
                state_name,
                poll_interval,
            )
            time.sleep(poll_interval)

        raise EngineError(
            message="Gemini file processing timed out",
            details=(
                f"File {uploaded.name} did not reach ACTIVE state "
                f"within {max_polls * poll_interval:.0f}s."
            ),
        )

    # ── Single-file transcription (inline or Files API) ─────

    def _transcribe_single(
        self,
        audio_path: Path,
        prompt: str,
        on_phase: object | None = None,
        on_segment: object | None = None,
    ) -> Transcript:
        """Transcribe a single (non-chunked) audio file.

        Routes to the inline path for files ≤ gemini_inline_max_mb, or the
        Gemini Files API for larger files.  Both paths use `_call_generate`
        which wraps generate_content with tenacity exponential-backoff retry
        and an automatic fallback model on quota exhaustion.
        """
        from google.genai import types

        file_size = audio_path.stat().st_size
        mime = get_mime(audio_path.suffix)
        size_mb = file_size / (1024 * 1024)

        logger.info(
            "Sending to Gemini: file=%s size=%.1fMB mime=%s model=%s",
            audio_path.name,
            size_mb,
            mime,
            self._model_name,
        )

        file_name_to_delete: str | None = None

        try:
            if file_size <= self._inline_max_bytes:
                # ── Inline path (small files) ────────────────────────
                logger.info(
                    "Using inline upload (%.1f MB ≤ %d MB threshold)",
                    size_mb,
                    self._inline_max_bytes // (1024 * 1024),
                )
                audio_bytes = audio_path.read_bytes()

                contents = [
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ]
            else:
                # ── Files API path (large files) ─────────────────────
                logger.info(
                    "Using Files API upload (%.1f MB > %d MB threshold)",
                    size_mb,
                    self._inline_max_bytes // (1024 * 1024),
                )
                file_uri, file_name_to_delete = self._upload_via_files_api(
                    audio_path, mime, on_phase
                )

                if on_phase and callable(on_phase):
                    on_phase("transcribing", "Transcribing...", 0.0)

                contents = [
                    types.Content(
                        parts=[
                            types.Part.from_uri(file_uri=file_uri, mime_type=mime),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ]

            response = self._call_generate(contents)

        finally:
            # Clean up the uploaded file immediately to free project storage
            # quota. Files auto-expire after 48 h anyway, but being explicit
            # is better hygiene when processing many files.
            if file_name_to_delete:
                try:
                    self._client.files.delete(name=file_name_to_delete)
                    logger.info("Deleted Files API entry: %s", file_name_to_delete)
                except Exception as del_exc:
                    # Non-fatal — the file will expire on its own.
                    logger.warning(
                        "Could not delete Files API entry %s: %s",
                        file_name_to_delete,
                        del_exc,
                    )

        transcript = self._parse_response(response, audio_path)
        if on_segment and callable(on_segment):
            for seg in transcript.segments:
                on_segment(seg)
        return transcript

    def _transcribe_chunked(
        self,
        audio_path: Path,
        prompt: str,
        on_phase: object | None = None,
        total_duration: float = 0.0,
        on_segment: object | None = None,
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
                    i + 1,
                    len(chunks),
                    time_offset,
                    chunk_path.name,
                )

                try:
                    cache_key = self._chunk_cache_key(
                        audio_path, i, time_offset,
                        model_name=self._model_name,
                        prompt=prompt,
                        language=None,
                        task="transcribe",
                        diarize=False,
                    )
                    transcript = self._load_chunk_from_cache(cache_key)
                    if transcript is not None:
                        logger.info(
                            "Chunk %d/%d loaded from cache (skipping API call)",
                            i + 1, len(chunks),
                        )
                        if on_phase and callable(on_phase):
                            on_phase(
                                "transcribing",
                                f"Chunk {i + 1}/{len(chunks)} loaded from cache",
                                (i + 1) / len(chunks),
                            )
                    else:
                        transcript = self._transcribe_single(
                            chunk_path,
                            prompt,
                            None,
                            None,
                        )
                        self._save_chunk_to_cache(cache_key, transcript)
                    chunk_results.append((transcript, time_offset))
                except EngineError as e:
                    logger.error(
                        "Chunk %d/%d failed — aborting run: %s",
                        i + 1,
                        len(chunks),
                        e,
                    )
                    raise
        finally:
            # Clean up chunk temp files (but not the original).
            if chunk_dir and chunk_dir != audio_path.parent:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        if not chunk_results:
            raise EngineError(
                message="All chunks failed",
                details="Every chunk transcription attempt failed.",
            )

        transcript = self._stitch_transcripts(chunk_results, audio_path)
        if on_segment and callable(on_segment):
            for seg in transcript.segments:
                on_segment(seg)
        return transcript

    def _dedup_chunk_boundary(self, prev_segments: list[Segment], next_segments: list[Segment]) -> list[Segment]:
        """Deduplicate segments exactly matching at the boundary of two 15-minute chunks."""
        if not prev_segments or not next_segments:
            return next_segments

        # Get the last 3 segments from previous chunk to compare against
        # first 3 segments of next chunk
        prev_tail = [s.text.strip().lower() for s in prev_segments[-3:]]

        drop_count = 0
        for i in range(min(3, len(next_segments))):
            next_text = next_segments[i].text.strip().lower()
            if next_text in prev_tail:
                drop_count += 1
                logger.info(f"Dropped duplicate boundary segment: '{next_text}'")
            else:
                break

        return next_segments[drop_count:]

    def _transcribe_chunked_text_only(
        self,
        audio_path: Path,
        prompt: str,
        on_phase: object | None = None,
        total_duration: float = 0.0,
        on_segment: object | None = None,
        language: str | None = None,
    ) -> Transcript:
        """Split long audio into 15-minute chunks and transcribe with text-only prompts."""
        import shutil

        from audiobench.transcribe.audio_converter import split_audio

        chunks = split_audio(
            audio_path,
            chunk_duration=_CHUNK_DURATION_ALIGN,
            overlap=_CHUNK_OVERLAP,
        )

        logger.info(
            "Text-only pipeline: Chunked %s into %d parts (15-min each)",
            audio_path.name,
            len(chunks),
        )

        all_segments: list[Segment] = []
        chunk_dir = chunks[0][0].parent if chunks else None

        detected_language = language or "en"
        total_words = 0

        try:
            for i, (chunk_path, time_offset) in enumerate(chunks):
                if on_phase and callable(on_phase):
                    on_phase(
                        "transcribing",
                        f"Transcribing chunk {i + 1}/{len(chunks)}...",
                        i / len(chunks),
                    )

                logger.info("Transcribing chunk %d/%d (15-min): %s", i + 1, len(chunks), chunk_path.name)

                try:
                    cache_key = self._chunk_cache_key(
                        audio_path, i, time_offset,
                        model_name=self._model_name,
                        prompt=prompt,
                        language=language,
                        task="transcribe",
                        diarize=False,
                    )
                    chunk_transcript = self._load_chunk_from_cache(cache_key)
                    if chunk_transcript is not None:
                        logger.info(
                            "Chunk %d/%d loaded from cache (skipping API call)",
                            i + 1, len(chunks),
                        )
                        if on_phase and callable(on_phase):
                            on_phase(
                                "transcribing",
                                f"Chunk {i + 1}/{len(chunks)} loaded from cache",
                                (i + 1) / len(chunks),
                            )
                    else:
                        chunk_transcript = self._transcribe_single(
                            chunk_path,
                            prompt,
                            None,
                            None,  # No on_segment yet because timestamps are 0.0
                        )
                        self._save_chunk_to_cache(cache_key, chunk_transcript)

                    if chunk_transcript.language and chunk_transcript.language != "en":
                        detected_language = chunk_transcript.language

                    raw_segments = chunk_transcript.segments

                    if all_segments:
                        raw_segments = self._dedup_chunk_boundary(all_segments, raw_segments)

                    if not raw_segments:
                        continue

                    for seg in raw_segments:
                        seg.id = len(all_segments)
                        # We don't have real timestamps yet, so we just use 0.0 or approximations
                        # The alignment pipeline will fix these.
                        seg.start = 0.0
                        seg.end = 0.0

                        all_segments.append(seg)
                        total_words += len(seg.text.split())

                except EngineError as e:
                    # Re-raise immediately — same reasoning as _transcribe_chunked:
                    # post-tenacity errors are session-fatal and skipping produces
                    # a silently incomplete text corpus fed to the alignment pipeline.
                    logger.error(
                        "Chunk %d/%d failed — aborting text-only run: %s",
                        i + 1,
                        len(chunks),
                        e,
                    )
                    raise

        finally:
            if chunk_dir and chunk_dir != audio_path.parent:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        if not all_segments:
            raise EngineError("All chunks failed in text-only pipeline.")

        return Transcript(
            text=" ".join(s.text.strip() for s in all_segments),
            segments=all_segments,
            language=detected_language,
            language_probability=0.99,
            duration_seconds=total_duration,
            word_count=total_words,
            file_name=audio_path.name,
            file_hash="",
        )

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
            len(chunk_results),
            len(all_segments),
            total_words,
            duration,
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
        search_region = raw[seg_match.end() :]

        # Find all closing braces that are followed by a comma or by
        # another opening brace (next segment) — these mark complete
        # segment boundaries.
        last_good = -1
        depth = 0
        i = 0
        while i < len(search_region):
            ch = search_region[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # This `}` closes a top-level object in the array.
                    last_good = seg_match.end() + i
            elif ch == '"':
                # Skip over strings to avoid counting braces inside them.
                i += 1
                while i < len(search_region) and search_region[i] != '"':
                    if search_region[i] == "\\":
                        i += 1  # skip escaped char
                    i += 1
            i += 1

        if last_good == -1:
            return None

        # Rebuild: everything up to and including the last good `}`,
        # then close the array and outer object.
        repaired = raw[: last_good + 1] + "\n  ]\n}"

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
            if raw[trim_pos] == "}":
                candidate = raw[: trim_pos + 1] + "\n  ]\n}"
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        return None

    def _parse_response(self, response, audio_path: Path) -> Transcript:
        """Parse Gemini's JSON response into a Transcript."""
        # ── Guard: check for blocked / empty responses ────────────────────────
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            finish_name = finish_reason.name if hasattr(finish_reason, "name") else str(finish_reason)
            if finish_name not in ("STOP", "1", "FINISH_REASON_STOP"):
                safety_ratings = getattr(candidate, "safety_ratings", [])
                logger.warning(
                    "Gemini response finish_reason=%s for %s — safety_ratings=%s",
                    finish_name,
                    audio_path.name,
                    safety_ratings,
                )
                if finish_name in ("SAFETY", "RECITATION"):
                    raise EngineError(
                        message=f"Gemini blocked the response (finish_reason={finish_name})",
                        details=(
                            f"File: {audio_path.name}. "
                            "The audio may have triggered a content safety filter. "
                            f"Safety ratings: {safety_ratings}"
                        ),
                    )

        # Safe text extraction — response.text raises if content is empty
        try:
            raw_text = response.text
        except Exception:
            raw_text = ""
            for cand in candidates:
                for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        raw_text += t

        if not raw_text or not raw_text.strip():
            logger.error(
                "Gemini returned empty text for %s. candidates=%s",
                audio_path.name,
                candidates,
            )
            raise EngineError(
                message="Gemini returned an empty transcription",
                details=(
                    f"No text content in response for '{audio_path.name}'. "
                    "This can happen due to safety filters, an unsupported audio "
                    "encoding, or a transient API error. Check the log for finish_reason."
                ),
            )

        raw_text = raw_text.strip()

        # Strip markdown fences if Gemini wraps the JSON.
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```\s*$", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(
                "Gemini response is not valid JSON (likely truncated at "
                "output-token limit). Attempting repair with json_repair…"
            )

            data = None
            try:
                import json_repair
                repaired = json_repair.repair_json(raw_text, return_objects=True)
                # json_repair may return a bare list — normalise it first.
                if isinstance(repaired, list):
                    repaired = {"segments": repaired}
                # json_repair returns {} or "" for completely unrecoverable input,
                # and may return a dict with no segments on partial failure.
                # Accept it only if it has at least one segment.
                if isinstance(repaired, dict) and repaired.get("segments"):
                    data = repaired
                    logger.warning(
                        "json_repair succeeded — recovered %d segments",
                        len(data["segments"]),
                    )
                else:
                    logger.warning(
                        "json_repair returned unusable result (%r), trying manual repair",
                        type(repaired).__name__,
                    )
            except ImportError:
                logger.warning("json-repair not installed")

            # Last resort: manual truncation repair
            if data is None:
                data = self._repair_truncated_json(raw_text)
                if isinstance(data, dict) and data.get("segments"):
                    logger.warning(
                        "Manual repair succeeded — recovered %d segments (tail may be missing)",
                        len(data["segments"]),
                    )
                else:
                    logger.error("JSON repair failed. First 500 chars: %s", raw_text[:500])
                    raise EngineError(
                        message="Failed to parse Gemini transcription response",
                        details=f"Invalid JSON: {e}. Raw response: {raw_text[:200]}...",
                    ) from e

        # Gemini occasionally returns a bare JSON array instead of the expected
        # {"language": ..., "segments": [...]} envelope.  Normalise it here so
        # the rest of the parsing code can always call data.get(...) safely.
        if isinstance(data, list):
            logger.warning(
                "Gemini returned a bare JSON array — wrapping into segments envelope"
            )
            data = {"segments": data}

        language = data.get("language", "en")
        raw_segments = data.get("segments", [])

        segments: list[Segment] = []
        full_text_parts: list[str] = []
        total_words = 0

        def _collapse_repetitions(text: str, max_repeats: int = 2) -> str:
            import re
            # Split by common sentence terminators and keep the delimiters attached.
            # Using a regex that splits after ., !, or ? followed by space(s).
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [p.strip() for p in parts if p.strip()]

            if not sentences:
                return text

            deduped = []
            current_sentence = None
            count = 0

            for s in sentences:
                s_lower = s.lower()
                if s_lower == current_sentence:
                    count += 1
                    if count <= max_repeats:
                        deduped.append(s)
                else:
                    current_sentence = s_lower
                    count = 1
                    deduped.append(s)

            # If the entire text was just one sentence repeated without punctuation,
            # this basic filter won't catch it, but most LLM loops include punctuation.
            return " ".join(deduped)

        def _safe_float(val: Any, default: float = 0.0) -> float:
            try:
                return float(val)
            except (ValueError, TypeError):
                # If the LLM generated something like "1.0.9", try to strip the second decimal
                if isinstance(val, str) and val.count(".") > 1:
                    parts = val.split(".")
                    try:
                        return float(f"{parts[0]}.{parts[1]}")
                    except (ValueError, TypeError):
                        pass
                return default

        def _safe_int(val: Any, default: int = 0) -> int:
            # Gemini sometimes returns [11] (a list) instead of 11 (int)
            if isinstance(val, list):
                val = val[0] if val else default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        last_seg_text = None
        for seg_data in raw_segments:
            seg_text = seg_data.get("text", "").strip()

            if seg_text:
                seg_text = _collapse_repetitions(seg_text)

            if not seg_text or seg_text.lower() == last_seg_text:
                continue

            last_seg_text = seg_text.lower()

            words: list[Word] = []
            for w in seg_data.get("words", []):
                words.append(
                    Word(
                        word=w.get("word", ""),
                        start=_safe_float(w.get("start", 0.0)),
                        end=_safe_float(w.get("end", 0.0)),
                        probability=1.0,
                    )
                )

            segment = Segment(
                id=_safe_int(seg_data.get("id", len(segments)), default=len(segments)),
                start=_safe_float(seg_data.get("start", 0.0)),
                end=_safe_float(seg_data.get("end", 0.0)),
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
