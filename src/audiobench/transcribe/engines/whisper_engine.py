"""Faster-whisper transcription engine implementation.

This is the primary engine, wrapping the faster-whisper library
(CTranslate2-based Whisper re-implementation) for high-performance
offline transcription.

Supports:
    - All Whisper model sizes (tiny → large-v3-turbo)
    - CPU (INT8) and CUDA (float16) inference
    - BatchedInferencePipeline for 3-5x CPU speedup
    - Word-level timestamps
    - Language auto-detection and 99+ languages
    - Both file paths and numpy arrays as input
    - Speed presets (fast / balanced / accurate)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from audiobench.core.error_types import EngineError, ModelLoadError, ModelNotFoundError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.audio_filters import collapse_repetitions, fix_broken_words
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.transcription_result import Segment, Transcript, Word

logger = get_logger("engines.whisper")

VALID_MODELS = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}


class WhisperEngine(TranscriptionEngine):
    """Transcription engine using faster-whisper (CTranslate2).

    Optimized for CPU via:
    - BatchedInferencePipeline (parallel segment processing)
    - cpu_threads tuning (match physical cores)
    - Configurable beam_size and VAD parameters
    """

    def __init__(self) -> None:
        self._model = None
        self._batched_pipeline = None
        self._model_name: str = ""
        self._device: str = "cpu"
        self._compute_type: str = "int8"
        self._cpu_threads: int = 4

    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ) -> None:
        """Load a Whisper model with optimized threading.

        Args:
            model_name: One of: tiny, base, small, medium, large-v3, large-v3-turbo.
            device: 'cpu' or 'cuda'.
            compute_type: 'int8' (CPU), 'float16' (CUDA), 'float32'.
            cpu_threads: Number of CPU threads for CTranslate2.
        """
        if model_name not in VALID_MODELS:
            raise ModelNotFoundError(model_name)

        logger.info(
            "Loading Whisper model: %s (device=%s, compute=%s, threads=%d)",
            model_name,
            device,
            compute_type,
            cpu_threads,
        )

        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )

            # Create batched pipeline for parallel segment processing
            self._batched_pipeline = BatchedInferencePipeline(model=self._model)

            self._model_name = model_name
            self._device = device
            self._compute_type = compute_type
            self._cpu_threads = cpu_threads

            logger.info("Model '%s' loaded successfully (batched pipeline ready)", model_name)

        except Exception as e:
            raise ModelLoadError(model_name, str(e)) from e

    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        beam_size: int = 3,
        batch_size: int = 4,
        temperature: float | list[float] = 0,
        compression_ratio_threshold: float = 2.4,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        condition_on_previous_text: bool = False,
        repetition_penalty: float = 1.1,
        initial_prompt: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
        on_segment: Callable[[Segment], None] | None = None,
    ) -> Transcript:
        """Transcribe audio using faster-whisper with batched inference.

        Args:
            audio: Path to WAV file or numpy array of audio samples.
            language: Language code (e.g., 'en', 'sw') or None for auto-detect.
            task: 'transcribe' or 'translate'.
            word_timestamps: Include word-level timestamps.
            beam_size: Beam search width (1=fastest, 5=most accurate).
            batch_size: Number of segments to process in parallel.
            temperature: Decoding temperature or fallback chain.
            compression_ratio_threshold: Re-decode if compression ratio exceeds this.
            no_speech_threshold: Skip segments with no_speech_prob above this.
            log_prob_threshold: Skip segments with avg_logprob below this.
            condition_on_previous_text: Feed previous segment text as context.
            repetition_penalty: Penalty for repeated tokens (1.0=off, >1=penalize).
            initial_prompt: Guide model with expected content/language context.
            progress_callback: Optional callback(percent: float) for progress updates.
            on_segment: Optional callback(segment: Segment) after each segment.

        Returns:
            Transcript with segments and word-level data.

        Raises:
            EngineError: If model not loaded or transcription fails.
        """
        if self._model is None:
            raise EngineError("No model loaded. Call load_model() first.")

        use_batched = batch_size > 1 and self._batched_pipeline is not None

        logger.info(
            "Transcribing: language=%s, task=%s, beam=%d, batch=%d, mode=%s, "
            "temperature=%s, repetition_penalty=%.1f, prompt=%s",
            language or "auto",
            task,
            beam_size,
            batch_size,
            "batched" if use_batched else "sequential",
            temperature,
            repetition_penalty,
            repr(initial_prompt[:50]) if initial_prompt else None,
        )

        start_time = time.perf_counter()

        try:
            # Optimized VAD parameters
            vad_params = dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
                min_speech_duration_ms=250,
            )

            # Common decoding parameters
            decode_params = dict(
                compression_ratio_threshold=compression_ratio_threshold,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=log_prob_threshold,
                condition_on_previous_text=condition_on_previous_text,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                initial_prompt=initial_prompt,
            )

            if use_batched:
                # ---- BATCHED MODE (3-5x faster on CPU) ----
                segments_gen, info = self._batched_pipeline.transcribe(
                    audio,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    word_timestamps=word_timestamps,
                    batch_size=batch_size,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    log_progress=False,
                    **decode_params,
                )
            else:
                # ---- SEQUENTIAL MODE (most accurate, slower) ----
                segments_gen, info = self._model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    word_timestamps=word_timestamps,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    **decode_params,
                )

            # Convert generator to our Pydantic models
            segments: list[Segment] = []
            skipped = 0
            for _idx, seg in enumerate(segments_gen):
                # Skip low-confidence garbage segments
                if seg.avg_logprob < -1.5:
                    logger.debug(
                        "Skipping low-confidence segment %d (avg_logprob=%.2f): %s",
                        _idx,
                        seg.avg_logprob,
                        seg.text[:60],
                    )
                    skipped += 1
                    continue

                # Apply text quality filters
                cleaned_text = collapse_repetitions(seg.text.strip())
                cleaned_text = fix_broken_words(cleaned_text)

                # Skip empty segments after filtering
                if not cleaned_text:
                    skipped += 1
                    continue

                words: list[Word] = []
                if word_timestamps and seg.words:
                    for w in seg.words:
                        words.append(
                            Word(
                                word=w.word,
                                start=round(w.start, 3),
                                end=round(w.end, 3),
                                probability=round(w.probability, 4),
                            )
                        )

                segments.append(
                    Segment(
                        id=len(segments),
                        text=cleaned_text,
                        start=round(seg.start, 3),
                        end=round(seg.end, 3),
                        words=words,
                        avg_logprob=round(seg.avg_logprob, 4),
                        no_speech_prob=round(seg.no_speech_prob, 4),
                    )
                )

                # Progress callback
                if progress_callback and hasattr(info, "duration"):
                    pct = min(seg.end / info.duration * 100, 100.0)
                    progress_callback(pct)

                # Live segment callback
                if on_segment:
                    on_segment(segments[-1])

            elapsed = time.perf_counter() - start_time

            # Build transcript
            duration = segments[-1].end if segments else 0.0
            transcript = Transcript(
                segments=segments,
                language=info.language,
                language_probability=round(info.language_probability, 4),
                duration_seconds=round(duration, 3),
                engine="faster-whisper",
                model_name=self._model_name,
            )

            # Speed metrics
            speed_ratio = duration / elapsed if elapsed > 0 else 0
            logger.info(
                "Transcription complete: %d segments, %d words, "
                "language=%s (%.1f%%), duration=%.1fs, "
                "processing=%.1fs, speed=%.1fx real-time",
                transcript.segment_count,
                transcript.word_count,
                transcript.language,
                transcript.language_probability * 100,
                transcript.duration_seconds,
                elapsed,
                speed_ratio,
            )

            return transcript

        except Exception as e:
            if isinstance(e, EngineError):
                raise
            raise EngineError(
                message="Transcription failed",
                details=str(e),
            ) from e

    def get_supported_languages(self) -> list[str]:
        """Return Whisper's supported language codes."""
        try:
            from faster_whisper.tokenizer import _LANGUAGE_CODES

            return sorted(_LANGUAGE_CODES)
        except ImportError:
            return [
                "af",
                "am",
                "ar",
                "as",
                "az",
                "ba",
                "be",
                "bg",
                "bn",
                "bo",
                "br",
                "bs",
                "ca",
                "cs",
                "cy",
                "da",
                "de",
                "el",
                "en",
                "es",
                "et",
                "eu",
                "fa",
                "fi",
                "fo",
                "fr",
                "gl",
                "gu",
                "ha",
                "haw",
                "he",
                "hi",
                "hr",
                "ht",
                "hu",
                "hy",
                "id",
                "is",
                "it",
                "ja",
                "jw",
                "ka",
                "kk",
                "km",
                "kn",
                "ko",
                "la",
                "lb",
                "ln",
                "lo",
                "lt",
                "lv",
                "mg",
                "mi",
                "mk",
                "ml",
                "mn",
                "mr",
                "ms",
                "mt",
                "my",
                "ne",
                "nl",
                "nn",
                "no",
                "oc",
                "pa",
                "pl",
                "ps",
                "pt",
                "ro",
                "ru",
                "sa",
                "sd",
                "si",
                "sk",
                "sl",
                "sn",
                "so",
                "sq",
                "sr",
                "su",
                "sv",
                "sw",
                "ta",
                "te",
                "tg",
                "th",
                "tk",
                "tl",
                "tr",
                "tt",
                "uk",
                "ur",
                "uz",
                "vi",
                "yi",
                "yo",
                "zh",
            ]

    def get_model_info(self) -> dict:
        """Return info about the loaded model."""
        return {
            "name": self._model_name,
            "engine": self.engine_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "cpu_threads": self._cpu_threads,
            "batched": self._batched_pipeline is not None,
            "is_loaded": self.is_loaded,
        }

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def engine_name(self) -> str:
        return "faster-whisper"
