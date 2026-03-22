"""Pipeline orchestrator — the main entry point for transcription.

Chains: load → transcribe → store → format → output

Emits phase callbacks for UI progress:
    on_phase("loading", "Loading model...")
    on_phase("converting", "Converting audio...")
    on_phase("transcribing", "Transcribing...", progress=0.42)
    on_phase("saving", "Saving to database...")
    on_phase("done", "Complete!")
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from audiobench.core.db_engine import init_db
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.audio_converter import AudioLoader
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.engines.engine_registry import create_engine
from audiobench.transcribe.transcription_result import AudioMetadata, Segment, Transcript

logger = get_logger("core.pipeline")

# Callback types
PhaseCallback = Callable[[str, str, float | None], None]
SegmentCallback = Callable[[Segment], None]


class TranscriptionPipeline:
    """Orchestrates the full transcription workflow."""

    def __init__(
        self,
        engine: TranscriptionEngine | None = None,
        repository: TranscriptionRepository | None = None,
    ) -> None:
        self._engine = engine
        self._repository = repository or TranscriptionRepository()
        self._settings = get_settings()
        self._db_initialized = False

    def _ensure_engine(
        self,
        on_phase: PhaseCallback | None = None,
        engine_name: str | None = None,
    ) -> TranscriptionEngine:
        """Lazy-init engine from settings if not provided."""
        if self._engine is None:
            selected = engine_name or self._settings.engine

            if on_phase:
                label = "Connecting to Gemini..." if selected == "gemini" else "Loading model..."
                on_phase("loading", label, None)

            self._engine = create_engine(
                engine_name=selected,
                model_name=(
                    self._settings.gemini_model
                    if selected == "gemini"
                    else self._settings.model_name
                ),
                device=self._settings.resolve_device(),
                compute_type=self._settings.resolve_compute_type(),
                cpu_threads=self._settings.resolve_cpu_threads(),
            )
        return self._engine

    def _ensure_db(self) -> None:
        """Ensure database tables exist."""
        if not self._db_initialized:
            init_db()
            self._db_initialized = True

    def transcribe_file(
        self,
        file_path: str | Path,
        language: str | None = None,
        output_format: str | None = None,
        output_path: str | None = None,
        word_timestamps: bool | None = None,
        skip_cache: bool = False,
        speed_preset: str | None = None,
        initial_prompt: str | None = None,
        translate: bool = False,
        enable_diarization: bool = False,
        on_phase: PhaseCallback | None = None,
        on_segment: SegmentCallback | None = None,
        filters: list[str] | None = None,
        engine_name: str | None = None,
    ) -> Transcript:
        """Transcribe an audio file through the full pipeline.

        Args:
            file_path: Path to audio/video file.
            language: Language code or None for auto-detect.
            output_format: Override default format (txt/srt/vtt/json).
            output_path: Write output to file; None = return only.
            word_timestamps: Override setting.
            skip_cache: If True, skip dedup check and re-transcribe.
            speed_preset: Override speed preset (fast/balanced/accurate).
            on_phase: Callback for phase updates (phase, message, progress).

        Returns:
            Transcript result.
        """
        self._ensure_db()
        engine = self._ensure_engine(on_phase, engine_name=engine_name)
        is_gemini = engine.engine_name == "gemini"

        fmt = output_format or self._settings.output_format
        word_ts = word_timestamps if word_timestamps is not None else self._settings.word_timestamps
        preset = speed_preset or self._settings.speed_preset
        beam = self._settings.resolve_beam_size(preset)
        batch = self._settings.resolve_batch_size(preset)
        temperature = self._settings.resolve_temperature(preset)
        condition_on_prev = self._settings.resolve_condition_on_previous_text(preset)

        # Step 1: Load audio
        if on_phase:
            on_phase("converting", "Converting audio...", None)

        logger.info(
            "Pipeline: loading %s (preset=%s, beam=%d, batch=%d)",
            file_path,
            preset,
            beam,
            batch,
        )

        with AudioLoader() as loader:
            wav_path, metadata = loader.load(file_path, filters=filters)

            # Step 1.5: Check cache
            if not skip_cache and metadata.file_hash:
                cached = self._repository.find_by_hash(metadata.file_hash)
                if cached:
                    logger.info("Pipeline: cache hit for hash %s", metadata.file_hash[:12])
                    cached_data = self._repository.get_by_id(cached.id)
                    if cached_data:
                        if on_phase:
                            on_phase("done", "Retrieved from cache", None)
                        return self._reconstruct_transcript(cached_data, metadata)

            # Step 2: Transcribe
            task = "translate" if translate else "transcribe"
            if on_phase:
                label = "Translating to English..." if translate else "Transcribing..."
                on_phase("transcribing", label, 0.0)

            logger.info("Pipeline: transcribing with %s (preset=%s)", engine.engine_name, preset)

            def progress_bridge(pct: float) -> None:
                if on_phase:
                    on_phase("transcribing", "Transcribing...", pct)

            # Gemini takes the raw file directly; whisper uses the converted WAV.
            audio_input = str(file_path) if is_gemini else wav_path

            if is_gemini:
                # Gemini only needs the core arguments.
                # Diarization is handled natively via prompt.
                transcript = engine.transcribe(
                    audio_input,
                    language=language or self._settings.language,
                    task=task,
                    word_timestamps=word_ts,
                    on_phase=on_phase,
                    diarize=enable_diarization,
                )
            else:
                transcript = engine.transcribe(
                    audio_input,
                    language=language or self._settings.language,
                    task=task,
                    word_timestamps=word_ts,
                    beam_size=beam,
                    batch_size=batch,
                    temperature=temperature,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=condition_on_prev,
                    repetition_penalty=1.1,
                    initial_prompt=initial_prompt,
                    progress_callback=progress_bridge,
                    on_segment=on_segment,
                )
            transcript.audio = metadata

            # Step 2.5: Diarization (optional, Whisper only — Gemini handles it natively)
            if enable_diarization and not is_gemini:
                if on_phase:
                    on_phase("diarizing", "Identifying speakers...", None)

                logger.info("Pipeline: running speaker diarization")
                try:
                    from audiobench.diarization.engine import PyannoteDiarizer

                    diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token)
                    transcript = diarizer.diarize(wav_path, transcript)
                    logger.info("Pipeline: diarization complete")
                except Exception as e:
                    logger.warning("Diarization failed (continuing without): %s", e)

            # Step 3: Store
            if on_phase:
                on_phase("saving", "Saving to database...", None)

            logger.info("Pipeline: saving to database")
            tx_id = self._repository.save_transcription(transcript, metadata)
            logger.info("Pipeline: saved as transcription #%d", tx_id)

            # Step 4: Format & output
            if output_path:
                self._write_output(transcript, fmt, output_path)
                logger.info("Pipeline: wrote %s output to %s", fmt, output_path)

            if on_phase:
                on_phase("done", "Complete!", None)

        return transcript

    def _write_output(self, transcript: Transcript, fmt: str, output_path: str) -> None:
        """Format transcript and write to file."""
        from audiobench.output.base import get_formatter

        formatter = get_formatter(fmt)
        content = formatter.format(transcript)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _reconstruct_transcript(self, data: dict, metadata: AudioMetadata) -> Transcript:
        """Reconstruct a Transcript from cached DB data."""
        from audiobench.transcribe.transcription_result import Segment

        segments = [
            Segment(
                id=s["index"],
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
            )
            for s in data.get("segments", [])
        ]

        return Transcript(
            segments=segments,
            language=data.get("language", "en"),
            language_probability=data.get("language_probability", 0.0),
            audio=metadata,
            duration_seconds=data.get("duration", 0.0),
            engine=data.get("engine", "faster-whisper"),
            model_name=data.get("model", "large-v3-turbo"),
        )
