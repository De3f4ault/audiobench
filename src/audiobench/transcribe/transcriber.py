"""Pipeline orchestrator — the main entry point for transcription.

Chains: load → transcribe → store → format → output

Emits phase callbacks for UI progress:
    on_phase("loading", "Loading model...")
    on_phase("converting", "Converting audio...")
    on_phase("transcribing", "Transcribing...", progress=0.42)
    on_phase("saving", "Saving to database...")
    on_phase("done", "Complete!")

Design principles applied in this refactor:
- transcribe_file() is an ~80-line orchestrator; heavy logic lives in private methods.
- ChapterInfo is the universal chapter currency; no raw dicts or ORM objects escape.
- get_chapter_repo() singleton avoids re-instantiating the repository on every call.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audiobench.core.db_engine import init_db
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.observatory.context import log_event
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.audio_converter import AudioLoader
from audiobench.transcribe.checkpoint_manager import CheckpointManager
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.engines.engine_registry import create_engine
from audiobench.transcribe.transcription_result import AudioMetadata, Segment, Transcript

_POST_PROCESS_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="postproc")

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

    # ── Public API ───────────────────────────────────────────────────────────

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
        map_speakers: str | None = None,
        auto_name: bool = False,
        on_phase: PhaseCallback | None = None,
        on_segment: SegmentCallback | None = None,
        filters: list[str] | None = None,
        engine_name: str | None = None,
        job_id: int | None = None,
        target_chapters: list[int] | None = None,
        resume: bool = False,
        strategy: str = "batch",
        pipeline_workers: int = 2,
        parallel: int = 1,
        skip_ghost: bool = True,
        chapter_id: int | None = None,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
        sensitive: bool = False,
        align: bool | None = None,
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
            job_id: Optional ID of the background job for emitting events.
            target_chapters: If set, only transcribe these chapter indices.
            resume: Skip chapters already marked 'completed'.
            parallel: Number of parallel chapter workers (1 = sequential).
            skip_ghost: Skip zero-duration ghost chapters.
            chapter_id: DB chapter ID if this call is for a single chapter.

        Returns:
            Transcript result.
        """
        self._ensure_db()
        engine = self._ensure_engine(on_phase, engine_name=engine_name)

        fmt = output_format or self._settings.output_format
        word_ts = word_timestamps if word_timestamps is not None else self._settings.word_timestamps
        preset = speed_preset or self._settings.speed_preset

        def emit(phase: str, message: str, progress: float | None = None) -> None:
            if on_phase:
                on_phase(phase, message, progress)
            kw = {"phase": phase}
            if progress is not None:
                kw["progress"] = int(progress * 100) if isinstance(progress, float) else progress
            self._emit_event(job_id, **kw)

        try:
            if target_chapters is not None:
                transcript = self._run_chapter_pipeline(
                    file_path=file_path,
                    target_chapters=target_chapters,
                    emit=emit,
                    language=language,
                    word_timestamps=word_ts,
                    skip_cache=skip_cache,
                    speed_preset=preset,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=enable_diarization,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_segment=on_segment,
                    filters=filters,
                    engine_name=engine_name,
                    job_id=job_id,
                    resume=resume,
                    strategy=strategy,
                    pipeline_workers=pipeline_workers,
                    parallel=parallel,
                    skip_ghost=skip_ghost,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
                    sensitive=sensitive,
                    align=align,
                )
            else:
                transcript = self._run_single_pipeline(
                    file_path=file_path,
                    engine=engine,
                    emit=emit,
                    language=language,
                    word_ts=word_ts,
                    preset=preset,
                    skip_cache=skip_cache,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=enable_diarization,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_segment=on_segment,
                    filters=filters,
                    chapter_id=chapter_id,
                    job_id=job_id,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
                    sensitive=sensitive,
                    align=align,
                )

            if output_path:
                self._write_output(transcript, fmt, output_path)
                logger.info("Pipeline: wrote %s output to %s", fmt, output_path)

            if job_id:
                from audiobench.jobs.repository import JobRepository

                JobRepository().mark_job_done(job_id)

            return transcript

        except Exception:
            if job_id:
                from audiobench.jobs.repository import JobRepository

                JobRepository().mark_job_failed(job_id, exit_code=1)
            raise

    # ── Private: Single-file pipeline ────────────────────────────────────────

    def _run_single_pipeline(
        self,
        file_path: str | Path,
        engine: TranscriptionEngine,
        emit: Callable,
        language: str | None,
        word_ts: bool,
        preset: str,
        skip_cache: bool,
        initial_prompt: str | None,
        translate: bool,
        enable_diarization: bool,
        map_speakers: str | None,
        auto_name: bool,
        on_segment: SegmentCallback | None,
        filters: list[str] | None,
        chapter_id: int | None,
        job_id: int | None,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
        sensitive: bool = False,
        align: bool | None = None,
    ) -> Transcript:
        """Load, transcribe, diarize, and save a single audio file."""
        is_gemini = engine.engine_name == "gemini"
        beam = self._settings.resolve_beam_size(preset)
        batch = self._settings.resolve_batch_size(preset)
        temperature = self._settings.resolve_temperature(preset)
        condition_on_prev = self._settings.resolve_condition_on_previous_text(preset)

        emit("converting", "Converting audio...")
        logger.info(
            "Pipeline: loading %s (preset=%s, beam=%d, batch=%d)", file_path, preset, beam, batch
        )

        with AudioLoader() as loader:
            wav_path, metadata = loader.load(file_path, filters=filters)

            # Cache check
            cached = self._check_cache(metadata, skip_cache)
            if cached:
                transcript = cached
                if (
                    enable_diarization
                    and not is_gemini
                    and not any(s.speaker for s in transcript.segments)
                ):
                    transcript = self._run_diarization(wav_path, transcript, emit, diarize_mode, diarize_threshold)

                # Heal cached Gemini transcripts that have all-zero timestamps.
                # This happens when:
                #   a) A previous alignment run failed (old 1D-array bug), or
                #   b) The run was interrupted before commit_alignment was called.
                # On the next invocation the file hits the cache and would be
                # returned with wrong timestamps — so we re-run alignment here.
                align_threshold = getattr(self._settings, "align_threshold_min", 0.5) * 60
                effective_duration = transcript.duration_seconds or metadata.duration_seconds
                needs_alignment = (
                    is_gemini
                    and align is not False
                    and effective_duration > align_threshold
                    and all(s.start == 0.0 and s.end == 0.0 for s in transcript.segments)
                )
                if needs_alignment:
                    cached_record = self._repository.find_by_hash(metadata.file_hash)
                    if cached_record:
                        emit("aligning", "Aligning timestamps...", 0.0)
                        logger.info("Cache hit: all timestamps are zero — healing via alignment worker")
                        self._run_alignment_worker(
                            tx_id=cached_record.id,
                            audio_path=file_path,
                            on_progress=on_segment,
                            transcript=transcript,
                        )
                        # Re-read aligned transcript from DB
                        refreshed = self._repository.get_by_id(cached_record.id)
                        if refreshed:
                            transcript = self._reconstruct_transcript_from_record(refreshed, metadata)

                emit("done", "Retrieved from cache")
                self._emit_event(
                    job_id,
                    phase="done",
                    words=transcript.word_count,
                    speakers=len(transcript.speaker_map),
                    duration=int(transcript.duration_seconds),
                )
                return transcript


            # Transcribe
            task = "translate" if translate else "transcribe"
            emit("transcribing", "Starting transcription pipeline...", 0.0)

            try:
                tx_id = self._repository.begin_transcription(
                    audio_metadata=metadata,
                    engine="gemini" if is_gemini else "faster-whisper",
                    model_name=getattr(engine, "_model_name", None) or self._settings.model_name,
                    chapter_id=chapter_id,
                    overwrite=skip_cache,
                )
            except ValueError as e:
                if "already exists" in str(e):
                    logger.info("Transcription already exists, skipping")
                    return None
                raise

            new_attempt_count = self._repository.increment_attempt_count(tx_id)
            logger.info("Starting attempt %d for tx_id %d", new_attempt_count, tx_id)

            import time
            last_heartbeat = [time.time()]
            def _progress(pct: float) -> None:
                emit("transcribing", "Transcribing...", pct)
                now = time.time()
                if now - last_heartbeat[0] > 60.0:
                    last_heartbeat[0] = now
                    try:
                        self._repository.touch_transcription(tx_id)
                    except Exception as e:
                        logger.debug("Failed to update transcription heartbeat: %s", e)
            def _do_transcribe():
                if is_gemini:
                    # Gemini is a cloud model — send the original, full-quality
                    # file rather than the Whisper-optimised 16 kHz mono WAV.
                    return engine.transcribe(
                        str(file_path),
                        language=language or self._settings.language,
                        task=task,
                        word_timestamps=word_ts,
                        on_phase=emit,
                        on_segment=on_segment,
                        diarize=enable_diarization,
                        align=align,
                    )
                else:
                    return engine.transcribe(
                        wav_path,
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
                        progress_callback=_progress,
                        on_segment=on_segment,
                    )

            diarize_device = self._settings.resolve_diarization_device()
            whisper_device_index = self._settings.resolve_device_index()
            whisper_dev = f"cuda:{whisper_device_index}" if isinstance(whisper_device_index, int) else f"cuda:{whisper_device_index[0]}"

            is_concurrent_capable = (
                enable_diarization
                and not is_gemini
                and diarize_mode == "accurate"
                and diarize_device.startswith("cuda")
                and (whisper_dev != diarize_device)
            )

            if is_concurrent_capable:
                logger.info("Running transcription and diarization concurrently on separate GPUs")
                try:
                    from audiobench.diarization.engine import PyannoteDiarizer
                    diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token, device=diarize_device)

                    with ThreadPoolExecutor(max_workers=2) as ex:
                        ft = ex.submit(_do_transcribe)
                        fd = ex.submit(diarizer.get_speaker_turns, wav_path)

                        transcript = ft.result()
                        transcript.audio = metadata

                        try:
                            turns = fd.result()
                            transcript = diarizer.assign_speakers(transcript, turns, audio_path=wav_path)
                            logger.info("Pipeline: diarization complete")
                        except Exception as e:
                            logger.warning("Concurrent diarization failed (continuing without): %s", e)
                except Exception as e:
                    logger.warning("Failed to initialize concurrent diarization (continuing without): %s", e)
                    transcript = _do_transcribe()
                    transcript.audio = metadata
            else:
                transcript = _do_transcribe()
                transcript.audio = metadata

            self._repository.commit_transcript_text(tx_id, transcript, chapter_id, privacy_tier=3 if sensitive else 0)
            logger.info("Pipeline: transcript text committed")

            post_processing_degraded = False

            if enable_diarization and not is_gemini and not is_concurrent_capable:
                try:
                    transcript = self._run_diarization(wav_path, transcript, emit, diarize_mode, diarize_threshold)
                    self._repository.commit_diarization(tx_id, transcript)
                except Exception as e:
                    logger.warning("Sequential diarization failed (continuing without): %s", e)
                    self._repository.mark_transcription_degraded(tx_id, str(e), "diarizing")
                    post_processing_degraded = True

            # Post-process: forced alignment (engine-agnostic)
            align_threshold = getattr(self._settings, "align_threshold_min", 0.5) * 60
            # Gemini transcripts arrive with duration_seconds=0 (the API doesn't echo timing).
            # Fall back to the audio file metadata duration for gate decisions.
            effective_duration = transcript.duration_seconds or (
                metadata.duration_seconds if metadata else 0.0
            )
            should_align = (
                align is True
                or (align is None
                    and is_gemini
                    and effective_duration > align_threshold)
            )

            if should_align:
                emit("aligning", "Aligning timestamps...", 0.0)
                self._run_alignment_worker(
                    tx_id=tx_id,
                    audio_path=file_path,
                    on_progress=on_segment,
                    transcript=transcript,
                )
                # Re-read the aligned transcript from DB so segment objects
                # have the updated timestamps for the rest of the pipeline.
                refreshed = self._repository.get_by_id(tx_id)
                if refreshed:
                    transcript = self._reconstruct_transcript_from_record(refreshed, metadata)
                logger.info("Pipeline: forced alignment committed")
            elif on_segment and is_gemini and align is not False and transcript.duration_seconds > align_threshold:
                # If align was false but we chunked, fire the segments now
                for seg in transcript.segments:
                    on_segment(seg)

            # Speaker naming
            self._apply_speaker_naming(
                transcript, map_speakers, auto_name, enable_diarization, emit
            )

            # Finalize
            emit("saving", "Finalizing transcription...")
            self._repository.finalize_transcription(
                tx_id=tx_id,
                transcript=transcript,
                chapter_id=chapter_id,
                on_phase=emit,
                privacy_tier=3 if sensitive else 0,
                audio_metadata=metadata,
                is_degraded=post_processing_degraded,
            )
            logger.info("Pipeline: saved as transcription #%d", tx_id)

            # Fire plugin hook
            try:
                from audiobench.events import get_bus
                get_bus().emit(
                    "transcription.complete",
                    tx_id=tx_id,
                    file_path=str(file_path),
                    duration_seconds=transcript.duration_seconds,
                    word_count=transcript.word_count,
                    language=transcript.language,
                )
            except Exception:
                logger.warning("EventBus emit failed (non-fatal)", exc_info=True)

            # Post-transcription naming (opt-in via --auto-name only)
            if auto_name and chapter_id is None and transcript.segments:
                from audiobench.transcribe.rename_service import spawn_auto_naming
                spawn_auto_naming(tx_id)

            emit("done", "Complete!")
            self._emit_event(
                job_id,
                phase="done",
                words=transcript.word_count,
                speakers=len(transcript.speaker_map),
                duration=int(transcript.duration_seconds),
            )
            return transcript

    # ── Private: Chapter pipeline ─────────────────────────────────────────────

    def _run_chapter_pipeline(
        self,
        file_path: str | Path,
        target_chapters: list[int] | str,
        emit: Callable,
        language: str | None,
        word_timestamps: bool,
        skip_cache: bool,
        speed_preset: str,
        initial_prompt: str | None,
        translate: bool,
        enable_diarization: bool,
        map_speakers: str | None,
        auto_name: bool,
        on_segment: SegmentCallback | None,
        filters: list[str] | None,
        engine_name: str | None,
        job_id: int | None,
        resume: bool,
        strategy: str,
        pipeline_workers: int,
        parallel: int,
        skip_ghost: bool,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
        sensitive: bool = False,
        align: bool | None = None,
    ) -> Transcript:
        """Split a file by chapter indices and transcribe each chunk."""
        from audiobench.chapters.cue_parser import ChapterInfo
        from audiobench.chapters.splitter import ChapterSplitter
        from audiobench.storage.chapter_repository import get_chapter_repo

        repo = get_chapter_repo()

        # Ensure the audio file exists in the library and chapters are detected
        audio_record = self._ensure_audio_record(file_path, emit)
        all_chapters = repo.get_chapters(audio_record.id if audio_record else 0)

        cm = CheckpointManager(file_path)

        # Filter to the requested indices
        if target_chapters == "all":
            chapters_to_process = all_chapters
        else:
            chapters_to_process = [c for c in all_chapters if c.index in target_chapters]

        if skip_ghost:
            chapters_to_process = [c for c in chapters_to_process if not c.is_ghost]

        if resume:
            chapters_to_process = [c for c in chapters_to_process if not cm.has_checkpoint(c.index)]
            if len(chapters_to_process) < len(all_chapters):
                logger.info("Resuming: Skipped %d already-completed chapters.", len(all_chapters) - len(chapters_to_process))

        if not chapters_to_process and target_chapters:
            # Everything is already done! Load all checkpoints.
            results = [cm.load_checkpoint(c.index) for c in all_chapters]
            results = [r for r in results if r is not None]
            if not results:
                raise RuntimeError("No checkpoints found, but resume filtered all chapters.")
            emit("done", "Loaded all from cache")
            return self._merge_transcripts(results)

        splitter = ChapterSplitter()
        source_path = Path(audio_record.file_path if audio_record else file_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            emit("converting", "Extracting chapters...", 0.0)
            chunk_paths = splitter.split(source_path, chapters_to_process, Path(tmp_dir), fmt="wav")

            def _process_chunk(
                i: int, chap: ChapterInfo, chunk_path: Path | None, do_diarize: bool
            ) -> Transcript | None:
                if chunk_path is None:
                    return None

                # Check checkpoint
                if resume and cm.has_checkpoint(chap.index):
                    res = cm.load_checkpoint(chap.index)
                    if res:
                        return res

                emit("transcribing", f"Transcribing chapter {chap.index}...", float(i) / len(chapters_to_process))

                result = self.transcribe_file(
                    file_path=chunk_path,
                    language=language,
                    output_format=None,
                    output_path=None,
                    word_timestamps=word_timestamps,
                    skip_cache=True,
                    speed_preset=speed_preset,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=do_diarize,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_phase=None,
                    on_segment=on_segment,
                    filters=filters,
                    engine_name=engine_name,
                    job_id=job_id,
                    target_chapters=None,
                    chapter_id=chap.id,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
                    sensitive=sensitive,
                    align=align,
                )

                # Shift timestamps
                offset = chap.start_time
                for seg in result.segments:
                    seg.start += offset
                    seg.end += offset
                    for word in seg.words:
                        word.start += offset
                        word.end += offset

                # Save checkpoint
                cm.save_checkpoint(chap.index, result)
                return result

            results = []

            if strategy == "batch":
                # Phase 1: Transcribe all
                for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                    r = _process_chunk(i, c, p, do_diarize=False)
                    if r: results.append(r)

                # Phase 2: Diarize all
                if enable_diarization:
                    emit("diarizing", "Diarizing all chapters...", 0.0)
                    for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                        res = cm.load_checkpoint(c.index)
                        if res and p and p.exists() and not any(s.speaker for s in res.segments):
                            emit("diarizing", f"Diarizing chapter {c.index}...", float(i) / len(chapters_to_process))
                            # Diarize and overwrite checkpoint
                            res = self._run_diarization(p, res, emit, diarize_mode, diarize_threshold)
                            cm.save_checkpoint(c.index, res)

            elif strategy == "concurrent":
                # Producer-Consumer pipeline for multi-GPU
                import queue
                import threading

                import torch

                diarize_device = self._settings.resolve_diarization_device()
                whisper_device_index = self._settings.resolve_device_index()
                whisper_dev = f"cuda:{whisper_device_index}" if isinstance(whisper_device_index, int) else f"cuda:{whisper_device_index[0]}"

                is_concurrent = (
                    enable_diarization
                    and diarize_mode == "accurate"
                    and diarize_device.startswith("cuda")
                    and (whisper_dev != diarize_device)
                )

                if is_concurrent:
                    logger.info("Using true producer-consumer concurrent chapter pipeline")
                    q = queue.Queue()
                    results = [None] * len(chapters_to_process)

                    def producer():
                        for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                            r = _process_chunk(i, c, p, do_diarize=False)
                            q.put((i, c, p, r))
                        q.put(None)  # Sentinel

                    def consumer():
                        while True:
                            item = q.get()
                            if item is None:
                                break
                            i, c, p, r = item
                            if r is not None:
                                if enable_diarization and p and p.exists() and not any(s.speaker for s in r.segments):
                                    emit("diarizing", f"Diarizing chapter {c.index}...", float(i) / len(chapters_to_process))
                                    r = self._run_diarization(p, r, emit, diarize_mode, diarize_threshold)
                                    cm.save_checkpoint(c.index, r)
                                results[i] = r
                            q.task_done()

                    t1 = threading.Thread(target=producer)
                    t2 = threading.Thread(target=consumer)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                    results = [r for r in results if r is not None]
                else:
                    # Single-GPU sequential fallback but using threads to share memory/IO efficiently
                    torch.set_num_threads(1)

                    def _process_concurrent(idx: int, c: ChapterInfo, p: Path) -> Transcript | None:
                        # Whisper
                        r = _process_chunk(idx, c, p, do_diarize=False)
                        if not r: return None

                        # Pyannote
                        if enable_diarization and p and p.exists() and not any(s.speaker for s in r.segments):
                            emit("diarizing", f"Diarizing chapter {c.index}...", float(idx) / len(chapters_to_process))
                            r = self._run_diarization(p, r, emit, diarize_mode, diarize_threshold)
                            cm.save_checkpoint(c.index, r)
                        return r

                    with ThreadPoolExecutor(max_workers=pipeline_workers) as ex:
                        futures = [ex.submit(_process_concurrent, i, c, p) for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))]
                        results = [f.result() for f in futures if f.result() is not None]

            else:
                # strategy == "chunk"
                if parallel > 1:
                    with ThreadPoolExecutor(max_workers=parallel) as ex:
                        futures = [ex.submit(_process_chunk, i, c, p, enable_diarization) for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))]
                        results = [f.result() for f in futures if f.result() is not None]
                else:
                    results = [r for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)) if (r := _process_chunk(i, c, p, enable_diarization)) is not None]

        # Load all checkpoints (including those skipped via resume)
        final_results = []
        for c in all_chapters:
            r = cm.load_checkpoint(c.index)
            if r: final_results.append(r)

        if not final_results:
            raise RuntimeError("No chapters were successfully transcribed.")

        return self._merge_transcripts(final_results)

    # ── Private: Helpers ──────────────────────────────────────────────────────

    def _ensure_audio_record(self, file_path: str | Path, emit: Callable):
        """Ensure the audio file has a DB record, importing it into the library if needed."""
        from audiobench.chapters.detector import ChapterDetector
        from audiobench.core.db_session import get_session
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import AudioFileRecord

        with AudioLoader() as loader:
            _, metadata = loader.load(file_path)

        if metadata and metadata.file_hash:
            with get_session() as session:
                record = session.query(AudioFileRecord).filter_by(file_hash=metadata.file_hash).first()
                if record:
                    return record

        # New file — import to library and detect chapters
        new_path = self._repository._import_to_library(str(file_path))
        with get_session() as session:
            audio_record = AudioFileRecord(
                file_path=new_path,
                file_name=metadata.file_name,
                file_size_bytes=metadata.file_size_bytes,
                format=metadata.format,
                duration_seconds=metadata.duration_seconds,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                file_hash=metadata.file_hash,
            )
            session.add(audio_record)
            session.commit()

            try:
                chapters = ChapterDetector().detect(Path(new_path))
                if chapters:
                    get_chapter_repo().save_chapters(audio_record.id, chapters)
            except Exception as e:
                logger.warning("Chapter detection failed for %s: %s", new_path, e)

            return session.query(AudioFileRecord).filter_by(id=audio_record.id).first()

    def _check_cache(self, metadata: AudioMetadata, skip_cache: bool) -> Transcript | None:
        """Return a cached Transcript if one exists, otherwise None."""
        if skip_cache or not metadata.file_hash:
            return None
        cached = self._repository.find_by_hash(metadata.file_hash)
        if not cached:
            return None
        logger.info("Pipeline: cache hit for hash %s", metadata.file_hash[:12])
        data = self._repository.get_by_id(cached.id)
        return self._reconstruct_transcript(data, metadata) if data else None

    def _run_alignment_worker(
        self,
        tx_id: int,
        audio_path,
        on_progress=None,
        transcript=None,
    ) -> None:
        """Run alignment in a fresh subprocess to isolate OOM crashes.

        alignment_worker.py loads its own Python interpreter (no Gemini engine,
        no daemon models in scope). If it crashes with SIGABRT/bad_alloc the
        parent catches the non-zero exit and continues — the transcript text is
        already in the DB at this point, so no data is lost.

        Results are committed to the DB by the worker itself; the caller should
        re-read the DB record if it needs the updated segment timestamps.
        """
        import subprocess
        import sys
        from pathlib import Path

        worker = Path(__file__).parent.parent / "daemon" / "alignment_worker.py"
        cmd = [sys.executable, str(worker), str(tx_id), str(audio_path)]

        logger.info(
            "Alignment: spawning subprocess — tx_id=%d, audio=%s", tx_id, audio_path
        )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,   # 1-hour hard limit
            )
            if result.returncode == 0:
                logger.info("Alignment worker completed successfully (tx_id=%d)", tx_id)
            else:
                logger.warning(
                    "Alignment worker exited with code %d (tx_id=%d) — "
                    "transcript saved without word-level timestamps.\n"
                    "stderr: %s",
                    result.returncode, tx_id, result.stderr[:500],
                )
                if on_progress and transcript:
                    for seg in transcript.segments:
                        on_progress(seg)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Alignment worker timed out after 1 hour (tx_id=%d)", tx_id
            )
            if on_progress and transcript:
                for seg in transcript.segments:
                    on_progress(seg)
        except Exception as exc:
            logger.warning("Failed to spawn alignment worker: %s", exc)
            if on_progress and transcript:
                for seg in transcript.segments:
                    on_progress(seg)

    def _reconstruct_transcript_from_record(self, data, metadata=None) -> Transcript:
        """Re-read a transcript from a DB record dict (post-alignment refresh)."""
        return self._reconstruct_transcript(data, metadata)

    def _run_diarization(self, wav_path: str, transcript: Transcript, emit: Callable, diarize_mode: str = "fast", diarize_threshold: float = 0.65) -> Transcript:
        """Run speaker diarization, returning updated transcript (or original on failure)."""
        emit("diarizing", "Identifying speakers...")
        try:
            if diarize_mode == "accurate":
                from audiobench.diarization.engine import PyannoteDiarizer
                diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token, device=self._settings.resolve_diarization_device())
            else:
                from audiobench.diarization.engine import LightweightDiarizer
                diarizer = LightweightDiarizer(distance_threshold=diarize_threshold, device=self._settings.resolve_diarization_device())

            result = diarizer.diarize(wav_path, transcript)
            logger.info("Pipeline: diarization complete")
            return result
        except Exception as e:
            logger.warning("Diarization failed (continuing without): %s", e)
            return transcript

    def _apply_speaker_naming(
        self,
        transcript: Transcript,
        map_speakers: str | None,
        auto_name: bool,
        enable_diarization: bool,
        emit: Callable,
    ) -> None:
        """Apply manual or automatic speaker name mapping in-place."""
        if map_speakers:
            emit("naming", "Applying manual speaker map...")
            try:
                for pair in map_speakers.split(","):
                    k, v = pair.split("=")
                    transcript.speaker_map[k.strip()] = v.strip()
                logger.info("Applied manual speaker map: %s", transcript.speaker_map)
            except Exception as e:
                logger.warning("Failed to parse map_speakers '%s': %s", map_speakers, e)
        elif auto_name and enable_diarization:
            emit("naming", "Auto-detecting speaker names...")
            try:
                self._auto_name_speakers(transcript)
                logger.info("Auto-detected speaker map: %s", transcript.speaker_map)
            except Exception as e:
                logger.warning("Auto-naming failed (continuing without): %s", e)

    def _merge_transcripts(self, results: list[Transcript]) -> Transcript:
        """Merge a list of chapter transcripts into a single unified Transcript."""
        merged = results[0]
        for r in results[1:]:
            merged.segments.extend(r.segments)
            merged.duration_seconds = max(merged.duration_seconds, r.duration_seconds)
        return merged

    # ── Private: Infrastructure ───────────────────────────────────────────────

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
                device_index=self._settings.resolve_device_index(),
            )
        return self._engine

    def _ensure_db(self) -> None:
        """Ensure database tables exist."""
        if not self._db_initialized:
            init_db()
            self._db_initialized = True

    def _emit_event(self, job_id: int | None, **kwargs) -> None:
        """Emit a structured pipeline event to the journal DB via log_event().

        Prior implementation wrote key=value lines to data_dir/job_logs/job_{id}.events.
        That flat file is no longer written — all events route through log_event() so
        the Observatory can observe them without polling a file on disk.
        """
        if not job_id:
            return
        phase = kwargs.pop("phase", "unknown")
        # Build a concise human-readable message from remaining kwargs
        extras = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        message = f"{phase}" + (f" ({extras})" if extras else "")
        log_event(
            "transcribe",
            phase,
            message,
            entity_type="job",
            entity_id=job_id,
            metadata=kwargs if kwargs else None,
        )

    def _write_output(self, transcript: Transcript, fmt: str, output_path: str) -> None:
        """Format transcript and write to file."""
        if fmt == "pdf":
            from audiobench.export.pdf import PDFExporter

            data = transcript.dict()
            data["file_name"] = Path(output_path).stem if output_path else "transcript"
            PDFExporter().export_transcript(data, output_path)
            return

        from audiobench.output.base import get_formatter

        formatter = get_formatter(fmt)
        content = formatter.format(transcript)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _auto_name_speakers(self, transcript: Transcript) -> None:
        """Use Gemini to detect actual speaker names from the transcript context."""
        import json
        import re

        from google import genai

        from audiobench.output.text import TextFormatter

        if not self._settings.gemini_api_key:
            logger.warning("Gemini API key not configured, skipping auto-name")
            return

        client = genai.Client(api_key=self._settings.gemini_api_key)

        # Use only the first ~5 minutes for speaker identification
        intro_segments = [s for s in transcript.segments if s.end <= 300][:50]
        intro = Transcript(
            segments=intro_segments,
            duration_seconds=intro_segments[-1].end if intro_segments else 0.0,
        )
        formatted_intro = TextFormatter().format(intro)

        prompt = (
            "Analyze the following transcript excerpt and identify the real names of the speakers "
            "based on the context (e.g. introductions, 'Welcome to the podcast, John').\n\n"
            f"{formatted_intro}\n\n"
            "Respond ONLY with a valid JSON object mapping the generic speaker labels to their detected names. "
            "If a name cannot be determined with high confidence, do not include it in the JSON.\n"
            'Example format:\n{\n  "Speaker 1": "Lex Fridman",\n  "Speaker 2": "Elon Musk"\n}'
        )

        response = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)

        try:
            detected = json.loads(raw)
            if isinstance(detected, dict):
                transcript.speaker_map.update(
                    {k: v for k, v in detected.items() if isinstance(k, str) and isinstance(v, str)}
                )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Gemini auto-name JSON: %s", e)

    def _spawn_refinement(self, tx_id: int, raw_text: str, segments: list) -> None:
        """Spawn a background thread to refine transcript segments using an LLM."""
        import threading

        def _refine() -> None:
            try:
                from audiobench.chat.providers.ollama_provider import OllamaClient
                from audiobench.transcribe.refiner import TranscriptRefiner

                client = OllamaClient(
                    base_url=self._settings.ollama_base_url,
                    model=self._settings.clean_model,
                )
                if not client.is_available():
                    logger.info("Ollama not available, skipping refinement for #%d", tx_id)
                    return

                refiner = TranscriptRefiner(client, model=self._settings.clean_model)
                seg_texts = [seg.text for seg in segments]
                cleaned, has_failures = refiner.refine_segments(seg_texts)

                if cleaned == seg_texts:
                    logger.info("Refinement produced no changes for #%d", tx_id)
                    return

                if not self._repository.update_segments(tx_id, cleaned):
                    logger.warning("update_segments failed for #%d", tx_id)
                    return

                refined_full = " ".join(t.strip() for t in cleaned if t.strip())
                self._repository.update_full_text(tx_id, refined_full, raw_text)
                logger.info("Segment refinement complete for #%d", tx_id)
            except Exception as e:
                logger.warning("Background refinement failed for #%d: %s", tx_id, e)

        _POST_PROCESS_POOL.submit(_refine)
        logger.info("Submitted background refinement task for #%d", tx_id)

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
            is_cached=True,
        )
