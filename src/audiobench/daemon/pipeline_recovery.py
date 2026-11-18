"""
pipeline_recovery.py - Background sweep to resume stuck transcriptions.
"""
import logging
import time
from pathlib import Path

from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.transcriber import TranscriptionPipeline
from audiobench.transcribe.transcription_result import Segment, Transcript

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 300  # 5 minutes
MAX_ATTEMPTS = 3

def _reconstruct_transcript(data: dict) -> Transcript:
    segments = [
        Segment(
            id=s["index"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker")
        ) for s in data.get("segments", [])
    ]
    return Transcript(
        text=data.get("full_text", ""),
        segments=segments,
        language=data.get("language", "en"),
        language_probability=data.get("language_probability", 1.0),
        duration_seconds=data.get("duration", 0.0),
        word_count=data.get("word_count", 0),
        speaker_map=data.get("speaker_map", {})
    )

def _is_system_idle() -> bool:
    """
    Gate: returns True only when it is safe to launch a backfill subprocess.
    """
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        if cpu > 80.0:
            logger.debug("Backfill gate: CPU %.1f%% > 80%%, skipping", cpu)
            return False

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if rss_mb > 4096:
            logger.debug("Backfill gate: RSS %.1fMB > 4096MB, skipping", rss_mb)
            return False
    except ImportError:
        pass  # psutil not available — skip resource checks, proceed

    # Check for active live transcription in the DB
    from datetime import UTC, datetime, timedelta

    from audiobench.core.db_session import get_session
    from audiobench.storage.models import TranscriptionRecord

    cutoff = datetime.now(UTC) - timedelta(minutes=10)
    active_phases = ("transcribing", "uploading", "processing")
    with get_session() as session:
        active = session.query(TranscriptionRecord).filter(
            TranscriptionRecord.pipeline_phase.in_(active_phases),
            TranscriptionRecord.updated_at > cutoff,
        ).first()
    if active:
        logger.debug("Backfill gate: active transcription tx=%d in phase=%s, skipping", active.id, active.pipeline_phase)
        return False

    return True

def _run_alignment_subprocess(wav_path: Path, transcript: Transcript, tx_id: int, duration_seconds: float = 0.0) -> Transcript:
    """
    Run forced alignment in an isolated subprocess.

    Raises RuntimeError on non-zero exit or subprocess.TimeoutExpired (1800s limit).
    TimeoutExpired is re-raised as RuntimeError so the caller's except-and-cap
    path (backfill_attempt_count / mark_backfill_exhausted) handles it uniformly.
    """
    import subprocess
    import sys
    worker = Path(__file__).parent / "alignment_worker.py"
    try:
        result = subprocess.run(
            [sys.executable, str(worker), str(tx_id), str(wav_path)],
            timeout=1800,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        duration_h = duration_seconds / 3600
        raise RuntimeError(
            f"alignment_timeout: file too large for CPU alignment "
            f"({duration_h:.2f}h audio, 1800s limit exceeded)"
        )
    if result.returncode != 0:
        raise RuntimeError(f"Alignment worker exited {result.returncode}: {result.stderr[-500:]}")
    # Caller will re-read from DB
    return transcript

def _do_sweep_once(transcriber: TranscriptionPipeline) -> None:
    """Check for stuck transcriptions and resume them."""
    repo = TranscriptionRepository()
    # Find incomplete records that haven't been updated in 5+ minutes
    stuck_records = repo.get_incomplete_transcriptions(max_age_minutes=5)

    if not stuck_records:
        logger.debug("Pipeline Sweep: No stuck transcriptions found.")
    else:
        from audiobench.core.settings import get_settings
        settings = get_settings()
        logger.info("Pipeline Sweep: Found %d stuck transcriptions", len(stuck_records))

        for record in stuck_records:
            if record.attempt_count >= MAX_ATTEMPTS:
                logger.warning("Pipeline Sweep: Tx %d has exceeded max attempts (%d), marking failed.", record.id, MAX_ATTEMPTS)
                repo.mark_transcription_failed(record.id, "Max attempts exceeded", record.pipeline_phase)
                continue

            logger.info("Pipeline Sweep: Resuming stuck tx %d (phase=%s, attempt=%d)", record.id, record.pipeline_phase, record.attempt_count + 1)

            try:
                repo.increment_attempt_count(record.id)

                if record.pipeline_phase == "transcribing":
                    # The Gemini step crashed before any data was committed.
                    # Do NOT auto-retry this expensive step.
                    logger.warning("Pipeline Sweep: Tx %d stuck during 'transcribing'. Marking failed.", record.id)
                    repo.mark_transcription_failed(record.id, "Crashed during extraction", record.pipeline_phase)
                    continue

                # Gemini jobs: diarization is handled inside the Gemini engine at
                # transcription time, and alignment runs via the subprocess worker
                # (alignment_worker.py). Neither should be re-run here by the sweep.
                # Attempting either would crash because the sweep only has the library
                # .m4a path, not the 16kHz mono WAV that torchaudio/faster-whisper expect.
                is_gemini = getattr(record, "engine", "") == "gemini"
                if is_gemini:
                    logger.debug("Pipeline Sweep: Tx %d is a Gemini job — skipping local alignment/diarization", record.id)
                    # Fall through to finalize below — just commit whatever state we have.

                # If phase is transcribed, aligned, diarized, we CAN resume because we have the transcript object!
                # We reconstruct it.
                data = repo.get_by_id(record.id, session=None)
                if not data or not data.get("file_path"):
                    logger.warning("Pipeline Sweep: Tx %d has no file path. Marking failed.", record.id)
                    repo.mark_transcription_failed(record.id, "No linked audio file", record.pipeline_phase)
                    continue

                wav_path = Path(data["file_path"])
                transcript = _reconstruct_transcript(data)

                # Since we are bypassing _run_single_pipeline, we run the remaining steps directly.
                def _dummy_emit(phase: str, msg: str, pct: float = 0.0) -> None:
                    pass

                if not is_gemini and record.pipeline_phase == "transcribed":
                    # Need to run alignment — trigger if file meets the configured minimum duration.
                    # Gemini jobs are excluded: their alignment runs via alignment_worker.py subprocess.
                    align_threshold_secs = settings.align_threshold_min * 60
                    if transcript.duration_seconds >= align_threshold_secs:
                        try:
                            transcript = transcriber._run_forced_alignment(
                                wav_path, transcript, _dummy_emit, None
                            )
                            repo.commit_alignment(record.id, transcript)
                            record.pipeline_phase = "aligned"
                        except Exception as exc:
                            logger.warning("Pipeline Sweep: Alignment failed for %d: %s", record.id, exc)
                            repo.mark_transcription_degraded(record.id, "alignment_failed", "aligning")
                            # Skip diarization if degraded? We can continue.

                if not is_gemini and record.pipeline_phase in ["transcribed", "aligned"]:
                    # Need to run diarization.
                    # Gemini jobs are excluded: diarization is handled inside the Gemini engine
                    # and the sweep only has the .m4a library path, not the 16kHz WAV that
                    # torchaudio requires.
                    if settings.enable_diarization and not transcript.speaker_map:
                        try:
                            transcript = transcriber._run_diarization(
                                wav_path, transcript, _dummy_emit,
                                diarize_mode="fast",
                                diarize_threshold=0.65
                            )
                            repo.commit_diarization(record.id, transcript)
                            record.pipeline_phase = "diarized"

                            # Release ECAPA-TDNN from daemon memory
                            from audiobench.diarization.verification import _classifiers
                            _device = settings.resolve_diarization_device()
                            _classifiers.pop(_device, None)
                            import gc; gc.collect()
                        except Exception as exc:
                            logger.warning("Pipeline Sweep: Diarization failed for %d: %s", record.id, exc)
                            repo.mark_transcription_degraded(record.id, "diarization_failed", "diarizing")

                # Finalize
                if record.pipeline_phase in ["transcribed", "aligned", "diarized"]:
                    # No map_speakers or auto_name context in the sweep — pass falsy values
                    transcriber._apply_speaker_naming(
                        transcript, None, False, settings.enable_diarization, _dummy_emit
                    )

                    # Probe audio metadata (ffprobe only — no decode, ~50ms).
                    # AudioLoader.load() was used here before but it performs a full
                    # ffmpeg decode just to get duration/rate/channels, then discards
                    # the WAV immediately — causing a double-decode with the biometric
                    # pass that finalize_transcription() schedules right after.
                    from audiobench.transcribe.audio_converter import probe
                    from audiobench.transcribe.transcription_result import AudioMetadata
                    audio_metadata = None
                    try:
                        _path = Path(wav_path)
                        _info = probe(_path)
                        audio_metadata = AudioMetadata(
                            file_path=str(_path),
                            file_name=_path.name,
                            file_size_bytes=_path.stat().st_size,
                            format=_path.suffix.lstrip(".").lower(),
                            duration_seconds=round(_info.duration, 3),
                            sample_rate=_info.sample_rate,
                            channels=_info.channels,
                        )
                    except Exception:
                        pass

                    repo.finalize_transcription(
                        tx_id=record.id,
                        transcript=transcript,
                        chapter_id=None,
                        on_phase=_dummy_emit,
                        privacy_tier=0,
                        audio_metadata=audio_metadata,
                        run_inline=True,  # daemon process — thread is safe, no shutdown race
                    )

                    logger.info("Pipeline Sweep: Successfully finalized tx %d", record.id)

                    # Fire event
                    try:
                        from audiobench.events import get_bus
                        get_bus().emit(
                            "transcription.complete",
                            tx_id=record.id,
                            file_path=str(wav_path),
                            duration_seconds=transcript.duration_seconds,
                            word_count=transcript.word_count,
                            language=transcript.language,
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.exception("Pipeline Sweep: Failed to resume tx %d: %s", record.id, e)

    # ── Backfill: re-align/re-diarize gracefully-degraded records ──────────
    if not _is_system_idle():
        return

    MAX_BACKFILL_ATTEMPTS = 3
    backfill_records = repo.get_degraded_for_backfill(max_backfill_attempts=MAX_BACKFILL_ATTEMPTS)
    if not backfill_records:
        return

    # Process one record per sweep tick. Shortest-duration first (ordered by query).
    record = backfill_records[0]
    logger.info("Backfill: attempting re-alignment for tx %d (attempt %d, duration=%.0fs)",
                record.id, record.backfill_attempt_count + 1, record.duration_seconds)

    # Exponential backoff: 30min → 2h → 6h
    backoff_seconds = [1800, 7200, 21600]
    delay = backoff_seconds[min(record.backfill_attempt_count, len(backoff_seconds) - 1)]
    repo.increment_backfill_attempt(record.id, next_attempt_delay_seconds=delay)

    data = repo.get_by_id(record.id, session=None)
    if not data or not data.get("file_path"):
        logger.warning("Backfill: tx %d has no file path, marking exhausted", record.id)
        repo.mark_backfill_exhausted(record.id)
        return

    wav_path = Path(data["file_path"])
    if not wav_path.exists():
        logger.warning("Backfill: audio file missing for tx %d: %s", record.id, wav_path)
        repo.mark_backfill_exhausted(record.id)
        return

    transcript = _reconstruct_transcript(data)

    try:
        is_diarization_failure = (
            record.failure_reason == "diarization_failed"
            or "diariz" in (record.failure_reason or "").lower()
        )
        is_gemini_record = getattr(record, "engine", "") == "gemini"

        if is_diarization_failure and not is_gemini_record:
            from audiobench.core.settings import get_settings
            from audiobench.transcribe.transcriber import TranscriptionPipeline

            def _dummy_emit(phase: str, msg: str, pct: float = 0.0) -> None:
                pass

            settings = get_settings()
            transcriber = TranscriptionPipeline()
            transcript = transcriber._run_diarization(
                wav_path, transcript, _dummy_emit,
                diarize_mode="fast",
                diarize_threshold=0.65
            )
            repo.commit_diarization(record.id, transcript)
            logger.info("Backfill: diarization committed for tx %d", record.id)

            # Post-diarization cleanup
            from audiobench.diarization.verification import _classifiers
            _device = settings.resolve_diarization_device()
            _classifiers.pop(_device, None)
            import gc; gc.collect()

            # Change phase so next normal sweep finalizes it
            from audiobench.core.db_session import get_session

            from audiobench.storage.models import TranscriptionRecord
            with get_session() as session:
                rec = session.query(TranscriptionRecord).get(record.id)
                rec.pipeline_phase = "diarized"
                session.commit()
        elif record.failure_reason == "timeline_clamped":
            logger.info(
                "Backfill: tx %d has failure_reason='timeline_clamped' — "
                "alignment succeeded with monotonic clamping; skipping retry to preserve diagnostic label.",
                record.id,
            )
            return
        else:
            if record.failure_reason not in ("alignment_failed",) and not (record.failure_reason or "").startswith("alignment_timeout"):
                logger.warning("Backfill: Unrecognized failure reason '%s' for tx %d, defaulting to alignment backfill",
                               record.failure_reason, record.id)
            _run_alignment_subprocess(wav_path, transcript, record.id, duration_seconds=record.duration_seconds)
            logger.info("Backfill: alignment committed for tx %d", record.id)
    except Exception as exc:
        new_count = record.backfill_attempt_count + 1
        logger.warning("Backfill: %s failed for tx %d (attempt %d/%d): %s",
                       record.failure_reason, record.id, new_count, MAX_BACKFILL_ATTEMPTS, exc)
        if new_count >= MAX_BACKFILL_ATTEMPTS:
            repo.mark_backfill_exhausted(record.id)
        return

def pipeline_recovery_sweep_loop() -> None:
    """Run the pipeline recovery sweep periodically."""
    logger.info("Pipeline Recovery Sweep loop started (interval=%ds)", SWEEP_INTERVAL_SECONDS)
    transcriber = TranscriptionPipeline()
    while True:
        try:
            _do_sweep_once(transcriber)
        except Exception as exc:
            logger.error("Pipeline Recovery Sweep loop: unexpected error: %s", exc)
        time.sleep(SWEEP_INTERVAL_SECONDS)
