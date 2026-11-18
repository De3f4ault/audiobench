"""
alignment_worker.py — Subprocess entry point for alignment.

Invoked by Transcriber._run_alignment_worker() via subprocess.run().
Also used by pipeline_recovery.py for daemon backfill.
Communicates result via DB, not stdout.
Exit code 0 = success, non-zero = failure (caller reads DB to confirm).

Two-phase design
----------------
Phase 1 (instant, crash-safe): proportional timestamps.
    Immediately writes evenly-distributed timestamps to the DB using only
    the total audio duration.  Even if the subprocess is killed later,
    users see non-zero timestamps (e.g. [0:02 → 0:08]) instead of [0:00].

Phase 2 (accurate, ~3–5 min for 1h30m): faster-whisper alignment.
    Re-transcribes the audio with the cached tiny model to get real segment
    timestamps, then maps those to the Gemini segments via character-count
    interpolation.  Overwrites Phase 1 timestamps if successful.

Duration resolution hierarchy (highest → lowest priority):
  1. AudioFileRecord.duration_seconds  — stored by AudioLoader at import time;
                                         always correct, never 0 for real files.
  2. torchaudio.info()                 — fast container-level probe, no decode.
  3. ffprobe CLI                       — robust fallback for any format.
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_audio_file_duration(audio_file_id: int | None) -> float:
    """Priority 1: read duration from AudioFileRecord already in the DB.

    AudioLoader writes this at import time from the actual container
    metadata — it is always correct and never 0 for real audio files.
    """
    if not audio_file_id:
        return 0.0
    try:
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AudioFileRecord

        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
            if rec and rec.duration_seconds > 0.0:
                return rec.duration_seconds
    except Exception:
        pass
    return 0.0


def _probe_file_duration(audio_path: Path) -> float:
    """Priority 2 + 3: probe the file directly when the DB value is missing."""
    # torchaudio — fast container-level metadata read (no full decode)
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        if info.num_frames > 0 and info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass

    # ffprobe — handles any format torchaudio can't
    try:
        import json
        import subprocess
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        probe = json.loads(result.stdout)
        duration = float(probe["format"]["duration"])
        if duration > 0.0:
            return duration
    except Exception:
        pass

    return 0.0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main():
    _configure_logging()

    if len(sys.argv) < 3:
        print("Usage: alignment_worker.py <tx_id> <audio_path> [--events-file <path>]", file=sys.stderr)
        sys.exit(1)

    import json

    tx_id = int(sys.argv[1])
    audio_path = Path(sys.argv[2])

    events_path: Path | None = None
    if "--events-file" in sys.argv:
        try:
            ef_idx = sys.argv.index("--events-file")
            events_path = Path(sys.argv[ef_idx + 1])
        except Exception:
            pass

    def emit_event(d: dict) -> None:
        if events_path:
            try:
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(d) + "\n")
                    f.flush()
            except Exception:
                pass

    try:
        from audiobench.daemon.pipeline_recovery import _reconstruct_transcript
        from audiobench.storage.repository import TranscriptionRepository
        from audiobench.transcribe.alignment_pipeline import (
            _estimate_initial_timestamps,
            align_transcript,
            load_align_model,
        )

        emit_event({"t": "phase", "phase": "aligning", "message": "Initializing alignment..."})

        repo = TranscriptionRepository()
        data = repo.get_by_id(tx_id, session=None)
        if not data:
            logger.error("TranscriptionRecord %d not found.", tx_id)
            emit_event({"t": "error", "message": f"TranscriptionRecord {tx_id} not found."})
            sys.exit(1)

        transcript = _reconstruct_transcript(data)

        # ── Resolve duration (critical for Gemini transcripts) ────────────────
        if transcript.duration_seconds == 0.0:
            duration = _get_audio_file_duration(data.get("audio_file_id"))
            if duration == 0.0:
                duration = _probe_file_duration(audio_path)

            if duration > 0.0:
                transcript.duration_seconds = duration
                logger.info(
                    "Duration patched: %.1fs (TranscriptionRecord had 0.0 — Gemini transcript)",
                    duration,
                )
            else:
                logger.warning(
                    "Could not determine audio duration — "
                    "alignment will produce proportional timestamps only."
                )

        # ── Phase 1: proportional timestamps (instant, crash-safe) ───────────
        emit_event({"t": "phase", "phase": "aligning", "message": "Phase 1: Estimating proportional timestamps..."})
        _estimate_initial_timestamps(transcript)
        if transcript.duration_seconds > 0.0:
            repo.commit_alignment(tx_id, transcript, phase="aligned_proportional")
            logger.info(
                "Phase 1: proportional timestamps committed for %d segments "
                "(%.0fs duration) — phase stamped 'aligned_proportional', "
                "subprocess continuing to Phase 2",
                len(transcript.segments),
                transcript.duration_seconds,
            )
        else:
            logger.warning(
                "Phase 1 skipped: duration is 0 — cannot estimate proportional timestamps."
            )

        # ── Phase 2: faster-whisper alignment (accurate, ~3–5 min) ───────────
        try:
            emit_event({"t": "phase", "phase": "aligning", "message": "Phase 2: Aligning word timestamps..."})
            model = load_align_model()
            transcript, has_clamped = align_transcript(transcript, audio_path, model)
            if has_clamped:
                repo.commit_alignment(tx_id, transcript, phase="degraded")
                repo.mark_transcription_degraded(tx_id, "timeline_clamped", "aligning")
                logger.warning(
                    "Phase 2: faster-whisper alignment committed with timeline clamping for %d segments "
                    "— phase stamped 'degraded' (failure_reason='timeline_clamped'), subprocess exiting",
                    len(transcript.segments),
                )
            else:
                repo.commit_alignment(tx_id, transcript, phase="aligned")
                logger.info(
                    "Phase 2: faster-whisper alignment committed for %d segments "
                    "— phase stamped 'aligned', subprocess exiting",
                    len(transcript.segments),
                )
        except Exception as exc:
            logger.warning(
                "Phase 2 (faster-whisper alignment) failed: %s — "
                "proportional timestamps from Phase 1 are retained "
                "(record stays 'aligned_proportional' until sweep times it out).",
                exc,
            )

        # ── Trigger biometric pass now that timestamps are committed ───────────
        try:
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import SegmentRecord
            with get_session() as s:
                seg_ids = [
                    row[0] for row in s.query(SegmentRecord.id)
                    .filter_by(transcription_id=tx_id)
                    .order_by(SegmentRecord.segment_index)
                    .all()
                ]
            if seg_ids and audio_path.exists():
                repo._schedule_biometric_pass(
                    seg_ids, str(audio_path), tx_id, run_inline=False
                )
                logger.info("Biometric pass scheduled post-alignment for %d segments", len(seg_ids))
        except Exception as bio_exc:
            logger.warning("Failed to schedule post-alignment biometric pass: %s", bio_exc)

        emit_event({"t": "done"})
        sys.exit(0)

    except Exception as exc:
        logger.error("Alignment worker failed for tx %d: %s", tx_id, exc)
        emit_event({"t": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
