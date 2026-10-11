"""Biometric worker — detached subprocess entry point.

Called by _schedule_biometric_pass() in the CLI paths to perform ECAPA-TDNN
voiceprint classification without blocking CLI exit or racing interpreter
shutdown against live C++ tensor kernels.

Usage (internal — do not call directly):
    python -m audiobench._biometric_worker <tx_id> <audio_path> <segment_ids_json>

Exit codes:
    0  — pass complete (or voiceprint not enrolled / audio missing — both are normal)
    1  — unexpected exception (logged to data/logs/biometric_worker.log)
"""

from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path


def _setup_logging() -> logging.Logger:
    """Configure file-based logging so failures are visible after the parent exits."""
    from audiobench.core.settings import get_settings

    log_dir = get_settings().data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "biometric_worker.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    # Stream to stderr as well so parent process / job runner captures output in job_<id>.log
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console_handler)
    return logging.getLogger("biometric_worker")


def main() -> int:
    """Entry point.  Returns an exit code."""
    if len(sys.argv) != 4:
        # Invoked incorrectly — nothing useful to log yet; just exit.
        return 1

    tx_id_str, audio_path_raw, segment_ids_json = sys.argv[1], sys.argv[2], sys.argv[3]
    logger = _setup_logging()

    try:
        tx_id = int(tx_id_str)
        audio_path = audio_path_raw if audio_path_raw else None
        segment_ids: list[int] = json.loads(segment_ids_json)
    except (ValueError, json.JSONDecodeError) as exc:
        # Argument parsing failure — log and exit; parent already returned cleanly.
        logger.error("Bad arguments: %s — argv=%s", exc, sys.argv[1:])
        return 1

    logger.info("Biometric worker started for tx #%d (%d segments)", tx_id, len(segment_ids))

    def _sigterm_handler(signum, frame):
        logger.info("Biometric worker received SIGTERM (preempted by transcription). Exiting cleanly.")
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except Exception:
        pass

    try:

        from audiobench.security.voiceprint import (
            _load_audio,
            _load_ecapa,
            is_enrolled,
            tag_segments_batch,
        )

        if not is_enrolled():
            logger.info("Voiceprint not enrolled — skipping tx #%d", tx_id)
            return 0

        if not audio_path or not Path(audio_path).exists():
            logger.warning(
                "Audio file missing or not specified for tx #%d — skipping biometric pass",
                tx_id,
            )
            return 0

        from audiobench.core.db_engine import init_db
        from audiobench.core.db_session import get_session as db_session
        from audiobench.storage.models import SegmentRecord

        from audiobench.core.settings import get_settings
        settings = get_settings()
        n_threads = settings.resolve_indexing_threads()
        try:
            import torch
            torch.set_num_threads(n_threads)
        except Exception:
            pass

        init_db()

        # ── Worker-Level Firewall Pre-Check ───────────────────────────────────
        try:
            from audiobench.media.classifier import classify_media
            from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

            with db_session() as s:
                tx_rec = s.get(TranscriptionRecord, tx_id)
                af_rec = s.get(AudioFileRecord, tx_rec.audio_file_id) if tx_rec and tx_rec.audio_file_id else None
                meta_dict = {}
                if af_rec:
                    try:
                        meta_dict["tags"] = json.loads(af_rec.tags) if af_rec.tags else {}
                    except Exception:
                        meta_dict["tags"] = {}
                    meta_dict["title"] = af_rec.track_title or af_rec.file_name
                    meta_dict["genre"] = af_rec.genre

                tags_data = meta_dict.get("tags") or {}
                is_confirmed = isinstance(tags_data, dict) and (tags_data.get("audiobook") or tags_data.get("user_confirmed"))

                cls_res = classify_media(
                    file_path=audio_path,
                    transcript_text=tx_rec.full_text if tx_rec else None,
                    metadata=meta_dict,
                    duration_seconds=tx_rec.duration_seconds if tx_rec else 0.0,
                    explicit_flag=bool(is_confirmed),
                )

                if cls_res.is_autonomous_bypass:
                    logger.info(
                        "Autonomous bypass inside worker for tx #%d: verified audiobook (%s)",
                        tx_id, ", ".join(cls_res.evidence),
                    )
                    if af_rec and isinstance(tags_data, dict):
                        tags_data["audiobook"] = True
                        tags_data["user_confirmed"] = True
                        af_rec.tags = json.dumps(tags_data)
                        s.commit()
                    return 0

                if cls_res.category == "class_b_soft_heuristic":
                    logger.info(
                        "Class B soft heuristic inside worker for tx #%d: %s — deferring to pending review",
                        tx_id, cls_res.review_reason,
                    )
                    if af_rec and isinstance(tags_data, dict):
                        tags_data["pending_privacy_review"] = cls_res.review_reason
                        af_rec.tags = json.dumps(tags_data)
                        s.commit()
                    return 0
        except Exception as exc:
            logger.debug("Worker firewall check skipped: %s", exc)

        waveform, sr = _load_audio(Path(audio_path))
        model = _load_ecapa()

        with db_session() as s:
            segs = (
                s.query(SegmentRecord)
                .filter(SegmentRecord.id.in_(segment_ids))
                .filter(SegmentRecord.privacy_tier == 0)
                .order_by(SegmentRecord.segment_index)
                .all()
            )
            slices: list = []
            ids: list[int] = []
            for seg in segs:
                start = int(seg.start_time * sr)
                end = int(seg.end_time * sr)
                audio_slice = waveform[start:end]
                if len(audio_slice) >= int(sr * 0.5):
                    slices.append(audio_slice)
                    ids.append(seg.id)

        if ids:
            chunk_size = 25
            total_tagged = 0
            for i in range(0, len(ids), chunk_size):
                chunk_ids = ids[i : i + chunk_size]
                chunk_slices = slices[i : i + chunk_size]
                tagged = tag_segments_batch(chunk_ids, chunk_slices, sr, model=model)
                total_tagged += len(tagged)

            logger.info(
                "Biometric pass complete for tx #%d (%d/%d segments classified, %d tagged)",
                tx_id,
                len(ids),
                len(segment_ids),
                total_tagged,
            )
        else:
            logger.info(
                "Biometric pass: no classifiable segments for tx #%d (all too short or already tagged)",
                tx_id,
            )

        return 0

    except Exception as exc:
        # Named accepted cost: failure leaves segment at privacy_tier=0 with no retry.
        # The log line here is the sole visibility mechanism; it is greppable.
        # If this fires repeatedly, the right follow-up is a biometric_status column
        # + daemon sweep (see: audiobench session 2026-07-27, Track 3 discussion).
        logger.error(
            "Biometric pass FAILED for tx #%d — segment(s) remain at privacy_tier=0: %s",
            tx_id,
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
