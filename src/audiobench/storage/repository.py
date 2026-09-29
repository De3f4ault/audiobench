"""Repository — CRUD operations for transcription data.

Provides a clean interface over SQLAlchemy for:
- Saving transcriptions with deduplication (by file hash)
- Querying transcription history
- Searching past transcriptions by text
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc

if TYPE_CHECKING:
    # Imported only for type annotations; never executed at runtime.
    # This avoids a circular import: storage.repository → cli.repl.session.
    from audiobench.cli.repl.session import ReplSession

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord
from audiobench.transcribe.transcription_result import AudioMetadata, Transcript

logger = get_logger("storage.repository")


class TranscriptionRepository:
    """CRUD operations for transcription persistence."""

    def _import_to_library(self, original_path: str, move: bool = False) -> str:
        """Import an audio file into the managed data/library directory.

        If ``move=True`` the original file is moved. By default (``move=False``),
        the file is copied, leaving the original intact. If a move succeeds
        but a later step raises, the file is moved back before re-raising so
        the caller never loses the original.

        Also moves any .cue / .srt / .vtt sidecars alongside the audio file.
        """
        import hashlib
        import shutil
        from pathlib import Path

        from audiobench.core.settings import get_settings

        settings = get_settings()
        library_dir = settings.data_dir / "library"
        library_dir.mkdir(parents=True, exist_ok=True)

        orig = Path(original_path).absolute()
        if library_dir in orig.parents or not orig.exists():
            return str(orig)

        file_id_hash = hashlib.md5(str(orig).encode()).hexdigest()[:8]
        target_name = f"{file_id_hash}_{orig.name}"
        target_path = library_dir / target_name

        moved_pairs: list[tuple] = []  # (dest, src) for rollback

        try:
            if not target_path.exists():
                if move:
                    try:
                        shutil.move(str(orig), str(target_path))
                        moved_pairs.append((target_path, orig))
                    except OSError as e:
                        logger.warning("Failed to move %s, falling back to copy: %s", orig, e)
                        shutil.copy2(str(orig), str(target_path))
                else:
                    shutil.copy2(str(orig), str(target_path))

            # Move sidecars alongside the audio file
            for ext in (".cue", ".srt", ".vtt", ".json"):
                sidecar = orig.with_suffix(ext)
                if sidecar.exists():
                    sidecar_target = target_path.with_suffix(ext)
                    if not sidecar_target.exists():
                        if move:
                            try:
                                shutil.move(str(sidecar), str(sidecar_target))
                                moved_pairs.append((sidecar_target, sidecar))
                            except OSError:
                                shutil.copy2(str(sidecar), str(sidecar_target))
                        else:
                            shutil.copy2(str(sidecar), str(sidecar_target))

            return str(target_path)

        except Exception:
            # Roll back: move each already-moved file back to its origin
            for dest, src in reversed(moved_pairs):
                try:
                    if Path(dest).exists():
                        shutil.move(str(dest), str(src))
                except Exception as rb_err:
                    logger.error("Rollback failed for %s → %s: %s", dest, src, rb_err)
            raise

    def save_transcription(
        self,
        transcript: Transcript,
        audio_metadata: AudioMetadata | None = None,
        chapter_id: int | None = None,
        on_phase: object | None = None,
        overwrite: bool = False,
        privacy_tier: int = 0,
    ) -> int:
        """
        Save a complete transcription to the database.

        Args:
            transcript:   The transcript to save.
            audio_metadata: Source audio metadata (for dedup by hash).
            chapter_id:   Optional chapter ID if this is a chapter transcription.
            on_phase:     Optional callback for phase progress.
            overwrite:    If True, deletes any existing transcription for this audio.
            privacy_tier: Minimum privacy tier to stamp on all segments.
                          0 = public (default, biometric pass may upgrade).
                          2 = intimate (--sensitive flag, stamps all segments Tier 2
                              regardless of voiceprint match).

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            # Find or create audio file record
            audio_record = None
            chapter_record = None

            if chapter_id:
                from audiobench.storage.models import ChapterRecord

                chapter_record = session.query(ChapterRecord).get(chapter_id)
                if chapter_record:
                    audio_record = session.query(AudioFileRecord).get(chapter_record.audio_file_id)
            elif audio_metadata and audio_metadata.file_hash:
                audio_record = (
                    session.query(AudioFileRecord)
                    .filter_by(file_hash=audio_metadata.file_hash)
                    .first()
                )

            # --- DEDUPLICATION ENFORCEMENT ---
            if audio_record and not chapter_id:
                existing_txs = session.query(TranscriptionRecord).filter_by(audio_file_id=audio_record.id).all()
                if existing_txs:
                    if not overwrite:
                        raise ValueError("Transcription already exists for this audio file.")

                    # Delete old transcriptions and their semantic vectors
                    from audiobench.memory.enums import SourceType
                    from audiobench.storage.expression_repository import ExpressionRepository

                    expr_repo = ExpressionRepository()
                    for old_tx in existing_txs:
                        expr_repo.delete_by_source(SourceType.AUDIO_TRANSCRIPT.value, old_tx.id)
                        session.delete(old_tx)
                    session.commit()

            if audio_record is None and audio_metadata and not chapter_id:
                # Import file to library
                new_path = self._import_to_library(audio_metadata.file_path)

                audio_record = AudioFileRecord(
                    file_path=new_path,
                    file_name=audio_metadata.file_name,
                    file_size_bytes=audio_metadata.file_size_bytes,
                    format=audio_metadata.format,
                    duration_seconds=audio_metadata.duration_seconds,
                    sample_rate=audio_metadata.sample_rate,
                    channels=audio_metadata.channels,
                    file_hash=audio_metadata.file_hash,
                )
                session.add(audio_record)
                session.flush()  # Get the ID

                # Auto-detect chapters
                from pathlib import Path

                from audiobench.chapters.detector import ChapterDetector

                try:
                    detector = ChapterDetector()
                    chapters_info = detector.detect(Path(new_path))
                    if chapters_info:
                        chap_dicts = [
                            {
                                "index": c.index,
                                "title": c.title,
                                "start_time": c.start_time,
                                "end_time": c.end_time,
                                "is_ghost": c.is_ghost,
                            }
                            for c in chapters_info
                        ]
                        # ChapterRepository manages its own session, but we are inside one.
                        # Wait, ChapterRepository.save_chapters opens its own `with get_session()`.
                        # Since we are already in a transaction, it might block if using sqlite with WAL,
                        # but get_session() typically handles nested or new connections.
                        # However, AudioFileRecord might not be committed yet!
                        # We must commit first so ChapterRepository can see the audio_file_id.
                        session.commit()

                        from audiobench.storage.chapter_repository import get_chapter_repo

                        get_chapter_repo().save_chapters(audio_record.id, chap_dicts)
                        logger.info(
                            "Auto-detected and saved %d chapters for %s",
                            len(chapters_info),
                            audio_record.file_name,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to auto-detect chapters for %s: %s", audio_record.file_name, e
                    )

            # Create transcription record
            tx_record = TranscriptionRecord(
                audio_file_id=audio_record.id if audio_record else None,
                source="file",
                file_name=audio_metadata.file_name if audio_metadata else "",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine=transcript.engine,
                model_name=transcript.model_name,
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                pipeline_phase="complete",
                speaker_map=json.dumps(transcript.speaker_map),
            )
            session.add(tx_record)
            session.flush()

            # Save segments — stamp privacy_tier from --sensitive flag (or 0 default)
            saved_segment_ids = []
            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                    chapter_id=chapter_id,
                    privacy_tier=privacy_tier,
                )
                session.add(seg_record)
                session.flush()  # get the ID
                saved_segment_ids.append(seg_record.id)

            if chapter_record:
                chapter_record.transcription_id = tx_record.id
                chapter_record.transcription_status = "completed"

            session.commit()
            logger.info(
                "Saved transcription #%d (%d segments, privacy_tier=%d)",
                tx_record.id, len(transcript.segments), privacy_tier,
            )

            # ── Background biometric pass ──────────────────────────────────────────
            # Only run if: voiceprint enrolled AND segments not already manually
            # stamped at Tier 2 via --sensitive (biometric pass would be redundant).
            # This is fire-and-forget in a daemon thread so it never blocks the REPL.
            if privacy_tier < 2:
                audio_path = audio_metadata.file_path if audio_metadata else None
                tx_id_for_pass = tx_record.id
                self._schedule_biometric_pass(
                    saved_segment_ids, audio_path, tx_id_for_pass, run_inline=False
                )



            return tx_record.id

    def _schedule_biometric_pass(
        self,
        segment_ids: list[int],
        audio_path: str | None,
        tx_id: int,
        run_inline: bool = True,
    ) -> None:
        """Launch a background biometric pass for newly saved segments.

        Args:
            segment_ids: DB IDs of SegmentRecord rows to classify.
            audio_path:  Path to the source audio file (needed for ECAPA-TDNN slicing).
            tx_id:       Transcription ID — used only for logging correlation.
            run_inline:  True  → daemon-thread path (safe in long-lived daemon process).
                         False → detached-subprocess path (safe for CLI; avoids racing
                                 C++ tensor teardown on interpreter shutdown, which
                                 causes SIGABRT — see: 2026-07-27 Track 3 diagnosis).

        Skips silently if no voiceprint is enrolled or no audio path given.

        Accepted cost (subprocess path): a subprocess crash leaves the segment at
        privacy_tier=0 with no automatic retry.  Failures are logged to
        data/logs/biometric_worker.log and are greppable.  If failures accumulate,
        the correct escalation is a biometric_status column + daemon sweep pickup.
        """
        if run_inline:
            # ── Daemon process path — thread is safe; process lifetime >> thread ──
            import threading

            def _run() -> None:
                try:
                    from pathlib import Path

                    from audiobench.security.voiceprint import (
                        _load_audio,
                        _load_ecapa,
                        is_enrolled,
                        tag_segments_batch,
                    )

                    if not is_enrolled() or not audio_path or not Path(audio_path).exists():
                        return

                    from audiobench.core.db_session import get_session as db_session
                    from audiobench.storage.models import SegmentRecord

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
                        slices = []
                        ids = []
                        for seg in segs:
                            start = int(seg.start_time * sr)
                            end = int(seg.end_time * sr)
                            audio_slice = waveform[start:end]
                            if len(audio_slice) >= int(sr * 0.5):
                                slices.append(audio_slice)
                                ids.append(seg.id)

                    tag_segments_batch(ids, slices, sr, model=model)
                    logger.info(
                        "Biometric pass complete for tx #%d (%d segments evaluated)",
                        tx_id, len(ids),
                    )
                except Exception as exc:
                    logger.warning("Background biometric pass failed for tx #%d: %s", tx_id, exc)

            t = threading.Thread(target=_run, daemon=True, name=f"bio-pass-tx{tx_id}")
            t.start()

        else:
            # ── CLI process path — detached subprocess avoids interpreter-shutdown race ──
            # Guard: check enrollment before paying subprocess startup cost.
            # is_enrolled() is a single file-existence check — effectively free.
            try:
                from audiobench.security.voiceprint import is_enrolled
                if not is_enrolled():
                    logger.debug("Biometric pass skipped for tx #%d — no voiceprint enrolled", tx_id)
                    return
            except Exception:
                # voiceprint module unavailable (SpeechBrain not installed) — skip silently.
                return

            import json
            from pathlib import Path

            from audiobench.jobs.scheduler import enqueue, ensure_worker

            try:
                fname = Path(audio_path).name if audio_path else f"tx #{tx_id}"
                enqueue(
                    job_type="biometric_pass",
                    args=[
                        "_biometric_worker",
                        str(tx_id),
                        audio_path or "",
                        json.dumps(segment_ids),
                    ],
                    slot="indexing",
                    file_label=fname,
                    command_display=f"biometric analysis {fname}",
                    # max_attempts=3: allows recovery across 2 preemptions before
                    # permanent failure. A preempted job is reset to pending by
                    # preempt_indexing_jobs() — startup_recovery checks attempt < max_attempts.
                    max_attempts=3,
                )
                ensure_worker()
                logger.info("Biometric pass queued via scheduler for tx #%d", tx_id)
            except Exception as exc:
                logger.warning(
                    "Failed to queue biometric worker for tx #%d: %s",
                    tx_id, exc,
                )



    def save_live_session(self, transcript: Transcript, on_phase: object | None = None) -> int:
        """Save a live transcription session to the database.

        Live sessions have no source audio file.

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            tx_record = TranscriptionRecord(
                audio_file_id=None,
                source="live",
                file_name="🎤 Live session",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine="faster-whisper",
                model_name=transcript.model_name if transcript.model_name else "base",
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                pipeline_phase="complete",
            )
            session.add(tx_record)
            session.flush()

            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                )
                session.add(seg_record)

            session.commit()
            logger.info(
                "Saved live session #%d (%d segments)", tx_record.id, len(transcript.segments)
            )



            return tx_record.id

    def find_by_hash(self, file_hash: str) -> TranscriptionRecord | None:
        """Find an existing transcription by audio file hash (deduplication).

        Returns the most recent *completed* transcription for the given file hash,
        or None.  Failed/in-progress records are intentionally excluded so they
        are never mistaken for a valid cache hit and silently skip re-transcription.
        """
        with get_session() as session:
            audio = session.query(AudioFileRecord).filter_by(file_hash=file_hash).first()
            if audio is None:
                return None

            return (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.audio_file_id == audio.id, TranscriptionRecord.pipeline_phase.in_(["complete", "completed"]))
                .order_by(desc(TranscriptionRecord.created_at))
                .first()
            )

    def get_history(
        self, limit: int = 20, offset: int = 0, chapter_mode: bool | None = None
    ) -> list[dict]:
        """Get recent transcription history.

        Args:
            chapter_mode: If True, returns only chapter transcripts. If False, returns only master transcripts. If None, returns all.
        Returns:
            List of dicts with transcription + audio metadata.
        """
        with get_session() as session:
            query = session.query(TranscriptionRecord)
            from audiobench.storage.models import ChapterRecord

            subquery = session.query(ChapterRecord.transcription_id).filter(
                ChapterRecord.transcription_id.isnot(None)
            )
            if chapter_mode is True:
                query = query.filter(TranscriptionRecord.id.in_(subquery))
            elif chapter_mode is False:
                query = query.filter(~TranscriptionRecord.id.in_(subquery))

            records = (
                query.order_by(desc(TranscriptionRecord.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )

            results = []
            for rec in records:
                if rec.file_name:
                    label = rec.file_name
                elif rec.source == "live":
                    label = "🎤 Live session"
                elif rec.source == "reimport":
                    audio = rec.audio_file
                    label = "📥 " + (audio.file_name if audio else "Imported transcript")
                else:
                    audio = rec.audio_file
                    label = audio.file_name if audio else "unknown"
                results.append(
                    {
                        "id": rec.id,
                        "file_name": label,
                        "source": rec.source,
                        "language": rec.language,
                        "model": rec.model_name,
                        "word_count": rec.word_count,
                        "duration": rec.duration_seconds,
                        "status": rec.pipeline_phase,
                        "audio_file_id": rec.audio_file_id,
                        "chapter_id": getattr(rec, "chapter_id", None),
                        "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                        "created_at": rec.created_at.isoformat() if rec.created_at else "",
                        "text_preview": rec.full_text[:100] + "..."
                        if len(rec.full_text) > 100
                        else rec.full_text,
                    }
                )

            return results

    def get_untranscribed_files(self) -> list[dict]:
        """Get audio files that have no transcription record (Awaiting Transcription)."""
        with get_session() as session:
            # Subquery to find audio_file_ids that HAVE transcriptions
            subq = session.query(TranscriptionRecord.audio_file_id).filter(
                TranscriptionRecord.audio_file_id.isnot(None)
            )

            # Find audio files NOT in that subquery
            records = (
                session.query(AudioFileRecord)
                .filter(~AudioFileRecord.id.in_(subq))
                .order_by(desc(AudioFileRecord.created_at))
                .all()
            )

            return [
                {
                    "id": rec.id,
                    "file_name": rec.file_name,
                    "file_path": rec.file_path,
                    "duration_seconds": rec.duration_seconds,
                    "file_size_bytes": rec.file_size_bytes,
                    "created_at": rec.created_at.isoformat() if rec.created_at else "",
                    "tags": rec.tags,
                }
                for rec in records
            ]

    def get_idle_transcripts(self) -> list[dict]:
        """Get transcripts that have not been semantically chunked (Idle Transcripts)."""
        with get_session() as session:
            # In a real scenario we'd check ExpressionRecord source_id, but we can approximate
            # by looking for completed transcriptions. If they exist but have no expressions.
            # For now, let's just query completed transcriptions. The Library UI will handle the logic.
            # Actually, audiobench doesn't have an easy ExpressionRecord join here since it's in a different module.
            # Let's just return all completed transcripts and let the UI query the daemon/expression repo.
            records = (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.pipeline_phase.in_(["complete", "completed"]))
                .order_by(desc(TranscriptionRecord.created_at))
                .all()
            )
            return [
                {
                    "id": rec.id,
                    "file_name": rec.file_name
                    or (rec.audio_file.file_name if rec.audio_file else "unknown"),
                    "audio_file_id": rec.audio_file_id,
                    "created_at": rec.created_at.isoformat() if rec.created_at else "",
                }
                for rec in records
            ]

    def get_file_by_path(self, file_path: str) -> dict | None:
        """Find an audio file record by its absolute path."""
        from pathlib import Path

        abs_path = str(Path(file_path).absolute())
        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(file_path=abs_path).first()
            if not rec:
                return None
            return {
                "id": rec.id,
                "file_path": rec.file_path,
                "file_name": rec.file_name,
                "duration_seconds": rec.duration_seconds,
                "format": rec.format,
                "file_size_bytes": rec.file_size_bytes,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
            }

    def get_audio_file(self, audio_file_id: int) -> dict | None:
        """Find an audio file record by its DB ID."""
        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
            if not rec:
                return None

            # Count transcriptions
            tx_count = (
                session.query(TranscriptionRecord).filter_by(audio_file_id=audio_file_id).count()
            )

            return {
                "id": rec.id,
                "file_path": rec.file_path,
                "file_name": rec.file_name,
                "duration_seconds": rec.duration_seconds,
                "format": rec.format,
                "file_size_bytes": rec.file_size_bytes,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
                "transcript_count": tx_count,
            }

    def get_or_create_file(self, file_path: str) -> int:
        """Find an audio file record by path, or create a stub if it doesn't exist."""
        import os
        from pathlib import Path

        # Import to library first
        abs_path = self._import_to_library(file_path)

        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(file_path=abs_path).first()
            if rec:
                return rec.id

            # Create a stub record. It will be filled in properly when transcribed.
            path_obj = Path(abs_path)
            file_size = os.path.getsize(abs_path) if path_obj.exists() else 0

            new_rec = AudioFileRecord(
                file_path=abs_path,
                file_name=path_obj.name,
                file_size_bytes=file_size,
                format=path_obj.suffix.lstrip("."),
            )
            session.add(new_rec)
            session.commit()
            return new_rec.id

    def get_latest_transcript_for_file(
        self,
        audio_file_id: int,
        session: "ReplSession | None" = None,
    ) -> dict | None:
        """Find the most recent completed transcription for a file."""
        with get_session() as db_session:
            rec = (
                db_session.query(TranscriptionRecord)
                .filter(
                    TranscriptionRecord.audio_file_id == audio_file_id,
                    TranscriptionRecord.pipeline_phase.in_(["complete", "completed"])
                )
                .order_by(desc(TranscriptionRecord.created_at))
                .first()
            )
            if not rec:
                return None
            return self.get_by_id(rec.id, session=session)

    def search(
        self,
        query: str,
        limit: int = 10,
        session: "ReplSession | None" = None,
    ) -> list[dict]:
        """Search transcriptions by text content.

        Args:
            query: Search string (case-insensitive LIKE).
            limit: Maximum number of results.
            session: The active ReplSession, or None.  When None (or locked at
                Tier 0), text_preview is replaced with "[REDACTED]" for any
                transcript that contains segments with privacy_tier >= 2.
                Uses a single EXISTS subquery per result — no N+1 penalty.

        Returns:
            List of matching transcription dicts.
        """
        with get_session() as db_session:
            from sqlalchemy import and_, exists

            effective_tier: int = session.effective_tier() if session is not None else 0

            tokens = [t.strip() for t in query.split() if t.strip()]
            filters = [TranscriptionRecord.full_text.ilike(f"%{t}%") for t in tokens]

            records = (
                db_session.query(TranscriptionRecord)
                .filter(and_(*filters) if filters else True)
                .order_by(desc(TranscriptionRecord.created_at))
                .limit(limit)
                .all()
            )

            # Pre-compute which transcript IDs have protected segments.
            # One EXISTS query per candidate — still O(N) but each is a fast
            # index scan on segments.privacy_tier (indexed column).
            if effective_tier < 2:
                protected_ids: set[int] = set()
                for rec in records:
                    has_protected = db_session.query(
                        exists().where(
                            SegmentRecord.transcription_id == rec.id,
                            SegmentRecord.privacy_tier >= 2,
                        )
                    ).scalar()
                    if has_protected:
                        protected_ids.add(rec.id)
            else:
                protected_ids = set()

            results = []
            for rec in records:
                if rec.id in protected_ids:
                    preview = "[REDACTED]"
                else:
                    preview = (rec.full_text or "")[:200]
                results.append(
                    {
                        "id": rec.id,
                        "file_name": rec.file_name
                        or (rec.audio_file.file_name if rec.audio_file else "unknown"),
                        "language": rec.language,
                        "text_preview": preview,
                        "created_at": rec.created_at.isoformat() if rec.created_at else "",
                    }
                )
            return results


    def search_transcripts(self, query: str = "", limit: int = 15) -> list[dict]:
        """Search transcripts by filename and full text for the /load picker.

        Only returns completed, non-empty transcripts.  When *query* is empty
        (or whitespace-only) the most-recent *limit* records are returned —
        the same behaviour as get_history() but lighter (no chapter filtering).

        Args:
            query: Free-text search string; matched against file_name and
                   full_text with case-insensitive LIKE.  Multiple words are
                   OR-combined on file_name and AND-combined on full_text so
                   a filename hit is always surfaced.
            limit: Maximum rows to return.

        Returns:
            List of dicts with id, file_name, language, model, word_count,
            duration, status, created_at — same shape as get_history() rows.
        """
        with get_session() as session:
            from sqlalchemy import and_, or_

            base = (
                session.query(TranscriptionRecord)
                .filter(
                    TranscriptionRecord.pipeline_phase.in_(["complete", "completed"]),
                    TranscriptionRecord.word_count > 0,
                )
            )

            tokens = [t.strip() for t in query.split() if t.strip()]
            if tokens:
                # Filename: any token matches (OR)
                name_filters = or_(
                    *[TranscriptionRecord.file_name.ilike(f"%{t}%") for t in tokens]
                )
                # Full-text: all tokens present (AND — more precise)
                text_filters = and_(
                    *[TranscriptionRecord.full_text.ilike(f"%{t}%") for t in tokens]
                )
                base = base.filter(or_(name_filters, text_filters))

            records = (
                base.order_by(desc(TranscriptionRecord.created_at))
                .limit(limit)
                .all()
            )

            results = []
            for rec in records:
                if rec.file_name:
                    label = rec.file_name
                elif rec.source == "live":
                    label = "🎤 Live session"
                elif rec.source == "reimport":
                    audio = rec.audio_file
                    label = "📥 " + (audio.file_name if audio else "Imported transcript")
                else:
                    audio = rec.audio_file
                    label = audio.file_name if audio else "unknown"

                results.append(
                    {
                        "id": rec.id,
                        "file_name": label,
                        "language": rec.language,
                        "model": rec.model_name,
                        "word_count": rec.word_count,
                        "duration": rec.duration_seconds,
                        "status": rec.pipeline_phase,
                        "created_at": rec.created_at.isoformat() if rec.created_at else "",
                    }
                )
            return results

    # ── Privacy-tier resolution helpers ───────────────────────────────────────
    # Used by derived-expression ingest sites (knowledge_ingester, memoir_writer,
    # search_ingester) to inherit the correct tier from source segments.
    # Both helpers are single-table, fully parameterized queries — no string
    # interpolation — so the pattern stays safe if copy-pasted to future sites.

    def get_max_privacy_tier_for_transcriptions(self, tx_ids: list[int]) -> int:
        """Return MAX(segments.privacy_tier) across all segments of the given
        transcription IDs.  Returns 0 when *tx_ids* is empty or no segments exist.

        Single indexed query — O(1) round-trips, no joins.

        Args:
            tx_ids: List of transcription primary-key IDs.
        """
        if not tx_ids:
            return 0
        from sqlalchemy import func
        with get_session() as db_session:
            result = (
                db_session.query(func.max(SegmentRecord.privacy_tier))
                .filter(SegmentRecord.transcription_id.in_(tx_ids))
                .scalar()
            )
            return int(result) if result else 0

    def get_max_privacy_tier_for_segments(self, segment_ids: list[int]) -> int:
        """Return MAX(segments.privacy_tier) for a specific list of segment IDs.
        Used by search_ingester, which works at segment-ID granularity directly.

        Returns 0 when *segment_ids* is empty or no segments exist.

        Single indexed query — O(1) round-trips, no joins.

        Args:
            segment_ids: List of segment primary-key IDs.
        """
        if not segment_ids:
            return 0
        from sqlalchemy import func
        with get_session() as db_session:
            result = (
                db_session.query(func.max(SegmentRecord.privacy_tier))
                .filter(SegmentRecord.id.in_(segment_ids))
                .scalar()
            )
            return int(result) if result else 0

    def get_by_id(
        self,
        transcription_id: int,
        session: "ReplSession | None" = None,
    ) -> dict | None:
        """Get full transcription by ID including all segments.

        Args:
            transcription_id: Primary key of the transcription.
            session: The active ReplSession, or None.  When None (all standalone
                CLI commands, API routes, daemon workers, and AI chat), the caller
                is treated as locked at Tier 0.  Segments whose privacy_tier >= 2
                have their text replaced with "[REDACTED]" and the top-level
                full_text / raw_text fields are fully replaced when any such
                segment exists — full replacement is deliberate (surgical splicing
                leaks structure/context even without the literal content).
        """
        with get_session() as db_session:
            rec = db_session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return None

            # Compute effective tier once — 0 when no session (always locked).
            effective_tier: int = session.effective_tier() if session is not None else 0

            # Build the segment list with per-segment redaction.
            sorted_segs = sorted(rec.segments, key=lambda s: s.segment_index)
            segments_out = []
            has_protected = False
            for seg in sorted_segs:
                is_protected = seg.privacy_tier >= 2 and seg.privacy_tier > effective_tier
                if is_protected:
                    has_protected = True
                segments_out.append(
                    {
                        "index":        seg.segment_index,
                        "text":         "[REDACTED]" if is_protected else seg.text,
                        "start":        seg.start_time,
                        "end":          seg.end_time,
                        "speaker":      seg.speaker,
                        "chapter_id":   seg.chapter_id,
                        # Always expose tier so callers can know *something* exists,
                        # and the redacted flag so callers don't need to recompute.
                        "privacy_tier": seg.privacy_tier,
                        "redacted":     is_protected,
                    }
                )

            # Redact full_text / raw_text if any segment was protected.
            # Full replacement (not surgical splicing) — a topic-shaped hole in the
            # full text leaks structure.  If the transcript contains any owner-voice
            # content and the caller is locked, the whole field becomes unusable.
            if has_protected:
                full_text_out = (
                    "[REDACTED: transcript contains protected segments. "
                    r"\unlock 2 to view.]"
                )
                raw_text_out = ""
            else:
                full_text_out = rec.full_text
                raw_text_out = rec.raw_text or ""

            data = {
                "id": rec.id,
                "file_name": rec.file_name
                or (
                    "🎤 Live session"
                    if rec.source == "live"
                    else (
                        "📥 " + (rec.audio_file.file_name if rec.audio_file else "Imported transcript")
                        if rec.source == "reimport"
                        else (rec.audio_file.file_name if rec.audio_file else "unknown")
                    )
                ),
                "file_path": (rec.audio_file.file_path if rec.audio_file else None),
                "audio_file_id": rec.audio_file_id,
                "source": rec.source,
                "full_text": full_text_out,
                "raw_text": raw_text_out,
                "language": rec.language,
                "language_probability": rec.language_probability,
                "engine": rec.engine,
                "model": rec.model_name,
                "duration": rec.duration_seconds,
                "word_count": rec.word_count,
                "segment_count": rec.segment_count,
                "status": rec.pipeline_phase,
                "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
                "segments": segments_out,
                "speaker_map": json.loads(rec.speaker_map) if rec.speaker_map else {},
                "chapters": [],
            }

            # Fetch chapters if we have an audio file ID
            if rec.audio_file_id:
                from audiobench.storage.chapter_repository import get_chapter_repo

                chapter_records = get_chapter_repo().get_chapters(rec.audio_file_id)
                data["chapters"] = [c.to_dict() for c in chapter_records]

            return data


    def update_text(self, transcription_id: int, new_text: str) -> bool:
        """Update the full text of a transcription (used by REPL .edit).

        Returns True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            rec.full_text = new_text
            rec.word_count = len(new_text.split())
            session.commit()
            logger.info("Updated text for transcription #%d", transcription_id)
            return True

    def update_full_text(
        self,
        transcription_id: int,
        refined_text: str,
        raw_text: str,
    ) -> bool:
        """Update transcript with LLM-refined text, preserving the raw version.

        If raw_text is currently empty (pre-existing transcription), seeds it
        from the current full_text before overwriting. Stamps refined_at.

        Args:
            transcription_id: The transcription to update.
            refined_text: LLM-cleaned transcript text.
            raw_text: Original Whisper output (preserved in raw_text column).

        Returns:
            True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            # Seed raw_text for pre-existing transcriptions that weren't yet cleaned
            if not rec.raw_text:
                rec.raw_text = raw_text
            rec.full_text = refined_text
            rec.word_count = len(refined_text.split())
            rec.refined_at = datetime.now(UTC)
            session.commit()
            logger.info(
                "Refined transcript #%d (%d → %d chars)",
                transcription_id,
                len(raw_text),
                len(refined_text),
            )
            return True

    def update_segments(
        self,
        transcription_id: int,
        cleaned_texts: list[str],
    ) -> bool:
        """Bulk-update segment texts for a transcription (timestamps unchanged).

        Args:
            transcription_id: The transcription whose segments to update.
            cleaned_texts: New text for each segment, in segment_index order.

        Returns:
            True if all segments were updated, False if count mismatch or not found.
        """
        with get_session() as session:
            segments = (
                session.query(SegmentRecord)
                .filter_by(transcription_id=transcription_id)
                .order_by(SegmentRecord.segment_index)
                .all()
            )
            if not segments:
                return False
            if len(segments) != len(cleaned_texts):
                logger.warning(
                    "update_segments: segment count mismatch for #%d (%d vs %d)",
                    transcription_id,
                    len(segments),
                    len(cleaned_texts),
                )
                return False
            for seg, new_text in zip(segments, cleaned_texts, strict=True):
                seg.text = new_text
            session.commit()
            logger.info(
                "Updated %d segments for transcription #%d",
                len(segments),
                transcription_id,
            )
            return True

    def get_refinement_status(self, transcription_id: int) -> dict | None:
        """Return refinement status for a transcription.

        Returns:
            Dict with keys: is_refined, refined_at, raw_text, full_text.
            None if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return None
            return {
                "id": rec.id,
                "is_refined": rec.refined_at is not None,
                "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                "raw_text": rec.raw_text or "",
                "full_text": rec.full_text or "",
            }

    def delete_by_id(self, transcription_id: int) -> bool:
        """Delete a transcription by ID.

        Returns True if found and deleted, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            from audiobench.memory.enums import SourceType
            from audiobench.storage.expression_repository import ExpressionRepository

            ExpressionRepository().delete_by_source(
                SourceType.AUDIO_TRANSCRIPT.value, transcription_id
            )
            session.delete(rec)
            session.commit()
            logger.info("Deleted transcription #%d", transcription_id)
            return True

    def delete_all(self) -> int:
        """Delete all transcriptions. Returns number deleted."""
        with get_session() as session:
            tx_ids = [r[0] for r in session.query(TranscriptionRecord.id).all()]
            from audiobench.memory.enums import SourceType
            from audiobench.storage.expression_repository import ExpressionRepository

            expr_repo = ExpressionRepository()
            for tx_id in tx_ids:
                expr_repo.delete_by_source(SourceType.AUDIO_TRANSCRIPT.value, tx_id)

            count = len(tx_ids)
            session.query(SegmentRecord).delete()
            session.query(TranscriptionRecord).delete()
            session.commit()
            logger.info("Deleted %d transcription(s)", count)
            return count

    def begin_transcription(
        self,
        audio_metadata: AudioMetadata | None,
        engine: str,
        model_name: str,
        chapter_id: int | None = None,
        overwrite: bool = False,
    ) -> int:
        """
        Insert a TranscriptionRecord with status='transcribing'.
        Returns the new tx_id. Called BEFORE the Gemini/Whisper API call.
        """
        with get_session() as session:
            audio_record = None
            chapter_record = None

            if chapter_id:
                chapter_record = session.query(ChapterRecord).get(chapter_id)
                if chapter_record:
                    audio_record = session.query(AudioFileRecord).get(chapter_record.audio_file_id)
            elif audio_metadata and audio_metadata.file_hash:
                audio_record = (
                    session.query(AudioFileRecord)
                    .filter_by(file_hash=audio_metadata.file_hash)
                    .first()
                )

            # --- DEDUPLICATION ENFORCEMENT ---
            if audio_record and not chapter_id:
                existing_txs = session.query(TranscriptionRecord).filter_by(audio_file_id=audio_record.id).all()
                # Separate completed records from incomplete/stuck ones.
                # Incomplete stubs (transcribing / failed / degraded) are ALWAYS
                # purged so that a re-run never accumulates ghost 0-word rows.
                # A completed record only blocks if the caller did not pass overwrite.
                has_completed = any(t.pipeline_phase in ("complete", "completed") for t in existing_txs)
                incomplete_txs = [t for t in existing_txs if t.pipeline_phase not in ("complete", "completed")]

                if has_completed and not overwrite:
                    raise ValueError("Transcription already exists for this audio file.")

                # Delete ALL existing records (completed only when overwrite=True,
                # incomplete stubs always) so we start with a clean slate.
                txs_to_delete = existing_txs if (has_completed and overwrite) else incomplete_txs
                if txs_to_delete:
                    from audiobench.memory.enums import SourceType
                    from audiobench.storage.expression_repository import ExpressionRepository

                    expr_repo = ExpressionRepository()
                    for old_tx in txs_to_delete:
                        expr_repo.delete_by_source(SourceType.AUDIO_TRANSCRIPT.value, old_tx.id)
                        session.delete(old_tx)
                    session.commit()

            if audio_record is None and audio_metadata and not chapter_id:
                # Import file to library
                new_path = self._import_to_library(audio_metadata.file_path)

                audio_record = AudioFileRecord(
                    file_path=new_path,
                    file_name=audio_metadata.file_name,
                    file_size_bytes=audio_metadata.file_size_bytes,
                    format=audio_metadata.format,
                    duration_seconds=audio_metadata.duration_seconds,
                    sample_rate=audio_metadata.sample_rate,
                    channels=audio_metadata.channels,
                    file_hash=audio_metadata.file_hash,
                )
                session.add(audio_record)
                session.flush()

                from pathlib import Path

                from audiobench.chapters.detector import ChapterDetector
                try:
                    detector = ChapterDetector()
                    chapters_info = detector.detect(Path(new_path))
                    if chapters_info:
                        chap_dicts = [
                            {
                                "index": c.index,
                                "title": c.title,
                                "start_time": c.start_time,
                                "end_time": c.end_time,
                                "is_ghost": c.is_ghost,
                            }
                            for c in chapters_info
                        ]
                        session.commit()
                        from audiobench.storage.chapter_repository import get_chapter_repo
                        get_chapter_repo().save_chapters(audio_record.id, chap_dicts)
                        logger.info("Auto-detected and saved %d chapters for %s", len(chapters_info), audio_record.file_name)
                except Exception as e:
                    logger.warning("Failed to auto-detect chapters for %s: %s", audio_record.file_name, e)

            # Create the record in transcribing status
            tx_record = TranscriptionRecord(
                audio_file_id=audio_record.id if audio_record else None,
                source="file",
                file_name=audio_metadata.file_name if audio_metadata else "",
                full_text="",
                language="en",
                language_probability=0.0,
                engine=engine,
                model_name=model_name,
                duration_seconds=0.0,
                word_count=0,
                segment_count=0,
                pipeline_phase="transcribing",  # mapped to DB status
                attempt_count=0,
                speaker_map="{}",
            )
            session.add(tx_record)
            session.flush()
            tx_id = tx_record.id
            session.commit()
            return tx_id

    def commit_transcript_text(
        self,
        tx_id: int,
        transcript: Transcript,
        chapter_id: int | None = None,
        privacy_tier: int = 0,
    ) -> None:
        """
        After engine.transcribe() completes: write full_text, segments, language, duration.
        Atomic UPDATE status='transcribed', attempt_count=0.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if not tx_record:
                raise ValueError(f"TranscriptionRecord {tx_id} not found.")

            tx_record.full_text = transcript.text
            tx_record.language = transcript.language
            tx_record.language_probability = transcript.language_probability
            tx_record.duration_seconds = transcript.duration_seconds
            tx_record.word_count = transcript.word_count
            tx_record.segment_count = transcript.segment_count
            tx_record.pipeline_phase = "transcribed"
            tx_record.attempt_count = 0

            # Delete old segments if this is a retry
            session.query(SegmentRecord).filter_by(transcription_id=tx_id).delete()

            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                    chapter_id=chapter_id,
                    privacy_tier=privacy_tier,
                )
                session.add(seg_record)

            session.commit()

    def commit_alignment(self, tx_id: int, transcript: Transcript, phase: str = "aligned") -> None:
        """
        After align_transcript() completes: UPDATE segment start_time/end_time.
        Atomic UPDATE pipeline_phase=phase, attempt_count=0.

        Args:
            phase: The pipeline phase to stamp on this record.
                   Use ``"aligned_proportional"`` after Phase 1 (proportional
                   timestamps — subprocess still running).
                   Use ``"aligned"`` (default) after Phase 2 (faster-whisper
                   timestamps — subprocess about to exit).
                   This distinction lets the sweep and stuck-detection logic
                   distinguish "in progress" from "fully done" without timing
                   heuristics.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if not tx_record:
                return

            # Update segments
            for seg in transcript.segments:
                session.query(SegmentRecord).filter_by(
                    transcription_id=tx_id, segment_index=seg.id
                ).update({
                    "start_time": seg.start,
                    "end_time": seg.end,
                })

            tx_record.pipeline_phase = phase
            tx_record.attempt_count = 0
            # Persist the corrected duration so get_by_id, summaries, and
            # embed queries all reflect the real audio length (Gemini stores 0.0).
            if transcript.duration_seconds > 0.0:
                tx_record.duration_seconds = transcript.duration_seconds
            session.commit()

    def commit_diarization(self, tx_id: int, transcript: Transcript) -> None:
        """
        After diarization completes: UPDATE segment.speaker, transcript.speaker_map.
        Atomic UPDATE status='diarized', attempt_count=0.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if not tx_record:
                return

            for seg in transcript.segments:
                session.query(SegmentRecord).filter_by(
                    transcription_id=tx_id, segment_index=seg.id
                ).update({"speaker": seg.speaker})

            tx_record.speaker_map = json.dumps(transcript.speaker_map)
            tx_record.pipeline_phase = "diarized"
            tx_record.attempt_count = 0
            session.commit()

    def finalize_transcription(
        self,
        tx_id: int,
        transcript: Transcript,
        chapter_id: int | None = None,
        on_phase: Any = None,
        privacy_tier: int = 0,
        audio_metadata: AudioMetadata | None = None,
        run_inline: bool = False,
        is_degraded: bool = False,
    ) -> None:
        """
        After speaker naming: UPDATE status='complete', attempt_count=0.
        Triggers _register_expressions() and background biometric pass.

        Args:
            run_inline: Passed to _schedule_biometric_pass.
                        False (default) → detached subprocess path, safe for CLI exit.
                        True            → daemon-thread path, safe in long-lived daemon.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if not tx_record:
                return

            tx_record.speaker_map = json.dumps(transcript.speaker_map)
            if not is_degraded:
                tx_record.pipeline_phase = "complete"
                tx_record.failure_reason = None
            tx_record.attempt_count = 0

            if chapter_id:
                chapter_record = session.query(ChapterRecord).get(chapter_id)
                if chapter_record:
                    chapter_record.transcription_id = tx_id
                    chapter_record.transcription_status = "completed"

            # Get saved segment ids for biometric pass
            saved_segment_ids = [s.id for s in session.query(SegmentRecord.id).filter_by(transcription_id=tx_id).all()]
            session.commit()

            # ── Background biometric pass ──────────────────────────────────────────
            if privacy_tier < 2:
                audio_path = audio_metadata.file_path if audio_metadata else None
                self._schedule_biometric_pass(
                    saved_segment_ids, audio_path, tx_id, run_inline=run_inline
                )



    def mark_transcription_failed(
        self,
        tx_id: int,
        reason: str,
        failed_phase: str,
    ) -> None:
        """
        Set status='failed' when transcribing phase fails.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record:
                tx_record.pipeline_phase = "failed"
                tx_record.failure_reason = reason
                session.commit()

    def mark_transcription_degraded(
        self,
        tx_id: int,
        reason: str,
        failed_phase: str,
    ) -> None:
        """
        Set status='degraded' when a post-processing phase fails repeatedly.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record:
                tx_record.pipeline_phase = "degraded"
                tx_record.failure_reason = reason
                session.commit()

    def increment_attempt_count(self, tx_id: int) -> int:
        """
        Atomically increments attempt_count. Returns new count.
        """
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record:
                tx_record.attempt_count += 1
                new_count = tx_record.attempt_count
                session.commit()
                return new_count
        return 0

    def touch_transcription(self, tx_id: int) -> None:
        """Update the updated_at timestamp on a TranscriptionRecord as a heartbeat."""
        from datetime import UTC, datetime
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record:
                tx_record.updated_at = datetime.now(UTC)
                session.commit()

    def get_incomplete_transcriptions(self, max_age_minutes: int = 5) -> list[TranscriptionRecord]:
        """
        Returns all records where pipeline_phase NOT IN ('complete', 'completed', 'degraded', 'failed')
        AND updated_at < now() - max_age_minutes, with phase-specific grace periods:
          - 'transcribing': 60 min (Gemini API uploads can be slow)
          - 'aligned_proportional': 45 min (Phase 2 faster-whisper alignment ceiling
            is 30 min per the 1800s subprocess hard cap; 45 min adds buffer for slow
            machines — NOT measured against real Phase 2 timing data, revisit if
            false-positive stuck-detections occur on long files)
          - everything else: max_age_minutes (default 5 min)
        """
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import and_, or_
        cutoff_default = datetime.now(UTC) - timedelta(minutes=max_age_minutes)
        cutoff_transcribing = datetime.now(UTC) - timedelta(minutes=max(60, max_age_minutes))
        # 45min = 30min alignment hard cap (subprocess timeout) + 15min buffer.
        # Not derived from measured Phase 2 duration distribution — revisit if
        # false-positive stuck-detections occur on legitimately long alignments.
        cutoff_aligning = datetime.now(UTC) - timedelta(minutes=max(45, max_age_minutes))
        with get_session() as session:
            return session.query(TranscriptionRecord).filter(
                ~TranscriptionRecord.pipeline_phase.in_(["complete", "completed", "degraded", "failed"]),
                or_(
                    and_(
                        TranscriptionRecord.pipeline_phase == "transcribing",
                        TranscriptionRecord.updated_at < cutoff_transcribing,
                    ),
                    and_(
                        TranscriptionRecord.pipeline_phase == "aligned_proportional",
                        TranscriptionRecord.updated_at < cutoff_aligning,
                    ),
                    and_(
                        ~TranscriptionRecord.pipeline_phase.in_(["transcribing", "aligned_proportional"]),
                        TranscriptionRecord.updated_at < cutoff_default,
                    ),
                )
            ).all()

    def get_degraded_for_backfill(self, max_backfill_attempts: int = 3) -> list[TranscriptionRecord]:
        """
        Returns degraded records eligible for background re-alignment or re-diarization.

        Records with ``failure_reason='timeline_clamped'`` are intentionally excluded.
        The Whisper timeline anomaly that caused the clamp is unidentified and deterministic —
        retrying the same audio would clamp again, and ``mark_backfill_exhausted`` would
        overwrite the diagnostic label with ``'alignment_exhausted'``, destroying the signal.
        These records remain readable and searchable but are never touched by the sweep.
        """
        from datetime import UTC, datetime

        from sqlalchemy import or_
        now = datetime.now(UTC)
        with get_session() as session:
            return session.query(TranscriptionRecord).filter(
                TranscriptionRecord.pipeline_phase == "degraded",
                TranscriptionRecord.failure_reason != "timeline_clamped",
                TranscriptionRecord.backfill_attempt_count < max_backfill_attempts,
                or_(
                    TranscriptionRecord.backfill_next_attempt_at == None,
                    TranscriptionRecord.backfill_next_attempt_at <= now,
                ),
            ).order_by(TranscriptionRecord.duration_seconds.asc()).all()

    def increment_backfill_attempt(self, tx_id: int, next_attempt_delay_seconds: int) -> None:
        """Increment backfill_attempt_count and set backfill_next_attempt_at."""
        from datetime import UTC, datetime, timedelta
        with get_session() as session:
            rec = session.query(TranscriptionRecord).get(tx_id)
            if rec:
                rec.backfill_attempt_count += 1
                rec.backfill_next_attempt_at = datetime.now(UTC) + timedelta(seconds=next_attempt_delay_seconds)
                session.commit()

    def mark_backfill_exhausted(self, tx_id: int) -> None:
        """Terminal state — all backfill attempts consumed, will never be retried."""
        with get_session() as session:
            rec = session.query(TranscriptionRecord).get(tx_id)
            if rec:
                rec.failure_reason = "alignment_exhausted"
                session.commit()
