"""Database upgrade and granular rollback engine for AudioBench transcriptions.

Implements the in-database archival schema, atomic record replacement,
single-record rollback, batch rollback, and post-hoc audit sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.memory.enums import SourceType
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import (
    ArchivedSegmentRecord,
    ArchivedTranscriptionRecord,
    AudioFileRecord,
    Base,
    BookmarkRecord,
    ChapterRecord,
    SegmentRecord,
    TranscriptionRecord,
)

logger = get_logger("transcribe.upgrader")


@dataclass
class UpgradeResult:
    """Summary of a single transcription upgrade operation."""

    audio_file_id: int
    old_transcription_id: int | None
    new_transcription_id: int
    upgrade_batch_id: str
    archived_segments_count: int
    new_segments_count: int
    chapters_linked_count: int


@dataclass
class RollbackResult:
    """Summary of a single transcription rollback operation."""

    audio_file_id: int
    restored_transcription_id: int
    archived_id: int
    upgrade_batch_id: str
    segments_restored_count: int


def ensure_archival_tables(session: Session) -> None:
    """Ensure archived_transcriptions and archived_segments exist in the database."""
    bind = session.get_bind()
    Base.metadata.create_all(
        bind=bind,
        tables=[
            ArchivedTranscriptionRecord.__table__,
            ArchivedSegmentRecord.__table__,
        ],
    )


def create_pre_upgrade_backup(db_path: Path | str | None = None, suffix: str = "") -> Path:
    """Create a physical safety snapshot of the SQLite database before mutation.

    Uses SQLite's online backup API to ensure a clean, transactionally consistent
    snapshot even under active WAL mode.
    Returns the absolute Path of the created backup file.
    """
    if db_path is None:
        url = get_settings().database_url
        if url.startswith("sqlite:///"):
            db_path = Path(url.replace("sqlite:///", "")).resolve()
        else:
            db_path = Path("data/transcriptions.db").resolve()
    else:
        db_path = Path(db_path).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    tag = f"_{suffix}" if suffix else ""
    backup_path = db_path.parent / f"{db_path.name}.bak_upgrade{tag}_{ts}"

    logger.info("Creating pre-flight safety snapshot: %s", backup_path.name)

    # Use SQLite's online backup API for safe, tear-free WAL snapshotting
    src_conn = sqlite3.connect(str(db_path))
    dst_conn = sqlite3.connect(str(backup_path))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError("Backup snapshot creation failed or zero-byte file created!")

    logger.info("Backup snapshot verified: %d bytes", backup_path.stat().st_size)
    return backup_path


class TranscriptUpgrader:
    """Handles archival, replacement, and rollback of transcriptions."""

    def __init__(self, session: Session) -> None:
        self.session = session
        ensure_archival_tables(self.session)
        self.expr_repo = ExpressionRepository()

    def _purge_segment_vectors(self, segment_ids: list[int]) -> None:
        """Purge segment vectors from LanceDB for the given segment IDs."""
        if not segment_ids:
            return
        from audiobench.core.settings import get_settings
        if get_settings().disable_memory:
            return
        try:
            from audiobench.memory.singletons import get_segment_store
            store = get_segment_store()
            store.delete_segments(segment_ids)
        except Exception as exc:
            logger.warning("Segment vector purge warning for %d segments: %s", len(segment_ids), exc)

    def archive_transcription(
        self,
        tx: TranscriptionRecord,
        upgrade_batch_id: str,
    ) -> ArchivedTranscriptionRecord:
        """Clone an existing live transcription and its segments into archival tables."""
        arch_tx = ArchivedTranscriptionRecord(
            original_transcription_id=tx.id,
            audio_file_id=tx.audio_file_id,
            source=tx.source,
            file_name=tx.file_name,
            full_text=tx.full_text,
            raw_text=tx.raw_text,
            language=tx.language,
            language_probability=tx.language_probability,
            engine=tx.engine,
            model_name=tx.model_name,
            duration_seconds=tx.duration_seconds,
            word_count=tx.word_count,
            segment_count=tx.segment_count,
            pipeline_phase=tx.pipeline_phase,
            failure_reason=tx.failure_reason,
            attempt_count=tx.attempt_count,
            backfill_attempt_count=tx.backfill_attempt_count,
            backfill_next_attempt_at=tx.backfill_next_attempt_at,
            speaker_map=tx.speaker_map,
            refined_at=tx.refined_at,
            is_indexed=tx.is_indexed,
            created_at=tx.created_at,
            updated_at=tx.updated_at,
            upgrade_batch_id=upgrade_batch_id,
            archived_at=datetime.now(UTC),
        )
        self.session.add(arch_tx)
        self.session.flush()

        # Archive segments
        for seg in tx.segments:
            arch_seg = ArchivedSegmentRecord(
                archived_transcription_id=arch_tx.id,
                original_segment_id=seg.id,
                transcription_id=tx.id,
                segment_index=seg.segment_index,
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker=seg.speaker,
                chapter_id=seg.chapter_id,
                privacy_tier=seg.privacy_tier,
            )
            self.session.add(arch_seg)

        self.session.flush()
        logger.debug(
            "Archived transcription #%d (%d segments) under batch %s",
            tx.id,
            len(tx.segments),
            upgrade_batch_id,
        )
        return arch_tx

    def upgrade_record(
        self,
        audio_file_id: int,
        incoming_data: dict[str, Any],
        upgrade_batch_id: str,
        keep_chapters: bool = True,
    ) -> UpgradeResult:
        """Atomically archive old transcription (if any), delete old vectors,

        and insert new transcription and segments with chapter linkage.
        """
        # Find audio record
        audio_rec = self.session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
        if not audio_rec:
            raise ValueError(f"Audio file record id={audio_file_id} not found.")

        # Find existing live transcription
        existing_tx = (
            self.session.query(TranscriptionRecord)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(TranscriptionRecord.id.desc())
            .first()
        )

        old_tx_id = None
        archived_seg_count = 0

        if existing_tx:
            old_tx_id = existing_tx.id
            archived_seg_count = len(existing_tx.segments)
            old_segment_ids = [seg.id for seg in existing_tx.segments]
            # 1. Archive
            self.archive_transcription(existing_tx, upgrade_batch_id)

            # 2. Purge vectors from LanceDB/HNSW
            try:
                self.expr_repo.delete_by_source(SourceType.AUDIO_TRANSCRIPT.value, existing_tx.id, session=self.session)
            except Exception as e:
                logger.warning("Expression vector purge encountered warning for tx %d: %s", existing_tx.id, e)

            # 2b. Purge segment vectors from LanceDB
            self._purge_segment_vectors(old_segment_ids)

            # 3. Delete live transcription (cascades to segments, SET NULL on bookmarks/chat/notes)
            self.session.delete(existing_tx)
            self.session.flush()

        # 4. Extract incoming data
        full_text = str(incoming_data.get("text") or incoming_data.get("full_text") or "").strip()
        engine = incoming_data.get("engine") or "whisper"
        model_name = incoming_data.get("model_name") or "large-v3-turbo"
        language = incoming_data.get("language") or "en"
        raw_segs = incoming_data.get("segments") or []
        dur = float(
            incoming_data.get("duration_seconds")
            or incoming_data.get("audio", {}).get("duration_seconds")
            or audio_rec.duration_seconds
            or 0.0
        )
        word_count = int(incoming_data.get("word_count") or len(full_text.split()))

        # 5. Insert new TranscriptionRecord
        new_tx = TranscriptionRecord(
            audio_file_id=audio_file_id,
            source="reimport",
            file_name=audio_rec.file_name,
            full_text=full_text,
            raw_text=full_text,
            language=language,
            engine=engine,
            model_name=model_name,
            duration_seconds=dur,
            word_count=word_count,
            segment_count=len(raw_segs),
            pipeline_phase="complete",
            is_indexed=0,  # Scheduled for daemon re-indexing
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.session.add(new_tx)
        self.session.flush()

        # 6. Query existing chapters for midpoint linking
        chapters = []
        if keep_chapters:
            chapters = (
                self.session.query(ChapterRecord)
                .filter_by(audio_file_id=audio_file_id, is_ghost=0)
                .order_by(ChapterRecord.start_time.asc())
                .all()
            )

        linked_chapters_count = 0
        new_segment_records = []

        for idx, seg_dict in enumerate(raw_segs):
            s_start = float(seg_dict.get("start", seg_dict.get("start_time", 0.0)))
            s_end = float(seg_dict.get("end", seg_dict.get("end_time", 0.0)))
            s_text = str(seg_dict.get("text", "")).strip()
            s_spk = seg_dict.get("speaker")

            # Midpoint chapter linkage
            assigned_chap_id = None
            if chapters:
                if len(chapters) == 1:
                    assigned_chap_id = chapters[0].id
                else:
                    midpoint = (s_start + s_end) / 2.0
                    for c_idx, ch in enumerate(chapters):
                        is_last = c_idx == len(chapters) - 1
                        if ch.start_time <= midpoint and (midpoint < ch.end_time or is_last):
                            assigned_chap_id = ch.id
                            break
                    if not assigned_chap_id and chapters:
                        # Fallback to closest start time
                        assigned_chap_id = min(chapters, key=lambda c: abs(c.start_time - midpoint)).id

            if assigned_chap_id:
                linked_chapters_count += 1

            seg_rec = SegmentRecord(
                transcription_id=new_tx.id,
                segment_index=idx,
                text=s_text,
                start_time=s_start,
                end_time=s_end,
                speaker=s_spk,
                chapter_id=assigned_chap_id,
                privacy_tier=0,
            )
            self.session.add(seg_rec)
            new_segment_records.append(seg_rec)

        self.session.flush()

        # Re-link bookmarks and chapters to new transcription
        self.session.query(BookmarkRecord).filter_by(audio_file_id=audio_file_id).update(
            {"transcription_id": new_tx.id}, synchronize_session="fetch"
        )
        self.session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).update(
            {"transcription_id": new_tx.id}, synchronize_session="fetch"
        )

        return UpgradeResult(
            audio_file_id=audio_file_id,
            old_transcription_id=old_tx_id,
            new_transcription_id=new_tx.id,
            upgrade_batch_id=upgrade_batch_id,
            archived_segments_count=archived_seg_count,
            new_segments_count=len(new_segment_records),
            chapters_linked_count=linked_chapters_count,
        )

    def rollback_record(self, identifier: int | str) -> RollbackResult:
        """Roll back a single audio file's transcription from its most recent archive."""
        arch_query = self.session.query(ArchivedTranscriptionRecord)

        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            target_id = int(identifier)
            arch_tx = (
                arch_query.filter(
                    (ArchivedTranscriptionRecord.audio_file_id == target_id)
                    | (ArchivedTranscriptionRecord.original_transcription_id == target_id)
                    | (ArchivedTranscriptionRecord.id == target_id)
                )
                .order_by(ArchivedTranscriptionRecord.id.desc())
                .first()
            )
        else:
            name_term = str(identifier).strip()
            # Try matching audio_file_name or file_name
            arch_tx = (
                arch_query.filter(
                    (ArchivedTranscriptionRecord.file_name.ilike(f"%{name_term}%"))
                )
                .order_by(ArchivedTranscriptionRecord.id.desc())
                .first()
            )

        if not arch_tx:
            raise ValueError(f"No archived transcription found matching identifier '{identifier}'.")

        audio_file_id = arch_tx.audio_file_id
        if not audio_file_id:
            raise ValueError(f"Archived record #{arch_tx.id} has no linked audio_file_id.")

        # Find current live transcription
        current_tx = (
            self.session.query(TranscriptionRecord)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(TranscriptionRecord.id.desc())
            .first()
        )

        if current_tx:
            current_seg_ids = [seg.id for seg in current_tx.segments]
            # Purge live vectors
            try:
                self.expr_repo.delete_by_source(SourceType.AUDIO_TRANSCRIPT.value, current_tx.id, session=self.session)
            except Exception as e:
                logger.warning("Expression vector purge warning during rollback: %s", e)

            # Purge segment vectors from LanceDB
            self._purge_segment_vectors(current_seg_ids)

            # Delete current live transcription
            self.session.delete(current_tx)
            self.session.flush()

        # Re-insert the original transcription
        restored_tx = TranscriptionRecord(
            id=arch_tx.original_transcription_id,
            audio_file_id=arch_tx.audio_file_id,
            source=arch_tx.source,
            file_name=arch_tx.file_name,
            full_text=arch_tx.full_text,
            raw_text=arch_tx.raw_text,
            language=arch_tx.language,
            language_probability=arch_tx.language_probability,
            engine=arch_tx.engine,
            model_name=arch_tx.model_name,
            duration_seconds=arch_tx.duration_seconds,
            word_count=arch_tx.word_count,
            segment_count=arch_tx.segment_count,
            pipeline_phase=arch_tx.pipeline_phase,
            failure_reason=arch_tx.failure_reason,
            attempt_count=arch_tx.attempt_count,
            backfill_attempt_count=arch_tx.backfill_attempt_count,
            backfill_next_attempt_at=arch_tx.backfill_next_attempt_at,
            speaker_map=arch_tx.speaker_map,
            refined_at=arch_tx.refined_at,
            is_indexed=arch_tx.is_indexed,
            created_at=arch_tx.created_at,
            updated_at=datetime.now(UTC),
        )
        self.session.add(restored_tx)
        self.session.flush()

        # Restore archived segments with original segment IDs
        arch_segments = (
            self.session.query(ArchivedSegmentRecord)
            .filter_by(archived_transcription_id=arch_tx.id)
            .order_by(ArchivedSegmentRecord.segment_index.asc())
            .all()
        )

        for aseg in arch_segments:
            seg_rec = SegmentRecord(
                id=aseg.original_segment_id,
                transcription_id=restored_tx.id,
                segment_index=aseg.segment_index,
                text=aseg.text,
                start_time=aseg.start_time,
                end_time=aseg.end_time,
                speaker=aseg.speaker,
                chapter_id=aseg.chapter_id,
                privacy_tier=aseg.privacy_tier,
            )
            self.session.add(seg_rec)

        self.session.flush()

        # Re-link bookmarks and chapters back to restored transcription
        self.session.query(BookmarkRecord).filter_by(audio_file_id=audio_file_id).update(
            {"transcription_id": restored_tx.id}, synchronize_session="fetch"
        )
        self.session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).update(
            {"transcription_id": restored_tx.id}, synchronize_session="fetch"
        )

        # Remove the archived record from archive
        self.session.delete(arch_tx)
        self.session.flush()

        return RollbackResult(
            audio_file_id=audio_file_id,
            restored_transcription_id=restored_tx.id,
            archived_id=arch_tx.id,
            upgrade_batch_id=arch_tx.upgrade_batch_id,
            segments_restored_count=len(arch_segments),
        )

    def rollback_batch(self, upgrade_batch_id: str) -> list[RollbackResult]:
        """Roll back all transcriptions associated with a specific upgrade_batch_id."""
        arch_records = (
            self.session.query(ArchivedTranscriptionRecord)
            .filter_by(upgrade_batch_id=upgrade_batch_id)
            .order_by(ArchivedTranscriptionRecord.id.desc())
            .all()
        )

        if not arch_records:
            raise ValueError(f"No archived transcriptions found for batch ID '{upgrade_batch_id}'.")

        # Deduplicate by audio_file_id, keeping the newest archived record per audio file
        seen_audio_ids = set()
        unique_arch_records = []
        for r in arch_records:
            if r.audio_file_id not in seen_audio_ids:
                seen_audio_ids.add(r.audio_file_id)
                unique_arch_records.append(r)

        results: list[RollbackResult] = []
        for arch in unique_arch_records:
            res = self.rollback_record(arch.id)
            results.append(res)

        return results

    def select_post_hoc_audit_samples(
        self,
        audio_file_ids: list[int],
        sample_size: int = 15,
    ) -> list[dict[str, Any]]:
        """Stratified sample selection across duration, titles, and authors
        for post-upgrade human verification.

        Stratifies across duration (<15m short, 15-60m medium, >1h long)
        while enforcing author, work, and genre diversity (avoiding multiple
        tracks from the same book or author).
        """
        if not audio_file_ids:
            return []

        audio_records = (
            self.session.query(AudioFileRecord)
            .filter(AudioFileRecord.id.in_(audio_file_ids))
            .all()
        )

        if not audio_records:
            return []

        # Divide into duration strata:
        # short: < 15 mins (900s)
        # medium: 15 - 60 mins (900s - 3600s)
        # long: > 60 mins (> 3600s)
        short_bucket = [a for a in audio_records if (a.duration_seconds or 0) < 900]
        med_bucket = [a for a in audio_records if 900 <= (a.duration_seconds or 0) <= 3600]
        long_bucket = [a for a in audio_records if (a.duration_seconds or 0) > 3600]

        # Sort each bucket deterministically by author presence, duration, and file name
        for b in (short_bucket, med_bucket, long_bucket):
            b.sort(key=lambda a: (
                0 if (a.work and a.work.author) else 1,
                a.duration_seconds or 0,
                a.file_name,
            ))

        strata = [b for b in (short_bucket, med_bucket, long_bucket) if b]
        samples: list[AudioFileRecord] = []
        seen_authors: set[str] = set()
        seen_works: set[str] = set()

        def try_pick(
            bucket: list[AudioFileRecord],
            allow_seen_author: bool = False,
            allow_seen_work: bool = False,
        ) -> AudioFileRecord | None:
            for r in bucket:
                if r in samples:
                    continue
                auth = (r.work.author or "").strip().lower() if r.work and r.work.author else ""
                work = (
                    str(r.work_id)
                    if r.work_id
                    else (r.work.title.strip().lower() if r.work and r.work.title else r.file_name.strip().lower())
                )
                if not allow_seen_author:
                    if auth and auth in seen_authors:
                        continue
                    if not auth and "unknown" in seen_authors:
                        continue

                if not allow_seen_work and work in seen_works:
                    continue

                samples.append(r)
                if auth:
                    seen_authors.add(auth)
                else:
                    seen_authors.add("unknown")
                seen_works.add(work)
                return r
            return None

        # Step 1: Guarantee at least 1 candidate per available stratum
        for b in strata:
            if len(samples) >= sample_size:
                break
            try_pick(b)

        # Step 2: Round-robin across strata with strict diversity (unseen author & unseen work)
        while len(samples) < sample_size:
            picked_any = False
            for b in strata:
                if len(samples) >= sample_size:
                    break
                if try_pick(b):
                    picked_any = True
            if not picked_any:
                break

        # Step 3: Relax author constraint if needed, preserving distinct works
        while len(samples) < sample_size:
            picked_any = False
            for b in strata:
                if len(samples) >= sample_size:
                    break
                if try_pick(b, allow_seen_author=True):
                    picked_any = True
            if not picked_any:
                break

        # Step 4: Fill any remaining capacity from all records
        while len(samples) < min(sample_size, len(audio_records)):
            picked_any = False
            for b in strata:
                if len(samples) >= sample_size:
                    break
                if try_pick(b, allow_seen_author=True, allow_seen_work=True):
                    picked_any = True
            if not picked_any:
                break

        out = []
        for s in samples:
            dur = s.duration_seconds or 0.0
            category = "short (<15m)" if dur < 900 else ("medium (15-60m)" if dur <= 3600 else "long (>1h)")
            author = s.work.author if (s.work and s.work.author) else "Unknown"
            work_title = s.work.title if (s.work and s.work.title) else (s.track_title or s.file_name)
            genre = s.genre or "Unknown"
            out.append({
                "audio_file_id": s.id,
                "file_name": s.file_name,
                "duration_seconds": dur,
                "stratum": category,
                "author": author,
                "work_title": work_title,
                "genre": genre,
            })
        return out
