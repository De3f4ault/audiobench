"""Repository — CRUD operations for chapter data.

All public methods return ``ChapterInfo`` dataclass instances (not ORM objects).
This eliminates the detached-session / dict-vs-object inconsistency that caused
most of the runtime bugs in the previous implementation.

Module-level singleton: use ``get_chapter_repo()`` instead of ``ChapterRepository()``
to avoid creating a new object (and redundant session-factory setup) on every call.
"""

from __future__ import annotations

import json
from functools import lru_cache

from sqlalchemy import asc

from audiobench.chapters.cue_parser import ChapterInfo
from audiobench.core import db_session


def get_session():
    """Get database session dynamically from db_session module to prevent mock freeze across tests."""
    return db_session.get_session()
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import ChapterRecord

logger = get_logger("storage.chapter_repository")


def _record_to_info(rec: ChapterRecord) -> ChapterInfo:
    """Convert an ORM ChapterRecord to a ChapterInfo dataclass (session-safe)."""
    return ChapterInfo(
        id=rec.id,
        index=rec.chapter_index,
        title=rec.title,
        start_time=rec.start_time,
        end_time=rec.end_time,
        is_ghost=bool(rec.is_ghost),
    )


class ChapterRepository:
    """CRUD operations for audio file chapters.

    Always returns ``ChapterInfo`` objects, never raw ORM instances.
    Use the module-level ``get_chapter_repo()`` singleton to avoid
    re-instantiation overhead on every call site.
    """

    # ── Write ────────────────────────────────────────────────

    def save_chapters(
        self, audio_file_id: int, chapters: list[ChapterInfo] | list[dict]
    ) -> list[ChapterInfo]:
        """Save chapters for an audio file, replacing any existing ones.

        Accepts either ``ChapterInfo`` instances or plain dicts with the same keys.
        Returns the saved ``ChapterInfo`` list with ``id`` fields populated.
        """
        with get_session() as session:
            session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).delete()

            records: list[ChapterRecord] = []
            for chap in chapters:
                # Normalise: accept both ChapterInfo and raw dict
                if isinstance(chap, ChapterInfo):
                    idx = chap.index
                    title = chap.title
                    start = chap.start_time
                    end = chap.end_time
                    ghost = chap.is_ghost
                else:
                    idx = chap["index"]
                    title = chap.get("title", "Untitled")
                    start = chap["start_time"]
                    end = chap["end_time"]
                    ghost = chap.get("is_ghost", False)

                status = "skipped" if ghost else "pending"
                rec = ChapterRecord(
                    audio_file_id=audio_file_id,
                    chapter_index=idx,
                    title=str(title)[:512],
                    start_time=start,
                    end_time=end,
                    is_ghost=1 if ghost else 0,
                    transcription_status=status,
                )
                session.add(rec)
                records.append(rec)

            session.commit()

            saved = (
                session.query(ChapterRecord)
                .filter_by(audio_file_id=audio_file_id)
                .order_by(asc(ChapterRecord.chapter_index))
                .all()
            )
            result = [_record_to_info(r) for r in saved]
            logger.info("Saved %d chapters for audio_file #%d", len(result), audio_file_id)
            return result

    # ── Read ─────────────────────────────────────────────────

    def get_chapters(self, audio_file_id: int) -> list[ChapterInfo]:
        """Get all chapters for an audio file, ordered by index."""
        with get_session() as session:
            records = (
                session.query(ChapterRecord)
                .filter_by(audio_file_id=audio_file_id)
                .order_by(asc(ChapterRecord.chapter_index))
                .all()
            )
            return [_record_to_info(r) for r in records]

    # Aliases used throughout the codebase (kept for backward compatibility)
    def list_for_file(self, audio_file_id: int) -> list[ChapterInfo]:
        """Alias for ``get_chapters``."""
        return self.get_chapters(audio_file_id)

    def get_chapters_for_file(self, audio_file_id: int) -> list[ChapterInfo]:
        """Alias for ``get_chapters``."""
        return self.get_chapters(audio_file_id)

    def get_ordered_chapters(
        self, work_id: int | None = None, audio_file_id: int | None = None
    ) -> list[ChapterInfo]:
        """Get chapters ordered by canonical sequence across a work or file.

        Ordering contract:
            - Files with explicit sequence_key sort by sequence_key ASC
            - Within each file, chapters sort by chapter_index ASC
            - Files without sequence_key sort by file_name ASC
        """
        from sqlalchemy import case
        from audiobench.storage.models import AudioFileRecord

        with get_session() as session:
            query = (
                session.query(ChapterRecord)
                .join(AudioFileRecord, ChapterRecord.audio_file_id == AudioFileRecord.id)
                .filter(ChapterRecord.is_ghost == 0)
            )
            if audio_file_id:
                query = query.filter(ChapterRecord.audio_file_id == audio_file_id)
            if work_id:
                query = query.filter(AudioFileRecord.work_id == work_id)

            has_seq = case(
                (AudioFileRecord.sequence_key.isnot(None), 0),
                else_=1,
            )

            records = (
                query.order_by(
                    has_seq,
                    AudioFileRecord.sequence_key.asc(),
                    ChapterRecord.chapter_index.asc(),
                    AudioFileRecord.file_name.asc(),
                ).all()
            )
            return [_record_to_info(r) for r in records]

    def get_chapter(self, chapter_id: int) -> ChapterInfo | None:
        """Get a single chapter by DB primary key."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            return _record_to_info(rec) if rec else None

    def get_chapter_by_index(self, audio_file_id: int, index: int) -> ChapterInfo | None:
        """Get a chapter by its sequential index within an audio file."""
        with get_session() as session:
            rec = (
                session.query(ChapterRecord)
                .filter_by(audio_file_id=audio_file_id, chapter_index=index)
                .first()
            )
            return _record_to_info(rec) if rec else None

    def get_transcription_id(self, chapter_id: int) -> int | None:
        """Return the transcription_id linked to a chapter, or None."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            return rec.transcription_id if rec else None

    # ── Update ───────────────────────────────────────────────

    def update_chapter_status(
        self,
        chapter_id: int,
        status: str,
        transcription_id: int | None = None,
    ) -> bool:
        """Update a chapter's transcription status."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            if not rec:
                return False
            rec.transcription_status = status
            if transcription_id is not None:
                rec.transcription_id = transcription_id
            session.commit()
            logger.info("Updated chapter #%d status → %s", chapter_id, status)
            return True

    def update_chapter_summary(self, chapter_id: int, summary_text: str) -> bool:
        """Update a chapter's AI-generated summary and mark dirty for re-embedding."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            if not rec:
                return False
            rec.summary = summary_text
            rec.last_embedded_at = None  # Dirty flag trigger
            session.commit()
            logger.info("Updated chapter #%d summary (marked dirty for embedding)", chapter_id)
            return True

    def update_chapter_title(self, chapter_id: int, title: str) -> bool:
        """Update a chapter's title and mark dirty for re-embedding."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            if not rec:
                return False
            rec.title = title
            rec.last_embedded_at = None  # Dirty flag trigger
            session.commit()
            logger.info("Updated chapter #%d title (marked dirty for embedding)", chapter_id)
            return True

    def update_chapter_tags(self, chapter_id: int, tags: list[str]) -> bool:
        """Update a chapter's tags (stored as JSON string)."""
        with get_session() as session:
            rec = session.query(ChapterRecord).filter_by(id=chapter_id).first()
            if not rec:
                return False
            rec.tags = json.dumps(tags)
            session.commit()
            return True

    # ── Delete ───────────────────────────────────────────────

    def delete_chapters(self, audio_file_id: int) -> int:
        """Delete all chapters for an audio file. Returns count deleted."""
        with get_session() as session:
            count = session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).delete()
            session.commit()
            logger.info("Deleted %d chapters for audio_file #%d", count, audio_file_id)
            return count


@lru_cache(maxsize=1)
def get_chapter_repo() -> ChapterRepository:
    """Module-level singleton. Use this instead of ``ChapterRepository()`` everywhere."""
    return ChapterRepository()
