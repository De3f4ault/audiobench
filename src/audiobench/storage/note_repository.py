"""Repository — CRUD operations and lifecycle management for Note data."""

from __future__ import annotations

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.daemon.client import DaemonClient
from audiobench.memory.enums import RelationType, SourceType
from audiobench.storage.models import (
    ExpressionRecord,
    ExpressionRelation,
    NoteCapture,
    NoteCollection,
)

logger = get_logger("storage.note_repository")


class NoteRepository:
    """CRUD operations for NoteCollection and NoteCapture."""

    def find_or_create_collection(self, audio_file_id: int | None, title: str) -> NoteCollection:
        """Find an existing collection by title+audio_file_id, or create a new one with its expression."""
        with get_session() as session:
            # 1. Look for existing
            if audio_file_id is not None:
                col = session.query(NoteCollection).filter_by(title=title, audio_file_id=audio_file_id).first()
            else:
                col = session.query(NoteCollection).filter_by(title=title).first()

            if col:
                session.expunge(col)
                return col

            # 2. Create new expression for the collection
            expr = ExpressionRecord(
                content=f"Collection: {title}",
                source_type=SourceType.JOURNAL_ENTRY.value,
                inference_status="active"
            )
            session.add(expr)
            session.flush() # get expr.id

            # 3. Create collection
            col = NoteCollection(
                title=title,
                audio_file_id=audio_file_id,
                expression_id=expr.id
            )
            session.add(col)
            session.flush()

            # 4. Link expression to source_id
            expr.source_id = col.id

            session.commit()
            session.refresh(col)
            session.expunge(col)
            return col

    def create_capture(
        self,
        collection_id: int,
        body: str,
        segment_id: int | None,
        transcript_expression_id: int | None,
        collection_expression_id: int | None
    ) -> NoteCapture:
        """Create a new note capture and link it semantically."""
        with get_session() as session:
            # 1. Create expression for the capture
            expr = ExpressionRecord(
                content=body,
                source_type=SourceType.JOURNAL_ENTRY.value,
                inference_status="active"
            )
            session.add(expr)
            session.flush()

            # 2. Create the capture
            cap = NoteCapture(
                collection_id=collection_id,
                expression_id=expr.id,
                segment_id=segment_id,
                body=body
            )
            session.add(cap)
            session.flush()

            # 3. Update expression source_id
            expr.source_id = cap.id

            # 4. Create explicit relations
            if transcript_expression_id:
                rel_t = ExpressionRelation(
                    from_expression_id=expr.id,
                    to_expression_id=transcript_expression_id,
                    relation_type=RelationType.INSPIRED_BY.value,
                    created_by="user"
                )
                session.add(rel_t)

            if collection_expression_id:
                rel_c = ExpressionRelation(
                    from_expression_id=expr.id,
                    to_expression_id=collection_expression_id,
                    relation_type=RelationType.BELONGS_TO.value,
                    created_by="user"
                )
                session.add(rel_c)

            session.commit()
            session.refresh(cap)

            expr_id = expr.id

        # 5. Emit to daemon
        daemon = DaemonClient()
        if daemon.ping():
            daemon.embed(expr_id, body, SourceType.JOURNAL_ENTRY)

        return cap

    def get_captures_for_segment(self, segment_id: int) -> list[NoteCapture]:
        with get_session() as session:
            caps = session.query(NoteCapture).filter_by(segment_id=segment_id).order_by(NoteCapture.created_at).all()
            for c in caps:
                session.expunge(c)
            return caps

    def get_captures_for_collection(self, collection_id: int) -> list[NoteCapture]:
        with get_session() as session:
            caps = session.query(NoteCapture).filter_by(collection_id=collection_id).order_by(NoteCapture.created_at).all()
            for c in caps:
                session.expunge(c)
            return caps

    def list_collections(self, limit: int = 50) -> list[NoteCollection]:
        with get_session() as session:
            cols = session.query(NoteCollection).order_by(NoteCollection.updated_at.desc()).limit(limit).all()
            for c in cols:
                session.expunge(c)
            return cols
