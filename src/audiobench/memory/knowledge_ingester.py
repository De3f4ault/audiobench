"""Ingests application models (chats, asks, bookmarks) into the Expression graph."""

import json

from audiobench.core.db_session import get_session
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import (
    AskEntry,
    AskLog,
    BookmarkRecord,
    ChatConversation,
    ChatMessage,
    ExpressionRecord,
    TranscriptionRecord,
)
from audiobench.storage.repository import TranscriptionRepository


class KnowledgeIngester:
    def __init__(self):
        self.repo = ExpressionRepository()
        self._tx_repo = TranscriptionRepository()

    def _get_transcript_expressions(self, session, transcription_ids):
        if not transcription_ids:
            return []
        return session.query(ExpressionRecord).filter(
            ExpressionRecord.source_type == "audio_transcript",
            ExpressionRecord.source_id.in_(transcription_ids)
        ).all()

    def _get_existing(self, session, source_type, source_id):
        return session.query(ExpressionRecord).filter_by(
            source_type=source_type, source_id=source_id
        ).first()

    def ingest_chat_message(self, message: ChatMessage, conversation: ChatConversation) -> ExpressionRecord:
        with get_session() as session:
            existing = self._get_existing(session, "chat_message", message.id)
            if existing:
                return existing

            try:
                t_ids = json.loads(conversation.transcript_ids)
            except Exception:
                t_ids = []

            # Inherit the highest privacy tier from any source transcript.
            privacy_tier = self._tx_repo.get_max_privacy_tier_for_transcriptions(t_ids)

            expr = self.repo.register(
                content=message.content,
                source_type="chat_message",
                source_id=message.id,
                session_type=conversation.session_type,
                session_id=conversation.id,
                speaker=message.role,
                privacy_tier=privacy_tier,
            )

            if t_ids:
                transcripts = self._get_transcript_expressions(session, t_ids)
                for t_expr in transcripts:
                    self.repo.link(expr.id, t_expr.id, "thematic")

            return expr

    def ingest_ask_entry(self, entry: AskEntry, log: AskLog) -> tuple[ExpressionRecord, ExpressionRecord]:
        with get_session() as session:
            q_existing = self._get_existing(session, "ask_query", entry.id)
            a_existing = self._get_existing(session, "ask_answer", entry.id)

            if q_existing and a_existing:
                return q_existing, a_existing

            t_records = session.query(TranscriptionRecord).filter_by(audio_file_id=log.audio_file_id).all()
            t_ids = [t.id for t in t_records]

            # Inherit the highest privacy tier from the audio file's transcripts.
            privacy_tier = self._tx_repo.get_max_privacy_tier_for_transcriptions(t_ids)

            q_id = q_existing.id if q_existing else self.repo.register(
                content=entry.question,
                source_type="ask_query",
                source_id=entry.id,
                session_type="ask",
                session_id=log.id,
                speaker="user",
                privacy_tier=privacy_tier,
            ).id
            a_id = a_existing.id if a_existing else self.repo.register(
                content=entry.answer,
                source_type="ask_answer",
                source_id=entry.id,
                session_type="ask",
                session_id=log.id,
                speaker="assistant",
                privacy_tier=privacy_tier,
            ).id

            if not (q_existing and a_existing):
                self.repo.link(a_id, q_id, "answers")

            if t_ids:
                transcripts = self._get_transcript_expressions(session, t_ids)
                for t_expr in transcripts:
                    if not q_existing:
                        self.repo.link(q_id, t_expr.id, "thematic")
                    if not a_existing:
                        self.repo.link(a_id, t_expr.id, "thematic")

            return self.repo.get_by_id(q_id), self.repo.get_by_id(a_id)

    def ingest_bookmark(self, bookmark: BookmarkRecord) -> ExpressionRecord:
        with get_session() as session:
            existing = self._get_existing(session, "bookmark", bookmark.id)
            if existing:
                return existing

            # Inherit the privacy tier from the bookmark's transcription, if any.
            tx_ids = [bookmark.transcription_id] if bookmark.transcription_id else []
            privacy_tier = self._tx_repo.get_max_privacy_tier_for_transcriptions(tx_ids)

            expr = self.repo.register(
                content=bookmark.notes or "Bookmark",
                source_type="bookmark",
                source_id=bookmark.id,
                session_type="bookmark",
                speaker="user",
                privacy_tier=privacy_tier,
            )

            if bookmark.transcription_id:
                transcripts = self._get_transcript_expressions(session, [bookmark.transcription_id])
                for t_expr in transcripts:
                    self.repo.link(expr.id, t_expr.id, "thematic")

            return expr
