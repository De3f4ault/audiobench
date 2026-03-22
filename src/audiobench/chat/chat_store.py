"""Repository — CRUD operations for AI chat conversations.

Provides persistence for chat sessions:
- Creating and updating conversations
- Saving messages with role/content/thinking
- Listing and resuming past conversations
- Deleting conversations
"""

from __future__ import annotations

import json

from sqlalchemy import desc

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import ChatConversation, ChatMessage

logger = get_logger("storage.chat_repository")


class ChatRepository:
    """CRUD operations for AI chat persistence."""

    def create_conversation(
        self,
        model: str,
        transcript_ids: list[int] | None = None,
        title: str = "Untitled Chat",
    ) -> int:
        """Create a new chat conversation.

        Args:
            model: The Ollama model name.
            transcript_ids: List of transcript IDs loaded as context.
            title: Conversation title (will be AI-generated later).

        Returns:
            The conversation ID.
        """
        with get_session() as session:
            conv = ChatConversation(
                title=title,
                model_name=model,
                transcript_ids=json.dumps(transcript_ids or []),
                message_count=0,
            )
            session.add(conv)
            session.commit()
            logger.info("Created conversation #%d (model=%s)", conv.id, model)
            return conv.id

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        thinking: str | None = None,
    ) -> int:
        """Add a message to a conversation.

        Args:
            conversation_id: The conversation to add to.
            role: Message role — "system", "user", or "assistant".
            content: The message text.
            thinking: Optional chain-of-thought text.

        Returns:
            The message ID.
        """
        with get_session() as session:
            msg = ChatMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                thinking=thinking,
                token_count=len(content) // 4,  # rough estimate
            )
            session.add(msg)

            # Update conversation message count
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.message_count = session.query(ChatMessage).filter_by(
                    conversation_id=conversation_id
                ).filter(ChatMessage.role != "system").count() + (1 if role != "system" else 0)

            session.commit()
            return msg.id

    def get_conversation(self, conversation_id: int) -> dict | None:
        """Get a conversation with all its messages.

        Returns:
            Dict with conversation metadata and messages, or None.
        """
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv is None:
                return None

            return {
                "id": conv.id,
                "title": conv.title,
                "model": conv.model_name,
                "transcript_ids": json.loads(conv.transcript_ids),
                "message_count": conv.message_count,
                "created_at": (conv.created_at.isoformat() if conv.created_at else ""),
                "updated_at": (conv.updated_at.isoformat() if conv.updated_at else ""),
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "thinking": msg.thinking,
                        "token_count": msg.token_count,
                        "created_at": (msg.created_at.isoformat() if msg.created_at else ""),
                    }
                    for msg in sorted(conv.messages, key=lambda m: m.created_at)
                ],
            }

    def get_messages_for_api(self, conversation_id: int) -> list[dict]:
        """Get messages formatted for the Ollama /api/chat endpoint.

        Returns:
            List of {"role": str, "content": str} dicts.
        """
        with get_session() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(conversation_id=conversation_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            return [{"role": msg.role, "content": msg.content} for msg in messages]

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """List recent conversations (summary view).

        Returns:
            List of conversation summary dicts.
        """
        with get_session() as session:
            convs = (
                session.query(ChatConversation)
                .order_by(desc(ChatConversation.updated_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "model": c.model_name,
                    "message_count": c.message_count,
                    "transcript_ids": json.loads(c.transcript_ids),
                    "created_at": (c.created_at.isoformat() if c.created_at else ""),
                    "updated_at": (c.updated_at.isoformat() if c.updated_at else ""),
                }
                for c in convs
            ]

    def update_title(self, conversation_id: int, title: str) -> None:
        """Update a conversation's title."""
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.title = title[:256]
                session.commit()
                logger.info(
                    "Updated title for conversation #%d: %s",
                    conversation_id,
                    title[:50],
                )

    def update_transcript_ids(self, conversation_id: int, transcript_ids: list[int]) -> None:
        """Update the transcript IDs associated with a conversation."""
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.transcript_ids = json.dumps(transcript_ids)
                session.commit()

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation and all its messages.

        Returns:
            True if found and deleted, False if not found.
        """
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv is None:
                return False
            session.delete(conv)
            session.commit()
            logger.info("Deleted conversation #%d", conversation_id)
            return True

    def delete_all_conversations(self) -> int:
        """Delete all conversations. Returns number deleted."""
        with get_session() as session:
            count = session.query(ChatConversation).count()
            session.query(ChatMessage).delete()
            session.query(ChatConversation).delete()
            session.commit()
            logger.info("Deleted %d conversation(s)", count)
            return count
