"""SQLAlchemy ORM models for persisting transcription, chat, and bookmark data.

Tables:
    audio_files: Source audio file metadata + SHA-256 hash for dedup
    transcriptions: Transcription results linked to audio files
    segments: Individual segments within a transcription
    chat_conversations: Persistent AI chat sessions
    chat_messages: Individual messages within a chat conversation
    bookmarks: Timestamp markers and region annotations for audio files
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class AudioFileRecord(Base):
    """Persisted audio file metadata."""

    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_name: Mapped[str] = mapped_column(String(256), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    format: Mapped[str] = mapped_column(String(16), default="unknown")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, default=0)
    channels: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    transcriptions: Mapped[list[TranscriptionRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[BookmarkRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<AudioFile(id={self.id}, name='{self.file_name}', "
            f"duration={self.duration_seconds:.1f}s)>"
        )


class TranscriptionRecord(Base):
    """Persisted transcription result."""

    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("audio_files.id"), nullable=True, index=True
    )
    source: Mapped[str] = mapped_column(String(20), default="file")
    file_name: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    full_text: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(10), default="en", index=True)
    language_probability: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64), default="faster-whisper")
    model_name: Mapped[str] = mapped_column(String(64), default="medium")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="transcriptions")
    segments: Mapped[list[SegmentRecord]] = relationship(
        back_populates="transcription", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Transcription(id={self.id}, lang='{self.language}', "
            f"words={self.word_count}, model='{self.model_name}')>"
        )


class SegmentRecord(Base):
    """Persisted segment within a transcription."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcription_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("transcriptions.id"), nullable=False, index=True
    )
    segment_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, default=0.0)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    transcription: Mapped[TranscriptionRecord] = relationship(back_populates="segments")

    def __repr__(self) -> str:
        return f"<Segment(id={self.id}, idx={self.segment_index}, text='{self.text[:30]}...')>"


class ChatConversation(Base):
    """A persistent AI chat conversation."""

    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), default="Untitled Chat")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    engine: Mapped[str] = mapped_column(String(64), default="ollama")
    transcript_ids: Mapped[str] = mapped_column(
        String(512), default="[]"
    )  # JSON list, e.g. "[3,5,7]"
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ChatConversation(id={self.id}, title='{self.title}', messages={self.message_count})>"
        )


class ChatMessage(Base):
    """A single message in a chat conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # system|user|assistant
    content: Mapped[str] = mapped_column(Text, default="")
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:40] if self.content else ""
        return f"<ChatMessage(id={self.id}, role='{self.role}', text='{preview}...')>"


class BookmarkRecord(Base):
    """Persisted bookmark or region marker for an audio file.

    Point bookmarks have only `timestamp`; region markers also set
    `end_timestamp` to define a span.
    """

    __tablename__ = "bookmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    transcription_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("transcriptions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    end_timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    name: Mapped[str] = mapped_column(String(512), default="Untitled")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    bookmark_type: Mapped[str] = mapped_column(String(16), default="bookmark")
    color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC),
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="bookmarks")
    transcription: Mapped[TranscriptionRecord | None] = relationship()

    @property
    def is_region(self) -> bool:
        """True if this bookmark defines a region (start + end)."""
        return self.end_timestamp is not None

    def __repr__(self) -> str:
        kind = "Region" if self.is_region else "Point"
        return (
            f"<Bookmark(id={self.id}, {kind}, "
            f"t={self.timestamp:.1f}, name='{self.name[:30]}')>"
        )
