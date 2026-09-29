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

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class WorkRecord(Base):
    """A semantic work that groups audio files and expressions."""

    __tablename__ = "works"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    audio_files: Mapped[list[AudioFileRecord]] = relationship(back_populates="work")
    expressions: Mapped[list[ExpressionRecord]] = relationship(back_populates="work")

    def __repr__(self) -> str:
        author_str = f", author='{self.author}'" if self.author else ""
        return f"<Work(id={self.id}, title='{self.title[:30]}'{author_str})>"


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
    tags: Mapped[str] = mapped_column(Text, default="[]")
    work_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("works.id", ondelete="SET NULL"), nullable=True, index=True
    )
    youtube_video_id: Mapped[str | None] = mapped_column(
        String(11), unique=True, nullable=True, index=True
    )
    youtube_channel_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )  # UC... ID — allows get_library_count() to scope to a specific channel

    # Relationships
    work: Mapped[WorkRecord | None] = relationship(back_populates="audio_files")
    chapters: Mapped[list[ChapterRecord]] = relationship(
        back_populates="audio_file",
        cascade="all, delete-orphan",
        order_by="ChapterRecord.chapter_index",
    )
    transcriptions: Mapped[list[TranscriptionRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[BookmarkRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )

    @property
    def has_chapters(self) -> bool:
        """True if the audio file has any chapters."""
        return len(self.chapters) > 0

    def __repr__(self) -> str:
        return (
            f"<AudioFile(id={self.id}, name='{self.file_name}', "
            f"duration={self.duration_seconds:.1f}s)>"
        )


class ChapterRecord(Base):
    """Persisted chapter metadata for an audio file."""

    __tablename__ = "chapters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chapter_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="Untitled", nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    transcription_status: Mapped[str] = mapped_column(String(20), default="pending")
    transcription_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("transcriptions.id", ondelete="SET NULL"), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str] = mapped_column(Text, default="[]")
    is_ghost: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="chapters")
    transcription: Mapped[TranscriptionRecord | None] = relationship()
    segments: Mapped[list[SegmentRecord]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[BookmarkRecord]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )
    # jobs relationship removed 2026-09-29: JobRecord (System 1 'jobs' table) retired.

    @property
    def is_real(self) -> bool:
        """True if this is a real chapter, false if it's a ghost chapter (start == end)."""
        return not bool(self.is_ghost)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def tags_list(self) -> list[str]:
        """Deserialise the JSON tags column to a Python list."""
        import json as _json

        try:
            return _json.loads(self.tags) if self.tags else []
        except Exception:
            return []

    def __repr__(self) -> str:
        return (
            f"<Chapter(id={self.id}, index={self.chapter_index}, title='{self.title[:30]}', "
            f"status='{self.transcription_status}')>"
        )


class TranscriptionRecord(Base):
    """Persisted transcription result."""

    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("audio_files.id"), nullable=True, index=True
    )
    # Provenance of the transcription. Valid values: "file" | "live" | "reimport"
    source: Mapped[str] = mapped_column(String(20), default="file")
    file_name: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    full_text: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(10), default="en", index=True)
    language_probability: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64), default="faster-whisper")
    model_name: Mapped[str] = mapped_column(String(64), default="large-v3-turbo")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    # Renamed status to pipeline_phase, mapping to the same DB column
    pipeline_phase: Mapped[str] = mapped_column("status", String(20), default="complete")
    failure_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    backfill_attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    backfill_next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    speaker_map: Mapped[str] = mapped_column(Text, default="{}")
    refined_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    is_indexed: Mapped[int] = mapped_column(Integer, default=0, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
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
    chapter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Security: 0=public, 1=relational, 2=intimate (voiceprint), 3=manual override
    privacy_tier: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)

    # Relationships
    transcription: Mapped[TranscriptionRecord] = relationship(back_populates="segments")
    chapter: Mapped[ChapterRecord | None] = relationship(back_populates="segments")

    def __repr__(self) -> str:
        return f"<Segment(id={self.id}, idx={self.segment_index}, text='{self.text[:30]}...')>"


class ChatConversation(Base):
    """A persistent AI chat conversation."""

    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), default="Untitled Chat")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    session_type: Mapped[str] = mapped_column(String(64), default="chat")
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
        Integer,
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcription_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("transcriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    end_timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    name: Mapped[str] = mapped_column(String(512), default="Untitled")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    bookmark_type: Mapped[str] = mapped_column(String(16), default="bookmark")
    color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    chapter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chapters.id", ondelete="CASCADE"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="bookmarks")
    transcription: Mapped[TranscriptionRecord | None] = relationship()
    chapter: Mapped[ChapterRecord | None] = relationship(back_populates="bookmarks")

    @property
    def is_region(self) -> bool:
        """True if this bookmark defines a region (start + end)."""
        return self.end_timestamp is not None

    def __repr__(self) -> str:
        kind = "Region" if self.is_region else "Point"
        return f"<Bookmark(id={self.id}, {kind}, t={self.timestamp:.1f}, name='{self.name[:30]}')>"



# JobRecord (jobs table, System 1) was removed 2026-09-29.
# Introduced 2026-05-31; last write 2026-08-31 (56 rows, all status='failed').
# runner.py (its only writer) had zero callers in src/. Data preserved in DB.


class ExpressionRecord(Base):
    """A semantic expression representing a unit of memory."""

    __tablename__ = "expressions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    session_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inference_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    work_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("works.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Security: mirrors segments.privacy_tier — inherited during expression creation
    privacy_tier: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    # Graph topology role — set by the RAG sweep to distinguish tiered nodes.
    # Values: 'sweep_document' (T1 full text), 'sweep_passage' (T2 context group),
    # 'sweep_chunk' (T3 embedded leaf). NULL = pre-Track-4 row or non-sweep origin
    # (chat, memoir, bookmark, etc.). Future consumers must treat NULL as
    # "role unknown / not a tiered sweep node" — not as an implicit tier value.
    graph_role: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    work: Mapped[WorkRecord | None] = relationship(back_populates="expressions")
    relations_from: Mapped[list[ExpressionRelation]] = relationship(
        "ExpressionRelation",
        foreign_keys="ExpressionRelation.from_expression_id",
        back_populates="source_expression",
        cascade="all, delete-orphan",
    )
    relations_to: Mapped[list[ExpressionRelation]] = relationship(
        "ExpressionRelation",
        foreign_keys="ExpressionRelation.to_expression_id",
        back_populates="target_expression",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ExpressionRecord(id={self.id}, source_type='{self.source_type}', len={len(self.content)})>"


class ExpressionSegmentMap(Base):
    """Bridge table linking expressions to their raw transcript segments."""

    __tablename__ = "expression_segment_map"

    expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), primary_key=True
    )
    segment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    def __repr__(self) -> str:
        return f"<ExpressionSegmentMap(expr={self.expression_id}, seg={self.segment_id})>"


class ExpressionRelation(Base):
    """Directed relation between two semantic expressions."""

    __tablename__ = "expression_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    to_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "from_expression_id",
            "to_expression_id",
            "relation_type",
            name="uq_expression_relation_edge",
        ),
    )

    # Relationships
    source_expression: Mapped[ExpressionRecord] = relationship(
        "ExpressionRecord", foreign_keys=[from_expression_id], back_populates="relations_from"
    )
    target_expression: Mapped[ExpressionRecord] = relationship(
        "ExpressionRecord", foreign_keys=[to_expression_id], back_populates="relations_to"
    )

    def __repr__(self) -> str:
        return f"<ExpressionRelation({self.from_expression_id} -> {self.to_expression_id}, type='{self.relation_type}')>"


class PendingRelation(Base):
    __tablename__ = "pending_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False
    )
    to_expression_id_hint: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    to_source_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, default="explicit")
    raw_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))



class AskLog(Base):
    """Log of questions and answers for a specific audio file."""

    __tablename__ = "ask_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    entry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    entries: Mapped[list[AskEntry]] = relationship(
        "AskEntry", back_populates="log", cascade="all, delete-orphan"
    )
    audio_file: Mapped[AudioFileRecord] = relationship()

    def __repr__(self) -> str:
        return f"<AskLog(id={self.id}, audio_file_id={self.audio_file_id}, entries={self.entry_count})>"


class AskEntry(Base):
    """A single question and answer in an ask log."""

    __tablename__ = "ask_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ask_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    question_expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    answer_expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    log: Mapped[AskLog] = relationship("AskLog", back_populates="entries")

    def __repr__(self) -> str:
        return f"<AskEntry(id={self.id}, log_id={self.log_id})>"


class ConversationSummary(Base):
    """Semantic summary and insight extraction from a chat conversation."""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    drift_phases: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    key_insights: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    open_threads: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    refined_title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    generated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    conversation: Mapped[ChatConversation] = relationship()

    def __repr__(self) -> str:
        return f"<ConversationSummary(id={self.id}, conv_id={self.conversation_id})>"


class StagingCartItem(Base):
    """A persisted item in the user's transcription staging cart."""

    __tablename__ = "staging_cart"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    engine: Mapped[str] = mapped_column(String(64), default="gemini")
    model_name: Mapped[str] = mapped_column(String(64), default="large-v3-turbo")
    speed_preset: Mapped[str] = mapped_column(String(64), default="balanced")
    strategy: Mapped[str] = mapped_column(String(64), default="batch")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship()

    def __repr__(self) -> str:
        return f"<StagingCartItem(id={self.id}, audio_id={self.audio_file_id}, engine='{self.engine}')>"


# JobQueueItem (job_queue table, System 2) was removed 2026-09-29.
# The table had its last write on 2026-08-08; queue_worker.py was its only
# writer and had zero callers in src/. Data is preserved in the DB.


class UnifiedJob(Base):
    """A single unit of work with a persistent lifecycle."""

    __tablename__ = "unified_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # What to execute
    command_display: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    args_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    # Lifecycle
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False, index=True
    )  # pending, running, done, failed, cancelled

    # Concurrency control
    slot: Mapped[str] = mapped_column(
        String(32), default="transcription", nullable=False, index=True
    )  # transcription, network, indexing

    # Batch identity
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    batch_label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    batch_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Process tracking
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    events_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Display
    file_label: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Deduplication fingerprint
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Outcome
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Retry
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Priority
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<UnifiedJob(id={self.id}, type='{self.job_type}', status='{self.status}', slot='{self.slot}')>"


class CommandEvent(Base):
    """Append-only log of every REPL command dispatch.

    Powers the intelligence layer: pattern detection, proactive suggestions,
    and named workflow capture. Written after every successful dispatch_command()
    call — one row, sub-millisecond, always-on.
    """

    __tablename__ = "command_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    command: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    args_json: Mapped[str] = mapped_column(Text, default="[]")       # JSON list of args
    context_file_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    context_tx_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    def __repr__(self) -> str:
        return f"<CommandEvent(id={self.id}, cmd='{self.command}', ts={self.created_at})>"


class Workflow(Base):
    """A named, replayable sequence of REPL commands.

    Created via \\workflow save <name>. Replayed via \\workflow run <name>.
    The steps field is a JSON array of {"command": str, "args": list[str]} objects.
    """

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    steps_json: Mapped[str] = mapped_column(Text, default="[]")      # JSON list of steps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return f"<Workflow(id={self.id}, name='{self.name}')>"


class NoteCollection(Base):
    """A collection of notes (captures) tied to an audio file or subject."""
    __tablename__ = "note_collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    audio_file_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="SET NULL"), nullable=True, index=True
    )
    expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord | None] = relationship()
    expression: Mapped[ExpressionRecord | None] = relationship()
    captures: Mapped[list[NoteCapture]] = relationship(
        "NoteCapture", back_populates="collection", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<NoteCollection(id={self.id}, title='{self.title}')>"


class NoteCapture(Base):
    """An individual note capture, addressable via segment_id and expression_id."""
    __tablename__ = "note_captures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("note_collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    segment_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("segments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    collection: Mapped[NoteCollection] = relationship(back_populates="captures")
    expression: Mapped[ExpressionRecord | None] = relationship()
    segment: Mapped[SegmentRecord | None] = relationship()

    def __repr__(self) -> str:
        return f"<NoteCapture(id={self.id}, coll_id={self.collection_id}, body='{self.body[:20]}')>"

class StudyProject(Base):
    """A project containing study sessions for a specific audio file."""

    __tablename__ = "study_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship()
    sessions: Mapped[list[StudySession]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

class StudySession(Base):
    """An active or completed study session over specific chapters."""

    __tablename__ = "study_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("study_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chat_conversations.id", ondelete="SET NULL"), nullable=True
    )
    chapter_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    memoir_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=None
    )

    # Relationships
    project: Mapped[StudyProject] = relationship(back_populates="sessions")
    memoir: Mapped[ExpressionRecord | None] = relationship()


class YouTubeChannel(Base):
    """A cached YouTube channel resolution."""

    __tablename__ = "youtube_channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"<YouTubeChannel(query='{self.query}', id='{self.channel_id}')>"


class YouTubeChannelNode(Base):
    """A first-class channel node — a standing relationship with an invited voice.

    IMPORTANT:
    - whiteboard_text is NEVER sent to LanceDB. Enforced in channel_store.py.
      It is an operational note (orients the user on return), not corpus content.
    - engagement_weight is NOT stored here. It is computed on read by
      channel_store.compute_engagement_weight() from last_visited_at.
      Storing it would create a stale number that never decays — the absence of
      events would not lower it, only new events would move it.
    """

    __tablename__ = "youtube_channel_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The channel whiteboard — operational note, never embedded, never in LanceDB.
    whiteboard_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    whiteboard_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Only stored engagement signal. Weight is derived from this on read, not stored.
    last_visited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationship to playlist cache (one-to-one)
    playlist_cache: Mapped[YouTubePlaylistCache | None] = relationship(
        back_populates="channel_node", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<YouTubeChannelNode(id='{self.channel_id}', title='{self.title[:40]}')>"


class YouTubePlaylistCache(Base):
    """Cached uploads playlist for a channel.

    Filled by Phase 2 (playlist.py). Schema locked here in Phase 1.

    videos_json is a JSON array of objects with this exact shape:
        {
            "video_id":    "vcrv2Sr988M",
            "title":       "We spend little time studying ourselves",
            "published_at": "2026-08-29",
            "duration_pt": "PT14M20S",
            "availability": "public"        # public | unlisted | private_or_removed
        }

    availability is captured at fetch time from the API response — NOT derived
    from the title string. "Private video" as a title is not a reliable signal.

    video_count_available (public + unlisted) is the denominator for the UI
    progress display ("89/312"). video_count_total is the raw API count and
    includes private/removed placeholders — it is NOT shown to the user as
    the target number.
    """

    __tablename__ = "youtube_playlist_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("youtube_channel_nodes.channel_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Raw count from API pageInfo.totalResults — includes private placeholders.
    video_count_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Fetchable count only (availability = public | unlisted). UI denominator.
    video_count_available: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # JSON array — see class docstring for entry shape.
    videos_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # nextPageToken from the last API call. None = fully fetched, token = partial.
    next_page_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship back to channel node
    channel_node: Mapped[YouTubeChannelNode] = relationship(back_populates="playlist_cache")

    def __repr__(self) -> str:
        return (
            f"<YouTubePlaylistCache(channel_id='{self.channel_id}', "
            f"available={self.video_count_available}/{self.video_count_total}, "
            f"fetched={self.fetched_at.date() if self.fetched_at else None})>"
        )
