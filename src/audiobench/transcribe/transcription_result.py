"""Pydantic domain models for transcription data.

These models represent the core data structures used throughout the application:
- Word: individual word with timestamp and confidence
- Segment: a continuous speech segment (sentence/phrase)
- Transcript: complete transcription result with metadata
- TranscriptionRequest: configuration for a transcription job
- AudioMetadata: metadata about the source audio file
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

# --- Enums ---


class ModelSize(StrEnum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"


class OutputFormat(StrEnum):
    """Supported output formats."""

    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


class TranscriptionStatus(StrEnum):
    """Status of a transcription job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(StrEnum):
    """Whisper task type."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"  # Translate to English


# --- Core Data Models ---


class Word(BaseModel):
    """A single transcribed word with timing and confidence."""

    word: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return round(self.end - self.start, 3)


class Segment(BaseModel):
    """A continuous speech segment (typically a sentence or phrase)."""

    id: int = Field(description="Segment index")
    text: str = Field(description="Transcribed text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    words: list[Word] = Field(default_factory=list, description="Word-level timestamps")
    speaker: str | None = Field(default=None, description="Speaker label from diarization")
    avg_logprob: float = Field(default=0.0, description="Average log probability")
    no_speech_prob: float = Field(default=0.0, description="Probability of no speech")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return round(self.end - self.start, 3)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        """Number of words in the segment."""
        return len(self.text.split())


class AudioMetadata(BaseModel):
    """Metadata about a source audio file."""

    file_path: str
    file_name: str
    file_size_bytes: int = Field(default=0, ge=0)
    format: str = Field(default="unknown", description="File extension without dot")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    sample_rate: int = Field(default=0, ge=0)
    channels: int = Field(default=0, ge=0)
    file_hash: str | None = Field(default=None, description="SHA-256 hash for deduplication")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_formatted(self) -> str:
        """Human-readable duration (HH:MM:SS)."""
        total_secs = int(self.duration_seconds)
        hours, remainder = divmod(total_secs, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of an audio file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class Transcript(BaseModel):
    """Complete transcription result."""

    segments: list[Segment] = Field(default_factory=list)
    language: str = Field(default="en", description="Detected or specified language code")
    language_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    audio: AudioMetadata | None = Field(default=None, description="Source audio metadata")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Total audio duration")
    engine: str = Field(default="whisper", description="Engine used for transcription")
    model_name: str = Field(default="medium", description="Model used")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """Full transcript text (all segments concatenated)."""
        return " ".join(seg.text.strip() for seg in self.segments)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        """Total word count across all segments."""
        return sum(seg.word_count for seg in self.segments)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def segment_count(self) -> int:
        """Number of segments."""
        return len(self.segments)


# --- Request / Configuration Models ---


class TranscriptionRequest(BaseModel):
    """Configuration for a transcription job."""

    file_path: str
    language: str | None = Field(
        default=None, description="ISO 639-1 language code or None for auto-detect"
    )
    model_name: ModelSize = Field(default=ModelSize.MEDIUM)
    task: TaskType = Field(default=TaskType.TRANSCRIBE)
    output_format: OutputFormat = Field(default=OutputFormat.TXT)
    word_timestamps: bool = Field(default=True)
    enable_diarization: bool = Field(default=False)
    beam_size: int = Field(default=5, ge=1, le=10)
    output_path: str | None = Field(default=None, description="Output file path; None = stdout")
