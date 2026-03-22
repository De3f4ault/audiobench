"""Error types — exception hierarchy for AudioBench.

Exception hierarchy:
    AudioBenchError
    ├── AudioLoadError       — Failed to load/convert audio file
    ├── UnsupportedFormatError — Audio format not supported
    ├── EngineError          — Transcription engine failure
    │   ├── ModelNotFoundError   — Requested model not available
    │   └── ModelLoadError       — Model failed to load
    ├── StorageError         — Database/persistence failure
    ├── StreamingError       — Real-time streaming failure
    ├── DiarizationError     — Speaker diarization failure
    └── OutputFormatError    — Output formatting failure
"""


class AudioBenchError(Exception):
    """Base exception for all AudioBench errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


# --- Audio Loading ---


class AudioLoadError(AudioBenchError):
    """Failed to load or convert an audio file."""

    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        super().__init__(
            message=f"Failed to load audio: {file_path}",
            details=reason,
        )


class UnsupportedFormatError(AudioBenchError):
    """Audio format is not supported."""

    def __init__(self, file_path: str, format_ext: str) -> None:
        self.file_path = file_path
        self.format_ext = format_ext
        super().__init__(
            message=f"Unsupported audio format: .{format_ext}",
            details=f"File: {file_path}. Supported formats: m4a, mp3, wav, flac, ogg, aac, "
            "wma, opus, aiff, webm, amr, mp4, mkv, avi, mov",
        )


# --- Engine ---


class EngineError(AudioBenchError):
    """Transcription engine failure."""


class ModelNotFoundError(EngineError):
    """Requested model is not available locally or for download."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            message=f"Model not found: {model_name}",
            details="Available models: tiny, base, small, medium, large-v3, large-v3-turbo",
        )


class ModelLoadError(EngineError):
    """Model exists but failed to load (e.g., corrupted, incompatible device)."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(
            message=f"Failed to load model: {model_name}",
            details=reason,
        )


# --- Storage ---


class StorageError(AudioBenchError):
    """Database or persistence layer failure."""


# --- Streaming ---


class StreamingError(AudioBenchError):
    """Real-time audio streaming failure."""


# --- Diarization ---


class DiarizationError(AudioBenchError):
    """Speaker diarization failure."""


# --- Output ---


class OutputFormatError(AudioBenchError):
    """Failed to format transcription output."""

    def __init__(self, format_name: str, reason: str) -> None:
        self.format_name = format_name
        super().__init__(
            message=f"Output format error ({format_name})",
            details=reason,
        )
