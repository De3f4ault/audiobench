"""Abstract base class for transcription engines.

All engines (Whisper, Vosk, etc.) implement this interface,
allowing the pipeline and factory to work with any backend.

Usage:
    class MyEngine(TranscriptionEngine):
        def load_model(self, model_name, device, compute_type):
            ...
        def transcribe(self, audio_path, ...):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from audiobench.transcribe.transcription_result import Transcript


class TranscriptionEngine(ABC):
    """Abstract base class for all transcription engines."""

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ) -> None:
        """Load a transcription model.

        Args:
            model_name: Model identifier (e.g., 'medium', 'large-v3').
            device: Compute device ('cpu' or 'cuda').
            compute_type: Quantization type ('int8', 'float16', 'float32').

        Raises:
            ModelNotFoundError: If the model cannot be found.
            ModelLoadError: If the model fails to load.
        """

    @abstractmethod
    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        beam_size: int = 5,
    ) -> Transcript:
        """Transcribe audio to text.

        Args:
            audio: Path to audio file (str) or raw PCM numpy array.
            language: ISO 639-1 language code, or None for auto-detect.
            task: 'transcribe' or 'translate' (translate to English).
            word_timestamps: Whether to include word-level timestamps.
            beam_size: Beam search width (higher = more accurate, slower).

        Returns:
            Transcript with segments, words, and metadata.

        Raises:
            EngineError: If transcription fails.
        """

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return information about the loaded model.

        Returns:
            Dict with keys like 'name', 'device', 'compute_type', 'size_mb'.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded and ready."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine name (e.g., 'faster-whisper')."""
