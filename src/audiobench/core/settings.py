"""Application settings — Pydantic-based configuration loading.

Configuration is loaded from (in priority order):
1. Environment variables (prefixed with AUDIOBENCH_)
2. .env file in project root
3. Default values defined here

Usage:
    from audiobench.core.settings import get_settings
    settings = get_settings()
    print(settings.model_name)  # "large-v3-turbo"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — computed from this file's location:
# core/settings.py → core/ → audiobench/ → src/ → project_root/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"


class AudioBenchSettings(BaseSettings):
    """Central configuration for the AudioBench."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIOBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Transcription Engine ---
    model_name: str = Field(
        default="large-v3-turbo",
        description="Whisper model size: tiny, base, small, medium, large-v3, large-v3-turbo",
    )
    device: str = Field(
        default="auto",
        description="Compute device: auto, cpu, cuda",
    )
    compute_type: str = Field(
        default="int8",
        description="Quantization: int8 (CPU), float16 (CUDA), float32",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code or None for auto-detect",
    )

    # --- Performance ---
    speed_preset: str = Field(
        default="balanced",
        description="Speed preset: fast, balanced, accurate",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Batch size for batched inference (higher = faster, more RAM)",
    )
    cpu_threads: int = Field(
        default=0,
        description="CPU threads for CTranslate2 (0 = auto-detect physical cores)",
    )
    beam_size: int = Field(default=3, ge=1, le=10, description="Beam search size")

    # --- Output ---
    output_format: str = Field(
        default="txt", description="Default output format: txt, srt, vtt, json"
    )
    word_timestamps: bool = Field(default=True, description="Enable word-level timestamps")

    # --- Features ---
    enable_diarization: bool = Field(default=False, description="Enable speaker diarization")

    # --- Database ---
    database_url: str = Field(
        default_factory=lambda: f"sqlite:///{_DATA_DIR / 'transcriptions.db'}",
        description="SQLAlchemy database URL",
    )

    # --- Storage ---
    models_dir: Path = Field(
        default=Path.home() / ".audiobench" / "models",
        description="Directory for downloaded models (shared, multi-GB)",
    )
    data_dir: Path = Field(
        default=_DATA_DIR,
        description="Base directory for project-local data (db, plugins, presets, logs)",
    )

    # --- Text-to-Speech ---
    tts_voice: str = Field(
        default="en_US-amy-medium",
        description="Default Piper TTS voice model name",
    )
    voices_dir: Path = Field(
        default=Path.home() / ".audiobench" / "voices",
        description="Directory for TTS voice models",
    )

    # --- AI / LLM ---
    ollama_model: str = Field(
        default="deepseek-v3.2:cloud",
        description="Default Ollama model for AI features",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )

    # --- Engine Selection ---
    engine: str = Field(
        default="whisper",
        description="Transcription engine: whisper (local) or gemini (cloud)",
    )

    # --- Google Gemini (optional cloud engine) ---
    gemini_api_key: str | None = Field(
        default=None,
        description="API key for Google Gemini (enable with --engine gemini)",
    )
    gemini_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini model to use for transcription",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Diarization ---
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token for pyannote model download",
    )

    # --- Validators ---

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: str | None) -> str | None:
        """Convert empty string to None (for auto-detect)."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        return v

    @field_validator("hf_token", "gemini_api_key", mode="before")
    @classmethod
    def validate_optional_token(cls, v: str | None) -> str | None:
        """Convert empty string to None."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        valid = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}
        if v not in valid:
            raise ValueError(f"Invalid model: {v}. Choose from: {', '.join(sorted(valid))}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Invalid device: {v}. Choose from: auto, cpu, cuda")
        return v

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        if v not in {"int8", "float16", "float32"}:
            raise ValueError(f"Invalid compute_type: {v}. Choose from: int8, float16, float32")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        if v not in {"txt", "srt", "vtt", "json"}:
            raise ValueError(f"Invalid output_format: {v}. Choose from: txt, srt, vtt, json")
        return v

    @field_validator("speed_preset")
    @classmethod
    def validate_speed_preset(cls, v: str) -> str:
        if v not in {"fast", "balanced", "accurate"}:
            raise ValueError(f"Invalid speed_preset: {v}. Choose from: fast, balanced, accurate")
        return v

    def ensure_dirs(self) -> None:
        """Create data and model directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device (cpu or cuda)."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_compute_type(self) -> str:
        """Auto-select compute type based on device."""
        if self.compute_type != "int8":
            return self.compute_type
        resolved_device = self.resolve_device()
        if resolved_device == "cuda":
            return "float16"
        return "int8"

    def resolve_cpu_threads(self) -> int:
        """Resolve CPU thread count (0 = auto-detect physical cores)."""
        if self.cpu_threads > 0:
            return self.cpu_threads
        try:
            return max(1, (os.cpu_count() or 4) // 2)
        except Exception:
            return 4

    def resolve_beam_size(self, preset: str | None = None) -> int:
        """Get beam size for a speed preset."""
        p = preset or self.speed_preset
        presets = {"fast": 1, "balanced": 3, "accurate": 5}
        return presets.get(p, self.beam_size)

    def resolve_batch_size(self, preset: str | None = None) -> int:
        """Get batch size for a speed preset."""
        p = preset or self.speed_preset
        presets = {"fast": 8, "balanced": 4, "accurate": 1}
        return presets.get(p, self.batch_size)

    def resolve_temperature(self, preset: str | None = None) -> float | list[float]:
        """Get temperature setting for a speed preset.

        fast: 0 (no fallback, fastest).
        balanced/accurate: fallback chain (re-decode on failure).
        """
        p = preset or self.speed_preset
        if p == "fast":
            return 0
        return [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def resolve_condition_on_previous_text(self, preset: str | None = None) -> bool:
        """Whether to condition on previous segment text.

        Only in accurate mode — adds coherence but costs speed.
        """
        p = preset or self.speed_preset
        return p == "accurate"


@lru_cache(maxsize=1)
def get_settings() -> AudioBenchSettings:
    """Get cached application settings (singleton)."""
    return AudioBenchSettings()
