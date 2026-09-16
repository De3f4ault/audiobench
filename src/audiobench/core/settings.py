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

from audiobench.core.constants import (
    COMPUTE_DEVICES,
    COMPUTE_TYPES,
    OUTPUT_FORMATS,
    SPEED_PRESETS,
    WHISPER_MODELS,
)

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
    device_index: str = Field(
        default="auto",
        description=(
            "GPU index for Whisper: 'auto' (uses GPU 0 by default, or all GPUs if >1), "
            "'0', '1', or comma-separated '0,1'. Ignored on CPU."
        ),
    )
    diarization_device: str = Field(
        default="auto",
        description=(
            "Device for pyannote: 'auto' (GPU:N-1 if multi-GPU, GPU:0 if single-GPU, "
            "cpu if no GPU). Accepts 'cpu', 'cuda:0', 'cuda:1', etc."
        ),
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

    # --- Play UI ---
    play_mode: str = Field(
        default="default", description="Active display mode: 'default' or 'enhanced'"
    )
    play_karaoke: bool = Field(
        default=False, description="Karaoke word highlight (linear interpolation)"
    )
    play_focus_gradient: bool = Field(
        default=True, description="4-level focus gradient dimming"
    )
    play_center_lock: bool = Field(
        default=True, description="Pin active line to vertical center"
    )
    play_speaker_badges: bool = Field(
        default=True, description="Show speaker change labels"
    )
    play_timestamps: bool = Field(
        default=False, description="Show [HH:MM:SS] gutter on each line"
    )
    play_show_remaining: bool = Field(
        default=False, description="Show remaining time on progress bar"
    )

    # --- Features ---
    enable_diarization: bool = Field(default=False, description="Enable speaker diarization")

    # --- Database ---
    # CANONICAL DATABASE: data/transcriptions.db (relative to project root).
    # The absolute path is computed from settings.py's own location — it does NOT
    # depend on the working directory, so it resolves identically from any invocation
    # context (CLI, daemon, test script). There is exactly one database. If you see
    # any other .db file at a plausible-looking path, it is a stub or artifact.
    database_url: str = Field(
        default_factory=lambda: f"sqlite:///{_DATA_DIR / 'transcriptions.db'}",
        description="SQLAlchemy database URL",
    )

    # --- Storage ---
    models_dir: Path = Field(
        default=Path.home() / ".audiobench" / "models",
        description="Directory for downloaded models (shared, multi-GB)",
    )
    offline_mode: bool = Field(
        default=False,
        description="Force offline mode (prevents HF/network calls for model loading)",
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
        default="gpt-oss:120b-cloud",
        description="Default Ollama model for AI features",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    bookmark_model: str = Field(
        default="qwen3-coder:480b-cloud",
        description="Ollama model for AI bookmark extraction (structured output)",
    )
    clean_model: str = Field(
        default="qwen4-next:110b-cloud",
        description="Ollama model for transcript cleaning (spelling/punctuation correction)",
    )

    # --- Daemon & Memory Layer ---
    disable_memory: bool = Field(
        default=False,
        description="Disable daemon auto-start and semantic memory embeddings entirely",
    )
    daemon_socket_path: Path = Field(
        default=Path("/tmp/audiobench-daemon.sock"),
        description="Path to daemon Unix socket",
    )
    daemon_pid_path: Path = Field(
        default=Path("/tmp/audiobench-daemon.pid"),
        description="Path to daemon PID file",
    )
    daemon_warmup_timeout: float = Field(
        default=120.0, description="Seconds to wait for daemon startup"
    )
    daemon_ping_timeout: float = Field(
        default=0.1, description="Seconds to wait for daemon health check (fail fast)"
    )
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="Primary embedding model for vector storage",
    )
    boundary_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Fast embedding model for semantic boundary detection",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for search result reranking",
    )
    embedding_dims: int = Field(default=768, description="Dimensions of primary embedding vectors")
    chunk_breakpoint_percentile: float = Field(
        default=85.0,
        description="Percentile threshold for semantic chunk boundaries",
    )
    chunk_max_tokens: int = Field(default=350, description="Max tokens per chunk fallback guard")
    chunk_sentence_group_size: int = Field(
        default=3, description="Number of sentences per boundary comparison block"
    )
    chunk_short_threshold: int = Field(
        default=600, description="Characters below which to skip chunking entirely"
    )
    chunk_long_threshold: int = Field(
        default=10_000, description="Characters above which to enforce max_tokens guard"
    )
    retrieval_top_k: int = Field(
        default=15, description="Initial candidates to retrieve from LanceDB"
    )
    rerank_top_k: int = Field(default=5, description="Candidates to keep after reranking")
    summary_min_turns: int = Field(
        default=3, description="Minimum chat turns required to trigger session summary"
    )
    lancedb_path: Path = Field(
        default_factory=lambda: _DATA_DIR / "lancedb",
        description="Path to embedded LanceDB directory",
    )
    lancedb_optimize_interval_days: int = Field(
        default=3,
        ge=-1,
        description=(
            "Days between automatic LanceDB optimize runs (startup check). "
            "Set to 0 to optimize on every startup. Set to -1 to disable startup check."
        ),
    )
    lancedb_optimize_write_threshold: int = Field(
        default=10_000,
        ge=0,
        description=(
            "Combined number of new expression + segment vector writes that trigger "
            "a background LanceDB optimize after a sweep pass completes. "
            "Set to 0 to disable write-count trigger."
        ),
    )

    # --- Engine Selection ---
    engine: str = Field(
        default="whisper",
        description="Transcription engine: whisper (local) or gemini (cloud)",
    )

    # --- YouTube (optional) ---
    youtube_api_key: str | None = Field(
        default=None,
        description="API key for YouTube Data API v3",
    )

    # --- Google Gemini (optional cloud engine) ---
    gemini_api_key: str | None = Field(
        default=None,
        description="API key for Google Gemini (enable with --engine gemini)",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use for transcription",
    )
    gemini_inline_max_mb: int = Field(
        default=100,
        ge=1,
        description=(
            "Files below this size (MB) are sent inline in the request body. "
            "Larger files are uploaded via the Gemini Files API. "
            "Google increased the inline payload limit from 20 MB to 100 MB in early 2026."
        ),
    )
    gemini_chunk_threshold_min: int = Field(
        default=2,
        ge=1,
        description=(
            "Audio duration in minutes above which to chunk for Gemini. "
            "Gemini models have an 8192 output token limit, which is exhausted quickly "
            "when word-level timestamps are enabled. Chunking prevents truncation."
        ),
    )
    align_threshold_min: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Audio duration in minutes above which to automatically trigger the hybrid "
            "forced alignment pipeline when using Gemini (e.g. 0.5 = 30 seconds)."
        ),
    )
    align_device: str = Field(
        default="cpu",
        description="Device to use for the forced alignment model (cpu, cuda).",
    )
    gemini_upload_fallback_model: str = Field(
        default="gemini-2.0-flash",
        description=(
            "Fallback Gemini model used when the primary model exhausts its quota "
            "(all tenacity retries fail with ResourceExhausted)."
        ),
    )
    gemini_max_retries: int = Field(
        default=6,
        ge=1,
        le=20,
        description=(
            "Maximum number of retry attempts on Gemini 429/ResourceExhausted errors. "
            "Uses exponential backoff with jitter (2s → 60s). "
            "Applies to both the primary and fallback model calls."
        ),
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")
    observatory: dict = Field(
        default_factory=dict,
        description="Configuration for observatory and thresholds (e.g. {'thresholds': {'transcribe.confidence': ['lt', 0.6, 'WARN']}})",
    )

    # --- Diarization ---
    hf_token: str | None = Field(
        default_factory=lambda: os.environ.get("HF_TOKEN"),
        description="HuggingFace token for pyannote model download",
    )

    # --- Security Layer ---
    voiceprint_threshold: float = Field(
        default=0.75,
        description=(
            "Cosine similarity threshold for SpeechBrain ECAPA-TDNN voiceprint matching. "
            "Higher = stricter (fewer false positives). Lower = looser (better recall). "
            "Range: 0.0–1.0. Recommended: 0.70–0.80."
        ),
    )
    idle_lock_seconds: int = Field(
        default=180,
        ge=0,
        description=(
            "Seconds of keyboard idle before the session auto-locks (drops unlocked_tier to 0). "
            "Set to 0 to disable idle locking entirely."
        ),
    )
    security_dir: Path = Field(
        default_factory=lambda: _DATA_DIR / "security",
        description=(
            "Directory for security artefacts: security.json (passphrase hashes) "
            "and voiceprint.npy (enrolled ECAPA-TDNN embedding)."
        ),
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
        if v not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid model: {v}. Choose from: {', '.join(sorted(WHISPER_MODELS))}"
            )
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in COMPUTE_DEVICES:
            raise ValueError(
                f"Invalid device: {v}. Choose from: {', '.join(sorted(COMPUTE_DEVICES))}"
            )
        return v

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        if v not in COMPUTE_TYPES:
            raise ValueError(
                f"Invalid compute_type: {v}. Choose from: {', '.join(sorted(COMPUTE_TYPES))}"
            )
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        if v not in OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format: {v}. Choose from: {', '.join(sorted(OUTPUT_FORMATS))}"
            )
        return v

    @field_validator("speed_preset")
    @classmethod
    def validate_speed_preset(cls, v: str) -> str:
        if v not in SPEED_PRESETS:
            raise ValueError(
                f"Invalid speed_preset: {v}. Choose from: {', '.join(sorted(SPEED_PRESETS))}"
            )
        return v

    def ensure_dirs(self) -> None:
        """Create data and model directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Save current settings to settings.json in the data directory."""
        self.ensure_dirs()
        json_path = self.data_dir / "settings.json"
        # We write using Pydantic's model_dump_json for clean serialization
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

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

    def resolve_device_index(self) -> int | list[int]:
        """Resolve 'auto' device index to actual device indices."""
        try:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n_gpus = 0

        if self.device_index != "auto":
            if isinstance(self.device_index, str) and "," in self.device_index:
                indices = [int(x.strip()) for x in self.device_index.split(",")]
                valid_indices = [i for i in indices if i < n_gpus]
                if not valid_indices:
                    return 0
                return valid_indices if len(valid_indices) > 1 else valid_indices[0]
            elif isinstance(self.device_index, str) and self.device_index.isdigit():
                idx = int(self.device_index)
                return idx if idx < n_gpus else 0
            return self.device_index if isinstance(self.device_index, int) and self.device_index < n_gpus else 0

        # Auto resolution
        resolved_device = self.resolve_device()
        if resolved_device == "cuda":
            if n_gpus > 1:
                return 0  # Default to 0 for Whisper to leave GPU 1 for Pyannote
            return 0
        return 0

    def resolve_diarization_device(self) -> str:
        """Resolve 'auto' diarization device to actual device."""
        try:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n_gpus = 0

        if self.diarization_device != "auto":
            if self.diarization_device.startswith("cuda:") and n_gpus > 0:
                try:
                    idx = int(self.diarization_device.split(":")[1])
                    if idx < n_gpus:
                        return self.diarization_device
                    return f"cuda:{n_gpus - 1}"
                except ValueError:
                    pass
            elif self.diarization_device == "cpu":
                return "cpu"

        if n_gpus == 0:
            return "cpu"
        elif n_gpus == 1:
            # Warn if VRAM is tight (e.g. < 8GB)
            try:
                import torch
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 8 * 1024 * 1024 * 1024:
                    from audiobench.core.logger_factory import get_logger
                    get_logger("settings").warning(
                        "Single GPU with < 8GB VRAM detected. Running Whisper and Pyannote "
                        "sequentially may cause OutOfMemory errors."
                    )
            except Exception:
                pass
            return "cuda:0"
        else:
            return f"cuda:{n_gpus - 1}"

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
    """Get cached application settings (singleton), loading from settings.json if present."""
    import json

    json_path = _DATA_DIR / "settings.json"
    json_data = {}

    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            # Fallback to default loading if JSON is corrupted
            print(f"Warning: Could not parse settings.json: {e}")

    # Remove keys from json_data if they exist in the environment,
    # so that environment variables take priority over settings.json
    for key in list(json_data.keys()):
        env_key = f"AUDIOBENCH_{key.upper()}"
        if env_key in os.environ:
            del json_data[key]

    # Also explicitly check for HF_TOKEN
    if "HF_TOKEN" in os.environ and "hf_token" in json_data:
        del json_data["hf_token"]

    return AudioBenchSettings(**json_data)


def invalidate_settings_cache() -> None:
    """Clear the cached settings so the next call to get_settings() reloads from disk."""
    get_settings.cache_clear()
