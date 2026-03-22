"""Engine factory — creates transcription engines from configuration.

Usage:
    from audiobench.transcribe.engines.engine_registry import create_engine

    engine = create_engine("whisper", model_name="large-v3-turbo", device="cpu")
    transcript = engine.transcribe("audio.wav", batch_size=4)
"""

from __future__ import annotations

from audiobench.core.error_types import EngineError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine

logger = get_logger("engines.factory")

# Registry of available engines
_ENGINE_REGISTRY: dict[str, type[TranscriptionEngine]] = {}


def register_engine(name: str, engine_class: type[TranscriptionEngine]) -> None:
    """Register an engine class by name."""
    _ENGINE_REGISTRY[name] = engine_class
    logger.debug("Registered engine: %s → %s", name, engine_class.__name__)


def _ensure_registered() -> None:
    """Lazy-register built-in engines on first use."""
    if "whisper" not in _ENGINE_REGISTRY:
        from audiobench.transcribe.engines.whisper_engine import WhisperEngine

        register_engine("whisper", WhisperEngine)

    # Vosk is optional — only register if available
    if "vosk" not in _ENGINE_REGISTRY:
        try:
            from audiobench.transcribe.engines.vosk_engine import VoskEngine

            register_engine("vosk", VoskEngine)
        except ImportError:
            pass

    # Gemini is optional — only register if google-genai is installed
    if "gemini" not in _ENGINE_REGISTRY:
        try:
            from audiobench.transcribe.engines.gemini_engine import GeminiEngine

            register_engine("gemini", GeminiEngine)
        except ImportError:
            pass


def create_engine(
    engine_name: str = "whisper",
    model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 4,
) -> TranscriptionEngine:
    """Create and initialize a transcription engine.

    Args:
        engine_name: Engine identifier ('whisper' or 'vosk').
        model_name: Model to load (e.g., 'large-v3-turbo', 'medium').
        device: Compute device ('cpu' or 'cuda').
        compute_type: Quantization ('int8', 'float16', 'float32').
        cpu_threads: CPU threads for CTranslate2.

    Returns:
        Initialized TranscriptionEngine ready for transcription.

    Raises:
        EngineError: If engine_name is unknown or loading fails.
    """
    _ensure_registered()

    if engine_name not in _ENGINE_REGISTRY:
        available = ", ".join(sorted(_ENGINE_REGISTRY.keys()))
        raise EngineError(
            message=f"Unknown engine: '{engine_name}'",
            details=f"Available engines: {available}",
        )

    logger.info(
        "Creating engine: %s (model=%s, device=%s, compute=%s, threads=%d)",
        engine_name,
        model_name,
        device,
        compute_type,
        cpu_threads,
    )

    engine_class = _ENGINE_REGISTRY[engine_name]
    engine = engine_class()
    engine.load_model(model_name, device, compute_type, cpu_threads=cpu_threads)

    return engine


def list_engines() -> list[str]:
    """Return names of all registered engines."""
    _ensure_registered()
    return sorted(_ENGINE_REGISTRY.keys())
