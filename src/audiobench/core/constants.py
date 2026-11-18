"""Centralized constants for AudioBench settings, validation, and mappings."""

# --- Whisper Engine Constants ---
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}

# --- Hardware Constants ---
COMPUTE_DEVICES = {"auto", "cpu", "cuda"}
COMPUTE_TYPES = {"int8", "float16", "float32"}

# --- Output & Formatting ---
OUTPUT_FORMATS = {"txt", "srt", "vtt", "json"}

# --- Application Presets ---
SPEED_PRESETS = {"fast", "balanced", "accurate"}

# --- Audio Processing Constants ---
# Map common audio extensions to MIME types.
MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
}


def get_mime(suffix: str) -> str:
    """Resolve MIME type from file extension."""
    return MIME_MAP.get(suffix.lower(), "audio/mpeg")


# --- Text & Search Constants ---
# Common English grammatical stop words that match broadly across corpora.
# Used for noise reduction while preserving 1-3 letter domain terms (fox, tax, law, war, net, etc.).
ENGLISH_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with",
}
