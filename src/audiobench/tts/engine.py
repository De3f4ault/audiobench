"""Piper TTS engine — offline text-to-speech synthesis.

Wraps piper-tts for high-quality, low-latency speech synthesis.
All processing happens locally via ONNX models — no cloud APIs.

Usage:
    from audiobench.tts.engine import PiperTTSEngine

    engine = PiperTTSEngine()
    engine.synthesize("Hello world", output_path="hello.wav")
    engine.play("Hello world")
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from audiobench.core.error_types import AudioBenchError
from audiobench.core.logger_factory import get_logger

logger = get_logger("tts.engine")

# Default voice model download URL base
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class TTSError(AudioBenchError):
    """Text-to-speech engine failure."""


@dataclass
class VoiceInfo:
    """Metadata about a Piper voice model."""

    name: str  # e.g., "en_US-amy-medium"
    language: str  # e.g., "en_US"
    quality: str  # e.g., "medium", "high", "low"
    model_path: Path  # path to .onnx file
    config_path: Path  # path to .onnx.json file


class PiperTTSEngine:
    """Text-to-speech engine using Piper (ONNX-based, offline).

    Handles model loading, synthesis, and playback. Voice models
    are stored in ~/.audiobench/voices/.
    """

    def __init__(self, voices_dir: str | Path | None = None) -> None:
        self._voices_dir = Path(voices_dir) if voices_dir else self._default_voices_dir()
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._voice = None  # Lazy-loaded PiperVoice instance
        self._voice_name: str | None = None

    @staticmethod
    def _default_voices_dir() -> Path:
        """Default directory for voice models."""
        return Path.home() / ".audiobench" / "voices"

    def _load_voice(self, voice_name: str) -> None:
        """Load a Piper voice model.

        Args:
            voice_name: Voice identifier (e.g., 'en_US-amy-medium').

        Raises:
            TTSError: If the voice model is not found or fails to load.
        """
        if self._voice is not None and self._voice_name == voice_name:
            return  # Already loaded

        try:
            from piper import PiperVoice
        except ImportError:
            raise TTSError(
                "piper-tts not installed",
                "Install with: pip install piper-tts\nOr: pip install -e '.[tts]'",
            ) from None

        model_path = self._voices_dir / f"{voice_name}.onnx"
        config_path = self._voices_dir / f"{voice_name}.onnx.json"

        if not model_path.exists():
            raise TTSError(
                f"Voice model not found: {voice_name}",
                f"Expected at: {model_path}\n"
                f"Download with: audiobench download-voice {voice_name}",
            )

        logger.info("Loading TTS voice: %s", voice_name)
        try:
            self._voice = PiperVoice.load(str(model_path), str(config_path))
            self._voice_name = voice_name
            logger.info("Voice loaded: %s", voice_name)
        except Exception as e:
            raise TTSError(
                f"Failed to load voice: {voice_name}",
                str(e),
            ) from e

    def synthesize(
        self,
        text: str,
        voice: str = "en_US-amy-medium",
        output_path: str | Path | None = None,
    ) -> Path:
        """Synthesize text to a WAV file.

        Args:
            text: Text to speak.
            voice: Piper voice name.
            output_path: Where to save the WAV. If None, auto-generates a path.

        Returns:
            Path to the generated WAV file.

        Raises:
            TTSError: If synthesis fails.
        """
        self._load_voice(voice)

        if output_path is None:
            output_path = Path.cwd() / "speech_output.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Synthesizing %d chars to %s", len(text), output_path)

        try:
            # Piper's synthesize_wav() handles WAV headers automatically
            with wave.open(str(output_path), "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)

            logger.info("Synthesized: %s (%d bytes)", output_path.name, output_path.stat().st_size)
            return output_path

        except Exception as e:
            raise TTSError("Synthesis failed", str(e)) from e

    def play(self, text: str, voice: str = "en_US-amy-medium") -> None:
        """Synthesize and play text through speakers.

        Args:
            text: Text to speak.
            voice: Piper voice name.

        Raises:
            TTSError: If playback fails.
        """
        self._load_voice(voice)

        try:
            import sounddevice as sd
        except ImportError:
            raise TTSError(
                "sounddevice not installed",
                "Install with: pip install sounddevice",
            ) from None

        logger.info("Playing TTS: %d chars with voice %s", len(text), voice)

        try:
            # Collect raw audio chunks from synthesize() generator
            audio_bytes: list[bytes] = []
            sample_rate = 22050  # Piper default, updated from first chunk

            for chunk in self._voice.synthesize(text):
                audio_bytes.append(chunk.audio_int16_bytes)
                sample_rate = chunk.sample_rate

            if not audio_bytes:
                logger.warning("TTS produced no audio")
                return

            # Convert to numpy and play
            import numpy as np

            raw_audio = b"".join(audio_bytes)
            samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            sd.play(samples, samplerate=sample_rate)
            sd.wait()  # Block until playback finishes

        except TTSError:
            raise
        except Exception as e:
            raise TTSError("Playback failed", str(e)) from e

    def list_voices(self) -> list[VoiceInfo]:
        """List locally available voice models.

        Returns:
            List of VoiceInfo for each downloaded voice.
        """
        voices: list[VoiceInfo] = []
        for model_path in sorted(self._voices_dir.glob("*.onnx")):
            if model_path.name.endswith(".onnx.json"):
                continue  # skip config files
            name = model_path.stem  # e.g., "en_US-amy-medium"
            parts = name.split("-")
            language = parts[0] if parts else "unknown"
            quality = parts[-1] if len(parts) > 2 else "unknown"

            voices.append(
                VoiceInfo(
                    name=name,
                    language=language,
                    quality=quality,
                    model_path=model_path,
                    config_path=model_path.with_suffix(".onnx.json"),
                )
            )

        return voices

    def download_voice(self, voice_name: str) -> Path:
        """Download a Piper voice model from HuggingFace.

        Args:
            voice_name: Voice identifier (e.g., 'en_US-amy-medium').

        Returns:
            Path to the downloaded .onnx model file.

        Raises:
            TTSError: If download fails.
        """
        import urllib.request

        model_path = self._voices_dir / f"{voice_name}.onnx"
        config_path = self._voices_dir / f"{voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            logger.info("Voice already downloaded: %s", voice_name)
            return model_path

        # Parse voice name to build URL path
        # e.g., "en_US-amy-medium" → "en/en_US/amy/medium/"
        parts = voice_name.split("-")
        if len(parts) < 3:
            raise TTSError(
                f"Invalid voice name format: {voice_name}",
                "Expected format: lang_COUNTRY-name-quality (e.g., en_US-amy-medium)",
            )

        lang_code = parts[0]  # e.g., "en_US"
        lang = lang_code.split("_")[0]  # e.g., "en"
        speaker = parts[1]  # e.g., "amy"
        quality = parts[2]  # e.g., "medium"

        base = f"{PIPER_VOICES_URL}/{lang}/{lang_code}/{speaker}/{quality}"
        model_url = f"{base}/{voice_name}.onnx"
        config_url = f"{base}/{voice_name}.onnx.json"

        logger.info("Downloading voice model: %s", model_url)

        try:
            for url, dest in [(config_url, config_path), (model_url, model_path)]:
                logger.info("Downloading: %s → %s", url, dest.name)
                urllib.request.urlretrieve(url, dest)

            logger.info(
                "Voice downloaded: %s (%d MB)",
                voice_name,
                model_path.stat().st_size // (1024 * 1024),
            )
            return model_path

        except Exception as e:
            # Cleanup partial downloads
            for p in [model_path, config_path]:
                if p.exists():
                    p.unlink()
            raise TTSError(f"Failed to download voice: {voice_name}", str(e)) from e
