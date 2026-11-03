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

import queue
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiobench.core.error_types import AudioBenchError
from audiobench.core.logger_factory import get_logger

logger = get_logger("tts.engine")

# Default voice model download URL base
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Module-level cache for loaded ONNX models (bounded to prevent excessive RAM usage)
_LOADED_VOICES: dict[str, Any] = {}


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


def split_speech_sentences(text: str) -> list[str]:
    """Split text into sentence units preserving punctuation for TTS alignment."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sentences: list[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p_clean = p.strip()
            if p_clean:
                sentences.append(p_clean)
    return sentences or ([text.strip()] if text.strip() else [])


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

        if voice_name in _LOADED_VOICES:
            self._voice = _LOADED_VOICES[voice_name]
            self._voice_name = voice_name
            return

        model_path = self._voices_dir / f"{voice_name}.onnx"
        config_path = self._voices_dir / f"{voice_name}.onnx.json"

        if not model_path.exists():
            raise TTSError(
                f"Voice model not found: {voice_name}",
                f"Expected at: {model_path}\nDownload with: audiobench download-voice {voice_name}",
            )

        logger.info("Loading TTS voice: %s", voice_name)
        try:
            self._voice = PiperVoice.load(str(model_path), str(config_path))
            self._voice_name = voice_name
            if len(_LOADED_VOICES) >= 2:
                oldest = next(iter(_LOADED_VOICES))
                del _LOADED_VOICES[oldest]
            _LOADED_VOICES[voice_name] = self._voice
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

    def synthesize_with_timestamps(
        self,
        text: str,
        voice: str = "en_US-amy-medium",
        output_path: str | Path | None = None,
    ) -> tuple[Path, list[dict]]:
        """Synthesize text to a WAV file with exact sentence-level timestamps.

        Args:
            text: Text to speak.
            voice: Piper voice name.
            output_path: Where to save the WAV. If None, auto-generates a path.

        Returns:
            Tuple of (output_wav_path, segments) where each segment has:
            {"start": float, "end": float, "text": str, "speaker": "assistant"}.

        Raises:
            TTSError: If synthesis fails.
        """
        self._load_voice(voice)

        if output_path is None:
            output_path = Path.cwd() / "speech_output.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sentences = split_speech_sentences(text)
        segments: list[dict] = []
        sample_rate = 22050
        current_time = 0.0

        logger.info(
            "Synthesizing %d sentences (%d chars) with timestamps to %s",
            len(sentences),
            len(text),
            output_path,
        )

        try:
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                for sentence in sentences:
                    sentence_chunks = list(self._voice.synthesize(sentence))
                    if not sentence_chunks:
                        continue

                    chunk_sr = sentence_chunks[0].sample_rate
                    if chunk_sr != sample_rate:
                        sample_rate = chunk_sr
                        wav_file.setframerate(sample_rate)

                    total_samples = 0
                    for chunk in sentence_chunks:
                        wav_file.writeframes(chunk.audio_int16_bytes)
                        total_samples += len(chunk.audio_int16_array)

                    duration = total_samples / float(sample_rate)
                    end_time = current_time + duration
                    segments.append({
                        "start": round(current_time, 2),
                        "end": round(end_time, 2),
                        "text": sentence,
                        "speaker": "assistant",
                    })
                    current_time = end_time

            logger.info(
                "Synthesized %d segments: %s (%d bytes)",
                len(segments),
                output_path.name,
                output_path.stat().st_size,
            )
            return output_path, segments

        except Exception as e:
            raise TTSError("Synthesis with timestamps failed", str(e)) from e

    _active_stop_event: threading.Event | None = None

    @classmethod
    def stop(cls) -> None:
        """Stop any active audio playback immediately."""
        if cls._active_stop_event is not None:
            cls._active_stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def play(self, text: str, voice: str = "en_US-amy-medium") -> None:
        """Synthesize and play text through speakers with pause/stop support.

        Args:
            text: Text to speak.
            voice: Piper voice name.

        Interactive Controls:
            - Space / P: Pause or resume playback
            - Q / Enter / Ctrl+C: Stop playback immediately

        Raises:
            TTSError: If playback fails.
        """
        self._load_voice(voice)

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as exc:
            raise TTSError(
                "Audio playback dependencies missing",
                "Install with: pip install sounddevice numpy",
            ) from exc

        logger.info("Playing TTS: %d chars with voice %s", len(text), voice)

        audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        synthesis_done = threading.Event()
        stop_requested = threading.Event()
        PiperTTSEngine._active_stop_event = stop_requested

        is_paused = False
        current_buf = np.array([], dtype=np.float32)
        buf_pos = 0
        sample_rate = 22050

        def audio_callback(outdata, frames, time_info, status):
            nonlocal current_buf, buf_pos
            if stop_requested.is_set():
                raise sd.CallbackStop
            if is_paused:
                outdata.fill(0)
                return

            filled = 0
            while filled < frames:
                if buf_pos < len(current_buf):
                    avail = len(current_buf) - buf_pos
                    needed = frames - filled
                    to_copy = min(avail, needed)
                    outdata[filled : filled + to_copy, 0] = current_buf[buf_pos : buf_pos + to_copy]
                    buf_pos += to_copy
                    filled += to_copy
                else:
                    try:
                        chunk = audio_queue.get_nowait()
                        if chunk is None:
                            # Sentinel for end of audio stream
                            outdata[filled:, 0] = 0
                            raise sd.CallbackStop
                        current_buf = chunk
                        buf_pos = 0
                    except queue.Empty:
                        if synthesis_done.is_set():
                            outdata[filled:, 0] = 0
                            raise sd.CallbackStop
                        else:
                            # Underflow while waiting for synthesizer chunk
                            outdata[filled:, 0] = 0
                            break

        def synthesize_worker():
            nonlocal sample_rate
            try:
                for chunk in self._voice.synthesize(text):
                    if stop_requested.is_set():
                        break
                    sample_rate = chunk.sample_rate
                    raw = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_queue.put(raw)
            except Exception as e:
                logger.error("TTS synthesis error in worker: %s", e)
            finally:
                synthesis_done.set()
                audio_queue.put(None)

        synth_thread = threading.Thread(target=synthesize_worker, daemon=True)
        synth_thread.start()

        # Wait briefly for first audio chunk or synthesis completion
        while audio_queue.empty() and not synthesis_done.is_set() and not stop_requested.is_set():
            time.sleep(0.01)

        if stop_requested.is_set() or (audio_queue.empty() and synthesis_done.is_set()):
            return

        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=audio_callback,
            ) as stream:
                if is_tty:
                    import select
                    import termios

                    fd = sys.stdin.fileno()
                    old_attr = termios.tcgetattr(fd)
                    try:
                        new_attr = termios.tcgetattr(fd)
                        new_attr[3] = new_attr[3] & ~termios.ICANON & ~termios.ECHO
                        new_attr[6][termios.VMIN] = 1
                        new_attr[6][termios.VTIME] = 0
                        termios.tcsetattr(fd, termios.TCSANOW, new_attr)

                        while stream.active and not stop_requested.is_set():
                            r, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if r:
                                ch = sys.stdin.read(1)
                                if ch in (" ", "p", "P"):
                                    is_paused = not is_paused
                                    if is_paused:
                                        print("\r  \033[36m[Voice]\033[0m \033[33m[Paused]\033[0m (Space: resume · q: stop)   ", end="", flush=True)
                                    else:
                                        print("\r  \033[36m[Voice]\033[0m \033[2mResumed.\033[0m                                ", end="", flush=True)
                                elif ch in ("q", "Q", "\r", "\n", "\x03"):
                                    stop_requested.set()
                                    print("\r  \033[36m[Voice]\033[0m \033[2mPlayback stopped.\033[0m                         ", end="", flush=True)
                                    break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
                        print()
                else:
                    while stream.active and not stop_requested.is_set():
                        time.sleep(0.05)

        except KeyboardInterrupt:
            stop_requested.set()
            try:
                sd.stop()
            except Exception:
                pass
            raise
        except TTSError:
            raise
        except Exception as e:
            if not stop_requested.is_set():
                raise TTSError("Playback failed", str(e)) from e
        finally:
            stop_requested.set()
            PiperTTSEngine._active_stop_event = None
            synth_thread.join(timeout=0.2)

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
