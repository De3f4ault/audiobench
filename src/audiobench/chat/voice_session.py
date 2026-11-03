"""Voice Chat Session — Native CLI hardware I/O and Gemini Live streaming.

Provides:
  - VoiceChatSession: Manages microphone capture, speaker output, Gemini Live connection,
    audio session recording, and turn timestamp tracking.
  - Supports single-turn push-to-talk queries (/voice) and continuous two-way live dialogue (/voice live).
  - Strictly no emojis in console output or database records.
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import AudioBenchSettings, get_settings

logger = get_logger("chat.voice_session")

# Audio format constants
INPUT_SAMPLE_RATE = 16000   # 16kHz PCM mono for mic capture & Gemini input
OUTPUT_SAMPLE_RATE = 24000  # 24kHz PCM mono from Gemini Live
GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


def is_internet_connected(timeout: float = 1.0) -> bool:
    """Fast check whether an internet connection is reachable."""
    import socket
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            return True
    except OSError:
        pass
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except OSError:
        return False


def is_local_stt_available() -> bool:
    """Check if local faster-whisper is installed."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


_LOADED_WHISPER_MODELS: dict[str, Any] = {}


def get_whisper_model(model_name: str = "base", device: str = "cpu", compute_type: str = "int8") -> Any:
    """Get or load a cached faster-whisper model."""
    key = f"{model_name}:{device}:{compute_type}"
    if key in _LOADED_WHISPER_MODELS:
        return _LOADED_WHISPER_MODELS[key]
    from faster_whisper import WhisperModel
    m = WhisperModel(model_name, device=device, compute_type=compute_type)
    if len(_LOADED_WHISPER_MODELS) >= 2:
        oldest = next(iter(_LOADED_WHISPER_MODELS))
        del _LOADED_WHISPER_MODELS[oldest]
    _LOADED_WHISPER_MODELS[key] = m
    return m


def capture_voice_query(
    console: Any = None,
    silence_timeout: float = 5.0,
    prompt_message: str = "Listening for query... (speak now, 5s silence or Enter to send)",
) -> tuple[str, bool]:
    """Capture a one-shot query from the microphone for search or chat.

    Returns:
        Tuple of (transcribed_text, is_offline_fallback).
        If offline and no STT is available, returns ("", True).
        If user cancels via Ctrl+C, returns ("", False).
    """
    from audiobench.cli.display.theme import console as default_console
    c = console or default_console

    # Check offline status first
    if not is_internet_connected():
        if not is_local_stt_available():
            c.print("  [yellow][Voice] Offline: speech recognition requires an internet connection (or local Whisper).[/yellow]")
            c.print("  [yellow][Voice] Falling back to standard keyboard input.[/yellow]")
            return "", True

    daemon_client = None
    session_id = f"voice_query_{os.getpid()}"
    try:
        from audiobench.daemon.factory import get_daemon_client
        daemon_client = get_daemon_client()
        if daemon_client and hasattr(daemon_client, "whisper_acquire"):
            s_obj = get_settings()
            daemon_client.whisper_acquire(session_id, model=s_obj.stt_model or "base")
    except Exception:
        daemon_client = None

    try:
        session = VoiceChatSession(conversation_id=0, console=c, record_audio=False)
        text, _, _ = session.record_single_turn(
            silence_timeout=silence_timeout,
            prompt_message=prompt_message,
        )
        return text, False
    except KeyboardInterrupt:
        c.print()
        c.print("  [dim][Voice] Cancelled.[/dim]")
        return "", False
    except Exception as exc:
        c.print(f"  [yellow][Voice] Voice capture failed ({exc}). Falling back to keyboard input.[/yellow]")
        return "", True
    finally:
        if daemon_client and hasattr(daemon_client, "whisper_release"):
            try:
                daemon_client.whisper_release(session_id)
            except Exception:
                pass


class VoiceChatSession:
    """Manages CLI voice interaction for Audiobench Chat."""

    def __init__(
        self,
        conversation_id: int,
        console: Console | None = None,
        settings: AudioBenchSettings | None = None,
        record_audio: bool = True,
        system_instruction: str | None = None,
    ):
        self.conversation_id = conversation_id
        self.console = console or Console()
        self.settings = settings or get_settings()
        self.record_audio = record_audio
        self.system_instruction = system_instruction

        self.turns: list[dict[str, Any]] = []
        self._session_start_time: float = 0.0
        self._audio_buffer = bytearray()
        self._stop_event = threading.Event()

    def _resample_24k_to_16k(self, audio_24k: np.ndarray) -> np.ndarray:
        """Resample 24kHz int16 PCM audio to 16kHz int16 using numpy interpolation."""
        if len(audio_24k) == 0:
            return np.array([], dtype=np.int16)
        target_len = int(len(audio_24k) * INPUT_SAMPLE_RATE / OUTPUT_SAMPLE_RATE)
        x_orig = np.arange(len(audio_24k))
        x_target = np.linspace(0, len(audio_24k) - 1, target_len)
        return np.interp(x_target, x_orig, audio_24k).astype(np.int16)

    def _save_session_wav(self) -> str | None:
        """Write recorded 16kHz mono audio buffer to a WAV file."""
        if not self.record_audio or len(self._audio_buffer) == 0:
            return None

        sessions_dir = self.settings.data_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = str(sessions_dir / f"voice_chat_{self.conversation_id}_{ts_str}.wav")

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(INPUT_SAMPLE_RATE)
                wf.writeframes(bytes(self._audio_buffer))

            logger.info("Saved voice session WAV: %s (%d bytes)", wav_path, len(self._audio_buffer))
            return wav_path
        except Exception as e:
            logger.error("Failed to save session WAV %s: %s", wav_path, e)
            return None

    # ── Single-Turn Push-to-Talk (/voice) ─────────────────────────

    def record_single_turn(
        self,
        silence_timeout: float = 5.0,
        prompt_message: str | None = None,
    ) -> tuple[str, str | None, float]:
        """Record a single spoken prompt from the microphone until silence or Enter.

        Transcribes via configured STT provider (local Whisper, cloud Gemini, or auto).

        Returns:
            Tuple of (transcribed_text, audio_file_path, duration_seconds).
        """
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError:
            self.console.print("  [red]Error: sounddevice not installed. Install with: pip install sounddevice[/red]")
            return "", None, 0.0

        raw_pcm, duration = self._capture_raw_audio(silence_timeout, prompt_message=prompt_message)
        if not raw_pcm or len(raw_pcm) < 3200:
            self.console.print("  [cyan][Voice][/cyan] [dim]No audio captured.[/dim]")
            return "", None, duration

        provider = (self.settings.stt_provider or "auto").lower()
        final_text = ""

        if provider == "local":
            whisper_m = self.settings.stt_model or self.settings.whisper_model or "base"
            self.console.print(f"  [cyan][Voice][/cyan] [dim]Recorded {duration:.1f}s. Transcribing with local Whisper [{whisper_m}]...[/dim]")
            final_text = self._transcribe_local(raw_pcm)
        elif provider == "cloud":
            self.console.print(f"  [cyan][Voice][/cyan] [dim]Recorded {duration:.1f}s. Transcribing with Gemini Cloud...[/dim]")
            try:
                final_text = self._transcribe_cloud(raw_pcm)
            except Exception as exc:
                logger.warning("Cloud transcription failed: %s, falling back to local Whisper", exc)
                self.console.print(f"  [yellow][Voice] Cloud transcription notice: {exc}, falling back to local Whisper...[/yellow]")
                final_text = self._transcribe_local(raw_pcm)
        else:  # "auto"
            online = is_internet_connected(timeout=0.8) and bool(self.settings.gemini_api_key)
            if online:
                self.console.print(f"  [cyan][Voice][/cyan] [dim]Recorded {duration:.1f}s. Transcribing with Gemini Cloud...[/dim]")
                try:
                    final_text = self._transcribe_cloud(raw_pcm)
                except Exception as exc:
                    logger.warning("Cloud transcription error in auto mode: %s, falling back to local", exc)
                    whisper_m = self.settings.stt_model or self.settings.whisper_model or "base"
                    self.console.print(f"  [yellow][Voice] Cloud failed. Falling back to local Whisper [{whisper_m}]...[/yellow]")
                    final_text = self._transcribe_local(raw_pcm)
            else:
                whisper_m = self.settings.stt_model or self.settings.whisper_model or "base"
                self.console.print(f"  [cyan][Voice][/cyan] [dim]Recorded {duration:.1f}s. Transcribing with local Whisper [{whisper_m}]...[/dim]")
                final_text = self._transcribe_local(raw_pcm)

        wav_path = None
        if self.record_audio and len(raw_pcm) > 0:
            self._audio_buffer = raw_pcm
            wav_path = self._save_session_wav()

        return final_text, wav_path, duration

    def _capture_raw_audio(
        self,
        silence_timeout: float = 5.0,
        prompt_message: str | None = None,
    ) -> tuple[bytearray, float]:
        """Record audio from the microphone until silence threshold or Enter key."""
        import sounddevice as sd
        import numpy as np

        provider = (self.settings.stt_provider or "auto").lower()
        if prompt_message:
            msg = prompt_message
        elif provider == "local":
            whisper_m = self.settings.stt_model or self.settings.whisper_model or "base"
            msg = f"Listening (local Whisper [{whisper_m}])... (speak now, 5s silence or Enter to submit)"
        elif provider == "cloud":
            msg = "Listening (Gemini Cloud)... (speak now, 5s silence or Enter to submit)"
        else:
            msg = "Listening... (speak now, 5s silence or Enter to submit)"

        self.console.print(f"  [cyan][Voice][/cyan] [dim]{msg}[/dim]")

        raw_pcm = bytearray()
        t_start = time.perf_counter()
        speech_detected = False
        last_speech = t_start
        stop_recording = False
        stop_event = threading.Event()

        def enter_listener():
            try:
                import select
                fd = sys.stdin.fileno() if sys.stdin and hasattr(sys.stdin, "fileno") else None
                if fd is None:
                    return
                while not stop_event.is_set():
                    r, _, _ = select.select([fd], [], [], 0.04)
                    if r and not stop_event.is_set():
                        try:
                            os.read(fd, 1024)
                        except Exception:
                            pass
                        stop_event.set()
                        break
            except Exception:
                pass

        enter_thread = threading.Thread(target=enter_listener, daemon=True)
        enter_thread.start()

        def mic_callback(indata, frames, time_info, status):
            nonlocal speech_detected, last_speech, stop_recording
            data_int16 = indata[:, 0]
            raw_pcm.extend(data_int16.tobytes())
            rms = np.sqrt(np.mean(data_int16.astype(float) ** 2)) if len(data_int16) > 0 else 0
            if rms > 450:  # Energy threshold for speech detection
                speech_detected = True
                last_speech = time.perf_counter()
            elif speech_detected and (time.perf_counter() - last_speech > silence_timeout):
                stop_recording = True

        stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=mic_callback,
        )

        try:
            with stream:
                while not stop_recording:
                    time.sleep(0.02)
                    if stop_event.is_set():
                        break
                    if time.perf_counter() - t_start > 60.0:
                        break
        except KeyboardInterrupt:
            stop_event.set()
            try:
                stream.abort()
                stream.stop()
            except Exception:
                pass
            raise
        finally:
            stop_event.set()

        duration = time.perf_counter() - t_start
        return raw_pcm, duration

    def _transcribe_local(self, raw_pcm: bytes | bytearray) -> str:
        """Transcribe raw PCM audio buffer using cached local faster-whisper."""
        if not raw_pcm:
            return ""

        whisper_tier = self.settings.stt_model or self.settings.whisper_model or "base"

        # 1. Try daemon transcription if daemon is running and has warm Whisper
        try:
            from audiobench.daemon.factory import get_daemon_client
            client = get_daemon_client()
            if client and hasattr(client, "whisper_transcribe"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    tmp_path = tmp_f.name
                try:
                    with wave.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(raw_pcm)

                    resp = client.whisper_transcribe(tmp_path, model=whisper_tier, language="en")
                    if resp and resp.get("ok"):
                        return resp.get("text", "")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        except Exception as daemon_exc:
            logger.debug("Daemon Whisper transcription unavailable (%s); using in-process model", daemon_exc)

        # 2. In-process fallback
        if not is_local_stt_available():
            self.console.print("  [yellow][Voice] faster-whisper is not installed for local STT.[/yellow]")
            return ""

        audio_f32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            model = get_whisper_model(
                whisper_tier,
                device=self.settings.resolve_device(),
                compute_type="int8",
            )
            segments, _ = model.transcribe(audio_f32, language="en")
            return " ".join(s.text.strip() for s in segments).strip()
        except KeyboardInterrupt:
            self.console.print("  [dim][Voice] Cancelled.[/dim]")
            raise
        except Exception as exc:
            logger.error("Local whisper transcription failed: %s", exc)
            self.console.print(f"  [yellow][Voice] Local Whisper failed: {exc}[/yellow]")
            return ""

    def _transcribe_cloud(self, raw_pcm: bytes | bytearray) -> str:
        """Transcribe raw PCM audio buffer using Gemini 2.5 Flash."""
        if not raw_pcm:
            return ""
        if not self.settings.gemini_api_key:
            raise RuntimeError("AUDIOBENCH_GEMINI_API_KEY is not configured")
        if not is_internet_connected(timeout=0.8):
            raise RuntimeError("Internet connection is offline")

        import io
        import wave
        from google import genai
        from google.genai import types

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(INPUT_SAMPLE_RATE)
            wf.writeframes(raw_pcm)
        wav_bytes = buf.getvalue()

        try:
            client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=15_000),  # 15 seconds deadline (min allowed by Gemini API is 10s)
            )
        except Exception:
            client = genai.Client(api_key=self.settings.gemini_api_key)

        cfg = types.GenerateContentConfig(
            system_instruction=(
                "You are an audio transcriber. Transcribe the user's spoken audio query accurately and verbatim. "
                "Preserve technical terms, programming jargon, and acronyms (e.g. SIMD, API, SQL) accurately. "
                "Never reply to the question or comment on it. "
                "Never add explanations or quotes. "
                "If the audio contains only silence or static, output nothing."
            ),
            temperature=0.0,
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                config=cfg,
            )
            return (response.text or "").strip()
        except KeyboardInterrupt:
            self.console.print("  [dim][Voice] Cancelled.[/dim]")
            raise

    def _record_single_turn_local(
        self,
        silence_timeout: float = 5.0,
        prompt_message: str | None = None,
    ) -> tuple[str, str | None, float]:
        """Compatibility wrapper for local single turn recording."""
        raw_pcm, duration = self._capture_raw_audio(silence_timeout, prompt_message=prompt_message)
        final_text = self._transcribe_local(raw_pcm)
        wav_path = None
        if self.record_audio and len(raw_pcm) > 0:
            self._audio_buffer = raw_pcm
            wav_path = self._save_session_wav()
        return final_text, wav_path, duration

    async def _record_single_turn_gemini(
        self,
        silence_timeout: float = 5.0,
        prompt_message: str | None = None,
    ) -> tuple[str, str | None, float]:
        """Compatibility wrapper for cloud single turn recording."""
        raw_pcm, duration = self._capture_raw_audio(silence_timeout, prompt_message=prompt_message)
        try:
            final_text = self._transcribe_cloud(raw_pcm)
        except Exception:
            final_text = self._transcribe_local(raw_pcm)
        wav_path = None
        if self.record_audio and len(raw_pcm) > 0:
            self._audio_buffer = raw_pcm
            wav_path = self._save_session_wav()
        return final_text, wav_path, duration

    # ── Continuous Live Dialogue (/voice live) ───────────────────

    def run_continuous_live(self) -> tuple[list[dict[str, Any]], str | None, float]:
        """Run continuous two-way live voice mode with Gemini Live.

        Blocks until user interrupts with Ctrl+C.
        Returns:
            Tuple of (turns_list, audio_file_path, total_duration).
        """
        try:
            import sounddevice as sd
        except ImportError:
            self.console.print("  [red]Error: sounddevice not installed. Install with: pip install sounddevice[/red]")
            return [], None, 0.0

        if not self.settings.gemini_api_key:
            self.console.print("  [yellow][Voice] Live two-way conversation requires AUDIOBENCH_GEMINI_API_KEY.[/yellow]")
            return [], None, 0.0

        return asyncio.run(self._run_continuous_live_async())

    async def _run_continuous_live_async(self) -> tuple[list[dict[str, Any]], str | None, float]:
        import sounddevice as sd
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.settings.gemini_api_key)
        audio_in_q: asyncio.Queue[bytes] = asyncio.Queue()
        audio_out_q: queue.Queue[bytes] = queue.Queue()
        loop = asyncio.get_running_loop()

        self._session_start_time = time.perf_counter()
        current_user_text: list[str] = []
        current_ai_text: list[str] = []
        user_turn_start: float = 0.0
        ai_turn_start: float = 0.0
        is_ai_speaking = False
        self._stop_event.clear()

        # Input callback (Microphone 16kHz)
        def mic_callback(indata, frames, time_info, status):
            b = indata.tobytes()
            # ECHO PREVENTION: Only send mic to Gemini if AI is not outputting audio
            if not is_ai_speaking:
                loop.call_soon_threadsafe(audio_in_q.put_nowait, b)
                if self.record_audio:
                    self._audio_buffer.extend(b)

        # Output callback (Speaker 24kHz)
        def speaker_callback(outdata, frames, time_info, status):
            bytes_needed = frames * 2  # 16-bit mono = 2 bytes per frame
            try:
                data = audio_out_q.get_nowait()
                if len(data) >= bytes_needed:
                    chunk = data[:bytes_needed]
                    leftover = data[bytes_needed:]
                    if leftover:
                        audio_out_q.put(leftover)
                else:
                    chunk = data.ljust(bytes_needed, b"\x00")
            except queue.Empty:
                chunk = b"\x00" * bytes_needed

            outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)

        input_stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=mic_callback,
        )

        output_stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=speaker_callback,
        )

        instruction = (
            self.system_instruction
            or "You are Audiobench Voice, a knowledgeable audio assistant. Be concise, clear, and conversational."
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=instruction,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
        )

        self.console.print()
        self.console.print("  [cyan][Voice: Live Mode][/cyan] Connected to Gemini Live (Puck)")
        self.console.print("  [dim]Speak freely into microphone. Press Ctrl+C to return to text chat.[/dim]")
        self.console.print()

        try:
            input_stream.start()
            output_stream.start()

            async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
                # Task A: Send microphone audio chunks
                async def sender():
                    while not self._stop_event.is_set():
                        try:
                            chunk = await asyncio.wait_for(audio_in_q.get(), timeout=0.1)
                            await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm"))
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error("Live send error: %s", e)
                            break

                # Task B: Receive server turns, transcripts, and audio
                async def receiver():
                    nonlocal is_ai_speaking, user_turn_start, ai_turn_start
                    async for response in session.receive():
                        if self._stop_event.is_set():
                            break

                        sc = response.server_content
                        if not sc:
                            continue

                        # Check interruption
                        if sc.interrupted:
                            is_ai_speaking = False
                            with audio_out_q.mutex:
                                audio_out_q.queue.clear()
                            self.console.print("\n  [yellow][Voice] (Interrupted)[/yellow]")

                        # User input transcription (Streaming live STT)
                        if sc.input_transcription and sc.input_transcription.text:
                            if not current_user_text:
                                user_turn_start = time.perf_counter() - self._session_start_time
                            current_user_text.append(sc.input_transcription.text)
                            self.console.print(f"  [dim]You:[/dim] {''.join(current_user_text)}", end="\r")

                        # AI Spoken response audio & caption
                        if sc.model_turn:
                            if not is_ai_speaking:
                                is_ai_speaking = True
                                ai_turn_start = time.perf_counter() - self._session_start_time
                                # Commit user's turn
                                u_text = "".join(current_user_text).strip()
                                if u_text:
                                    self.console.print(f"\n  [cyan][User][/cyan] {u_text}")
                                    self.turns.append({
                                        "role": "user",
                                        "text": u_text,
                                        "start_time": user_turn_start,
                                        "end_time": time.perf_counter() - self._session_start_time,
                                    })
                                    current_user_text.clear()

                            for part in sc.model_turn.parts:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    audio_bytes = part.inline_data.data
                                    audio_out_q.put(audio_bytes)

                                    if self.record_audio:
                                        # Resample 24kHz to 16kHz for unified session recording
                                        data_24k = np.frombuffer(audio_bytes, dtype=np.int16)
                                        data_16k = self._resample_24k_to_16k(data_24k)
                                        self._audio_buffer.extend(data_16k.tobytes())

                        if sc.output_transcription and sc.output_transcription.text:
                            current_ai_text.append(sc.output_transcription.text)
                            self.console.print(f"  [dim]Assistant:[/dim] {''.join(current_ai_text)}", end="\r")

                        if sc.turn_complete:
                            is_ai_speaking = False
                            ai_end = time.perf_counter() - self._session_start_time
                            ai_text = "".join(current_ai_text).strip()
                            if ai_text:
                                self.console.print(f"\n  [green][Assistant][/green] {ai_text}")
                                self.turns.append({
                                    "role": "assistant",
                                    "text": ai_text,
                                    "start_time": ai_turn_start,
                                    "end_time": ai_end,
                                    "voice_name": "Puck",
                                })
                                current_ai_text.clear()

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())

                # Wait until interrupted
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)

                send_task.cancel()
                recv_task.cancel()

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        except Exception as e:
            logger.error("Live continuous session error: %s", e)
            self.console.print(f"\n  [red][Voice] Error in session: {e}[/red]")
        finally:
            self._stop_event.set()
            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()

        total_duration = time.perf_counter() - self._session_start_time
        wav_path = self._save_session_wav() if self.record_audio else None

        self.console.print()
        self.console.print(f"  [cyan][Voice][/cyan] Session complete: {len(self.turns)} turn(s) recorded ({total_duration:.1f}s)")
        return self.turns, wav_path, total_duration
