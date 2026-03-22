"""Live transcription session — custom pipeline.

Pipeline: sounddevice (mic) → queue → Silero VAD (speech boundary) → faster-whisper

Based on silero-vad official examples and proven patterns from research.
Chunk size: 512 samples (32ms) at 16kHz — required by Silero VAD.

Architecture:
- Audio callback thread: sounddevice captures mic → enqueues chunks
- Main thread: reads queue → runs VAD → detects speech boundaries
- Transcription thread: transcribes completed utterances asynchronously
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np
import torch

from audiobench.core.error_types import StreamingError
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import AudioBenchSettings
from audiobench.transcribe.transcription_result import Segment, Transcript

logger = get_logger("streaming.session")

# Silero VAD requires exactly 512 samples at 16kHz (32ms)
SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # 32ms — mandatory for silero-vad


class LiveSession:
    """Live mic transcription: sounddevice → silero VAD → faster-whisper.

    Transcription runs in a background thread so VAD keeps processing
    audio while Whisper works — no missed speech.
    """

    def __init__(
        self,
        settings: AudioBenchSettings,
        on_text: callable | None = None,
        on_recording_start: callable | None = None,
        on_recording_stop: callable | None = None,
        translate: bool = False,
        language: str | None = None,
    ) -> None:
        self._settings = settings
        self._on_text = on_text
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._translate = translate
        self._language = language

        self._segments: list[Segment] = []
        self._start_time: float = 0.0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ── Model loading ──────────────────────────────────────────────

    def _load_vad(self):
        """Load Silero VAD from cached torch hub."""
        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            force_reload=False,
            verbose=False,
        )
        logger.info("Silero VAD loaded")
        return model

    def _load_whisper(self):
        """Load faster-whisper model."""
        from faster_whisper import WhisperModel

        device = self._settings.resolve_device()
        compute = self._settings.resolve_compute_type()
        model = WhisperModel(
            self._settings.model_name,
            device=device,
            compute_type=compute,
            cpu_threads=4,
        )
        logger.info("Whisper loaded: %s (%s/%s)", self._settings.model_name, device, compute)
        return model

    # ── Transcription (runs in background thread) ──────────────────

    def _transcribe_worker(self, whisper, transcribe_q: queue.Queue):
        """Background thread: transcribe audio buffers as they arrive."""
        while not self._stop_event.is_set():
            try:
                audio_int16 = transcribe_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if audio_int16 is None:  # Poison pill
                break

            try:
                audio_f32 = audio_int16.astype(np.float32) / 32768.0
                task = "translate" if self._translate else "transcribe"
                lang = self._language or self._settings.language or "en"

                segments_gen, info = whisper.transcribe(
                    audio_f32,
                    language=lang,
                    task=task,
                    beam_size=self._settings.resolve_beam_size(),
                    vad_filter=False,
                    initial_prompt=("Clear English speech with proper punctuation."),
                )

                text = " ".join(s.text.strip() for s in segments_gen).strip()
                if text:
                    self._add_segment(text)

            except Exception as e:
                logger.error("Transcription error: %s", e)

    # ── Main entry point ───────────────────────────────────────────

    def run(self) -> Transcript:
        """Start live transcription. Blocks until Ctrl+C."""
        self._start_time = time.perf_counter()

        try:
            import sounddevice as sd
        except ImportError:
            raise StreamingError(
                "sounddevice not installed",
                "Install with: pip install sounddevice",
            ) from None

        # Load models (print status so user knows what's happening)
        print("  Loading models...", end="", flush=True, file=sys.stderr)
        try:
            vad = self._load_vad()
            whisper = self._load_whisper()
        except Exception as e:
            print(f" failed: {e}", file=sys.stderr)
            raise StreamingError(
                "Failed to load models",
                str(e),
            ) from e
        print(" ready!", file=sys.stderr)

        # Audio queue: mic callback → VAD thread
        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        # Transcription queue: VAD loop → transcription thread
        transcribe_q: queue.Queue[np.ndarray | None] = queue.Queue()

        def mic_callback(indata, frames, time_info, status):
            """Runs in sounddevice thread — just enqueue raw audio."""
            if status:
                logger.debug("Mic: %s", status)
            audio_q.put((indata[:, 0] * 32767).astype(np.int16).copy())

        # Start transcription worker thread
        tx_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(whisper, transcribe_q),
            daemon=True,
        )
        tx_thread.start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=mic_callback,
        )

        try:
            stream.start()
            logger.info("Mic started (%d Hz)", SAMPLE_RATE)
            self._vad_loop(vad, audio_q, transcribe_q)
        except KeyboardInterrupt:
            logger.info("Session interrupted")
        finally:
            self._stop_event.set()
            # Stop mic
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            # Stop transcription thread
            transcribe_q.put(None)
            tx_thread.join(timeout=5)

        elapsed = time.perf_counter() - self._start_time
        logger.info("Session ended: %d segments, %.1fs", len(self._segments), elapsed)
        return self._build_transcript(elapsed)

    # ── VAD loop (main thread) ─────────────────────────────────────

    def _vad_loop(self, vad, audio_q, transcribe_q):
        """Process audio through VAD. Sends speech buffers to transcription thread.

        Uses energy-based pre-filter + Silero VAD for robust detection.
        Forces transcription after max_utterance_s to avoid infinite buffering.
        """
        speech_buffer: list[np.ndarray] = []
        is_speaking = False
        silence_start = 0.0
        speech_start = 0.0

        # Tunable thresholds
        speech_threshold = 0.5
        silence_end_s = 0.4  # seconds of silence to end utterance
        min_speech_s = 0.25  # minimum speech to bother transcribing
        max_utterance_s = 4.0  # force-transcribe every N seconds (keeps output flowing)
        energy_floor = 200  # RMS below this = silence (ignore VAD)

        while not self._stop_event.is_set():
            try:
                chunk_int16 = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if len(chunk_int16) < VAD_CHUNK_SAMPLES:
                continue

            # Energy pre-filter: skip VAD for very quiet audio
            rms = np.sqrt(np.mean(chunk_int16.astype(np.float64) ** 2))
            if rms < energy_floor:
                # Treat as silence even if VAD might disagree
                if is_speaking:
                    now = time.perf_counter()
                    if silence_start == 0:
                        silence_start = now
                    if now - silence_start >= silence_end_s:
                        self._flush_speech(
                            speech_buffer,
                            speech_start,
                            now,
                            min_speech_s,
                            transcribe_q,
                            vad,
                        )
                        is_speaking = False
                continue

            # VAD inference
            chunk_f32 = torch.from_numpy(chunk_int16.astype(np.float32) / 32768.0)
            try:
                speech_prob = vad(chunk_f32, SAMPLE_RATE).item()
            except Exception:
                continue

            now = time.perf_counter()
            is_speech = speech_prob >= speech_threshold

            if is_speech:
                if not is_speaking:
                    # Speech just started
                    is_speaking = True
                    speech_start = now
                    speech_buffer.clear()
                    silence_start = 0
                    if self._on_recording_start:
                        self._on_recording_start()

                speech_buffer.append(chunk_int16)
                silence_start = 0  # Reset silence timer

                # Force-transcribe if utterance is too long
                speech_dur = now - speech_start
                if speech_dur >= max_utterance_s:
                    self._flush_speech(
                        speech_buffer,
                        speech_start,
                        now,
                        min_speech_s,
                        transcribe_q,
                        vad,
                        end_of_speech=False,  # Still speaking
                    )
                    # Stay in speaking mode for next chunk
                    speech_start = now

            elif is_speaking:
                # Was speaking, now quiet
                speech_buffer.append(chunk_int16)

                if silence_start == 0:
                    silence_start = now

                silence_dur = now - silence_start

                if silence_dur >= silence_end_s:
                    self._flush_speech(
                        speech_buffer,
                        speech_start,
                        now,
                        min_speech_s,
                        transcribe_q,
                        vad,
                        end_of_speech=True,  # Done speaking
                    )
                    is_speaking = False

    def _flush_speech(
        self, speech_buffer, speech_start, now, min_speech_s, transcribe_q, vad, end_of_speech=True
    ):
        """Send accumulated speech buffer to transcription thread."""
        speech_dur = now - speech_start

        if end_of_speech and self._on_recording_stop:
            self._on_recording_stop()

        if speech_dur >= min_speech_s and speech_buffer:
            audio = np.concatenate(speech_buffer)
            transcribe_q.put(audio)
            logger.info("Queued %.1fs audio for transcription", len(audio) / SAMPLE_RATE)

        speech_buffer.clear()
        if end_of_speech:
            vad.reset_states()

    # ── Segment management ─────────────────────────────────────────

    def _add_segment(self, text: str) -> None:
        """Thread-safe: add a transcribed segment."""
        elapsed = time.perf_counter() - self._start_time

        with self._lock:
            segment = Segment(
                id=len(self._segments),
                text=text,
                start=max(0.0, elapsed),
                end=max(0.0, elapsed),
                words=[],
            )
            self._segments.append(segment)

        logger.info("Segment %d: %s", segment.id, text[:80])

        if self._on_text:
            self._on_text(text)

    def _build_transcript(self, duration: float) -> Transcript:
        """Build final Transcript from collected segments."""
        with self._lock:
            full_text = " ".join(seg.text for seg in self._segments)
            segments = list(self._segments)

        return Transcript(
            text=full_text,
            segments=segments,
            language=self._language or self._settings.language or "auto",
            duration_seconds=round(duration, 3),
        )
