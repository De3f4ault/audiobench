"""mpv IPC controller — control mpv playback via JSON IPC socket.

Provides a Python wrapper around mpv's --input-ipc-server protocol for
programmatic control of audio playback (pause, seek, speed, position,
A-B looping, audio filters, and chapter navigation).

Architecture notes
------------------
mpv is booted once in ``--idle=yes`` mode (persistent daemon). Subsequent
track changes use ``load_file()`` rather than spawning a new process, which
eliminates socket reconnect latency and process-teardown overhead.

A background reader thread (``_reader_loop``) continuously consumes mpv's
output stream and dispatches messages to two destinations:

* **Push events** (``"event"`` key): ``property-change`` events update cached
  scalar state (``_cached_pos``, ``_cached_paused``, ``_cached_speed``,
  ``_cached_duration``).  These are read by callers *without* holding
  ``_lock`` — this is intentional.  CPython's GIL guarantees that simple
  float/bool/int attribute assignment and read are atomic.

  .. warning::
     Any future cached field that is a **non-scalar** (list, dict, or custom
     object) **MUST** be read and written under ``self._lock``.  The
     "intentionally unlocked" contract applies only to the four named scalar
     fields above.

* **Command responses** (``"request_id"`` key): placed into ``_responses``
  dict and the matching ``threading.Event`` is set so the caller can unblock.
  Both ``_responses`` and ``_response_events`` are guarded by ``_lock``.

Q1 decision (confirmed by experiment 2026-10-30):
  mpv's ``save-position-on-quit`` writes watch-later state *only* on graceful
  process exit, not on ``loadfile replace``.  In persistent-daemon mode, only
  the last track in a session would receive an mpv-managed resume position.
  AudioBench therefore manages its own position persistence in
  ``PlaybackService`` (Phase 2A).  This controller does not touch
  ``save-position-on-quit`` per-call.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from audiobench.core.logger_factory import get_logger

logger = get_logger("playback.mpv")


class MpvTimeoutError(RuntimeError):
    """Raised when mpv does not respond to an IPC command within the deadline.

    Distinct from a ``None`` mpv response (which is a legitimate value for
    properties that have no current value, such as ``time-pos`` in idle mode).

    Callers that need to distinguish "mpv said nothing" from "mpv returned
    null" should catch this explicitly::

        try:
            val = mpv.get_property("chapter-list")
        except MpvTimeoutError:
            # mpv unresponsive — different recovery than val is None
            ...

    ``set_property`` absorbs this internally and returns ``False``, since for
    write operations timeout == failure and the bool contract is preserved.
    """


class MpvController:
    """Control an mpv instance via its JSON IPC Unix socket.

    Boot once with ``start()``, then call ``load_file()`` for each track
    change.  ``quit()`` tears the process down at end of session.
    """

    def __init__(self, socket_path: str | None = None) -> None:
        self._socket_path = socket_path or f"/tmp/audiobench_mpv_{os.getpid()}.sock"
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None

        # ── Reader thread ──────────────────────────────────────────────────
        self._reader_thread: threading.Thread | None = None
        self._running: bool = False

        # Command responses — guarded by _lock
        self._responses: dict[int, dict] = {}
        self._response_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._next_request_id: int = 100  # Start above the observe_property IDs

        # Cached live state — intentionally read/written without _lock.
        # See module docstring for the CPython-atomic contract.
        self._cached_pos: float = 0.0
        self._cached_speed: float = 1.0
        self._cached_paused: bool = False
        self._cached_duration: float = 0.0

        # Callbacks invoked from the reader thread when mpv signals end-of-file.
        # Callers append; reader thread iterates without holding _lock (list
        # is only appended before playback, never modified concurrently).
        self._eof_callbacks: list[Callable[[str], None]] = []

        # Register cleanup to prevent orphaned mpv processes
        atexit.register(self._cleanup)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, *, speed: float = 1.0) -> None:
        """Spawn mpv in idle/daemon mode and connect to its IPC socket.

        mpv launches without a file argument (``--idle=yes``).  Use
        ``load_file()`` to begin playback of the first (and subsequent) tracks.
        """
        Path(self._socket_path).unlink(missing_ok=True)

        cmd = [
            "mpv",
            "--idle=yes",           # Stay alive between tracks
            "--keep-open=yes",      # Pause at EOF instead of exiting
            "--no-video",
            "--terminal=no",
            "--no-input-terminal",
            "--really-quiet",
            f"--input-ipc-server={self._socket_path}",
            f"--speed={speed}",
        ]

        logger.info("Starting mpv daemon: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for socket file to appear (up to 3 s, checking process health)
        for _ in range(150):
            if self._proc.poll() is not None:
                self._cleanup()
                raise RuntimeError(
                    f"mpv process exited prematurely (code {self._proc.returncode})"
                )
            if Path(self._socket_path).exists():
                break
            time.sleep(0.02)
        else:
            self._cleanup()
            raise RuntimeError(f"mpv IPC socket not created at {self._socket_path}")

        # Connect with retry (socket file may appear before mpv starts listening)
        connected = False
        connect_deadline = time.time() + 1.5
        while time.time() < connect_deadline:
            if self._proc.poll() is not None:
                self._cleanup()
                raise RuntimeError(
                    f"mpv exited during socket connect (code {self._proc.returncode})"
                )
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            try:
                sock.connect(self._socket_path)
                self._sock = sock
                connected = True
                break
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                sock.close()
                time.sleep(0.02)

        if not connected:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to mpv IPC socket at {self._socket_path}")

        logger.info("Connected to mpv IPC at %s", self._socket_path)

        # Start background reader thread
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="mpv-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # Subscribe to push property updates
        self._subscribe_properties()

    def load_file(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float | None = None,
    ) -> None:
        """Load a new audio file into the running mpv daemon.

        This replaces the current playlist entry without spawning a new
        process.  ``start()`` must have been called first.

        Position persistence is the caller's responsibility (``PlaybackService``
        writes a SQLite resume record before this call — see Decision 1 in the
        implementation plan).
        """
        if not self.is_running():
            raise RuntimeError("mpv daemon is not running; call start() first")

        opts: dict[str, str] = {}
        if start_pos > 0.0:
            opts["start"] = f"{start_pos:.3f}"
        if speed is not None:
            opts["speed"] = f"{speed:.3f}"

        if opts:
            # mpv 0.38+ supports per-file options: ["loadfile", url, "replace", -1, options]
            # This applies start/speed directly at demuxer initialization before playback,
            # eliminating the timing race condition of a decoupled seek command.
            self._send_command_nowait(["loadfile", file_path, "replace", -1, opts])
        else:
            self._send_command_nowait(["loadfile", file_path, "replace"])

        if speed is not None:
            self._set_property_nowait("speed", speed)

        logger.info("Loaded file: %s (start=%.1f)", file_path, start_pos)

    def quit(self) -> None:
        """Gracefully quit mpv and tear down the daemon."""
        self._running = False
        try:
            self._send_command_nowait(["quit"])
        except Exception:  # noqa: BLE001
            pass
        if self._proc:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1)
        self._cleanup()

    def terminate(self) -> None:
        """Force-terminate mpv."""
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1)
        self._cleanup()

    def _cleanup(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        Path(self._socket_path).unlink(missing_ok=True)
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                pass

    # ── Background Reader Thread ───────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Background thread: read all mpv output and dispatch messages."""
        buf = b""
        while self._running and self._sock:
            try:
                r, _, _ = select.select([self._sock], [], [], 0.05)
                if not r:
                    continue
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "event" in data:
                        self._dispatch_event(data)
                    else:
                        rid = data.get("request_id")
                        if rid is not None:
                            with self._lock:
                                self._responses[rid] = data
                                ev = self._response_events.get(rid)
                            if ev is not None:
                                ev.set()
            except OSError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Reader loop exception: %s", exc)
                break
        logger.debug("mpv reader thread exited")

    def _dispatch_event(self, event: dict) -> None:
        """Update cached state from mpv push events."""
        name = event.get("event")
        if name == "property-change":
            prop = event.get("name")
            val = event.get("data")
            if prop == "time-pos" and val is not None:
                self._cached_pos = float(val)
            elif prop == "speed" and val is not None:
                self._cached_speed = float(val)
            elif prop == "pause" and val is not None:
                self._cached_paused = bool(val)
            elif prop == "duration" and val is not None:
                self._cached_duration = float(val)
        elif name == "end-file":
            reason = event.get("reason", "eof")
            for cb in self._eof_callbacks:
                try:
                    cb(reason)
                except Exception:  # noqa: BLE001
                    pass

    def _subscribe_properties(self) -> None:
        """Subscribe to push-event updates from mpv once on connect.

        IDs 1–4 are reserved for these subscriptions.
        """
        self._send_command_nowait(["observe_property", 1, "time-pos"])
        self._send_command_nowait(["observe_property", 2, "speed"])
        self._send_command_nowait(["observe_property", 3, "pause"])
        self._send_command_nowait(["observe_property", 4, "duration"])

    # ── IPC Communication ──────────────────────────────────────────────────

    def _next_rid(self) -> int:
        """Atomically generate the next request ID."""
        with self._lock:
            rid = self._next_request_id
            self._next_request_id += 1
        return rid

    def _send_command_nowait(self, command: list) -> None:
        """Fire a command with no response tracking (best-effort)."""
        if not self._sock:
            return
        msg = json.dumps({"command": command}) + "\n"
        try:
            self._sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, OSError):
            pass

    def _set_property_nowait(self, name: str, value: object) -> None:
        """Set a property with no response tracking."""
        self._send_command_nowait(["set_property", name, value])

    def _send_command(self, command: list, timeout: float = 2.0) -> dict | None:
        """Send a JSON command and wait for the matching response.

        Uses a threading.Event keyed by request_id so the reader thread can
        unblock us without busy-polling the socket.

        Raises:
            MpvTimeoutError: mpv did not respond within ``timeout`` seconds.
                Distinct from ``None`` — which is a legitimate mpv response
                for properties that have no current value.
        """
        if not self._sock:
            return None

        rid = self._next_rid()
        ev = threading.Event()
        with self._lock:
            self._response_events[rid] = ev

        msg = json.dumps({"command": command, "request_id": rid}) + "\n"
        try:
            self._sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, OSError):
            with self._lock:
                self._response_events.pop(rid, None)
            return None

        fired = ev.wait(timeout=timeout)
        with self._lock:
            self._response_events.pop(rid, None)
            resp = self._responses.pop(rid, None)

        if not fired:
            logger.warning("MpvTimeoutError: no response to %s within %.1fs", command[0], timeout)
            raise MpvTimeoutError(
                f"mpv did not respond to {command[0]!r} within {timeout}s"
            )
        return resp

    # ── Property Access ────────────────────────────────────────────────────

    def get_property(self, name: str) -> object:
        """Get an mpv property value via synchronous IPC.

        Returns ``None`` if mpv responds with no data (e.g., ``time-pos``
        in idle mode).

        Raises:
            MpvTimeoutError: mpv did not respond within the 2s deadline.
                Callers should distinguish this from a ``None`` response,
                which is valid. Example::

                    try:
                        chapters = mpv.get_property("chapter-list")
                    except MpvTimeoutError:
                        chapters = []   # different recovery path
        """
        if not self.is_running():
            return None
        resp = self._send_command(["get_property", name])
        if resp and resp.get("error") == "success":
            return resp.get("data")
        return None

    def set_property(self, name: str, value: object) -> bool:
        """Set an mpv property value via synchronous IPC.

        Returns ``True`` on success, ``False`` on failure *or* timeout.
        For write operations, timeout == failure: the property was not set,
        which is the semantically correct bool result. ``MpvTimeoutError``
        is absorbed here — callers that need to distinguish timeout from
        mpv rejection should call ``_send_command`` directly.
        """
        if not self.is_running():
            return False
        try:
            resp = self._send_command(["set_property", name, value])
        except MpvTimeoutError:
            return False
        return resp is not None and resp.get("error") == "success"

    # ── Cached State Reads (zero-latency, from reader thread) ─────────────

    def get_position(self) -> float:
        """Current playback position in seconds (cached, zero-latency)."""
        return self._cached_pos

    def get_duration(self) -> float:
        """Total duration in seconds (cached, zero-latency)."""
        return self._cached_duration

    def get_speed(self) -> float:
        """Current playback speed (cached, zero-latency)."""
        return self._cached_speed

    def is_paused(self) -> bool:
        """True if playback is currently paused (cached, zero-latency)."""
        return self._cached_paused

    def get_playback_state(self) -> tuple[float, float, bool]:
        """Return (position_secs, speed, is_paused) from cached state.

        Zero-latency — no IPC round-trip.  Values are updated continuously
        by the background reader thread via ``observe_property`` push events.
        """
        return self._cached_pos, self._cached_speed, self._cached_paused

    # ── Playback Controls ──────────────────────────────────────────────────

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new pause state."""
        new_state = not self._cached_paused
        self.set_property("pause", new_state)
        return new_state

    def seek(self, offset: float) -> None:
        """Seek by relative offset in seconds."""
        if self.is_running():
            self._send_command_nowait(["seek", str(offset), "relative"])

    def seek_absolute(self, position: float) -> None:
        """Seek to absolute position in seconds."""
        if self.is_running():
            self._send_command_nowait(["seek", str(position), "absolute", "exact"])

    def set_speed(self, speed: float) -> None:
        """Set playback speed (clamped to 0.25–4.0)."""
        speed = max(0.25, min(speed, 4.0))
        self._set_property_nowait("speed", speed)

    def is_running(self) -> bool:
        """True if the mpv daemon process is alive."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def stop(self) -> None:
        """Clear current file and return mpv to idle state.

        Sends mpv's ``stop`` command which clears the playlist and pauses
        the engine without terminating the process.  The daemon stays alive
        for the next ``load_file()`` call.  This is the persistent-daemon
        alternative to ``quit()`` for user-facing "stop playback" actions.
        """
        if self.is_running():
            self._send_command_nowait(["stop"])
            logger.debug("mpv stopped (returned to idle)")

    # ── A-B Segment Looping (Phase 1C) ────────────────────────────────────

    def set_loop(self, start: float, end: float) -> None:
        """Engage native mpv A-B hardware loop for the given time range.

        mpv loops the segment gaplessly in C with no Python involvement.
        The loop persists across seeks but is cleared on ``load_file()``.
        """
        self._set_property_nowait("ab-loop-a", start)
        self._set_property_nowait("ab-loop-b", end)
        logger.debug("A-B loop set: %.2f → %.2f", start, end)

    def clear_loop(self) -> None:
        """Cancel the active A-B loop.

        mpv clears the loop when either boundary is set to the string
        ``"no"`` — not False, not null.
        """
        self._set_property_nowait("ab-loop-a", "no")
        self._set_property_nowait("ab-loop-b", "no")
        logger.debug("A-B loop cleared")

    # ── Live DSP Audio Filters (Phase 1D) ─────────────────────────────────

    def set_audio_filter(self, filter_str: str) -> None:
        """Set the libavfilter audio filter graph.

        An empty string clears all active filters.  Engaging a filter causes
        a ~50ms audio latency spike as libavfilter re-buffers — expected.

        Note: loudnorm and speech isolation are mutually exclusive in a single
        ``af`` string as written.  If both need to be active simultaneously,
        compose a compound filter graph before calling this method.
        """
        self._set_property_nowait("af", filter_str)
        logger.debug("Audio filter set: %r", filter_str)

    def toggle_loudnorm(self, enabled: bool) -> None:
        """Toggle EBU R128 loudness normalization.

        Uses libavfilter's ``loudnorm`` filter (built into all modern mpv).
        Target: -16 LUFS, true-peak -1.5 dBTP, LRA 11 LU.
        """
        self.set_audio_filter("lavfi=[loudnorm=I=-16:TP=-1.5:LRA=11]" if enabled else "")

    def toggle_speech_isolation(self, enabled: bool) -> None:
        """Toggle 200 Hz–3400 Hz bandpass filter for speech clarity.

        Cuts HVAC hum (below 200 Hz) and tape hiss (above 3400 Hz).
        Effective for field recordings and old audiobooks.
        """
        self.set_audio_filter("highpass=f=200,lowpass=f=3400" if enabled else "")

    # ── Chapter Navigation (Phase 1E) ─────────────────────────────────────

    def get_chapters(self) -> list[dict]:
        """Query mpv for embedded container chapters.

        Returns a list of ``{"title": str, "time": float}`` dicts ordered by
        time, or an empty list if:

        - the file has no chapter metadata (mpv returns ``[]`` or ``null``), or
        - mpv did not respond within the timeout (``MpvTimeoutError`` is caught
          here and logged as a warning — callers receive ``[]`` in both cases
          but the log distinguishes them).

        Note: mpv's ``chapter-list`` property uses ``"time"`` (not
        ``"start_time"``).  The caller is responsible for computing
        ``end_time`` from the next chapter's ``"time"`` (or file duration).
        """
        try:
            data = self.get_property("chapter-list")
        except MpvTimeoutError:
            logger.warning("get_chapters: timed out querying chapter-list — returning []")
            return []
        if not isinstance(data, list):
            return []
        return [
            {
                "title": ch.get("title") or f"Chapter {i + 1}",
                "time": float(ch.get("time", 0.0)),
            }
            for i, ch in enumerate(data)
        ]

    def set_chapter(self, index: int) -> None:
        """Jump to a chapter by its 0-based index in mpv's internal list."""
        self._set_property_nowait("chapter", index)

    # ── Context Manager ────────────────────────────────────────────────────

    def __enter__(self) -> MpvController:
        return self

    def __exit__(self, *args) -> None:
        self.terminate()
