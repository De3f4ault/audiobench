"""DaemonPlaybackProxy — Duck-typed PlaybackService adapter for daemon clients.

Provides a PlaybackService-compatible interface for interactive TUIs (like ChapterReaderApp)
when attached to a running AudioBench daemon.

Key features:
- Extrapolated dead-reckoning position in memory between socket polls.
- 2 Hz status polling (poll_interval = 0.5s) to avoid socket spamming at 20 Hz.
- Immediate status sync on user action commands (seek, toggle, speed, loop, filter).
- Protocol-grounded EOF dispatching using daemon status ``last_eof_reason == 'eof'``.
- Idempotent unregister tokens for registered EOF listeners.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class DaemonPlaybackProxy:
    """Duck-typed adapter presenting PlaybackService interface over DaemonClient."""

    def __init__(self, client: Any, poll_interval: float = 0.5) -> None:
        self._client = client
        self._poll_interval = poll_interval
        self._lock = threading.Lock()

        self._last_sync_time: float = 0.0
        self._last_status: dict[str, Any] = {}
        self._eof_listeners: list[tuple[object, Callable[[str], None]]] = []
        self._eof_fired_for_reason: str | None = None

        # Initial synchronous status fetch
        self._sync_status_unlocked()

    def _sync_status_unlocked(self) -> dict[str, Any]:
        """Fetch fresh status from daemon, update timestamp, and check EOF."""
        try:
            st = self._client.playback_status()
            if not isinstance(st, dict):
                st = {}
        except Exception:
            st = self._last_status or {}

        self._last_status = st
        self._last_sync_time = time.monotonic()

        eof_reason = st.get("last_eof_reason")
        if eof_reason == "eof":
            if self._eof_fired_for_reason != "eof":
                self._eof_fired_for_reason = "eof"
                self._dispatch_eof_unlocked("eof")
        elif eof_reason == "stop":
            self._eof_fired_for_reason = "stop"
        elif st.get("playing"):
            # New or active track playing, reset EOF latch
            self._eof_fired_for_reason = None

        return st

    def _dispatch_eof_unlocked(self, reason: str) -> None:
        """Dispatch EOF event to all registered listeners."""
        for _token, cb in list(self._eof_listeners):
            try:
                cb(reason)
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        """Return current status with dead-reckoned position extrapolation."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_sync_time >= self._poll_interval:
                self._sync_status_unlocked()

            st = dict(self._last_status)
            if not st:
                return {
                    "playing": False,
                    "paused": False,
                    "state": "stopped",
                    "position": 0.0,
                    "duration": 0.0,
                    "speed": 1.0,
                    "file": None,
                    "audio_file_id": None,
                    "transcription_id": None,
                    "loop_a": None,
                    "loop_b": None,
                    "filter": "",
                    "last_eof_reason": None,
                }

            if not st.get("playing") or st.get("paused"):
                return st

            # Dead-reckoning position extrapolation
            base_pos = float(st.get("position", 0.0))
            dur = float(st.get("duration", 0.0))
            speed = float(st.get("speed", 1.0))
            elapsed = max(0.0, now - self._last_sync_time)
            extrapolated_pos = base_pos + elapsed * speed
            if dur > 0.0:
                extrapolated_pos = min(dur, extrapolated_pos)

            st["position"] = round(extrapolated_pos, 2)
            return st

    def seek(self, position: float) -> dict[str, Any]:
        """Seek to absolute position in seconds and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_seek(position)
            except Exception:
                pass
            return self._sync_status_unlocked()

    def seek_relative(self, offset: float) -> dict[str, Any]:
        """Seek by relative offset in seconds and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_seek_relative(offset)
            except Exception:
                pass
            return self._sync_status_unlocked()

    def toggle(self) -> dict[str, Any]:
        """Toggle pause/play and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_toggle()
            except Exception:
                pass
            return self._sync_status_unlocked()

    def pause(self) -> dict[str, Any]:
        """Pause playback and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_pause()
            except Exception:
                pass
            return self._sync_status_unlocked()

    def resume(self) -> dict[str, Any]:
        """Resume playback and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_resume()
            except Exception:
                pass
            return self._sync_status_unlocked()

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Set playback speed and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_speed(speed)
            except Exception:
                pass
            return self._sync_status_unlocked()

    def loop_segment(self, start: float, end: float) -> dict[str, Any]:
        """Set A-B loop and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_loop_segment(start, end)
            except Exception:
                pass
            return self._sync_status_unlocked()

    def clear_loop(self) -> dict[str, Any]:
        """Clear A-B loop and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_clear_loop()
            except Exception:
                pass
            return self._sync_status_unlocked()

    def set_filter(self, filter_str: str) -> dict[str, Any]:
        """Set DSP audio filter and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_set_filter(filter_str)
            except Exception:
                pass
            return self._sync_status_unlocked()

    def stop(self) -> dict[str, Any]:
        """Stop playback and immediately resync status."""
        with self._lock:
            try:
                self._client.playback_stop()
            except Exception:
                pass
            return self._sync_status_unlocked()

    def register_eof_listener(self, cb: Callable[[str], None]) -> Callable[[], None]:
        """Register an EOF callback. Returns an idempotent unregister function."""
        token = object()
        with self._lock:
            self._eof_listeners.append((token, cb))

        def unregister() -> None:
            with self._lock:
                self._eof_listeners[:] = [
                    entry for entry in self._eof_listeners if entry[0] is not token
                ]

        return unregister
