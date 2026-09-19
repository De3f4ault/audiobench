"""PlaybackService — daemon-hosted audio playback engine.

Manages a singleton MpvController for the daemon process. All AudioBench
modules (Chat, REPL, Search, Play) communicate through the daemon IPC to
control playback — one mpv process, shared across the entire session.

Design decisions:
  - Segments are loaded from DB by transcription_id (not passed over the wire)
  - Every mutating method returns status() so the client always has current state
  - Thread-safe via a threading.Lock (the daemon handles each request in a thread)
  - mpv socket uses a daemon-specific path so it never collides with play --lyrics
  - Supports external session synchronization for modes like play --lyrics without
    launching duplicate mpv instances
"""

from __future__ import annotations

import bisect
import os
import threading
from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.playback import MpvController

logger = get_logger("playback.service")

# Daemon-specific mpv socket — separate from play --lyrics which uses pid-based paths
_DAEMON_MPV_SOCKET = "/tmp/audiobench_daemon_mpv.sock"


class PlaybackService:
    """Singleton playback engine for the AudioBench daemon.

    Owns one MpvController. State persists for the lifetime of the daemon
    process — playback survives across multiple chat/search sessions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mpv: MpvController | None = None
        self._external: bool = False
        self._synced_position: float = 0.0
        self._current_file: str | None = None
        self._audio_file_id: int | None = None
        self._transcription_id: int | None = None
        self._segments: list[dict] = []
        self._segment_starts: list[float] = []  # parallel list for bisect

    # ── Database Helpers ─────────────────────────────────────

    @staticmethod
    def _load_segments_from_db(transcription_id: int) -> list[dict]:
        """Fetch ordered segment rows from the DB for a transcription."""
        try:
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import SegmentRecord

            with get_session() as session:
                rows = (
                    session.query(SegmentRecord)
                    .filter_by(transcription_id=transcription_id)
                    .order_by(SegmentRecord.segment_index)
                    .all()
                )
                return [
                    {
                        "start": r.start_time,
                        "end": r.end_time,
                        "text": r.text,
                        "speaker": r.speaker,
                    }
                    for r in rows
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load segments for tx #%s: %s", transcription_id, exc)
            return []

    # ── Internal Helpers ─────────────────────────────────────

    def _check_mpv_alive(self) -> bool:
        """Check if mpv process is alive. If dead, clear handle.

        MUST be called with self._lock held.
        """
        if self._mpv is not None and not self._mpv.is_running():
            self._mpv = None
        return self._mpv is not None

    def _status_unlocked(self) -> dict[str, Any]:
        """Return status dict — MUST be called with self._lock held.

        Returns:
            Dictionary containing:
                playing (bool): True if playing or actively synced externally.
                paused (bool): True if paused.
                position (float): Current position in seconds.
                duration (float): Total audio duration in seconds.
                speed (float): Playback speed multiplier.
                file (str | None): Absolute path to audio file.
                audio_file_id (int | None): Associated audio file ID.
                transcription_id (int | None): Associated transcription ID.
        """
        if self._external:
            return {
                "playing": True,
                "paused": False,
                "position": round(self._synced_position, 2),
                "duration": 0.0,
                "speed": 1.0,
                "file": self._current_file,
                "audio_file_id": self._audio_file_id,
                "transcription_id": self._transcription_id,
            }

        if not self._check_mpv_alive():
            return {
                "playing": False,
                "paused": False,
                "position": 0.0,
                "duration": 0.0,
                "speed": 1.0,
                "file": None,
                "audio_file_id": None,
                "transcription_id": None,
            }

        pos, speed, paused = self._mpv.get_playback_state()  # type: ignore[union-attr]
        return {
            "playing": True,
            "paused": paused,
            "position": round(pos, 2),
            "duration": round(self._mpv.get_duration(), 2),  # type: ignore[union-attr]
            "speed": round(speed, 2),
            "file": self._current_file,
            "audio_file_id": self._audio_file_id,
            "transcription_id": self._transcription_id,
        }

    # ── Lifecycle & Playback Controls ─────────────────────────

    def play(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        audio_file_id: int | None = None,
        transcription_id: int | None = None,
    ) -> dict[str, Any]:
        """Start playback or seek within the current file.

        If the same file is already playing, only seeks to start_pos.
        Otherwise terminates the current mpv and starts a new one.
        Segments are loaded from the DB by transcription_id.
        """
        with self._lock:
            self._external = False
            same_file = (
                self._check_mpv_alive()
                and self._current_file == os.path.abspath(file_path)
            )

            if same_file:
                self._mpv.seek_absolute(start_pos)  # type: ignore[union-attr]
                if speed != 1.0:
                    self._mpv.set_speed(speed)  # type: ignore[union-attr]
            else:
                if self._mpv is not None:
                    try:
                        self._mpv.quit()
                    except Exception:  # noqa: BLE001
                        pass
                    self._mpv = None

                self._mpv = MpvController(socket_path=_DAEMON_MPV_SOCKET)
                try:
                    self._mpv.start(
                        file_path,
                        start_pos=start_pos,
                        speed=speed,
                        save_position=False,
                    )
                except Exception as exc:
                    logger.error("Failed to start mpv: %s", exc)
                    self._mpv = None
                    raise

                self._current_file = os.path.abspath(file_path)

            self._audio_file_id = audio_file_id
            self._transcription_id = transcription_id
            if transcription_id is not None:
                segs = self._load_segments_from_db(transcription_id)
                self._segments = segs
                self._segment_starts = [s["start"] for s in segs]
            else:
                self._segments = []
                self._segment_starts = []

            return self._status_unlocked()

    def sync_session(
        self,
        file_path: str,
        *,
        transcription_id: int | None = None,
        position: float = 0.0,
        audio_file_id: int | None = None,
    ) -> dict[str, Any]:
        """Register metadata from an external playback session (e.g. lyrics mode).

        Updates active file, position, and transcript segments without spawning
        a secondary daemon mpv process.
        """
        with self._lock:
            if self._mpv:
                try:
                    self._mpv.quit()
                except Exception:  # noqa: BLE001
                    pass
                self._mpv = None

            self._external = True
            self._synced_position = max(0.0, position)
            self._current_file = os.path.abspath(file_path) if file_path else None
            self._audio_file_id = audio_file_id

            if transcription_id != self._transcription_id:
                self._transcription_id = transcription_id
                if transcription_id is not None:
                    segs = self._load_segments_from_db(transcription_id)
                    self._segments = segs
                    self._segment_starts = [s["start"] for s in segs]
                else:
                    self._segments = []
                    self._segment_starts = []

            return self._status_unlocked()

    def seek(self, position: float) -> dict[str, Any]:
        """Seek to absolute position in seconds."""
        with self._lock:
            if self._external:
                self._synced_position = max(0.0, position)
            elif self._check_mpv_alive():
                self._mpv.seek_absolute(max(0.0, position))  # type: ignore[union-attr]
            return self._status_unlocked()

    def seek_relative(self, offset: float) -> dict[str, Any]:
        """Seek by relative offset in seconds."""
        with self._lock:
            if self._external:
                self._synced_position = max(0.0, self._synced_position + offset)
            elif self._check_mpv_alive():
                self._mpv.seek(offset)  # type: ignore[union-attr]
            return self._status_unlocked()

    def pause(self) -> dict[str, Any]:
        """Pause playback."""
        with self._lock:
            if not self._external and self._check_mpv_alive() and not self._mpv.is_paused():  # type: ignore[union-attr]
                self._mpv.toggle_pause()  # type: ignore[union-attr]
            return self._status_unlocked()

    def resume(self) -> dict[str, Any]:
        """Resume playback."""
        with self._lock:
            if not self._external and self._check_mpv_alive() and self._mpv.is_paused():  # type: ignore[union-attr]
                self._mpv.toggle_pause()  # type: ignore[union-attr]
            return self._status_unlocked()

    def toggle(self) -> dict[str, Any]:
        """Toggle pause/resume."""
        with self._lock:
            if not self._external and self._check_mpv_alive():
                self._mpv.toggle_pause()  # type: ignore[union-attr]
            return self._status_unlocked()

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Set playback speed."""
        with self._lock:
            if not self._external and self._check_mpv_alive():
                self._mpv.set_speed(max(0.25, min(4.0, speed)))  # type: ignore[union-attr]
            return self._status_unlocked()

    def stop(self) -> dict[str, Any]:
        """Stop playback and terminate mpv. Returns uniform status schema."""
        with self._lock:
            if self._mpv:
                try:
                    self._mpv.quit()
                except Exception:  # noqa: BLE001
                    pass
                self._mpv = None
            self._external = False
            self._synced_position = 0.0
            self._current_file = None
            self._audio_file_id = None
            self._transcription_id = None
            self._segments = []
            self._segment_starts = []
            return {
                "playing": False,
                "paused": False,
                "position": 0.0,
                "duration": 0.0,
                "speed": 1.0,
                "file": None,
                "audio_file_id": None,
                "transcription_id": None,
            }

    def status(self) -> dict[str, Any]:
        """Return current playback state (thread-safe)."""
        with self._lock:
            return self._status_unlocked()

    # ── Transcript Context ───────────────────────────────────

    def get_segment_at(self, position: float | None = None) -> dict | None:
        """Return the transcript segment active at position (or current position)."""
        with self._lock:
            if not self._segments:
                return None
            if position is None:
                if self._external:
                    position = self._synced_position
                elif self._check_mpv_alive():
                    position = self._mpv.get_position()  # type: ignore[union-attr]
            if position is None:
                return None
            idx = bisect.bisect_right(self._segment_starts, position) - 1
            if 0 <= idx < len(self._segments):
                return self._segments[idx]
            return None

    def get_context_window(
        self, position: float | None = None, window: int = 3
    ) -> dict[str, Any]:
        """Return transcript segments around target position for explanation context.

        Args:
            position: Target timestamp in seconds (defaults to current playback position).
            window: Number of segments before and after active segment (±window, up to
                2*window + 1 total segments).

        Returns:
            Dictionary containing:
                segments: List of segment dicts within the window.
                current_idx: Index of active segment inside the returned segments list.
                position: Timestamp used for centering.
                transcription_id: Active transcription ID.
        """
        with self._lock:
            if not self._segments:
                return {"segments": [], "current_idx": -1, "position": position or 0.0}

            if position is None:
                if self._external:
                    position = self._synced_position
                elif self._check_mpv_alive():
                    position = self._mpv.get_position()  # type: ignore[union-attr]
            if position is None:
                return {"segments": [], "current_idx": -1, "position": 0.0}

            active_idx = bisect.bisect_right(self._segment_starts, position) - 1
            active_idx = max(0, min(active_idx, len(self._segments) - 1))

            start_idx = max(0, active_idx - window)
            end_idx = min(len(self._segments), active_idx + window + 1)
            window_segs = self._segments[start_idx:end_idx]
            current_in_window = active_idx - start_idx

            return {
                "segments": window_segs,
                "current_idx": current_in_window,
                "position": round(position, 2),
                "transcription_id": self._transcription_id,
            }
