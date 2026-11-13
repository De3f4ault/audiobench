"""PlaybackService — daemon-hosted audio playback engine.

Manages a singleton MpvController for the daemon process. All AudioBench
modules (Chat, REPL, Search, Play) communicate through the daemon IPC to
control playback — one mpv process, shared across the entire session.

Design decisions:
  - mpv is booted once and kept alive for the daemon lifetime (persistent-daemon
    mode). Track changes use MpvController.load_file() rather than quit/recreate,
    eliminating socket reconnect latency (~200ms per track change).
  - Position persistence is AudioBench's own responsibility (Q1 experiment
    confirmed mpv only writes watch-later data on graceful process exit, not on
    loadfile replace). PlaybackService writes a SQLite resume bookmark:
      a) Before each load_file() call (crash-safe — survives SIGTERM)
      b) Every 30 seconds via a background timer (covers crash mid-chapter)
    The 30s timer write is unconditional — if audio is playing in DECOUPLED
    reader mode the position IS advancing, so writes remain useful.
  - load_file() (which applies per-file atomic demuxer start_pos) is called
    outside self._lock to avoid holding the lock during blocking I/O.
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
import time
from typing import Any, Callable

from audiobench.core.logger_factory import get_logger
from audiobench.playback import MpvController

logger = get_logger("playback.service")

# Daemon-specific mpv socket — separate from play --lyrics which uses pid-based paths
_DAEMON_MPV_SOCKET = "/tmp/audiobench_daemon_mpv.sock"

_shared_playback_service: PlaybackService | None = None
_shared_lock = threading.Lock()


def get_playback_service() -> PlaybackService:
    """Return the shared PlaybackService instance, creating it if needed."""
    global _shared_playback_service
    if _shared_playback_service is None:
        with _shared_lock:
            if _shared_playback_service is None:
                _shared_playback_service = PlaybackService()
    return _shared_playback_service


class PlaybackService:
    """Singleton playback engine for the AudioBench daemon.

    Owns one MpvController that persists for the daemon lifetime.
    Track changes issue loadfile replace rather than tearing down mpv.
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
        self._duration: float = 0.0
        self._interrupted_playback: dict[str, Any] | None = None

        # Position persistence timer — fires every 30s while playing
        self._pos_timer: threading.Timer | None = None
        # External EOF listeners registered via register_eof_listener().
        # Each entry is (token, cb). Persisted across mpv reboots via _ensure_mpv.
        self._pending_eof_listeners: list[tuple[object, Any]] = []

        # Loop and filter tracking
        self._loop_a: float | None = None
        self._loop_b: float | None = None
        self._filter: str = ""

        # Last EOF reason string ('eof', 'stop', 'error', etc.)
        self._last_eof_reason: str | None = None

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

    @staticmethod
    def _write_resume_position(
        transcription_id: int | None,
        position: float,
        audio_file_id: int | None,
        duration: float,
    ) -> None:
        """Write a resume bookmark to SQLite (best-effort; never raises)."""
        if not transcription_id:
            return
        try:
            from audiobench.storage.bookmark_repository import BookmarkRepository

            BookmarkRepository().save_resume_position(
                transcription_id,
                position,
                audio_file_id=audio_file_id,
                duration=duration,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not save resume position for tx #%s: %s", transcription_id, exc)

    def _maybe_seed_chapters(self, mpv: MpvController, audio_file_id: int) -> None:
        """Seed chapter metadata from mpv's embedded container chapters.

        Called in a background daemon thread after ``load_file()`` — mpv needs
        a moment to demux the container before ``chapter-list`` is populated.
        Tries immediately, then retries once after 500ms to handle the demux
        race for large files.

        Guard: skips seeding if the file already has chapters with
        ``chapter_index > 0`` in the DB.  A sole index-0 row is the ffprobe
        placeholder ghost chapter (replaceable); any row with ``chapter_index
        > 0`` means real chapter data is already present and must not be
        overwritten.
        """
        try:
            from audiobench.storage.chapter_repository import get_chapter_repo
            from audiobench.chapters.cue_parser import ChapterInfo

            repo = get_chapter_repo()

            # Guard: skip if real chapter data already exists
            existing = repo.get_chapters(audio_file_id)
            if any(ch.index > 0 for ch in existing):
                logger.debug(
                    "Chapter seeding skipped for af#%d — %d real chapters already present",
                    audio_file_id,
                    len(existing),
                )
                return

            # First attempt (mpv may still be demuxing)
            raw = mpv.get_chapters()
            if not raw:
                # Retry after 500ms — covers most demux latencies
                import time
                time.sleep(0.5)
                raw = mpv.get_chapters()

            if not raw:
                logger.debug(
                    "Chapter seeding skipped for af#%d — mpv returned no chapters", audio_file_id
                )
                return

            duration = mpv.get_duration() or 0.0
            chapters: list[ChapterInfo] = []
            for i, ch in enumerate(raw):
                start = float(ch.get("time", 0.0))
                end = float(raw[i + 1]["time"]) if i + 1 < len(raw) else duration
                chapters.append(
                    ChapterInfo(
                        index=i,
                        title=ch.get("title") or f"Chapter {i + 1}",
                        start_time=start,
                        end_time=max(start, end),
                        is_ghost=False,
                    )
                )

            repo.save_chapters(audio_file_id, chapters)
            logger.info(
                "Seeded %d chapters for af#%d from mpv container metadata",
                len(chapters),
                audio_file_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chapter seeding failed for af#%d: %s", audio_file_id, exc)

    @staticmethod
    def get_chapter_telemetry(segment_ids: list[int]) -> dict[str, dict]:
        """Map a list of segment IDs to their containing chapter's metadata.

        Used by the reader TUI and search result rendering to show which
        chapter each search hit belongs to.

        Args:
            segment_ids: List of SegmentRecord primary keys from search results.

        Returns:
            Dict mapping segment_id (as str) to a chapter dict::

                {"<segment_id>": {
                    "chapter_id": int,
                    "chapter_index": int,
                    "title": str,
                    "start_time": float,
                    "end_time": float,
                }}

            Keys are strings so the dict is safe to pass directly over both the
            in-process (local_client) and JSON IPC (daemon) paths without
            per-path conversion.  Segment IDs with no chapter association
            (chapter_id IS NULL) are absent from the returned dict.
        """
        if not segment_ids:
            return {}
        try:
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import ChapterRecord, SegmentRecord

            with get_session() as session:
                rows = (
                    session.query(
                        SegmentRecord.id,
                        ChapterRecord.id,
                        ChapterRecord.chapter_index,
                        ChapterRecord.title,
                        ChapterRecord.start_time,
                        ChapterRecord.end_time,
                    )
                    .join(ChapterRecord, SegmentRecord.chapter_id == ChapterRecord.id)
                    .filter(SegmentRecord.id.in_(segment_ids))
                    .all()
                )
                return {
                    str(seg_id): {
                        "chapter_id": ch_id,
                        "chapter_index": ch_idx,
                        "title": title,
                        "start_time": float(start),
                        "end_time": float(end),
                    }
                    for seg_id, ch_id, ch_idx, title, start, end in rows
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_chapter_telemetry failed: %s", exc)
            return {}

    def register_eof_listener(self, cb: Any) -> "Callable[[], None]":
        """Register a callback to be called when the current track ends naturally.

        The callback receives the mpv ``reason`` string (e.g. ``'eof'``).
        Only ``reason == 'eof'`` indicates a natural track end; other values
        (``'stop'``, ``'error'``) are also passed through so callers can filter.

        **Multi-subscriber**: appends to the list — the service's own ``_on_eof``
        is never displaced.

        **Lifecycle across mpv reboots**: the listener is persisted in
        ``_pending_eof_listeners`` and re-registered whenever ``_ensure_mpv``
        boots a new controller, so callers survive unexpected mpv crashes.

        Returns:
            A zero-arg callable that removes this listener.  Callers should
            invoke it in their ``on_unmount`` / teardown path::

                unregister = svc.register_eof_listener(my_cb)
                ...
                unregister()  # safe to call multiple times
        """
        token = object()  # unique identity key — survives lambda rebinding
        entry = (token, cb)

        with self._lock:
            self._pending_eof_listeners.append(entry)
            if self._mpv is not None and self._mpv.is_running():
                self._mpv._eof_callbacks.append(cb)

        def _unregister() -> None:
            with self._lock:
                self._pending_eof_listeners[:] = [
                    e for e in self._pending_eof_listeners if e[0] is not token
                ]
                if self._mpv is not None:
                    try:
                        self._mpv._eof_callbacks.remove(cb)
                    except ValueError:
                        pass  # already removed or never registered on current mpv

        return _unregister

    # ── Internal Helpers ─────────────────────────────────────

    def _ensure_mpv(self) -> MpvController:
        """Return the live MpvController, booting mpv if necessary.

        MUST be called with self._lock held.
        """
        if self._mpv is not None and self._mpv.is_running():
            return self._mpv

        if self._mpv is not None:
            # Stale handle — process died unexpectedly
            logger.warning("mpv daemon died unexpectedly; rebooting")
            self._mpv = None

        mpv = MpvController(socket_path=_DAEMON_MPV_SOCKET)
        mpv.start()
        # Wire up service EOF handler first — service's own handler must run first
        mpv._eof_callbacks.append(self._on_eof)
        # Re-register any external listeners (e.g. ChapterReaderApp) added via
        # register_eof_listener() — ensures they survive mpv reboots
        for _token, cb in self._pending_eof_listeners:
            mpv._eof_callbacks.append(cb)
        self._mpv = mpv
        logger.info("mpv daemon started (persistent mode)")
        return mpv

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
                "state": "playing",
                "position": round(self._synced_position, 2),
                "duration": 0.0,
                "speed": 1.0,
                "file": self._current_file,
                "audio_file_id": self._audio_file_id,
                "transcription_id": self._transcription_id,
                "loop_a": self._loop_a,
                "loop_b": self._loop_b,
                "filter": self._filter,
                "last_eof_reason": self._last_eof_reason,
            }

        if not self._check_mpv_alive() or self._current_file is None:
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
                "last_eof_reason": self._last_eof_reason,
            }

        pos, speed, paused = self._mpv.get_playback_state()  # type: ignore[union-attr]
        if self._duration <= 0.0:
            try:
                self._duration = float(self._mpv.get_duration() or 0.0)  # type: ignore[union-attr]
            except Exception:
                pass
        return {
            "playing": True,
            "paused": paused,
            "state": "paused" if paused else "playing",
            "position": round(pos, 2),
            "duration": round(self._duration, 2),
            "speed": round(speed, 2),
            "file": self._current_file,
            "audio_file_id": self._audio_file_id,
            "transcription_id": self._transcription_id,
            "loop_a": self._loop_a,
            "loop_b": self._loop_b,
            "filter": self._filter,
            "last_eof_reason": self._last_eof_reason,
        }

    # ── Position Persistence Timer ───────────────────────────

    def _cancel_pos_timer(self) -> None:
        """Cancel the 30s position timer if running.

        MUST be called with self._lock held.
        """
        if self._pos_timer is not None:
            self._pos_timer.cancel()
            self._pos_timer = None

    def _schedule_pos_timer(self) -> None:
        """Arm the 30s position timer.

        MUST be called with self._lock held.
        """
        self._cancel_pos_timer()
        t = threading.Timer(30.0, self._on_pos_timer_fired)
        t.daemon = True
        t.start()
        self._pos_timer = t

    def _on_pos_timer_fired(self) -> None:
        """Periodic 30s callback — write current position to SQLite."""
        with self._lock:
            if not self._check_mpv_alive() or not self._transcription_id:
                return
            pos = self._mpv.get_position()  # type: ignore[union-attr]
            dur = self._mpv.get_duration()  # type: ignore[union-attr]
            tx_id = self._transcription_id
            af_id = self._audio_file_id
            # Re-arm for next interval while still playing
            if not self._mpv.is_paused():  # type: ignore[union-attr]
                self._schedule_pos_timer()

        # DB write outside the lock
        self._write_resume_position(tx_id, pos, af_id, dur)

    # ── EOF Callback (from MpvController reader thread) ──────

    def _on_eof(self, reason: str) -> None:
        """Handle track-end event pushed from the mpv reader thread.

        Called on the reader thread — must not hold _lock for long.
        ``reason`` is mpv's end-file reason string (e.g. 'eof', 'stop').
        """
        interrupted = None
        finished_file = None

        with self._lock:
            # If playback was already stopped (current_file is None), do not allow
            # a trailing/stale EOF event from the reader thread to overwrite state or last_eof_reason.
            if self._current_file is None:
                return

            self._cancel_pos_timer()
            self._last_eof_reason = reason
            if reason not in ("eof",):
                # stop/error/redirect — not a natural completion
                return

            # Write final position (0 = track finished — save_resume_position
            # handles the "at/near end → reset to 0" logic automatically)
            finished_file = self._current_file
            tx_id = self._transcription_id
            af_id = self._audio_file_id
            dur = self._duration
            pos = self._mpv.get_position() if self._mpv else 0.0  # type: ignore[union-attr]

            interrupted = self._interrupted_playback
            self._interrupted_playback = None

            # If no interrupted track is waiting to resume, mark current track ended
            if interrupted is None:
                self._current_file = None

        if tx_id:
            self._write_resume_position(tx_id, pos, af_id, dur)

        if interrupted and finished_file:
            logger.info(
                "EOF: resuming interrupted playback %s at %.2fs",
                interrupted["file_path"],
                interrupted["start_pos"],
            )
            try:
                self.play(
                    interrupted["file_path"],
                    start_pos=interrupted["start_pos"],
                    speed=interrupted["speed"],
                    audio_file_id=interrupted.get("audio_file_id"),
                    transcription_id=interrupted.get("transcription_id"),
                    auto_resume=False,
                )
            except Exception as exc:
                logger.warning("Failed to resume interrupted playback: %s", exc)

    # ── Lifecycle & Playback Controls ─────────────────────────

    def play(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        audio_file_id: int | None = None,
        transcription_id: int | None = None,
        segments: list[dict] | None = None,
        auto_resume: bool = False,
    ) -> dict[str, Any]:
        """Start playback or seek within the current file.

        If the same file is already playing, only seeks to start_pos.
        Otherwise issues loadfile replace to the persistent mpv daemon.
        Segments can be loaded from DB by transcription_id or passed directly.
        If auto_resume is True and another track is active, stashes it to resume
        on EOF.

        Lock discipline: self._lock is held only for state reads/writes.
        All blocking I/O (DB write, load_file call) runs
        outside the lock.
        """
        start_pos = float(start_pos or 0.0)
        speed = float(speed or 1.0)
        abs_target = os.path.abspath(file_path)

        # ── Phase 1: capture current state under lock ─────────────────────
        with self._lock:
            self._external = False
            self._last_eof_reason = None
            same_file = (
                self._check_mpv_alive()
                and self._current_file == abs_target
            )

            if same_file:
                # Seek within current file — no track change, stay in lock
                self._mpv.seek_absolute(start_pos)  # type: ignore[union-attr]
                if speed != 1.0:
                    self._mpv.set_speed(speed)  # type: ignore[union-attr]
                self._audio_file_id = audio_file_id
                self._transcription_id = transcription_id
                self._load_segment_state(transcription_id, segments)
                return self._status_unlocked()

            # Track change: capture what we need, then release lock for I/O
            if auto_resume and self._check_mpv_alive() and self._current_file:
                try:
                    cur_pos, cur_spd, _ = self._mpv.get_playback_state()  # type: ignore[union-attr]
                except Exception:
                    cur_pos, cur_spd = 0.0, 1.0
                self._interrupted_playback = {
                    "file_path": self._current_file,
                    "start_pos": max(0.0, float(cur_pos)),
                    "speed": float(cur_spd),
                    "audio_file_id": self._audio_file_id,
                    "transcription_id": self._transcription_id,
                }
                logger.info(
                    "Stashed interrupted playback: %s at %.2fs",
                    self._current_file,
                    cur_pos,
                )
            elif not auto_resume:
                self._interrupted_playback = None

            # Capture state for out-of-lock writes
            prev_tx_id = self._transcription_id
            prev_af_id = self._audio_file_id
            prev_pos = self._mpv.get_position() if self._check_mpv_alive() else 0.0  # type: ignore[union-attr]
            prev_dur = self._duration

            # Cancel periodic timer — will rearm after load
            self._cancel_pos_timer()

            # Ensure mpv daemon is alive (boots if needed)
            mpv = self._ensure_mpv()

        # ── Phase 2: blocking I/O outside the lock ────────────────────────

        # Write resume position for the *previous* track before switching
        if prev_tx_id is not None and prev_pos > 0.0:
            self._write_resume_position(prev_tx_id, prev_pos, prev_af_id, prev_dur)

        try:
            mpv.load_file(abs_target, start_pos=start_pos, speed=speed)
        except Exception as exc:
            logger.error("Failed to load file %s: %s", abs_target, exc)
            raise

        # Seed chapters from mpv container metadata — background, non-blocking.
        # Only when audio_file_id is known; without it there's no DB row to write.
        if audio_file_id is not None:
            t = threading.Thread(
                target=self._maybe_seed_chapters,
                args=(mpv, audio_file_id),
                daemon=True,
                name="mpv-chapter-seed",
            )
            t.start()

        # ── Phase 3: update state under lock, rearm timer ─────────────────
        with self._lock:
            self._current_file = abs_target
            self._duration = 0.0
            self._loop_a = None
            self._loop_b = None
            self._filter = ""
            self._audio_file_id = audio_file_id
            self._transcription_id = transcription_id
            self._load_segment_state(transcription_id, segments)
            if transcription_id is not None:
                self._schedule_pos_timer()
            return self._status_unlocked()

    def _load_segment_state(
        self,
        transcription_id: int | None,
        segments: list[dict] | None,
    ) -> None:
        """Update _segments and _segment_starts.

        MUST be called with self._lock held.
        """
        if transcription_id:
            segs = self._load_segments_from_db(transcription_id)
            self._segments = segs
            self._segment_starts = [s["start"] for s in segs]
        elif segments is not None:
            self._segments = list(segments)
            self._segment_starts = [float(s.get("start", 0.0)) for s in segments]
        else:
            self._segments = []
            self._segment_starts = []

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
        pos = float(position or 0.0)
        with self._lock:
            if self._mpv and self._mpv.is_running():
                self._mpv.stop()

            self._cancel_pos_timer()
            self._external = True
            self._synced_position = max(0.0, pos)
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
        pos = float(position or 0.0)
        with self._lock:
            if self._external:
                self._synced_position = max(0.0, pos)
            elif self._check_mpv_alive():
                self._mpv.seek_absolute(max(0.0, pos))  # type: ignore[union-attr]
            return self._status_unlocked()

    def seek_relative(self, offset: float) -> dict[str, Any]:
        """Seek by relative offset in seconds."""
        off = float(offset or 0.0)
        with self._lock:
            if self._external:
                self._synced_position = max(0.0, self._synced_position + off)
            elif self._check_mpv_alive():
                self._mpv.seek(off)  # type: ignore[union-attr]
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
                # Re-arm position timer: _on_pos_timer_fired doesn't re-arm when paused,
                # so the timer goes silent. Restart it now so writes resume with playback.
                if self._transcription_id is not None:
                    self._schedule_pos_timer()
            return self._status_unlocked()

    def toggle(self) -> dict[str, Any]:
        """Toggle pause/resume."""
        with self._lock:
            if not self._external and self._check_mpv_alive():
                self._mpv.toggle_pause()  # type: ignore[union-attr]
            return self._status_unlocked()

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Set playback speed."""
        spd = float(speed or 1.0)
        with self._lock:
            if not self._external and self._check_mpv_alive():
                self._mpv.set_speed(max(0.25, min(4.0, spd)))  # type: ignore[union-attr]
            return self._status_unlocked()

    def stop(self) -> dict[str, Any]:
        """Stop playback, clear state, return mpv to idle.

        Unlike the old implementation, the mpv daemon process is NOT terminated
        — it returns to --idle state and remains ready for the next play() call.
        """
        with self._lock:
            self._cancel_pos_timer()
            self._interrupted_playback = None
            if self._mpv and self._mpv.is_running():
                self._mpv.stop()  # send 'stop' command, keep daemon alive
            self._external = False
            self._synced_position = 0.0
            self._current_file = None
            self._audio_file_id = None
            self._transcription_id = None
            self._segments = []
            self._segment_starts = []
            self._loop_a = None
            self._loop_b = None
            self._filter = ""
            self._last_eof_reason = "stop"
            return self._status_unlocked()

    def status(self) -> dict[str, Any]:
        """Return current playback state (thread-safe)."""
        with self._lock:
            return self._status_unlocked()

    # ── Loop and Filter Delegation (Phase 2C) ────────────────

    def loop_segment(self, start: float, end: float) -> dict[str, Any]:
        """Engage A-B loop between start and end seconds."""
        with self._lock:
            self._loop_a = float(start)
            self._loop_b = float(end)
            if not self._external and self._check_mpv_alive():
                self._mpv.set_loop(start, end)  # type: ignore[union-attr]
            return self._status_unlocked()

    def clear_loop(self) -> dict[str, Any]:
        """Cancel active A-B loop."""
        with self._lock:
            self._loop_a = None
            self._loop_b = None
            if not self._external and self._check_mpv_alive():
                self._mpv.clear_loop()  # type: ignore[union-attr]
            return self._status_unlocked()

    def set_filter(self, filter_str: str) -> dict[str, Any]:
        """Set live DSP audio filter (empty string to clear)."""
        with self._lock:
            self._filter = str(filter_str)
            if not self._external and self._check_mpv_alive():
                self._mpv.set_audio_filter(filter_str)  # type: ignore[union-attr]
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
