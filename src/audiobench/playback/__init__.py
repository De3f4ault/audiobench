"""mpv IPC controller — control mpv playback via JSON IPC socket.

Provides a thin Python wrapper around mpv's --input-ipc-server protocol
for programmatic control of audio playback (pause, seek, speed, position).
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from audiobench.core.logger_factory import get_logger

logger = get_logger("playback.mpv")


class MpvController:
    """Control an mpv instance via its JSON IPC Unix socket."""

    def __init__(self, socket_path: str | None = None) -> None:
        self._socket_path = socket_path or f"/tmp/audiobench_mpv_{os.getpid()}.sock"
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        # Register cleanup to prevent orphaned mpv processes
        atexit.register(self._cleanup)

    # ── Lifecycle ───────────────────────────────────────────

    def start(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        save_position: bool = True,
    ) -> None:
        """Spawn mpv and connect to its IPC socket."""
        # Clean up stale socket
        Path(self._socket_path).unlink(missing_ok=True)

        cmd = [
            "mpv",
            "--no-video",
            "--terminal=no",
            "--no-input-terminal",
            "--really-quiet",
            f"--input-ipc-server={self._socket_path}",
            f"--speed={speed}",
        ]
        if start_pos > 0:
            cmd.append(f"--start={start_pos}")
        if save_position:
            cmd.append("--save-position-on-quit")
        else:
            cmd.append("--no-resume-playback")

        cmd.append(file_path)

        logger.info("Starting mpv: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for socket to appear (mpv needs a moment)
        for _ in range(50):  # 0.5 seconds total
            if Path(self._socket_path).exists():
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"mpv IPC socket not created at {self._socket_path}")

        # Connect
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)
        self._sock.settimeout(0.1)  # 100ms — fast enough for real-time sync

        logger.info("Connected to mpv IPC at %s", self._socket_path)

    def quit(self) -> None:
        """Gracefully quit mpv."""
        try:
            self._send_command(["quit"])
        except Exception:  # noqa: BLE001
            pass
        if self._proc:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._cleanup()

    def terminate(self) -> None:
        """Force-terminate mpv."""
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._cleanup()

    def _cleanup(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        Path(self._socket_path).unlink(missing_ok=True)
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    # ── IPC Communication ──────────────────────────────────

    def _send_command(self, command: list, request_id: int = 0) -> dict | None:
        """Send a JSON command to mpv and return the response."""
        if not self._sock:
            return None

        msg = json.dumps({"command": command, "request_id": request_id}) + "\n"
        try:
            self._sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, OSError):
            return None

        return self._read_response(request_id)

    def _read_response(self, request_id: int = 0) -> dict | None:
        """Read until we get the response matching our request_id."""
        if not self._sock:
            return None

        buf = b""
        try:
            for _ in range(20):  # max iterations to find our response
                chunk = self._sock.recv(4096)
                if not chunk:
                    return None
                buf += chunk
                # Process line by line
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Skip events, find our response
                    if "event" not in data and data.get("request_id") == request_id:
                        return data
        except TimeoutError:
            # Try to parse what we have
            for raw_line in buf.split(b"\n"):
                if not raw_line.strip():
                    continue
                try:
                    data = json.loads(raw_line)
                    if "event" not in data:
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    # ── Property Access ────────────────────────────────────

    def get_property(self, name: str) -> float | str | bool | None:
        """Get an mpv property value."""
        if not self.is_running():
            return None
        resp = self._send_command(["get_property", name])
        if resp and resp.get("error") == "success":
            return resp.get("data")
        return None

    def set_property(self, name: str, value: float | str | bool) -> bool:
        """Set an mpv property value."""
        if not self.is_running():
            return False
        resp = self._send_command(["set_property", name, value])
        return resp is not None and resp.get("error") == "success"

    # ── Playback Controls ──────────────────────────────────

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        val = self.get_property("time-pos")
        return float(val) if val is not None else 0.0

    def get_duration(self) -> float:
        """Get total duration in seconds."""
        val = self.get_property("duration")
        return float(val) if val is not None else 0.0

    def get_speed(self) -> float:
        """Get current playback speed."""
        val = self.get_property("speed")
        return float(val) if val is not None else 1.0

    def is_paused(self) -> bool:
        """Check if playback is paused."""
        val = self.get_property("pause")
        return bool(val)

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new pause state."""
        paused = self.is_paused()
        self.set_property("pause", not paused)
        return not paused

    def seek(self, offset: float) -> None:
        """Seek by relative offset in seconds."""
        if self.is_running():
            self._send_command(["seek", str(offset), "relative"])

    def seek_absolute(self, position: float) -> None:
        """Seek to absolute position in seconds."""
        if self.is_running():
            self._send_command(["seek", str(position), "absolute"])

    def set_speed(self, speed: float) -> None:
        """Set playback speed (0.25 to 4.0)."""
        speed = max(0.25, min(speed, 4.0))
        self.set_property("speed", speed)

    def is_running(self) -> bool:
        """Check if mpv process is still alive."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def get_playback_state(self) -> tuple[float, float, bool]:
        """Batch-read position, speed, and pause state in one burst.

        Sends all 3 queries without waiting, then reads all responses.
        Returns (position_secs, speed, is_paused).
        """
        if not self.is_running() or not self._sock:
            return (0.0, 1.0, False)

        # Send all 3 queries in rapid succession with unique request_ids
        queries = [
            ({"command": ["get_property", "time-pos"], "request_id": 1}, "time-pos"),
            ({"command": ["get_property", "speed"], "request_id": 2}, "speed"),
            ({"command": ["get_property", "pause"], "request_id": 3}, "pause"),
        ]

        try:
            for q, _ in queries:
                msg = json.dumps(q) + "\n"
                self._sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, OSError):
            return (0.0, 1.0, False)

        # Read all responses
        results: dict[int, object] = {}
        buf = b""
        try:
            for _ in range(30):
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "event" in data:
                        continue
                    rid = data.get("request_id", 0)
                    if rid in (1, 2, 3) and data.get("error") == "success":
                        results[rid] = data.get("data")
                if len(results) >= 3:
                    break
        except TimeoutError:
            pass

        pos = float(results.get(1, 0) or 0)
        spd = float(results.get(2, 1) or 1)
        paused = bool(results.get(3, False))
        return (pos, spd, paused)

    # ── Context Manager ────────────────────────────────────

    def __enter__(self) -> MpvController:
        return self

    def __exit__(self, *args) -> None:
        self.terminate()
