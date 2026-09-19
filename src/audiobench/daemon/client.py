"""Daemon client — sends JSON commands to the running daemon over Unix socket."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType

logger = get_logger("daemon.client")

# Timeout for normal requests (seconds)
_DEFAULT_TIMEOUT = 300.0


class DaemonClient:
    """Synchronous client for the AudioBench daemon."""

    def __init__(self, socket_path: Path | None = None) -> None:
        settings = get_settings()
        self._socket_path = socket_path or Path(settings.daemon_socket_path)

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _ensure_daemon_running(self) -> None:
        """Silently spawn the daemon if it's not running, and wait for it."""
        import subprocess
        import sys
        import time

        logger.info("Spawning background daemon...")
        log_dir = Path(settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "daemon_autostart.log", "w")
        subprocess.Popen(
            [sys.executable, "-m", "audiobench", "daemon", "start"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

        # Poll until responsive
        for _ in range(120):  # Wait up to 60 seconds (120 * 0.5s)
            time.sleep(0.5)
            # Use raw ping check to avoid infinite recursion
            if self._raw_ping():
                logger.info("Background daemon is now responsive.")
                return

        raise RuntimeError("Failed to auto-start daemon: timed out waiting for ping.")

    def _raw_ping(self) -> bool:
        """Internal ping that doesn't trigger auto-start."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                sock.connect(str(self._socket_path))
                request_id = str(uuid.uuid4())
                payload = json.dumps({"cmd": "ping", "args": {}, "request_id": request_id}) + "\n"
                sock.sendall(payload.encode("utf-8"))
                raw = sock.recv(4096)
                response = json.loads(raw)
                return bool(response.get("success"))
            finally:
                sock.close()
        except Exception:
            return False

    from collections.abc import Generator
    from typing import Any

    def _stream(self, cmd: str, args: dict[str, Any], _is_retry: bool = False) -> Generator[dict[str, Any], None, None]:
        """Send one JSON request and yield JSON response frames as they arrive."""
        request_id = str(uuid.uuid4())
        payload = json.dumps({"cmd": cmd, "args": args, "request_id": request_id}) + "\n"

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_DEFAULT_TIMEOUT)
        try:
            sock.connect(str(self._socket_path))
            sock.sendall(payload.encode("utf-8"))

            buf = bytearray()
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue

                    response = json.loads(line.decode("utf-8"))
                    yield response
                    if response.get("status") in ("ok", "error") or ("success" in response and response.get("status") != "progress"):
                        return

        except (ConnectionRefusedError, FileNotFoundError):
            sock.close()
            if not _is_retry:
                logger.debug("Daemon not reachable, attempting auto-start...")
                self._ensure_daemon_running()
                yield from self._stream(cmd, args, _is_retry=True)
                return
            raise
        finally:
            sock.close()

    def _send(self, cmd: str, args: dict[str, Any], _is_retry: bool = False) -> dict[str, Any]:
        """Send one JSON request and read one JSON response.

        Raises:
            ConnectionRefusedError: if socket is absent or daemon is not running (after retry).
            TimeoutError: if the daemon doesn't respond within the timeout.
            RuntimeError: if the daemon returns an error response.
        """
        terminal_response = None
        for response in self._stream(cmd, args, _is_retry):
            if response.get("status") == "progress":
                continue
            terminal_response = response
            break

        if terminal_response is None:
            raise RuntimeError(f"Daemon error [{cmd}]: Connection closed before terminal response")

        if terminal_response.get("status") == "error" or not terminal_response.get("success", True):
            err = terminal_response.get("error", "unknown")
            if isinstance(err, dict):
                err = err.get("message", str(err))
            raise RuntimeError(f"Daemon error [{cmd}]: {err}")

        return dict(terminal_response.get("data", {}))

    # ------------------------------------------------------------------
    # RetrievalClient interface
    # ------------------------------------------------------------------

    def daemon_is_healthy(self) -> dict[str, Any] | None:
        """Ping daemon and return the full health response dict.
        
        Returns None if daemon is not running or unreachable.
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                sock.connect(str(self._socket_path))
                request_id = str(uuid.uuid4())
                payload = json.dumps({"cmd": "ping", "args": {}, "request_id": request_id}) + "\n"
                sock.sendall(payload.encode("utf-8"))

                buf = bytearray()
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if b"\n" in buf:
                        line, _ = buf.split(b"\n", 1)
                        response = json.loads(line.decode("utf-8"))
                        if response.get("status") == "ok" or response.get("success"):
                            data = response.get("data", response)
                            return dict(data)
                        break
            finally:
                sock.close()
        except Exception:
            pass
        return None

    def ping(self) -> bool:
        """Return True if the daemon is alive and responsive."""
        return bool(self.daemon_is_healthy())

    def reload_settings(self) -> None:
        """Tell the daemon to drop its cached settings and reload from disk.
        Silently returns if the daemon is not running. Warns if the daemon is running but the IPC fails.
        """
        import logging
        if not self.ping():
            return

        try:
            self._send("reload_settings", {})
        except Exception as e:
            logging.warning(f"Failed to reload daemon settings (IPC error: {e}). You may need to restart the daemon.")


    def search(
        self,
        query: str,
        top_k: int = 5,
        speaker_filter: str | None = None,
        audio_file_id: int | None = None,
        work_id: int | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search over memory store via daemon."""
        args = {
            "query": query,
            "top_k": top_k,
            "speaker_filter": speaker_filter,
            "audio_file_id": audio_file_id,
            "work_id": work_id,
            "hyde_document": hyde_document,
            "use_bm25": use_bm25,
            "use_dense": use_dense,
            "use_colbert": use_colbert,
        }
        data = self._send("search", args)
        return list(data.get("results", []))

    def embed(
        self,
        expression_id: int,
        content: str,
        source_type: SourceType,
        speaker: str | None = None,
    ) -> None:
        """Embed and persist an expression to LanceDB."""
        args: dict = {
            "expression_id": expression_id,
            "content": content,
            "source_type": source_type.value,
        }
        if speaker is not None:
            args["speaker"] = speaker
        self._send("embed", args)

    def delete(self, expression_id: int) -> None:
        """Remove an expression from LanceDB."""
        self._send("delete", {"expression_id": expression_id})

    def delete_batch(self, expression_ids: list[int]) -> None:
        """Remove a batch of expressions from LanceDB."""
        if not expression_ids:
            return
        self._send("delete_batch", {"expression_ids": [int(eid) for eid in expression_ids]})

    def status(self) -> dict:
        """Return daemon status dict."""
        return self._send("status", {})

    def chunk(
        self, text: str, audio_file_id: int, diarized: bool, segments: list[dict] | None = None
    ) -> list[ChunkResult]:
        """Chunk text via the daemon's chunking pipeline (Phase 4)."""
        args = {"text": text, "audio_file_id": audio_file_id, "diarized": diarized}
        if segments is not None:
            args["segments"] = segments
        data = self._send("chunk", args)
        return list(data.get("chunks", []))

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Rerank a list of documents against a query using the daemon's ML model."""
        data = self._send("rerank", {"query": query, "docs": docs})
        return list(data.get("scores", []))

    def operate_on(self, target: int | str, verb: str, context: dict | None = None) -> dict:
        """Execute a universal semantic operation via the daemon."""
        args = {
            "target": target,
            "verb": verb,
            "context": context or {},
        }
        data = self._send("operate", args)
        return dict(data)

    def operate_stream(self, target: int | str, verb: str, context: dict | None = None):
        """Execute a universal semantic operation and stream progress frames."""
        args = {
            "target": target,
            "verb": verb,
            "context": context or {},
        }

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60.0)
        sock.connect(str(self._socket_path))

        request_id = str(uuid.uuid4())
        payload = json.dumps({"cmd": "operate", "args": args, "request_id": request_id}) + "\n"
        sock.sendall(payload.encode("utf-8"))

        buf = bytearray()
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    response = json.loads(line.decode("utf-8"))

                    if response.get("status") == "progress":
                        yield response
                    elif response.get("status") == "ok":
                        return response.get("data")
                    elif response.get("status") == "error":
                        raise RuntimeError(f"Operation failed: {response.get('error')}")
        finally:
            sock.close()

    def pipeline_stream(
        self,
        steps: list[dict],
        timeout: float = 120.0,
    ):
        """Execute a multi-step pipeline and stream progress frames back.

        Yields every ``{"status": "progress", ...}`` frame as it arrives.
        Consumes the terminal ``{"status": "ok" | "error"}`` frame internally;
        raises RuntimeError on error.

        Args:
            steps: List of ``{verb, params}`` dicts defining the pipeline.
            timeout: Per-socket timeout in seconds (default 120 s for long runs).
        """
        args = {"steps": steps}
        for response in self._stream("pipeline", args):
            status = response.get("status")
            if status == "progress":
                yield response
            elif status == "ok":
                return
            elif status == "error":
                raise RuntimeError(
                    f"Pipeline error: {response.get('error', response)}"
                )

    def autocomplete(self, prefix: str, top_k: int = 5) -> list[dict]:
        """Fetch semantic autocomplete results via the daemon fast-path."""
        data = self._send("autocomplete", {"prefix": prefix, "top_k": top_k})
        if "error" in data:
            return []
        return data.get("results", [])

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check semantic cache via daemon."""
        data = self._send("check_cache", {"query": query, "distance_threshold": distance_threshold})
        return dict(data) if data else None

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write to semantic cache via daemon."""
        self._send("write_cache", {"query": query, "answer": answer, "hyde_document": hyde_document})

    def confirm_inference(self, expression_id: int) -> dict:
        """Confirm a system inference."""
        return self._send("confirm_inference", {"expression_id": expression_id})

    def reject_inference(self, expression_id: int) -> dict:
        """Reject a system inference."""
        return self._send("reject_inference", {"expression_id": expression_id})

    def authorize_proposal(self, expression_id: int) -> dict:
        """Authorize a daemon proposal to hot-register an operator."""
        return self._send("authorize_proposal", {"expression_id": expression_id})

    def get_inferences(self) -> list[dict]:
        """Return proposed system inferences (system_inference, drift_observation, potential_relation)."""
        result = self._send("get_inferences", {})
        return result.get("inferences", []) if result else []

    def get_proposals(self) -> list[dict]:
        """Return pending daemon proposals (proposed or deferred)."""
        result = self._send("get_proposals", {})
        return result.get("proposals", []) if result else []

    def register_process(self, name: str, pid: int | None = None) -> None:
        """Register this process in the Observatory managed_processes table."""
        args: dict = {"name": name}
        if pid is not None:
            args["pid"] = pid
        self._send("register_process", args)

    def unregister_process(self, name: str, exit_code: int = 0) -> None:
        """Mark this process stopped in the Observatory managed_processes table."""
        self._send("unregister_process", {"name": name, "exit_code": exit_code})

    def log_events(self, payloads: list[dict]) -> None:
        """Forward a batch of observability event payloads to the daemon.

        The daemon records them via its in-process ObservabilitySubscriber,
        writing to journal.db without the CLI process needing its own
        SQLite connection.

        Raises on connection errors so callers can fall back to local SQLite.
        """
        self._send("log_events", {"payloads": payloads})

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using the daemon's warm Nomic model.

        Returns a 768-dimensional float vector. Using the daemon means the
        model stays resident in memory between calls rather than cold-loading
        in the CLI process on every invocation.
        """
        data = self._send("embed_query", {"text": text})
        return list(data["vector"])

    def embed_segment(
        self,
        segment_id: int,
        text: str,
        start_time: float,
        end_time: float,
        source_file: str,
    ) -> None:
        """Embed a single audio segment and write it to the segment_vectors LanceDB table.

        The daemon generates the vector using its warm Nomic model, then writes
        the SegmentVectorNode record. Idempotent — safe to call multiple times
        for the same segment_id.
        """
        self._send(
            "embed_segment",
            {
                "segment_id": segment_id,
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
                "source_file": source_file,
            },
        )

    def optimize(self) -> dict:
        """Request LanceDB optimization from the daemon."""
        return self._send("optimize", {})

    # ------------------------------------------------------------------
    # Universal Playback interface
    # ------------------------------------------------------------------

    def playback_play(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        audio_file_id: int | None = None,
        transcription_id: int | None = None,
    ) -> dict[str, Any]:
        """Start playback or seek within the current file.

        Note: None-valued optional arguments are omitted from payload to minimize wire size.
        """
        args: dict[str, Any] = {
            "file_path": file_path,
            "start_pos": start_pos,
            "speed": speed,
        }
        if audio_file_id is not None:
            args["audio_file_id"] = audio_file_id
        if transcription_id is not None:
            args["transcription_id"] = transcription_id
        return self._send("playback_play", args)

    def playback_sync_session(
        self,
        file_path: str,
        *,
        transcription_id: int | None = None,
        position: float = 0.0,
        audio_file_id: int | None = None,
    ) -> dict[str, Any]:
        """Sync playback session metadata without spawning a daemon mpv process."""
        args: dict[str, Any] = {
            "file_path": file_path,
            "position": position,
        }
        if transcription_id is not None:
            args["transcription_id"] = transcription_id
        if audio_file_id is not None:
            args["audio_file_id"] = audio_file_id
        return self._send("playback_sync_session", args)

    def playback_seek(self, position: float) -> dict[str, Any]:
        """Seek to absolute position in seconds."""
        return self._send("playback_seek", {"position": position})

    def playback_seek_relative(self, offset: float) -> dict[str, Any]:
        """Seek by relative offset in seconds."""
        return self._send("playback_seek_relative", {"offset": offset})

    def playback_pause(self) -> dict[str, Any]:
        """Pause playback."""
        return self._send("playback_pause", {})

    def playback_resume(self) -> dict[str, Any]:
        """Resume playback."""
        return self._send("playback_resume", {})

    def playback_toggle(self) -> dict[str, Any]:
        """Toggle pause/play."""
        return self._send("playback_toggle", {})

    def playback_speed(self, speed: float) -> dict[str, Any]:
        """Set playback speed."""
        return self._send("playback_speed", {"speed": speed})

    def playback_stop(self) -> dict[str, Any]:
        """Stop playback and tear down mpv."""
        return self._send("playback_stop", {})

    def playback_status(self) -> dict[str, Any]:
        """Return current playback status."""
        return self._send("playback_status", {})

    def playback_context(
        self, position: float | None = None, window: int = 3
    ) -> dict[str, Any]:
        """Return transcript segments around current/given position."""
        args: dict[str, Any] = {"window": window}
        if position is not None:
            args["position"] = position
        return self._send("playback_context", args)
