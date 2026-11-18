"""Process watcher. Loops to keep processes alive and logs their output."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import IO

from audiobench.core.logger_factory import get_logger
from audiobench.observatory.context import log_event
from audiobench.supervisor.backoff import ExponentialBackoff
from audiobench.supervisor.registry import upsert_process

logger = get_logger("supervisor.watcher")

# Global flag to signal shut down to all watchers
_SHUTDOWN_EVENT = threading.Event()

def _read_stream(stream: IO[bytes], name: str, level: str) -> None:
    """Read lines from a stream and emit them as events."""
    try:
        for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            log_event(
                subsystem="supervisor",
                event_type="process_output",
                message=line,
                level=level,
                process=name,
            )
            if level in ("WARN", "ERROR"):
                import sys
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            else:
                import sys
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
    except Exception:
        pass


def watch(name: str, cmd: list[str]) -> None:
    """Watch a process, restarting it if it crashes."""
    backoff = ExponentialBackoff()

    while not _SHUTDOWN_EVENT.is_set():
        log_event(
            subsystem="supervisor",
            event_type=f"{name}_started",
            message=f"Starting process '{name}'",
            level="INFO",
            process=name,
            metadata={"cmd": cmd, "restart_count": backoff.restart_count},
        )

        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        backoff.mark_started()

        upsert_process(
            name=name,
            state="running",
            pid=proc.pid,
            started_at=time.time(),
            restart_count=backoff.restart_count,
        )

        stdout_thread = threading.Thread(target=_read_stream, args=(proc.stdout, name, "INFO"), daemon=True)
        stderr_thread = threading.Thread(target=_read_stream, args=(proc.stderr, name, "WARN"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to exit
        while proc.poll() is None:
            if _SHUTDOWN_EVENT.is_set():
                # We need to stop the process
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(0.1)

        rc = proc.returncode

        # Cleanup threads
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)

        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()

        if _SHUTDOWN_EVENT.is_set():
            upsert_process(name=name, state="stopped", pid=None, stopped_at=time.time(), last_exit_code=rc)
            break

        # Process exited unexpectedly or gracefully on its own.
        # -15 (SIGTERM) or -9 (SIGKILL) usually means we stopped it.
        # But if we didn't set _SHUTDOWN_EVENT, it might be external.

        # Determine if it's an error
        level = "INFO" if rc == 0 else "ERROR"
        log_event(
            subsystem="supervisor",
            event_type="process_exited",
            message=f"Process '{name}' exited with code {rc}",
            level=level,
            process=name,
            metadata={"exit_code": rc},
        )

        upsert_process(
            name=name,
            state="stopped",
            pid=None,
            stopped_at=time.time(),
            last_exit_code=rc,
            last_error=f"Exit code {rc}" if rc != 0 else None,
        )

        if rc != 0 and rc != -15:
            is_fatal, delay = backoff.mark_failed()
            if is_fatal:
                upsert_process(name=name, state="fatal", pid=None, restart_count=backoff.restart_count)
                log_event(
                    subsystem="supervisor",
                    event_type="process_fatal",
                    message=f"Process '{name}' reached FATAL state after multiple failures.",
                    level="CRITICAL",
                    process=name,
                    metadata={"restart_count": backoff.restart_count},
                )
                break

            upsert_process(name=name, state="backoff", pid=None, restart_count=backoff.restart_count)
            # Sleep for backoff delay, checking for shutdown
            t0 = time.monotonic()
            while time.monotonic() - t0 < delay:
                if _SHUTDOWN_EVENT.is_set():
                    break
                time.sleep(0.1)
        else:
            # If it exited cleanly (0) but we aren't shutting down, we probably should restart it anyway (it's a daemon)
            # Or maybe just stop? The daemon shouldn't exit with 0 unless stopped.
            # We'll treat 0, -15 as intentional stops if they happen externally, but to be safe, if we didn't initiate it, we restart clean.
            backoff.clear()
            if rc in (0, -15):
                # If we get SIGTERM externally, don't restart.
                break

