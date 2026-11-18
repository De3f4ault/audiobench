"""Supervisor commands for managing processes."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from audiobench.observatory.context import log_event
from audiobench.supervisor.registry import get_process, upsert_process
from audiobench.supervisor.watcher import _SHUTDOWN_EVENT, watch

_watchers: dict[str, threading.Thread] = {}

def get_command_for(name: str) -> list[str]:
    """Return the shell command to run for a given managed process."""
    if name == "daemon":
        # Launch using python -c to directly call the server run method
        return [sys.executable, "-u", "-c", "from audiobench.daemon.server import run; run()"]
    elif name == "worker":
        # Launch the unified worker daemon
        return [sys.executable, "-m", "audiobench.jobs.worker_daemon"]
    raise ValueError(f"Unknown supervised process: {name}")


def start(name: str) -> None:
    """Start a supervised process in a background thread."""
    if name in _watchers and _watchers[name].is_alive():
        print(f"Process {name} is already supervised in this process.")
        return

    cmd = get_command_for(name)
    _SHUTDOWN_EVENT.clear()

    t = threading.Thread(target=watch, args=(name, cmd), name=f"watch-{name}", daemon=True)
    t.start()
    _watchers[name] = t


def stop(name: str) -> None:
    """Stop a supervised process gracefully."""
    proc_info = get_process(name)
    if not proc_info or not proc_info.get("pid"):
        print(f"Process {name} not found or no PID registered.")
        return

    pid = proc_info["pid"]

    log_event(
        subsystem="supervisor",
        event_type="stop_requested",
        message=f"Requested stop for process '{name}' (PID {pid})",
        level="INFO",
        process=name,
    )

    # Send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"Process {pid} not found.")
        upsert_process(name, state="stopped", pid=None, stopped_at=time.time())
        return
    except PermissionError:
        print(f"Permission denied to kill PID {pid}.")
        return

    # Wait up to 5s
    for _ in range(50):
        info = get_process(name)
        if not info or info.get("state") in ("stopped", "fatal") or not info.get("pid"):
            return
        # Or check OS
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            upsert_process(name, state="stopped", pid=None, stopped_at=time.time())
            return
        time.sleep(0.1)

    # Force kill
    log_event(
        subsystem="supervisor",
        event_type="force_kill",
        message=f"Process '{name}' did not exit cleanly, sending SIGKILL",
        level="WARN",
        process=name,
    )
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def restart(name: str) -> None:
    """Restart a supervised process."""
    stop(name)
    # The watcher thread will naturally restart it if _SHUTDOWN_EVENT is not set.
    # But wait, stop() sends SIGTERM.
    # If the watcher gets SIGTERM and _SHUTDOWN_EVENT is not set, it might just restart it!
    # Let's verify watcher behavior: if rc == -15 and _SHUTDOWN_EVENT is not set, watcher breaks.
    # So if we want to restart, we just call start again after watcher exits.

    # Wait for old watcher to exit if it exists
    if name in _watchers:
        _watchers[name].join(timeout=2.0)

    start(name)


def clear_fatal(name: str) -> None:
    """Clear fatal state for a supervised process."""
    upsert_process(name, state="stopped", pid=None, restart_count=0)
    log_event(
        subsystem="supervisor",
        event_type="fatal_cleared",
        message=f"Fatal state cleared for process '{name}'",
        level="INFO",
        process=name,
    )

def wait_all() -> None:
    """Block until all supervised watchers exit. Useful for CLI."""
    for t in list(_watchers.values()):
        if t.is_alive():
            try:
                # Handle signals manually if we are blocking the main thread
                while t.is_alive():
                    t.join(timeout=0.5)
            except KeyboardInterrupt:
                _SHUTDOWN_EVENT.set()
                # Stop them all
                for name in _watchers:
                    stop(name)
                break
