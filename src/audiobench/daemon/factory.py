"""Daemon client factory — handles auto-starting and fallback to local mode.

Two distinct entry points:

  ensure_daemon_running()
    Fire-and-forget. Forks the daemon process in the background and returns
    immediately. Called from the REPL at startup so the daemon is warm by
    the time the user types their first command. Never blocks, never prints.

  get_daemon_client()
    Fast-path client resolver. Checks if the socket is live and returns a
    DaemonClient if so, otherwise falls back to LocalRetrievalClient without
    attempting to start anything. The daemon should already be up (via
    ensure_daemon_running) by the time this is called in a session.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.client import DaemonClient
from audiobench.daemon.interface import RetrievalClient
from audiobench.daemon.local_client import LocalRetrievalClient

logger = get_logger("daemon.factory")

# How long get_daemon_client() will wait for the socket after a recent
# ensure_daemon_running() call. Needs to be long enough for slow
# environments to download models from HF and load them into GPU VRAM
# during cold starts.
_FAST_PATH_TIMEOUT = 180.0  # seconds


def _is_socket_alive(socket_path: Path) -> bool:
    """Check if the socket exists and a daemon is listening."""
    if not socket_path.exists():
        return False
    client = DaemonClient(socket_path)
    return client.ping()


def _clean_stale_socket(socket_path: Path) -> None:
    """Remove socket if it exists but nobody is listening and daemon is dead."""
    import fcntl
    import os

    settings = get_settings()
    pid_path = Path(settings.daemon_pid_path)
    lock_path = pid_path.with_suffix(".lock")

    # If daemon lock file is locked by a live process, DO NOT unlink the socket!
    if lock_path.exists():
        try:
            fd = os.open(str(lock_path), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (BlockingIOError, OSError):
                # Lock is actively held by running daemon!
                logger.warning(
                    "Socket %s ping timed out but daemon lock is held; not unlinking.",
                    socket_path,
                )
                return
            finally:
                os.close(fd)
        except Exception:
            pass

    # Check if PID is alive
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            logger.warning(
                "Socket %s ping timed out but daemon PID %d is alive; not unlinking.",
                socket_path,
                pid,
            )
            return
        except (ValueError, ProcessLookupError, OSError):
            pass

    try:
        socket_path.unlink()
        logger.debug("Removed stale socket at %s", socket_path)
    except Exception:
        pass


def _fork_daemon(socket_path: Path) -> bool:
    """Fork the daemon process and return immediately (non-blocking).

    The daemon loads its models in the background. The caller should
    poll the socket later (e.g. in get_daemon_client) to confirm readiness.

    Returns True if the fork succeeded, False on error.
    """
    import os
    settings = get_settings()
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "daemon_startup.log"

    try:
        env = os.environ.copy()
        # Ensure the 'src' directory is in PYTHONPATH so the -m audiobench command works
        # even if not pip-installed (e.g. in environments where sys.path is hacked).
        src_path = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"

        log_file = open(str(log_file_path), "a")
        subprocess.Popen(
            [sys.executable, "-m", "audiobench", "daemon", "start"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )
        logger.info("Daemon process forked (background), log: %s", log_file_path)
        return True
    except Exception as e:
        logger.error("Failed to fork daemon process: %s", e)
        return False



# Timestamp (monotonic) at which we last successfully forked a daemon process.
# ``None`` means we have never tried to start one this session.
_daemon_forked_at: float | None = None

# After a recent fork, poll for up to this many seconds before giving up.
_DAEMON_FORK_WAIT = _FAST_PATH_TIMEOUT  # 180 s — same value as before

# When no fork is pending, do a short grace-period to catch an already-running daemon.
_DAEMON_QUICK_WAIT = 2.0


def ensure_daemon_running() -> None:
    """Ensure the daemon is running, starting it in the background if needed.

    This is a fire-and-forget call — it returns immediately regardless of
    whether the daemon is ready yet.  Call it early (e.g. at REPL startup)
    to give the daemon maximum warm-up time before it is first needed.

    Safe to call repeatedly — no-ops if the daemon is already alive.
    """
    global _daemon_forked_at

    settings = get_settings()
    if settings.disable_memory:
        return

    socket_path = Path(settings.daemon_socket_path)

    if _is_socket_alive(socket_path):
        logger.debug("Daemon already running, nothing to do.")
        return

    if socket_path.exists():
        _clean_stale_socket(socket_path)

    logger.info("Daemon not running — forking in background...")
    if _fork_daemon(socket_path):
        _daemon_forked_at = time.monotonic()


def get_daemon_client() -> RetrievalClient:
    """Return a connected DaemonClient, or fall back to LocalRetrievalClient.

    Checks whether the daemon is alive.  If it was recently started via
    ensure_daemon_running(), waits up to _DAEMON_FORK_WAIT seconds for it
    to become ready (models usually load within 3-8 s on warm hardware).
    If no fork is pending, only waits _DAEMON_QUICK_WAIT seconds so that
    DenseStream / ColBERTStream calls don't block for 3 minutes when the
    daemon was never started.
    If it never comes up, falls back to the in-process client silently.
    """
    settings = get_settings()
    if settings.disable_memory:
        raise RuntimeError(
            "Semantic memory is disabled (AUDIOBENCH_DISABLE_MEMORY is set). "
            "Cannot get daemon client."
        )

    socket_path = Path(settings.daemon_socket_path)

    # 1. Quick check — already alive?
    if _is_socket_alive(socket_path):
        return DaemonClient(socket_path)

    if socket_path.exists():
        _clean_stale_socket(socket_path)

    # 2. Decide how long to wait:
    #    • If we forked a daemon during this session → wait the full timeout.
    #    • Otherwise → short grace period so we don't freeze for 3 minutes.
    now = time.monotonic()
    if _daemon_forked_at is not None and (now - _daemon_forked_at) < _DAEMON_FORK_WAIT:
        poll_deadline = _daemon_forked_at + _DAEMON_FORK_WAIT
        timeout_used = _DAEMON_FORK_WAIT
    else:
        poll_deadline = now + _DAEMON_QUICK_WAIT
        timeout_used = _DAEMON_QUICK_WAIT

    client = DaemonClient(socket_path)
    while time.monotonic() < poll_deadline:
        if client.ping():
            logger.info("Daemon became ready after %.1fs", time.monotonic() - now)
            return client
        time.sleep(0.25)

    # 3. Not up yet — fall back to local in-process models.
    logger.warning(
        "Daemon not ready within %.1fs — using local in-process client. "
        "Run `audiobench daemon start` for persistent background service.",
        timeout_used,
    )
    return LocalRetrievalClient()

