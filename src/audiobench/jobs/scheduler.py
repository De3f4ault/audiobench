"""Unified job scheduler and execution coordinator.

Provides a single entry point for enqueuing, running, and managing
asynchronous or sequenced tasks across AudioBench.
"""

from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.models import UnifiedJob

logger = get_logger("jobs.scheduler")


def is_alive(pid: int | None) -> bool:
    """Check if a process is alive using cross-platform os.kill."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_os_lock(lock_file_path: Path):
    """Acquire a non-blocking exclusive OS file lock.

    Returns the file object if successful, None if already locked.
    """
    try:
        lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_file_path, "w")
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except (BlockingIOError, OSError):
        try:
            f.close()
        except Exception:
            pass
        return None


def get_slot_capacity(slot: str) -> int:
    """Return maximum concurrent executions allowed for a given slot."""
    if slot == "transcription":
        settings = get_settings()
        engine = getattr(settings, "engine", "faster-whisper")
        if engine == "gemini":
            return 3  # API-based: network/rate-limit bound
        return 1  # CPU or GPU local model: strictly sequential
    return {
        "network": 3,
        "indexing": 1,
    }.get(slot, 1)


def enqueue(
    *,
    job_type: str,
    args: list[str],
    slot: str = "transcription",
    command_display: str | None = None,
    file_label: str | None = None,
    batch_id: str | None = None,
    batch_label: str | None = None,
    batch_index: int | None = None,
    batch_total: int | None = None,
    priority: int = 100,
    max_attempts: int = 1,
) -> int:
    """Insert one pending job row into unified_jobs.

    Returns the new job ID. Never starts any worker process.
    """
    cmd_disp = command_display or (" ".join(args) if args else job_type)
    with get_session() as session:
        job = UnifiedJob(
            job_type=job_type,
            command_display=cmd_disp,
            args_json=json.dumps(args),
            status="pending",
            slot=slot,
            batch_id=batch_id,
            batch_label=batch_label,
            batch_index=batch_index,
            batch_total=batch_total,
            file_label=file_label,
            attempt=1,
            max_attempts=max_attempts,
            priority=priority,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    logger.info("Enqueued %s job #%d (slot: %s)", job_type, job_id, slot)
    return job_id


def ensure_worker() -> None:
    """Ensure a background worker daemon is running (idempotent).

    If the worker OS lock is held, does nothing.
    If not held, launches a detached daemon process.
    """
    settings = get_settings()
    lock_path = Path(settings.data_dir) / "worker.lock"

    lock_file = acquire_os_lock(lock_path)
    if lock_file is None:
        # Worker is already actively running
        return

    # Lock was available; close it so the spawned daemon can acquire it
    try:
        lock_file.close()
    except Exception:
        pass

    _spawn_daemon()


def _spawn_daemon() -> None:
    """Spawn the worker daemon in the background detached from this terminal."""
    settings = get_settings()
    cmd = [sys.executable, "-m", "audiobench.jobs.worker_daemon"]

    log_dir = Path(settings.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "worker_daemon.log"

    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = 0x00000200

    with open(log_file, "a") as out:
        subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    logger.info("Spawned worker daemon in background.")


def startup_recovery() -> int:
    """Mark stale running jobs with dead PIDs as failed or reset for retry."""
    recovered = 0
    with get_session() as session:
        running_jobs = session.query(UnifiedJob).filter_by(status="running").all()
        for job in running_jobs:
            if not is_alive(job.pid):
                recovered += 1
                if job.attempt < job.max_attempts:
                    job.status = "pending"
                    job.attempt += 1
                    job.pid = None
                    job.started_at = None
                    logger.info("Recovered job #%d -> pending (attempt %d)", job.id, job.attempt)
                else:
                    job.status = "failed"
                    job.ended_at = datetime.now(UTC)
                    job.exit_code = -1
                    job.error_summary = f"Process died (PID {job.pid})"
                    logger.warning("Marked job #%d failed (dead PID %s)", job.id, job.pid)
        session.commit()
    return recovered


def claim_next_pending(slot: str) -> UnifiedJob | None:
    """Atomically claim the highest priority pending job for a given slot."""
    with get_session() as session:
        job = (
            session.query(UnifiedJob)
            .filter_by(status="pending", slot=slot)
            .order_by(UnifiedJob.priority.asc(), UnifiedJob.id.asc())
            .with_for_update(skip_locked=True)
            .first()
        )
        if not job:
            return None

        job.status = "running"
        job.started_at = datetime.now(UTC)
        job.pid = os.getpid()
        session.commit()
        session.refresh(job)
        return job


def has_any_pending() -> bool:
    """Check if any pending jobs exist across any slot."""
    with get_session() as session:
        return session.query(UnifiedJob).filter_by(status="pending").count() > 0


def drain_slot(
    slot: str = "transcription",
    *,
    timeout_per_job: int = 3600,
) -> int:
    """Process all pending jobs in this slot, sequentially in the foreground.

    Returns the number of jobs processed.
    """
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, SUCCESS, WARNING, console

    settings = get_settings()
    lock_path = Path(settings.data_dir) / "worker.lock"

    lock_file = acquire_os_lock(lock_path)
    if not lock_file:
        console.print(
            f"  [{WARNING}]A background worker daemon is already running.[/]\n"
            f"  [{DIM}]Your jobs have been queued and will be processed automatically.[/]\n"
            f"  [{DIM}]Use [bold]audiobench jobs[/bold] or [bold]audiobench jobs fg <id>[/bold] to monitor.[/]"
        )
        return 0

    processed = 0
    try:
        while True:
            job = claim_next_pending(slot)
            if not job:
                # Check if other slots have pending jobs, ensure background worker picks them up
                if has_any_pending():
                    ensure_worker()
                break

            # Print batch header if part of a batch
            if job.batch_total and job.batch_total > 1:
                with get_session() as session:
                    done_count = (
                        session.query(UnifiedJob)
                        .filter_by(batch_id=job.batch_id, status="done")
                        .count()
                    )
                idx = done_count + 1
                total = job.batch_total
                label = job.file_label or job.command_display
                console.print(f"\n  [{BOLD}][{ACCENT}── File {idx} / {total}  ({label}) ──[/]][/]\n")

            # Parse argv and format command
            try:
                raw_args = json.loads(job.args_json)
            except Exception:
                raw_args = []

            cmd: list[str]
            if raw_args and raw_args[0] == "_biometric_worker":
                cmd = [sys.executable, "-m", "audiobench._biometric_worker", *raw_args[1:]]
            elif raw_args and raw_args[0].startswith("audiobench."):
                cmd = [sys.executable, "-m", raw_args[0], *raw_args[1:]]
            elif raw_args and raw_args[0] == "audiobench":
                cmd = [sys.executable, "-m", "audiobench", *raw_args[1:]]
            else:
                cmd = [sys.executable, "-m", "audiobench", *raw_args]

            # Ensure --job-id is set for transcribe jobs
            if job.job_type == "transcribe" and "--job-id" not in cmd:
                cmd.extend(["--job-id", str(job.id)])

            job_id = job.id
            try:
                result = subprocess.run(cmd, timeout=timeout_per_job)
                with get_session() as session:
                    j = session.get(UnifiedJob, job_id)
                    if j:
                        if result.returncode == 0:
                            j.status = "done"
                            j.exit_code = 0
                        else:
                            j.status = "failed"
                            j.exit_code = result.returncode
                            j.error_summary = f"Process exited with code {result.returncode}"
                        j.ended_at = datetime.now(UTC)
                        session.commit()
                processed += 1
            except subprocess.TimeoutExpired:
                logger.error("Job #%d timed out after %ds", job_id, timeout_per_job)
                with get_session() as session:
                    j = session.get(UnifiedJob, job_id)
                    if j:
                        j.status = "failed"
                        j.exit_code = -1
                        j.error_summary = f"Timed out after {timeout_per_job}s"
                        j.ended_at = datetime.now(UTC)
                        session.commit()
            except KeyboardInterrupt:
                console.print(f"\n  [{WARNING}]Interrupted by user.[/]")
                with get_session() as session:
                    j = session.get(UnifiedJob, job_id)
                    if j:
                        j.status = "pending"
                        j.started_at = None
                        j.pid = None
                        session.commit()

                # Check if there are still pending jobs in this slot
                if has_any_pending():
                    import click

                    if click.confirm("\nSend remaining jobs to background daemon?", default=True):
                        ensure_worker()
                        console.print(f"  [{SUCCESS}]Remaining jobs sent to background worker.[/]")
                break
            finally:
                gc.collect()
    finally:
        try:
            lock_file.close()
        except Exception:
            pass

    return processed


def cancel_job(job_id: int) -> bool:
    """Cancel a pending or running job. Kills its process group if running."""
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        if not job:
            return False

        if job.status in ("done", "failed", "cancelled"):
            return True

        pid = job.pid
        if job.status == "running" and pid and is_alive(pid):
            try:
                if os.name == "posix":
                    os.killpg(pid, signal.SIGTERM)
                    time.sleep(1.0)
                    if is_alive(pid):
                        os.killpg(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        job.status = "cancelled"
        job.exit_code = 130
        job.ended_at = datetime.now(UTC)
        session.commit()
        return True


def cancel_batch(batch_id: str) -> int:
    """Cancel all pending or running jobs in a batch."""
    cancelled = 0
    with get_session() as session:
        jobs = session.query(UnifiedJob).filter_by(batch_id=batch_id).all()
        for job in jobs:
            if job.status in ("pending", "running"):
                if cancel_job(job.id):
                    cancelled += 1
    return cancelled


def cancel_all() -> int:
    """Cancel all pending and running jobs."""
    cancelled = 0
    with get_session() as session:
        jobs = session.query(UnifiedJob).filter(UnifiedJob.status.in_(["pending", "running"])).all()
        for job in jobs:
            if cancel_job(job.id):
                cancelled += 1
    return cancelled


def retry_job(job_id: int) -> int | None:
    """Reset a failed or cancelled job to pending status."""
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        if not job or job.status not in ("failed", "cancelled"):
            return None

        job.status = "pending"
        job.attempt = 1
        job.pid = None
        job.started_at = None
        job.ended_at = None
        job.exit_code = None
        job.error_summary = None
        session.commit()
        return job.id


def prune_jobs(prune_all: bool = False) -> int:
    """Delete finished job records and associated log/event files."""
    removed = 0
    with get_session() as session:
        query = session.query(UnifiedJob)
        if prune_all:
            jobs = query.all()
        else:
            jobs = query.filter(UnifiedJob.status.in_(["done", "failed", "cancelled"])).all()

        for job in jobs:
            for path_str in (job.log_path, job.events_path):
                if path_str:
                    p = Path(path_str)
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            session.delete(job)
            removed += 1
        session.commit()
    return removed


def deduplicate_files(file_paths: list[Path]) -> list[Path]:
    """Filter out files that already have pending, running, or done transcription jobs."""
    with get_session() as session:
        existing = (
            session.query(UnifiedJob.file_label)
            .filter(
                UnifiedJob.status.in_(["pending", "running", "done"]),
                UnifiedJob.job_type == "transcribe",
            )
            .all()
        )
        existing_names = {r[0] for r in existing if r[0]}

    return [f for f in file_paths if f.name not in existing_names]
