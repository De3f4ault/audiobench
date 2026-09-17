"""Background worker daemon for UnifiedJob sequential and bounded concurrent execution.

Monitors the unified_jobs table, dispatches tasks according to slot capacities,
reaps completed/failed tasks, and releases OS locks cleanly on shutdown.
"""

from __future__ import annotations

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
from audiobench.jobs.scheduler import (
    acquire_os_lock,
    claim_next_pending,
    get_slot_capacity,
    is_alive,
    startup_recovery,
)
from audiobench.storage.models import UnifiedJob

logger = get_logger("jobs.worker_daemon")

SLOT_ORDER = ["transcription", "network", "indexing"]


def run_daemon(max_idle_seconds: float = 3.0) -> None:
    """Main daemon loop. Runs until all queues are empty and all active jobs finish."""
    settings = get_settings()
    lock_path = Path(settings.data_dir) / "worker.lock"

    lock_file = acquire_os_lock(lock_path)
    if not lock_file:
        logger.info("Another worker daemon holds the OS lock. Exiting.")
        return

    logger.info("Worker daemon started (PID %d)", os.getpid())
    startup_recovery()

    # job_id -> (proc, log_path, events_path, deadline_timestamp)
    running_procs: dict[int, tuple[subprocess.Popen, Path, Path, float]] = {}
    processed_count = 0
    idle_start: float | None = None

    job_logs_dir = Path(settings.data_dir) / "job_logs"
    job_logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            # 1. Reap processes managed by this daemon
            finished_ids = []
            for job_id, (proc, log_path, _events_path, deadline) in running_procs.items():
                ret = proc.poll()
                if ret is not None:
                    finished_ids.append(job_id)
                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if j:
                            if ret == 0:
                                j.status = "done"
                                j.exit_code = 0
                            else:
                                j.status = "failed"
                                j.exit_code = ret
                                err = f"Exited with code {ret}"
                                if log_path.exists():
                                    try:
                                        content = log_path.read_text().strip()
                                        if content:
                                            err = content[-500:]
                                    except Exception:
                                        pass
                                j.error_summary = err
                            j.ended_at = datetime.now(UTC)
                            session.commit()
                    processed_count += 1
                    logger.info("Job #%d finished with code %d", job_id, ret)
                elif time.time() > deadline:
                    finished_ids.append(job_id)
                    logger.warning("Job #%d exceeded deadline. Terminating.", job_id)
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGTERM)
                            time.sleep(0.5)
                            if is_alive(proc.pid):
                                os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except Exception:
                        pass

                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if j:
                            j.status = "failed"
                            j.exit_code = -1
                            j.error_summary = "Timed out"
                            j.ended_at = datetime.now(UTC)
                            session.commit()

            for jid in finished_ids:
                running_procs.pop(jid, None)

            # 2. Check for any orphaned DB running jobs not in running_procs
            with get_session() as session:
                active_in_db = session.query(UnifiedJob).filter_by(status="running").all()
                for job in active_in_db:
                    if job.id not in running_procs and not is_alive(job.pid):
                        job.status = "failed"
                        job.ended_at = datetime.now(UTC)
                        job.exit_code = -1
                        job.error_summary = f"Process died unexpectedly (PID {job.pid})"
                        session.commit()

            # 3. Schedule next pending jobs for each slot
            spawned_any = False
            for slot in SLOT_ORDER:
                capacity = get_slot_capacity(slot)
                with get_session() as session:
                    running_count = (
                        session.query(UnifiedJob)
                        .filter_by(status="running", slot=slot)
                        .count()
                    )
                available = capacity - running_count

                while available > 0:
                    job = claim_next_pending(slot)
                    if not job:
                        break

                    job_id = job.id
                    log_path = job_logs_dir / f"job_{job_id}.log"
                    events_path = job_logs_dir / f"job_{job_id}.events"
                    log_path.touch()
                    events_path.touch()

                    try:
                        raw_args = json.loads(job.args_json)
                    except Exception:
                        raw_args = []

                    if raw_args and raw_args[0] == "_biometric_worker":
                        cmd = [sys.executable, "-m", "audiobench._biometric_worker", *raw_args[1:]]
                    elif raw_args and raw_args[0].startswith("audiobench."):
                        cmd = [sys.executable, "-m", raw_args[0], *raw_args[1:]]
                    elif raw_args and raw_args[0] == "audiobench":
                        cmd = [sys.executable, "-m", "audiobench", *raw_args[1:]]
                    else:
                        cmd = [sys.executable, "-m", "audiobench", *raw_args]

                    if job.job_type == "transcribe":
                        if "--job-id" not in cmd:
                            cmd.extend(["--job-id", str(job_id)])
                        if "--events-file" not in cmd:
                            cmd.extend(["--events-file", str(events_path)])

                    # Spawn detached process
                    kwargs: dict = {}
                    if os.name == "posix":
                        kwargs["start_new_session"] = True
                    else:
                        kwargs["creationflags"] = 0x00000200

                    log_out = open(log_path, "a", buffering=1)
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=log_out,
                            stderr=log_out,
                            stdin=subprocess.DEVNULL,
                            **kwargs,
                        )
                    finally:
                        log_out.close()

                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if j:
                            j.pid = proc.pid
                            j.log_path = str(log_path)
                            j.events_path = str(events_path)
                            session.commit()

                    timeout_duration = 3600.0
                    deadline = time.time() + timeout_duration
                    running_procs[job_id] = (proc, log_path, events_path, deadline)
                    available -= 1
                    spawned_any = True
                    logger.info("Spawned process %d for job #%d (%s)", proc.pid, job_id, slot)

            # 4. Termination check
            with get_session() as session:
                pending_count = session.query(UnifiedJob).filter_by(status="pending").count()

            if not running_procs and pending_count == 0:
                if idle_start is None:
                    idle_start = time.time()
                elif time.time() - idle_start >= max_idle_seconds:
                    logger.info("All queues empty and idle for %.1fs. Exiting.", max_idle_seconds)
                    break
            else:
                idle_start = None

            time.sleep(0.5)

        # Notify upon completion
        if processed_count > 0:
            try:
                if sys.platform == "darwin":
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'display notification "All queued jobs have completed." with title "AudioBench"',
                        ],
                        check=False,
                    )
                elif sys.platform == "linux":
                    subprocess.run(
                        ["notify-send", "AudioBench", "All queued jobs have completed."],
                        check=False,
                    )
            except Exception:
                pass

    finally:
        try:
            lock_file.close()
        except Exception:
            pass
        logger.info("Worker daemon stopped.")


if __name__ == "__main__":
    run_daemon()
