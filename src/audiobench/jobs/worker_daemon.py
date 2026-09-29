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
    has_active_local_compute,
    has_active_transcription,
    is_alive,
    startup_recovery,
)
from audiobench.storage.models import UnifiedJob

logger = get_logger("jobs.worker_daemon")

SLOT_ORDER = [
    "transcription:local",
    "transcription:cloud",
    "transcription:align",
    "transcription",
    "network",
    "indexing",
]


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
                            if j.status == "pending":
                                logger.info("Preempted job #%d process yielded cleanly.", job_id)
                                continue
                            if ret == 0:
                                failed_msg = None
                                if log_path.exists():
                                    try:
                                        content = log_path.read_text().strip()
                                        if "✗ No files found" in content or "No supported audio/video files" in content:
                                            failed_msg = "No supported audio/video files matched input"
                                        elif "is already running for" in content or "is already pending for" in content:
                                            failed_msg = "Self-interception duplicate loop detected"
                                        elif "✗ Failed:" in content:
                                            failed_msg = "Execution failed (see log)"
                                        elif "Retrieved from cache" in content:
                                            j.error_summary = "Retrieved from cache"
                                    except Exception:
                                        pass

                                if failed_msg:
                                    j.status = "failed"
                                    j.exit_code = 1
                                    j.error_summary = failed_msg
                                else:
                                    j.status = "done"
                                    j.exit_code = 0

                                if j.job_type == "align":
                                    try:
                                        raw_args = json.loads(j.args_json)
                                        if len(raw_args) > 1:
                                            tx_id = int(raw_args[1])
                                            from audiobench.storage.models import TranscriptionRecord
                                            tx_rec = session.get(TranscriptionRecord, tx_id)
                                            if tx_rec:
                                                if failed_msg:
                                                    tx_rec.pipeline_phase = "degraded"
                                                elif tx_rec.pipeline_phase != "degraded":
                                                    tx_rec.pipeline_phase = "complete"
                                    except Exception:
                                        pass
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

                                if j.job_type == "align":
                                    try:
                                        raw_args = json.loads(j.args_json)
                                        if len(raw_args) > 1:
                                            tx_id = int(raw_args[1])
                                            from audiobench.storage.models import TranscriptionRecord
                                            tx_rec = session.get(TranscriptionRecord, tx_id)
                                            if tx_rec:
                                                tx_rec.pipeline_phase = "degraded"
                                    except Exception:
                                        pass
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

            # 2.5 Preempt running indexing processes ONLY if urgent local transcription is active
            with get_session() as session:
                has_urgent_local = (
                    session.query(UnifiedJob)
                    .filter(
                        UnifiedJob.slot.in_(["transcription", "transcription:local"]),
                        UnifiedJob.status.in_(["pending", "running"]),
                    )
                    .count()
                    > 0
                )
            if has_urgent_local:
                for jid in list(running_procs.keys()):
                    with get_session() as session:
                        j = session.get(UnifiedJob, jid)
                        if j and j.slot == "indexing":
                            proc, _, _, _ = running_procs.pop(jid)
                            logger.info("Worker daemon preempting indexing job #%d for transcription priority", jid)
                            try:
                                if os.name == "posix":
                                    os.killpg(proc.pid, signal.SIGTERM)
                                else:
                                    proc.terminate()
                                proc.wait(timeout=1.0)
                            except Exception:
                                try:
                                    if os.name == "posix":
                                        os.killpg(proc.pid, signal.SIGKILL)
                                    else:
                                        proc.kill()
                                except Exception:
                                    pass
                            j.status = "pending"
                            j.pid = None
                            j.started_at = None
                            j.ended_at = None
                            j.exit_code = None
                            j.error_summary = "Preempted by transcription workload"
                            session.commit()

            # 3. Schedule next pending jobs for each slot
            spawned_any = False
            for slot in SLOT_ORDER:
                if slot == "indexing" and has_active_local_compute():
                    continue
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
                    elif raw_args and raw_args[0] == "align":
                        worker_script = Path(__file__).parent.parent / "daemon" / "alignment_worker.py"
                        cmd = [sys.executable, str(worker_script), *raw_args[1:]]
                    elif raw_args and raw_args[0].startswith("audiobench."):
                        cmd = [sys.executable, "-m", raw_args[0], *raw_args[1:]]
                    elif raw_args and raw_args[0] == "audiobench":
                        cmd = [sys.executable, "-m", "audiobench", *raw_args[1:]]
                    else:
                        cmd = [sys.executable, "-m", "audiobench", *raw_args]

                    if job.job_type in ("transcribe", "align"):
                        if "--job-id" not in cmd and job.job_type == "transcribe":
                            cmd.extend(["--job-id", str(job_id)])
                        if "--events-file" not in cmd:
                            cmd.extend(["--events-file", str(events_path)])

                    # Spawn detached process
                    kwargs: dict = {}
                    if os.name == "posix":
                        kwargs["start_new_session"] = True
                    else:
                        kwargs["creationflags"] = 0x00000200

                    env = os.environ.copy()
                    settings = get_settings()
                    if job.slot == "indexing":
                        n_threads = str(settings.resolve_indexing_threads())
                        env["OMP_NUM_THREADS"] = n_threads
                        env["MKL_NUM_THREADS"] = n_threads
                    elif job.slot == "transcription:align":
                        n_threads = str(settings.resolve_alignment_threads())
                        env["OMP_NUM_THREADS"] = n_threads
                        env["MKL_NUM_THREADS"] = n_threads
                    elif job.slot == "transcription:local":
                        n_threads = str(settings.resolve_cpu_threads())
                        env["OMP_NUM_THREADS"] = n_threads
                        env["MKL_NUM_THREADS"] = n_threads
                    kwargs["env"] = env

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
