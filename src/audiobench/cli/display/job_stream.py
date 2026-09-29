"""Shared job event streaming and detachable execution display.

Allows foreground commands (like transcribe) and management commands (like jobs tail)
to render structured events from running jobs with detachable Ctrl+C semantics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    ERROR,
    SUCCESS,
    console,
    format_duration,
    format_elapsed,
)
from audiobench.core.db_session import get_session
from audiobench.storage.models import UnifiedJob


def render_job_event(evt: dict) -> None:
    """Format and render a structured event object from the events stream."""
    etype = evt.get("t")
    if etype == "phase":
        phase = evt.get("phase", "")
        msg = evt.get("message")
        prog = evt.get("progress")
        if msg:
            if prog is not None and isinstance(prog, (int, float)) and prog > 0:
                pct = int(prog * 100) if prog <= 1.0 else int(prog)
                console.print(f"  [{ACCENT}]*[/]  {msg} [{DIM}]({pct}%)[/]")
            else:
                console.print(f"  [{ACCENT}]*[/]  {msg}")
        else:
            console.print(f"  [{ACCENT}]*[/]  {phase}...")
    elif etype == "segment":
        text = evt.get("text", "").strip()
        s = int(evt.get("start", 0))
        e = int(evt.get("end", 0))
        ts = f"[{s // 60}:{s % 60:02d} -> {e // 60}:{e % 60:02d}]"
        console.print(f"  [{DIM}]{ts}[/]  {text}")
    elif etype == "error":
        err_msg = evt.get("message", "An error occurred during execution.")
        console.print(f"\n  [{ERROR}]✗ Failed: {err_msg}[/]")
    elif etype == "done":
        words = evt.get("words")
        if words:
            dur = evt.get("duration")
            dur_str = f" · {format_duration(dur)}" if dur else ""
            console.print(f"\n  [{SUCCESS}]✓ Completed successfully[/] [{DIM}]({words:,} words{dur_str})[/]")
        else:
            console.print(f"\n  [{SUCCESS}]✓ Completed successfully[/]")


def tail_job_events(job_id: int, events_path: Path) -> bool:
    """Stream live events from an events file until job completion.

    Returns True if completed successfully, False if detached or failed.
    """
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        status = job.status if job else "unknown"

    console.print(f"  [{ACCENT}][Streaming Job #{job_id}][/] [{DIM}]Ctrl+C to detach[/]\n")

    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as f:
            # Read existing lines
            for line in f:
                if line.strip():
                    try:
                        evt = json.loads(line)
                        render_job_event(evt)
                        if evt.get("t") == "error":
                            return False
                    except Exception:
                        pass

            if status not in ("pending", "running"):
                return status == "done"

            while True:
                line = f.readline()
                if line:
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            render_job_event(evt)
                            if evt.get("t") == "error":
                                return False
                        except Exception:
                            pass
                else:
                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if not j or j.status != "running":
                            final_status = j.status if j else "unknown"
                            if final_status == "done":
                                return True
                            elif final_status == "failed":
                                err = j.error_summary if j else "Unknown error"
                                console.print(f"\n  [{ERROR}]✗ Job #{job_id} failed: {err}[/]")
                                return False
                            elif final_status == "cancelled":
                                console.print(f"\n  [{DIM}]⊘ Job #{job_id} cancelled[/]")
                                return False
                            break
                    time.sleep(0.2)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}]Detached from Job #{job_id}. Job continues running in background.[/]")
        console.print(f"  [{DIM}]To re-attach: [bold]audiobench jobs tail {job_id}[/bold][/]\n")
        return False

    return True


def _tail_log(job_id: int, log_path: Path, status: str) -> bool:
    """Stream execution log until job leaves running state or user detaches."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                console.print(line, end="", highlight=False)
            if status != "running":
                return status == "done"
            while True:
                line = f.readline()
                if line:
                    console.print(line, end="", highlight=False)
                else:
                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if not j or j.status != "running":
                            return j.status == "done" if j else False
                    time.sleep(0.2)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}]Detached from Job #{job_id}. Job continues running in background.[/]")
        return False


def _report_terminal(job_id: int, status: str, error_summary: str | None = None) -> bool:
    """Report terminal status of an already completed or failed job."""
    if status == "done":
        console.print(f"  [{SUCCESS}]Job #{job_id} already finished successfully.[/]")
        return True
    err = error_summary or f"Status: {status}"
    console.print(f"  [{ERROR}]Job #{job_id} finished ({err}).[/]")
    return False


def _wait_for_job(job_id: int, slot: str) -> tuple[str, str | None, str | None, str | None] | None:
    """Wait for a pending job to transition to running or terminal status.

    Returns (status, events_path, log_path, error_summary) on status change,
    or None if user detached (Ctrl+C).
    """
    last_wait_msg = ""
    last_wait_time = 0.0

    try:
        while True:
            with get_session() as session:
                job = session.get(UnifiedJob, job_id)
                if not job:
                    console.print(f"  [{ERROR}]Job #{job_id} not found.[/]")
                    return None

                if job.status == "running":
                    if job.events_path and Path(job.events_path).exists():
                        return (job.status, job.events_path, job.log_path, job.error_summary)
                    time.sleep(0.2)
                    continue

                if job.status in ("done", "failed", "cancelled"):
                    return (job.status, job.events_path, job.log_path, job.error_summary)

                running_job = (
                    session.query(UnifiedJob)
                    .filter(UnifiedJob.slot == slot, UnifiedJob.status == "running")
                    .first()
                )

            now = time.time()
            if now - last_wait_time >= 2.0 or not last_wait_msg:
                last_wait_time = now
                if running_job:
                    cur_label = running_job.file_label or running_job.command_display or "?"
                    if len(cur_label) > 30:
                        cur_label = cur_label[:27] + "..."
                    cur_elapsed = format_elapsed(running_job.started_at)
                    msg = (
                        f"  [{ACCENT}][P][/] Job #{job_id} queued at priority — "
                        f"waiting for slot (running: #{running_job.id} '{cur_label}', {cur_elapsed}) "
                        f"[{DIM}]Ctrl+C to detach[/]"
                    )
                else:
                    msg = (
                        f"  [{ACCENT}][P][/] Job #{job_id} queued at priority — "
                        f"waiting for worker daemon... [{DIM}]Ctrl+C to detach[/]"
                    )

                if msg != last_wait_msg:
                    console.print(msg)
                    last_wait_msg = msg

            time.sleep(0.5)

    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}]Detached from Job #{job_id}. Job remains queued in background.[/]")
        console.print(f"  [{DIM}]To monitor: [bold]audiobench jobs tail {job_id}[/bold][/]\n")
        return None


def _execute_wait_and_stream_job(job_id: int) -> bool:
    """Internal implementation of waiting and streaming a single job."""
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        if not job:
            console.print(f"  [{ERROR}]Job #{job_id} not found.[/]")
            return False
        events_str = job.events_path
        log_str = job.log_path
        status = job.status
        slot = job.slot
        err = job.error_summary

    events_path = Path(events_str) if events_str else None
    if events_path and events_path.exists() and events_path.stat().st_size > 0:
        return tail_job_events(job_id, events_path)

    log_path = Path(log_str) if log_str else None
    if log_path and log_path.exists():
        console.print(f"  [{DIM}][No event stream found for Job #{job_id} — streaming execution log instead][/]\n")
        return _tail_log(job_id, log_path, status)

    if status in ("done", "failed", "cancelled"):
        return _report_terminal(job_id, status, err)

    wait_result = _wait_for_job(job_id, slot)
    if not wait_result:
        return False

    status, events_str, log_str, err = wait_result
    if status in ("done", "failed", "cancelled"):
        return _report_terminal(job_id, status, err)

    events_path = Path(events_str) if events_str else None
    if events_path and events_path.exists():
        return tail_job_events(job_id, events_path)

    log_path = Path(log_str) if log_str else None
    if log_path and log_path.exists():
        console.print(f"  [{DIM}][Watching Job #{job_id} output log][/] [{DIM}]Ctrl+C to detach[/]\n")
        return _tail_log(job_id, log_path, "running")

    return True


def wait_and_stream_job(job_id: int) -> bool:
    """Wait for a pending job to start, then stream its events live.

    Handles informative waiting UI and clean detachable Ctrl+C interruption.
    If an alignment successor job was chained, automatically continues streaming it.
    Returns True if job completed successfully, False if detached or failed.
    """
    ok = _execute_wait_and_stream_job(job_id)
    if ok:
        with get_session() as session:
            align_job = (
                session.query(UnifiedJob)
                .filter(
                    UnifiedJob.batch_id == f"align_{job_id}",
                    UnifiedJob.status.in_(["pending", "running"]),
                )
                .first()
            )
            align_id = align_job.id if align_job else None

        if align_id:
            console.print(f"\n  [{ACCENT}][P][/] Continuing to stream alignment for Job #{job_id} (Alignment Job #{align_id})...\n")
            return wait_and_stream_job(align_id)

    return ok


def stream_jobs_sequence(job_ids: list[int]) -> bool:
    """Stream a sequence of jobs one after another as they execute.

    If the user detaches (Ctrl+C) or a job fails, stops following the sequence.
    Returns True if all jobs completed, False if detached or failed.
    """
    for idx, jid in enumerate(job_ids, start=1):
        if len(job_ids) > 1:
            console.print(f"\n  [{BOLD} {ACCENT}]── Streaming File {idx}/{len(job_ids)} (Job #{jid}) ──[/]\n")
        ok = wait_and_stream_job(jid)
        if not ok:
            return False
    return True
