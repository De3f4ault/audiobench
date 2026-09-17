"""Jobs command — manage background tasks and batch transcriptions."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    error_panel,
    format_duration,
    make_table,
)
from audiobench.core.db_session import get_session
from audiobench.jobs.scheduler import (
    cancel_all,
    cancel_batch,
    cancel_job,
    ensure_worker,
    is_alive,
    prune_jobs,
    retry_job,
    startup_recovery,
)
from audiobench.storage.models import UnifiedJob


@click.group(invoke_without_command=True)
@click.option("-i", "--interactive", is_flag=True, help="Force interactive mode")
@click.pass_context
def jobs(ctx: click.Context, interactive: bool) -> None:
    """Manage background transcription and worker jobs.

    Run without arguments to enter the interactive Jobs console.
    """
    import sys
    from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS

    if not SUPPORTS_BACKGROUND_JOBS:
        console.print(f"  [{WARNING}]Background jobs are only supported on Linux/macOS.[/]")
        sys.exit(1)

    startup_recovery()

    if ctx.invoked_subcommand is None:
        if interactive or sys.stdout.isatty():
            _interactive_jobs_repl()
        else:
            _list_jobs()


@jobs.command(name="list")
def list_cmd() -> None:
    """List active and recent jobs."""
    _list_jobs()


def _format_elapsed(started_at: datetime | None, ended_at: datetime | None = None) -> str:
    """Format elapsed execution duration."""
    if not started_at:
        return "—"
    end = ended_at or datetime.now(UTC)
    if started_at.tzinfo is None and end.tzinfo is not None:
        started_at = started_at.replace(tzinfo=UTC)
    elif started_at.tzinfo is not None and end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    delta = (end - started_at).total_seconds()
    return format_duration(max(0.0, delta))


def _list_jobs() -> None:
    """Render unified view of active and recent jobs with batch progress."""
    with get_session() as session:
        active_jobs = (
            session.query(UnifiedJob)
            .filter(UnifiedJob.status.in_(["running", "pending"]))
            .order_by(UnifiedJob.priority.asc(), UnifiedJob.id.asc())
            .all()
        )
        recent_jobs = (
            session.query(UnifiedJob)
            .filter(UnifiedJob.status.in_(["done", "failed", "cancelled"]))
            .order_by(UnifiedJob.id.desc())
            .limit(15)
            .all()
        )

        counts = {
            "running": session.query(UnifiedJob).filter_by(status="running").count(),
            "pending": session.query(UnifiedJob).filter_by(status="pending").count(),
            "done": session.query(UnifiedJob).filter_by(status="done").count(),
            "failed": session.query(UnifiedJob).filter_by(status="failed").count(),
        }

    if not active_jobs and not recent_jobs:
        console.print(f"  [{DIM}]No jobs found in database.[/]")
        return

    # 1. Active Jobs
    if active_jobs:
        # Check for batches
        batches: dict[str, list[UnifiedJob]] = {}
        standalone: list[UnifiedJob] = []

        for j in active_jobs:
            if j.batch_id and j.batch_total and j.batch_total > 1:
                batches.setdefault(j.batch_id, []).append(j)
            else:
                standalone.append(j)

        for batch_id, b_jobs in batches.items():
            first = b_jobs[0]
            with get_session() as session:
                done_in_b = (
                    session.query(UnifiedJob)
                    .filter_by(batch_id=batch_id, status="done")
                    .count()
                )
            total = first.batch_total or len(b_jobs)
            label = first.batch_label or "Batch"

            # Progress bar
            filled = int((done_in_b / max(1, total)) * 10)
            bar = "█" * filled + "░" * (10 - filled)
            console.print(
                f"\n  [{BOLD}]Batch: {label}[/]  [{ACCENT}]\\[{bar}\\][/]  [{BOLD}]{done_in_b}/{total}[/]"
            )

            table = make_table(
                f"Batch {batch_id[:8]}",
                [
                    ("ID", {"width": 6, "justify": "right"}),
                    ("Status", {"width": 12}),
                    ("Item", {}),
                    ("Elapsed", {"width": 10}),
                ],
            )
            # Show running or first 3 pending
            shown = 0
            for item in b_jobs:
                if item.status == "running" or shown < 3:
                    st_disp = (
                        f"[{ACCENT}]● running[/]"
                        if item.status == "running"
                        else f"[{DIM}]○ pending[/]"
                    )
                    elapsed = _format_elapsed(item.started_at) if item.status == "running" else "—"
                    name = item.file_label or item.command_display
                    if len(name) > 40:
                        name = name[:37] + "..."
                    table.add_row(f"#{item.id}", st_disp, name, elapsed)
                    shown += 1

            console.print(table)
            rem = len(b_jobs) - shown
            if rem > 0:
                console.print(f"  [{DIM}]… and {rem} more pending in this batch[/]")

        if standalone:
            table = make_table(
                "Active Tasks",
                [
                    ("ID", {"width": 6, "justify": "right"}),
                    ("Type", {"width": 14}),
                    ("Status", {"width": 12}),
                    ("Item / Command", {}),
                    ("Elapsed", {"width": 10}),
                ],
            )
            for item in standalone:
                st_disp = (
                    f"[{ACCENT}]● running[/]"
                    if item.status == "running"
                    else f"[{DIM}]○ pending[/]"
                )
                elapsed = _format_elapsed(item.started_at) if item.status == "running" else "—"
                cmd = item.file_label or item.command_display
                if len(cmd) > 40:
                    cmd = cmd[:37] + "..."
                table.add_row(f"#{item.id}", item.job_type, st_disp, cmd, elapsed)
            console.print(table)

    # 2. Recent Jobs
    if recent_jobs:
        console.print()
        table = make_table(
            "Recent Jobs",
            [
                ("ID", {"width": 6, "justify": "right"}),
                ("Type", {"width": 14}),
                ("Status", {"width": 12}),
                ("Item / Command", {}),
                ("Duration", {"width": 10}),
            ],
        )
        for item in recent_jobs:
            if item.status == "done":
                st_disp = f"[{SUCCESS}]✓ done[/]"
            elif item.status == "failed":
                st_disp = f"[{WARNING}]✗ failed[/]"
            else:
                st_disp = f"[{DIM}]⊘ cancelled[/]"

            dur = _format_elapsed(item.started_at, item.ended_at)
            cmd = item.file_label or item.command_display
            if len(cmd) > 40:
                cmd = cmd[:37] + "..."
            table.add_row(f"#{item.id}", item.job_type, st_disp, cmd, dur)
        console.print(table)

    # 3. Summary
    console.print(
        f"\n  [{DIM}]Summary:[/] [{ACCENT}]{counts['running']} running[/] · "
        f"[{DIM}]{counts['pending']} pending[/] · "
        f"[{SUCCESS}]{counts['done']} done[/] · "
        f"[{WARNING}]{counts['failed']} failed[/]"
    )


def _status_jobs() -> None:
    """Compact running-only view — ideal for quick status checks."""
    with get_session() as session:
        running_jobs = (
            session.query(UnifiedJob)
            .filter_by(status="running")
            .order_by(UnifiedJob.id.asc())
            .all()
        )
        counts = {
            "running": session.query(UnifiedJob).filter_by(status="running").count(),
            "pending": session.query(UnifiedJob).filter_by(status="pending").count(),
            "done": session.query(UnifiedJob).filter_by(status="done").count(),
            "failed": session.query(UnifiedJob).filter_by(status="failed").count(),
        }

    if not running_jobs:
        console.print(
            f"  [{DIM}]No jobs currently running.[/]  "
            f"[{DIM}]{counts['pending']} pending · {counts['done']} done · {counts['failed']} failed[/]"
        )
        return

    table = make_table(
        "Running Now",
        [
            ("ID", {"width": 6, "justify": "right"}),
            ("Type", {"width": 14}),
            ("Item / Command", {}),
            ("Elapsed", {"width": 10}),
        ],
    )
    for item in running_jobs:
        cmd = item.file_label or item.command_display
        if len(cmd) > 40:
            cmd = cmd[:37] + "..."
        table.add_row(
            f"#{item.id}",
            item.job_type,
            cmd,
            _format_elapsed(item.started_at),
        )
    console.print(table)
    console.print(
        f"  [{ACCENT}]{counts['running']} running[/] · "
        f"[{DIM}]{counts['pending']} pending · {counts['done']} done · {counts['failed']} failed[/]"
    )


def _ls_jobs() -> None:
    """Ultra-compact one-liner: summary + name of the currently running job(s)."""
    with get_session() as session:
        running_jobs = (
            session.query(UnifiedJob)
            .filter_by(status="running")
            .order_by(UnifiedJob.id.asc())
            .all()
        )
        counts = {
            "running": session.query(UnifiedJob).filter_by(status="running").count(),
            "pending": session.query(UnifiedJob).filter_by(status="pending").count(),
            "done": session.query(UnifiedJob).filter_by(status="done").count(),
            "failed": session.query(UnifiedJob).filter_by(status="failed").count(),
        }

    summary = (
        f"[{ACCENT}]{counts['running']} running[/] · "
        f"[{DIM}]{counts['pending']} pending · {counts['done']} done · {counts['failed']} failed[/]"
    )
    console.print(f"  {summary}")

    for item in running_jobs:
        name = item.file_label or item.command_display or "?"
        elapsed = _format_elapsed(item.started_at)
        console.print(f"    [{ACCENT}]#[/][bold]{item.id}[/]  [{DIM}]{item.job_type}[/]  {name}  [{DIM}]{elapsed}[/]")


@jobs.command(name="tail")
@click.argument("job_id", type=int)
def tail(job_id: int) -> None:
    """Stream live events or logs from a job in real-time (like journalctl -f)."""
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        if not job:
            console.print(error_panel("Not Found", f"Job #{job_id} does not exist"))
            return
        events_path_str = job.events_path
        log_path_str = job.log_path
        status = job.status

    events_path = Path(events_path_str) if events_path_str else None
    has_events = events_path and events_path.exists() and events_path.stat().st_size > 0

    # If no structured events file exists or it is empty, fall back directly to streaming the log
    if not has_events:
        if log_path_str and Path(log_path_str).exists():
            console.print(f"  [{DIM}][No event stream found for Job #{job_id} — streaming execution log instead][/]\n")
            fg.callback(job_id=str(job_id))
            return
        console.print(
            error_panel("No Output Available", f"No live events or log file found for Job #{job_id}.")
        )
        return

    console.print(f"  [{ACCENT}][Tailing events for Job #{job_id}][/] [{DIM}]Ctrl+C to detach[/]\n")

    def _render_event(evt: dict) -> None:
        etype = evt.get("t")
        if etype == "phase":
            console.print(f"  [{ACCENT}]◐[/]  {evt.get('phase', '')}...")
        elif etype == "segment":
            text = evt.get("text", "").strip()
            s = int(evt.get("start", 0))
            e = int(evt.get("end", 0))
            ts = f"[{s // 60}:{s % 60:02d} → {e // 60}:{e % 60:02d}]"
            console.print(f"  [{DIM}]{ts}[/]  {text}")
        elif etype == "done":
            console.print(f"\n  [{SUCCESS}]✓ Completed successfully[/]")

    try:
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        _render_event(json.loads(line))
                    except Exception:
                        pass

            if status != "running":
                return

            while True:
                line = f.readline()
                if line:
                    if line.strip():
                        try:
                            _render_event(json.loads(line))
                        except Exception:
                            pass
                else:
                    with get_session() as session:
                        j = session.get(UnifiedJob, job_id)
                        if j and j.status != "running":
                            console.print(f"\n  [{SUCCESS if j.status == 'done' else WARNING}][Job {j.status}][/]")
                            break
                    time.sleep(0.2)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}][Detached from Job #{job_id}][/]")


@jobs.command(name="fg")
@click.argument("job_id", type=str)
def fg(job_id: str) -> None:
    """Tail raw log file of a background job."""
    try:
        jid = int(job_id.lstrip("#Q"))
    except ValueError:
        console.print(error_panel("Invalid ID", f"Could not parse job ID: {job_id}"))
        return

    with get_session() as session:
        job = session.get(UnifiedJob, jid)
        if not job:
            console.print(error_panel("Not Found", f"Job #{jid} does not exist"))
            return
        log_path_str = job.log_path
        status = job.status

    if not log_path_str or not Path(log_path_str).exists():
        console.print(error_panel("Not Found", f"Log file not found for Job #{jid}"))
        return

    log_path = Path(log_path_str)
    console.print(f"  [{ACCENT}][Watching Job #{jid} logs][/] [{DIM}]Ctrl+C to detach[/]\n")

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                console.print(line, end="", highlight=False)

            if status != "running":
                return

            while True:
                line = f.readline()
                if line:
                    console.print(line, end="", highlight=False)
                else:
                    with get_session() as session:
                        j = session.get(UnifiedJob, jid)
                        if j and j.status != "running":
                            console.print(
                                f"\n  [{SUCCESS if j.status == 'done' else WARNING}][Job {j.status}][/]"
                            )
                            break
                    time.sleep(0.2)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}][Detached from Job #{jid}][/]")


@jobs.command(name="cancel")
@click.argument("job_id", type=int, required=False)
@click.option("--batch", "batch_id", type=str, default=None, help="Cancel all jobs in a batch")
@click.option("--all", "all_jobs", is_flag=True, help="Cancel all pending and running jobs")
def cancel_cmd(job_id: int | None, batch_id: str | None, all_jobs: bool) -> None:
    """Cancel a running or pending job."""
    if all_jobs:
        count = cancel_all()
        console.print(f"  [{SUCCESS}]Cancelled {count} job(s).[/]")
        return

    if batch_id:
        count = cancel_batch(batch_id)
        console.print(f"  [{SUCCESS}]Cancelled {count} job(s) in batch {batch_id}.[/]")
        return

    if job_id is None:
        console.print(error_panel("Missing Argument", "Specify a job ID, --batch <id>, or --all"))
        return

    success = cancel_job(job_id)
    if success:
        console.print(f"  [{SUCCESS}]Job #{job_id} cancelled.[/]")
    else:
        console.print(error_panel("Not Found", f"Job #{job_id} not found."))


@jobs.command(name="retry")
@click.argument("job_id", type=int)
def retry_cmd(job_id: int) -> None:
    """Re-enqueue a failed or cancelled job."""
    jid = retry_job(job_id)
    if jid:
        ensure_worker()
        console.print(f"  [{SUCCESS}]Job #{job_id} re-enqueued as pending.[/]")
    else:
        console.print(
            error_panel("Cannot Retry", f"Job #{job_id} is not in a failed or cancelled state.")
        )


@jobs.command(name="prune")
@click.option("--all", "prune_all", is_flag=True, help="Also remove running jobs (dangerous)")
def prune_cmd(prune_all: bool) -> None:
    """Delete completed, failed, and cancelled jobs."""
    count = prune_jobs(prune_all=prune_all)
    if count > 0:
        console.print(f"  [{SUCCESS}]Pruned {count} job(s).[/]")
    else:
        console.print(f"  [{DIM}]Nothing to prune.[/]")


# ── Interactive Jobs REPL ───────────────────────────────────────────────────


def _interactive_jobs_repl() -> None:
    """Launch the interactive Jobs REPL console."""
    from prompt_toolkit.formatted_text import ANSI
    from audiobench.cli.shared.repl_shell import make_prompt_session
    from audiobench.core.settings import get_settings
    from audiobench.jobs.jobs_completer import JobsAutoSuggest, JobsCompleter
    from audiobench.jobs.jobs_meta import render_jobs_help

    # Header banner
    console.print()
    console.print(
        f"  [{BOLD}]AudioBench Jobs[/]  [{DIM}]· Type [bold]/help[/bold] for commands or [bold]/exit[/bold] to quit[/]"
    )
    _list_jobs()

    settings = get_settings()
    hist_path = settings.data_dir / "jobs_history.txt"

    completer = JobsCompleter()
    auto_suggest = JobsAutoSuggest()
    pt_session = make_prompt_session(
        history_path=hist_path,
        slash_completer=completer,
        auto_suggest=auto_suggest,
    )

    prompt_str = ANSI("\033[38;5;48mjobs >>> \033[0m")


    while True:
        try:
            user_input = pt_session.prompt(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{DIM}]Goodbye![/]\n")
            break

        if not user_input:
            continue

        raw_cmd = user_input.split(None, 1)[0].lower()
        args_str = user_input[len(raw_cmd):].strip()

        # Handle slash commands or direct commands without slash
        cmd = raw_cmd if raw_cmd.startswith("/") else f"/{raw_cmd}"

        if cmd in ("/exit", "/quit", "/q"):
            console.print(f"  [{DIM}]Goodbye![/]\n")
            break

        elif cmd in ("/help", "/?"):
            render_jobs_help()

        elif cmd in ("/clear", "/cls"):
            console.clear()
            _list_jobs()

        elif cmd == "/list":
            # Full view: Active Tasks table + Recent Jobs table + summary
            console.print()
            _list_jobs()

        elif cmd == "/status":
            # Compact: running jobs only + one-line summary — quick health check
            console.print()
            _status_jobs()

        elif cmd == "/refresh":
            # Full repaint with timestamp — useful when polling
            import datetime as _dt
            now = _dt.datetime.now().strftime("%H:%M:%S")
            console.print(f"\n  [{DIM}]Refreshed at[/] [bold]{now}[/]")
            _list_jobs()

        elif cmd == "/ls":
            # Ultra-compact: one-line summary + currently running filenames only
            console.print()
            _ls_jobs()

        elif cmd in ("/tail", "/follow"):
            if not args_str:
                console.print(error_panel("Missing ID", "Usage: /tail <job_id>"))
                continue
            clean_id = args_str.split()[0].lstrip("#Q")
            try:
                jid = int(clean_id)
                tail.callback(job_id=jid)
            except ValueError:
                console.print(error_panel("Invalid ID", f"Could not parse job ID: {args_str}"))

        elif cmd in ("/log", "/fg", "/logs"):
            if not args_str:
                console.print(error_panel("Missing ID", "Usage: /log <job_id>"))
                continue
            clean_id = args_str.split()[0]
            fg.callback(job_id=clean_id)

        elif cmd in ("/retry", "/re-run", "/rerun"):
            parts = args_str.split()
            if not parts:
                console.print(error_panel("Missing Argument", "Usage: /retry <job_id> or /retry --all"))
                continue
            if "--all" in parts or parts[0] == "all":
                with get_session() as session:
                    failed = (
                        session.query(UnifiedJob)
                        .filter(UnifiedJob.status.in_(["failed", "cancelled"]))
                        .all()
                    )
                    if not failed:
                        console.print(f"  [{DIM}]No failed or cancelled jobs found to retry.[/]")
                        continue
                    count = 0
                    for j in failed:
                        if retry_job(j.id):
                            count += 1
                    if count > 0:
                        ensure_worker()
                        console.print(f"  [{SUCCESS}]Re-enqueued {count} job(s) as pending.[/]")
                    else:
                        console.print(f"  [{DIM}]No jobs were re-enqueued.[/]")
            else:
                clean_id = parts[0].lstrip("#Q")
                try:
                    jid = int(clean_id)
                    retry_cmd.callback(job_id=jid)
                except ValueError:
                    console.print(error_panel("Invalid ID", f"Could not parse job ID: {args_str}"))

        elif cmd in ("/cancel", "/stop", "/kill"):
            parts = args_str.split()
            if not parts:
                console.print(error_panel("Missing Argument", "Usage: /cancel <job_id>, /cancel --all, or /cancel --batch <id>"))
                continue
            if "--all" in parts or parts[0] == "all":
                cancel_cmd.callback(job_id=None, batch_id=None, all_jobs=True)
            elif "--batch" in parts:
                idx = parts.index("--batch")
                if idx + 1 < len(parts):
                    batch_id = parts[idx + 1]
                    cancel_cmd.callback(job_id=None, batch_id=batch_id, all_jobs=False)
                else:
                    console.print(error_panel("Missing Batch ID", "Usage: /cancel --batch <batch_id>"))
            else:
                clean_id = parts[0].lstrip("#Q")
                try:
                    jid = int(clean_id)
                    cancel_cmd.callback(job_id=jid, batch_id=None, all_jobs=False)
                except ValueError:
                    console.print(error_panel("Invalid ID", f"Could not parse job ID: {args_str}"))

        elif cmd in ("/clean", "/prune"):
            parts = args_str.split()
            prune_all = "--all" in parts
            prune_cmd.callback(prune_all=prune_all)

        else:
            console.print(
                f"  [{WARNING}]Unknown command:[/] [bold]{raw_cmd}[/] · "
                f"[{DIM}]Type [bold]/help[/bold] for available commands or press [bold]Tab[/bold][/]"
            )

