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
    format_elapsed,
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
    with get_session() as session:
        has_pending = session.query(UnifiedJob).filter_by(status="pending").count() > 0
    if has_pending:
        ensure_worker()

    if ctx.invoked_subcommand is None:
        if interactive or sys.stdout.isatty():
            _interactive_jobs_repl()
        else:
            _list_jobs()


@jobs.command(name="list")
def list_cmd() -> None:
    """List active and recent jobs."""
    _list_jobs()


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
                    is_prio = item.priority < 100
                    if item.status == "running":
                        st_disp = f"[{ACCENT}]● running [P][/]" if is_prio else f"[{ACCENT}]● running[/]"
                    else:
                        st_disp = f"[{ACCENT}]○ [P] pending[/]" if is_prio else f"[{DIM}]○ pending[/]"
                    elapsed = format_elapsed(item.started_at) if item.status == "running" else "—"
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
                    ("Status", {"width": 14}),
                    ("Item / Command", {}),
                    ("Elapsed", {"width": 10}),
                ],
            )
            for item in standalone:
                is_prio = item.priority < 100
                if item.status == "running":
                    st_disp = f"[{ACCENT}]● running [P][/]" if is_prio else f"[{ACCENT}]● running[/]"
                else:
                    st_disp = f"[{ACCENT}]○ [P] pending[/]" if is_prio else f"[{DIM}]○ pending[/]"
                elapsed = format_elapsed(item.started_at) if item.status == "running" else "—"
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
                if item.error_summary == "Retrieved from cache":
                    st_disp = f"[{SUCCESS}]✓ cached[/]"
                else:
                    st_disp = f"[{SUCCESS}]✓ done[/]"
            elif item.status == "failed":
                st_disp = f"[{WARNING}]✗ failed[/]"
            else:
                st_disp = f"[{DIM}]⊘ cancelled[/]"

            dur = format_elapsed(item.started_at, item.ended_at)
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
            format_elapsed(item.started_at),
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
        elapsed = format_elapsed(item.started_at)
        console.print(f"    [{ACCENT}]#[/][bold]{item.id}[/]  [{DIM}]{item.job_type}[/]  {name}  [{DIM}]{elapsed}[/]")


@jobs.command(name="tail")
@click.argument("job_id", type=int)
def tail(job_id: int) -> None:
    """Stream live events or logs from a job in real-time (like journalctl -f)."""
    from audiobench.cli.display.job_stream import wait_and_stream_job

    wait_and_stream_job(job_id)


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


@jobs.command(name="diff")
@click.argument("job_id_1", type=int)
@click.argument("job_id_2", type=int)
def diff_cmd(job_id_1: int, job_id_2: int) -> None:
    """Compare two jobs side-by-side to detect duplicates and parameter differences."""
    render_jobs_diff(job_id_1, job_id_2)


def render_jobs_diff(job_id_1: int, job_id_2: int) -> None:
    """Render side-by-side comparison of two jobs."""
    from rich.table import Table
    from audiobench.cli.display.theme import BOX_STYLE
    from audiobench.jobs.scheduler import compute_job_fingerprint

    with get_session() as session:
        j1 = session.get(UnifiedJob, job_id_1)
        j2 = session.get(UnifiedJob, job_id_2)

    if not j1:
        console.print(error_panel("Not Found", f"Job #{job_id_1} does not exist."))
        return
    if not j2:
        console.print(error_panel("Not Found", f"Job #{job_id_2} does not exist."))
        return

    table = Table(
        title=f"Comparison: Job #{j1.id} vs Job #{j2.id}",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("Property", style=BOLD)
    table.add_column(f"Job #{j1.id}")
    table.add_column(f"Job #{j2.id}")
    table.add_column("Match", style=DIM)

    def add_row(prop: str, val1: str, val2: str):
        match = f"[{SUCCESS}]✓ identical[/]" if str(val1) == str(val2) else f"[{WARNING}]≠ different[/]"
        table.add_row(prop, str(val1), str(val2), match)

    add_row("Job Type", j1.job_type, j2.job_type)
    add_row("File Label", j1.file_label or "—", j2.file_label or "—")
    add_row("Status", j1.status, j2.status)
    add_row("Slot", j1.slot, j2.slot)

    # Parse arguments
    try:
        args1 = json.loads(j1.args_json)
    except Exception:
        args1 = []
    try:
        args2 = json.loads(j2.args_json)
    except Exception:
        args2 = []

    def get_arg_val(args: list[str], flag: str) -> str:
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
        return "default"

    m1 = get_arg_val(args1, "--model")
    m2 = get_arg_val(args2, "--model")
    add_row("Model", m1, m2)

    e1 = get_arg_val(args1, "--engine")
    e2 = get_arg_val(args2, "--engine")
    add_row("Engine", e1, e2)

    lang1 = get_arg_val(args1, "--language")
    lang2 = get_arg_val(args2, "--language")
    add_row("Language", lang1, lang2)

    fp1 = j1.fingerprint or compute_job_fingerprint(j1.job_type, args1, j1.file_label)
    fp2 = j2.fingerprint or compute_job_fingerprint(j2.job_type, args2, j2.file_label)
    add_row("Fingerprint", fp1[:12] + "...", fp2[:12] + "...")

    console.print()
    console.print(table)

    # Overall verdict
    if fp1 == fp2:
        console.print(f"\n  [{SUCCESS}]Verdict: EXACT DUPLICATE (100% Identity)[/]\n")
    elif (j1.file_label and j1.file_label == j2.file_label) or (
        j1.job_type == j2.job_type and (m1 != m2 or e1 != e2)
    ):
        console.print(f"\n  [{WARNING}]Verdict: SIMILAR (Same target audio, different options)[/]\n")
    else:
        console.print(f"\n  [{DIM}]Verdict: DIFFERENT (Distinct target audio or task)[/]\n")


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

        elif cmd in ("/diff", "/compare"):
            parts = args_str.split()
            if len(parts) < 2:
                console.print(error_panel("Missing Argument", "Usage: /diff <job_id_1> <job_id_2>"))
                continue
            try:
                j1_id = int(parts[0].lstrip("#Q"))
                j2_id = int(parts[1].lstrip("#Q"))
                render_jobs_diff(j1_id, j2_id)
            except ValueError:
                console.print(error_panel("Invalid ID", f"Could not parse job IDs: {args_str}"))

        elif cmd in ("/resume", "/start"):
            ensure_worker()
            console.print(f"  [{SUCCESS}]Worker daemon signaled. Checking and running pending queue...[/]")

        elif cmd in ("/retry", "/re-run", "/rerun"):
            parts = args_str.split()
            if not parts:
                console.print(error_panel("Missing Argument", "Usage: /retry <job_id> or /retry --all"))
                continue
            if "--all" in parts or parts[0] == "all":
                include_cancelled = "--include-cancelled" in parts or "--cancelled" in parts
                target_statuses = ["failed", "cancelled"] if include_cancelled else ["failed"]
                with get_session() as session:
                    failed = (
                        session.query(UnifiedJob)
                        .filter(UnifiedJob.status.in_(target_statuses))
                        .all()
                    )
                    if not failed:
                        target_label = "failed or cancelled" if include_cancelled else "failed"
                        console.print(f"  [{DIM}]No {target_label} jobs found to retry.[/]")
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

