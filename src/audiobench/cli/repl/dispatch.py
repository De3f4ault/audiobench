"""REPL command dispatch — routes user input to Click commands.

Provides:
    - dispatch_command(): Run a Click command from the REPL
    - try_capture_last_id(): Auto-capture new transcript ID after transcribe
    - print_context_summary(): Compact one-line context display
"""

from __future__ import annotations

import click

from audiobench.cli.display.theme import (
    ACCENT,
    DIM,
    SUCCESS,
    WARNING,
    console,
    error_panel,
    format_duration,
)
from audiobench.cli.repl.session import ReplSession


def dispatch_command(session: ReplSession, args: list[str]) -> None:
    """Dispatch a command to the Click CLI group."""
    args = session.expand_vars(args)
    args = session.auto_inject_id(args)

    try:
        session.cli_group(args, standalone_mode=False)

        # After transcribe, auto-capture the new ID
        if args and args[0] == "transcribe":
            try_capture_last_id(session)

        session._command_count += 1

    except click.exceptions.Exit:
        pass
    except click.exceptions.Abort:
        console.print(f"  [{DIM}]Aborted[/]")
    except click.exceptions.UsageError as e:
        msg = str(e)
        console.print(f"  [{WARNING}]{msg}[/]")
        # Only suggest .use when the missing param is a transcript ID
        if "Missing" in msg and session.last_id is None:
            param_name = msg.lower()
            if any(kw in param_name for kw in ("transcript", "transcription")):
                console.print(
                    f"  [{DIM}]Tip: Set context with .use <ID> to auto-fill transcript IDs[/]"
                )
    except SystemExit:
        pass
    except Exception as e:
        console.print(error_panel("Error", str(e)))


def try_capture_last_id(session: ReplSession) -> None:
    """After a transcribe command, grab the newest ID."""
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=1)
        if records:
            new_id = records[0]["id"]
            session.set_context(new_id)
            print_context_summary(session)
            # Refresh navigation IDs
            session._load_history_ids()
    except Exception:
        pass


def print_context_summary(session: ReplSession) -> None:
    """Print a compact summary when context changes."""
    rec = session.context_record
    if not rec:
        return
    duration = rec.get("duration", 0) or 0
    dur_str = format_duration(duration)
    console.print(
        f"  [{SUCCESS}]✓[/] Context: "
        f"[{ACCENT}]#{session.last_id}[/] — "
        f"{rec.get('file_name', '?')} "
        f"[{DIM}]({rec.get('word_count', 0):,} words, "
        f"{dur_str})[/]"
    )
