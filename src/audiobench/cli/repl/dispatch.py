"""REPL command dispatch — routes user input to Click commands or \\ handlers.

Provides:
    - register_backslash(): Decorator to register a \\command handler
    - dispatch_backslash(): Route \\cmd [args] to a registered handler
    - dispatch_command(): Run a Click command from the REPL
    - try_capture_last_id(): Auto-capture new transcript ID after transcribe
    - print_context_summary(): Compact one-line context display
"""

from __future__ import annotations

import difflib
import uuid
from collections.abc import Callable

# ── Module-level session ID — generated once per REPL process ─────────────────
SESSION_ID: str = str(uuid.uuid4())[:8]

# ── Backslash Handler Registry ───────────────────────────────
#
# Handlers are plain functions: fn(session: ReplSession, args: str) -> None
# Register them with @register_backslash("name").
# The registry is module-level so importing backslash_commands.py
# is enough to populate it.

_BACKSLASH_HANDLERS: dict[str, Callable] = {}
_RESUME_HANDLERS: dict[str, Callable] = {}

def register_resume(context_name: str) -> Callable:
    """Decorator to register a resume handler for a specific stack context."""

    def decorator(fn: Callable) -> Callable:
        _RESUME_HANDLERS[context_name] = fn
        return fn

    return decorator


def register_backslash(name: str) -> Callable:
    """Decorator to register a \\command handler by name."""

    def decorator(fn: Callable) -> Callable:
        _BACKSLASH_HANDLERS[name] = fn
        return fn

    return decorator


def _suggest_backslash(cmd: str) -> str | None:
    """Return the closest registered \\command name, or None."""
    close = difflib.get_close_matches(
        cmd, list(_BACKSLASH_HANDLERS.keys()), n=1, cutoff=0.5
    )
    return close[0] if close else None


def dispatch_backslash(session, line: str) -> bool:
    """Route a \\command line to the registered handler.

    ``line`` is the text *after* the leading backslash, e.g. ``"focus 42"``.
    Returns True on success, False on unknown command or error.
    """
    from audiobench.cli.display.theme import ACCENT, DIM, WARNING, console

    parts = line.strip().split(None, 1)
    if not parts:
        # bare backslash → show help
        handler = _BACKSLASH_HANDLERS.get("help")
        if handler:
            handler(session, "")
        return True

    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = _BACKSLASH_HANDLERS.get(cmd)
    if handler is None:
        suggestion = _suggest_backslash(cmd)
        console.print(f"  [{WARNING}]Unknown: \\{cmd}[/]")
        if suggestion:
            console.print(f"  [{DIM}]Did you mean: [{ACCENT}]\\{suggestion}[/]?[/]")
        else:
            console.print(f"  [{DIM}]Type [{ACCENT}]\\help[/] for available commands.[/]")
        return False

    session._command_count += 1
    try:
        handler(session, args)
        return True
    except Exception as e:
        console.print(f"  [{WARNING}]Error in \\{cmd}: {e}[/]")
        return False

import time

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

    t_start = time.monotonic()
    try:
        session.cli_group(args, standalone_mode=False)

        # After transcribe, auto-capture the new ID
        if args and args[0] == "transcribe":
            try_capture_last_id(session)

        session._command_count += 1
        duration_ms = int((time.monotonic() - t_start) * 1000)
        _log_command_event(session, args, duration_ms)

    except click.exceptions.Exit:
        pass
    except click.exceptions.Abort:
        console.print(f"  [{DIM}]Aborted[/]")
    except click.exceptions.UsageError as e:
        msg = str(e)
        console.print(f"  [{WARNING}]{msg}[/]")
        # Only suggest \focus when the missing param is a transcript ID
        if "Missing" in msg and session.last_id is None:
            param_name = msg.lower()
            if any(kw in param_name for kw in ("transcript", "transcription")):
                console.print(
                    f"  [{DIM}]Tip: Use [{ACCENT}]\\focus <id>[/] to set a transcript context[/]"
                )
    except SystemExit:
        pass
    except Exception as e:
        console.print(error_panel("Error", str(e)))


def _log_command_event(session: ReplSession, args: list[str], duration_ms: int) -> None:
    """Log command dispatch to Observatory journal. Silent on any failure."""
    try:
        from audiobench.observatory.context import log_event

        cmd = args[0] if args else "unknown"
        file_id = session.focus.id if session.focus and session.focus.type == "file" else None
        tx_id = session.last_id

        log_event(
            subsystem="repl",
            event_type="command_dispatched",
            message=f"{cmd} {' '.join(args[1:]) if len(args) > 1 else ''}".strip(),
            level="INFO",
            duration_ms=duration_ms,
            metadata={
                "command": cmd,
                "args": args[1:] if len(args) > 1 else [],
                "context_file_id": file_id,
                "context_tx_id": tx_id,
            },
            session_id=SESSION_ID,
        )
    except Exception:
        pass  # Command graph is advisory — never break the REPL


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
            # Refresh completion cache
            from audiobench.cli.repl.completion import _load_transcript_cache
            _load_transcript_cache(session)
    except Exception:
        pass


def print_context_summary(session: ReplSession) -> None:
    """Print a compact summary when context changes."""
    if not session.focus:
        return

    repo = session._get_repo()

    if session.focus.type == "file":
        audio_file = repo.get_audio_file(session.focus.id)
        if audio_file:
            dur_str = format_duration(audio_file.get("duration_seconds", 0) or 0)
            console.print(
                f"  [{SUCCESS}]✓[/] Focused on: "
                f"[{ACCENT}]{audio_file.get('file_name', '?')}[/] "
                f"[{DIM}]({dur_str})[/]"
            )
        else:
            console.print(
                f"  [{SUCCESS}]✓[/] Focused on file: [{ACCENT}]#{session.focus.id}[/]"
            )

        # Mention transcript if there is one
        tx_id = session.last_id
        if tx_id:
            rec = repo.get_by_id(tx_id, session=session)
            if rec:
                words = rec.get("word_count", 0) or 0
                console.print(f"      [{DIM}]↳ Active transcript: #{tx_id} ({words:,} words)[/]")

    elif session.focus.type == "transcript":
        rec = repo.get_by_id(session.focus.id, session=session)
        if rec:
            words = rec.get("word_count", 0) or 0
            dur_str = format_duration(rec.get("duration", 0) or 0)
            console.print(
                f"  [{SUCCESS}]✓[/] Focused on transcript: "
                f"[{ACCENT}]#{session.focus.id}[/] — "
                f"{rec.get('file_name', '?')} "
                f"[{DIM}]({words:,} words, {dur_str})[/]"
            )
        else:
            console.print(
                f"  [{SUCCESS}]✓[/] Focused on transcript: [{ACCENT}]#{session.focus.id}[/]"
            )

    elif session.focus.type == "school":
        from audiobench.core.db_session import get_session
        from sqlalchemy import text as _sql

        with get_session() as db:
            row = db.execute(
                _sql("SELECT name, node_count FROM expression_communities WHERE community_id = :cid"),
                {"cid": session.focus.id},
            ).fetchone()
            if row:
                name, count = row[0], row[1]
                console.print(
                    f"  [{SUCCESS}]✓[/] Focused on school: "
                    f"[{ACCENT}]#{session.focus.id}[/] — "
                    f"{name} "
                    f"[{DIM}]({count:,} chunks)[/]"
                )
            else:
                console.print(
                    f"  [{SUCCESS}]✓[/] Focused on school: [{ACCENT}]#{session.focus.id}[/]"
                )

    elif session.focus.type == "work":
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import WorkRecord

        with get_session() as db:
            work = db.query(WorkRecord).filter_by(id=session.focus.id).first()
            if work:
                author_str = f" by {work.author}" if work.author else ""
                console.print(
                    f"  [{SUCCESS}]✓[/] Focused on work: "
                    f"[{ACCENT}]#{session.focus.id}[/] — "
                    f"{work.title}{author_str}"
                )
            else:
                console.print(
                    f"  [{SUCCESS}]✓[/] Focused on work: [{ACCENT}]#{session.focus.id}[/]"
                )

