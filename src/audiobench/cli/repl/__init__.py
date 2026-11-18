"""REPL — context-aware interactive shell for AudioBench.

The REPL is the nerve center of AudioBench. It provides:
  - Context-aware ID injection: `show` auto-uses current transcript
  - Dot-commands for quick actions on the active transcript
  - Auto-context: transcribing sets context automatically
  - Onboarding: shows recent history + tips on first launch
  - Full command dispatch: every CLI command works inside the REPL
  - Shell escape, tab completion, persistent history
  - .play, .edit, .find, .path, .open — deep system integration

Package modules:
  - session: ReplSession state management
  - dispatch: Command dispatch + context summary
  - dot_commands: .stats, .show, .play, etc.
  - slash_commands: /help, /exit, /commands
  - completion: Tab completion setup
  - banner: Banner, onboarding, help text, goodbye

Usage:
    $ audiobench repl
    $ audiobench repl 42     ← start with transcript #42 as context
"""

from __future__ import annotations

import difflib
import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path

import click
from prompt_toolkit import PromptSession

import audiobench.cli.repl.backslash_commands as _  # noqa: F401 — registers \handlers
import audiobench.cli.repl.dot_commands as _dot  # noqa: F401 — registers .handlers
from audiobench.cli.display.theme import ACCENT, DIM, WARNING, console, error_panel
from audiobench.cli.repl.banner import print_banner, print_goodbye, print_onboarding
from audiobench.cli.repl.completion import setup_completion
from audiobench.cli.repl.dispatch import dispatch_backslash, dispatch_command, print_context_summary
from audiobench.cli.repl.dot_commands import dispatch_dot
from audiobench.cli.repl.session import ReplSession
from audiobench.cli.repl.slash_commands import ALIASES, handle_slash_command


def _dispatch_single(session: ReplSession, user_input: str) -> bool:
    """Dispatch one command string through the full input pipeline.

    Returns True if the REPL should exit (/exit in a chain), False otherwise.
    Used by both the main loop and semicolon chaining.
    """
    from audiobench.cli.repl.dispatch import _BACKSLASH_HANDLERS
    from audiobench.cli.repl.structural_router import handle_structural_question

    user_input = user_input.strip()
    if not user_input:
        return False

    # ? AI shorthand
    if user_input.startswith("?"):
        question = user_input[1:].strip()
        if not question:
            console.print(f"  [{DIM}]Usage: ? what are the key points?[/]")
            return False
        if handle_structural_question(session, question):
            return False
        tx_id = session.last_id
        if tx_id is None:
            console.print(f"  [{WARNING}]No context. Use [{ACCENT}]\\focus <id>[/] first.[/]")
            return False
        dispatch_command(session, ["ask", str(tx_id), question])
        return False

    # \backslash commands
    if user_input.startswith("\\"):
        dispatch_backslash(session, user_input[1:])
        return False

    # .dot commands
    if user_input.startswith(".") and len(user_input) > 1 and not user_input.startswith(".."):
        dispatch_dot(session, user_input[1:])
        return False

    # !shell escape
    if user_input.startswith("!"):
        shell_cmd = user_input[1:].strip()
        if shell_cmd:
            try:
                subprocess.run(shell_cmd, shell=True)
            except Exception as e:
                console.print(f"  [{WARNING}]{e}[/]")
        return False

    # /slash commands — transparently route to \backslash if registered, else meta slash command
    if user_input.startswith("/"):
        body = user_input[1:].strip()
        cmd_name = body.split()[0].lower() if body else ""
        if cmd_name in _BACKSLASH_HANDLERS:
            dispatch_backslash(session, body)
            return False
        should_exit = handle_slash_command(user_input, session)
        return should_exit

    # bare aliases
    if user_input.lower() in ALIASES:
        mapped = ALIASES[user_input.lower()]
        if mapped == "/exit":
            return True
        handle_slash_command(mapped, session)
        return False

    # Parse as CLI command
    try:
        args = shlex.split(user_input)
    except ValueError as e:
        console.print(f"  [{WARNING}]Parse error: {e}[/]")
        return False

    if not args:
        return False

    # Background job &
    if args[-1] == "&":
        from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS
        if not SUPPORTS_BACKGROUND_JOBS:
            console.print(f"  [{WARNING}]Background jobs only supported on Linux/macOS.[/]")
            return False
        from audiobench.jobs.scheduler import enqueue, ensure_worker

        args = args[:-1]
        if not args:
            return False
        args = session.expand_vars(args)
        args = session.auto_inject_id(args)

        slot = "network" if len(args) > 0 and args[0] == "youtube" else "transcription"
        job_type = args[0] if args else "generic"

        job_id = enqueue(
            job_type=job_type,
            slot=slot,
            args=args,
            command_display=" ".join(args),
        )
        ensure_worker()
        cmd_str = " ".join(args)
        if len(cmd_str) > 40:
            cmd_str = cmd_str[:37] + "..."
        console.print(f"  [{ACCENT}][#{job_id}][/] Job submitted — {cmd_str}")
        return False

    # Strip redundant audiobench prefix
    cmd_name = args[0]
    if cmd_name == "audiobench" and len(args) > 1:
        args = args[1:]
        cmd_name = args[0]

    # Block nested REPL
    if cmd_name == "repl":
        console.print(f"  [{DIM}]Already in REPL.[/]")
        return False

    # work <file> shorthand
    if cmd_name == "work":
        if len(args) < 2:
            console.print(f"  [{WARNING}]Usage: work <file_path>[/]")
            return False
        from audiobench.core.focused_entity import FocusedEntity
        from audiobench.storage.repository import TranscriptionRepository
        repo = TranscriptionRepository()
        file_id = repo.get_or_create_file(args[1])
        session.focus = FocusedEntity(type="file", id=file_id, label=Path(args[1]).name)
        return False

    # use <id> shorthand
    if cmd_name == "use":
        if len(args) < 2:
            console.print(f"  [{WARNING}]Usage: use <transcript_id>[/]")
            return False
        dispatch_dot(session, f"use {args[1]}")
        return False

    # Unknown command — ranked correction
    if cmd_name not in session.cli_group.commands:
        from audiobench.cli.repl.completion import _TRANSITION_MATRIX
        pool = (
            list(session.cli_group.commands.keys()) +
            [f"\\{k}" for k in _BACKSLASH_HANDLERS.keys()]
        )
        candidates = difflib.get_close_matches(cmd_name, pool, n=5, cutoff=0.3)

        def _score(c):
            cmp = c[1:] if c.startswith("\\") else c
            base = difflib.SequenceMatcher(None, cmd_name, cmp).ratio()
            return base + (0.15 if cmp.startswith(cmd_name[0]) else 0) + (0.1 if c in _TRANSITION_MATRIX.values() else 0)

        close = sorted(candidates, key=_score, reverse=True)[:3]
        console.print(
            f"  [{WARNING}]Unknown command:[/] {cmd_name}  "
            f"[{DIM}]— type[/] [{ACCENT}]help[/] [{DIM}]or[/] [{ACCENT}]/commands[/]"
        )
        if close:
            console.print(f"  [{DIM}]Did you mean?[/]")
            for c in close:
                if c.startswith("\\"):
                    handler = _BACKSLASH_HANDLERS.get(c[1:])
                    help_txt = (handler.__doc__ or "").strip().split("\n")[0] if handler else ""
                else:
                    help_txt = session.cli_group.commands[c].short_help or ""
                console.print(f"    [{ACCENT}]{c:<10}[/] — [{DIM}]{help_txt}[/]")
        return False

    dispatch_command(session, args)
    return False


@click.command("repl")
@click.argument("transcript_id", required=False, type=int, default=None)
def repl(transcript_id: int | None) -> None:
    """Start the interactive AudioBench shell.

    \\b
    Start fresh:
      audiobench repl

    \\b
    Start with a transcript loaded:
      audiobench repl 42

    \\b
    Inside the REPL, every command works without 'audiobench' prefix.
    Context-aware commands auto-fill the transcript ID.
    Type 'help' for the full guide.
    """
    ctx = click.get_current_context()
    cli_group = ctx.parent.command if ctx.parent else None

    if cli_group is None or not isinstance(cli_group, click.Group):
        console.print(error_panel("REPL Error", "Cannot find parent CLI group"))
        return

    session = ReplSession(cli_group)

    # ── Security gate (runs BEFORE any UI is shown) ───────────────────────────
    # First-run: prompts to set passphrase + recovery key.
    # Returning user: 3-attempt passphrase check; recovery key fallback.
    # If auth fails fatally, exit before touching any data.
    from audiobench.core.settings import get_settings as _get_settings
    from audiobench.security.auth import run_startup_gate
    if not run_startup_gate():
        raise SystemExit(1)

    # Start idle-lock watcher (daemon thread; no-op if idle_lock_seconds=0)
    _security_settings = _get_settings()
    session.start_idle_lock_watcher(idle_seconds=_security_settings.idle_lock_seconds)

    session.maybe_resume()
    session._load_history_ids()
    setup_completion(session)

    # ── Daemon warm-up ────────────────────────────────────────────────────────
    # Fire-and-forget: fork the daemon in the background so semantic memory
    # is ready by the time the user types their first question.
    # This runs concurrently with banner + onboarding rendering (~1-2 s),
    # giving the daemon a head start. Completely silent — no status messages.
    import threading as _threading

    from audiobench.daemon.factory import ensure_daemon_running as _ensure_daemon
    _threading.Thread(target=_ensure_daemon, daemon=True, name="daemon-warmup").start()

    print_banner(session)

    from audiobench.cli.repl.dispatch import SESSION_ID
    from audiobench.observatory.context import log_event
    log_event(subsystem="repl", event_type="session_started", message="REPL session started", level="INFO", session_id=SESSION_ID)

    # Pre-load context if transcript ID given
    if transcript_id is not None:
        session.set_context(transcript_id)
        if session.focus:
            print_context_summary(session)
            console.print()

    # Onboarding: show recent transcripts
    print_onboarding(session)

    # ── Input loop ────────────────────────────────────────────────────────────
    from prompt_toolkit.completion import FuzzyCompleter

    from audiobench.cli.repl.completion import AudioBenchAutoSuggest, AudioBenchCompleter

    prompt_session = PromptSession(
        history=session.get_history(),
        auto_suggest=AudioBenchAutoSuggest(session),
        completer=FuzzyCompleter(AudioBenchCompleter(session)),
        complete_while_typing=False,  # only on Tab — never interrupt typing
    )

    from datetime import UTC
    from datetime import datetime as _dt

    while True:
        try:
            user_input = prompt_session.prompt(session.prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            log_event(subsystem="repl", event_type="session_ended", message="REPL session ended", level="INFO", session_id=SESSION_ID)
            print_goodbye(session)
            break

        # ── Security: check idle-lock BEFORE recording the new keypress ──────
        # Order matters: if we update last_keystroke_at first, the sync check
        # in effective_tier() will see idle=0 and never fire, letting an unlocked
        # session survive a sleep/wake cycle if the user types immediately on wake.
        if session._idle_lock_fired:
            console.print(f"  [{DIM}]● Auto-locked (idle).[/]")
            session._idle_lock_fired = False
        session.last_keystroke_at = _dt.now(UTC)


        if not user_input:
            continue

        # ── Semicolon chaining: \focus 152; ask "key points?"; \note ─────────
        if ";" in user_input:
            segments = [s.strip() for s in user_input.split(";") if s.strip()]
            if len(segments) > 1:
                for seg in segments:
                    should_exit = _dispatch_single(session, seg)
                    if should_exit:
                        log_event(subsystem="repl", event_type="session_ended", message="REPL session ended", level="INFO", session_id=SESSION_ID)
                        print_goodbye(session)
                        return  # exit REPL from within a chain
                continue

        # Single command — full dispatch
        should_exit = _dispatch_single(session, user_input)
        if should_exit:
            log_event(subsystem="repl", event_type="session_ended", message="REPL session ended", level="INFO", session_id=SESSION_ID)
            print_goodbye(session)
            break
