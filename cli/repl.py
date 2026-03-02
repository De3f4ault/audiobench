"""REPL — context-aware interactive shell for AudioBench.

The REPL is the nerve center of AudioBench. It provides:
  - Context-aware ID injection: `show` auto-uses current transcript
  - Dot-commands for quick actions on the active transcript
  - Auto-context: transcribing sets context automatically
  - Onboarding: shows recent history + tips on first launch
  - Full command dispatch: every CLI command works inside the REPL
  - Shell escape, tab completion, persistent history
  - .play, .edit, .find, .path, .open — deep system integration

Usage:
    $ audiobench repl
    $ audiobench repl 42     ← start with transcript #42 as context
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import readline
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from cli.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    error_panel,
    format_duration,
    make_table,
)


# ── Commands that accept a transcript ID as first positional arg ──
# When context is set and no ID is given, the REPL auto-injects it.

CONTEXT_AWARE_COMMANDS = {
    "show",  # show <id>
    "ask",  # ask <id> <question>
    "summarize",  # summarize <id>
    "vocab",  # vocab <id>
    "export",  # export <id>
    "delete",  # delete <id>
    "chat",  # chat <id...>  (nargs=-1, optional)
    "speak",  # speak <id>
}


# ── REPL Session State ──────────────────────────────────────


class ReplSession:
    """Holds mutable REPL state across the interactive session."""

    def __init__(self, cli_group: click.Group) -> None:
        self.cli_group = cli_group
        self.last_id: int | None = None
        self.context_record: dict | None = None
        self._history_file = Path.home() / ".audiobench" / "repl_history"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._repo = None
        self._command_count = 0
        self._history_ids: list[int] = []  # for .next / .prev navigation
        self._history_cursor: int = -1

    def _get_repo(self):
        """Lazy-init the transcription repository."""
        if self._repo is None:
            from src.audiobench.storage.database import init_db
            from src.audiobench.storage.repository import TranscriptionRepository

            init_db()
            self._repo = TranscriptionRepository()
        return self._repo

    def _load_history_ids(self) -> None:
        """Load all transcript IDs for .next/.prev navigation."""
        try:
            repo = self._get_repo()
            records = repo.get_history(limit=500)
            # Oldest first so .next goes forward in time
            self._history_ids = [r["id"] for r in reversed(records)]
        except Exception:
            self._history_ids = []

    # ── Prompt ──

    @property
    def prompt(self) -> str:
        if self.last_id is not None:
            name = ""
            if self.context_record:
                fname = self.context_record.get("file_name", "")
                if fname and len(fname) > 25:
                    fname = fname[:22] + "..."
                name = f" {fname}" if fname else ""
            return f"\001\033[36m\002ab\001\033[0m\002 #{self.last_id}{name} > "
        return "\001\033[36m\002ab\001\033[0m\002 > "

    # ── Variable expansion ──

    def expand_vars(self, args: list[str]) -> list[str]:
        """Replace $last / $last_id / $id with the current context ID."""
        if self.last_id is None:
            return args
        sid = str(self.last_id)
        return [sid if a in ("$last", "$last_id", "$id") else a for a in args]

    # ── Context auto-injection ──

    def auto_inject_id(self, args: list[str]) -> list[str]:
        """For context-aware commands, inject the current ID if missing."""
        if not args or self.last_id is None:
            return args

        cmd_name = args[0]
        if cmd_name not in CONTEXT_AWARE_COMMANDS:
            return args

        rest = args[1:]

        # If the next arg is already a number, user specified an ID
        if rest and rest[0].isdigit():
            return args

        # If next arg is a flag, or there's no next arg — inject ID
        return [cmd_name, str(self.last_id)] + rest

    # ── Set context ──

    def set_context(self, record_id: int) -> None:
        """Set the current context to a transcription ID."""
        repo = self._get_repo()
        record = repo.get_by_id(record_id)
        if record:
            self.last_id = record_id
            self.context_record = record
            # Update navigation cursor
            if record_id in self._history_ids:
                self._history_cursor = self._history_ids.index(record_id)
        else:
            console.print(f"  [{WARNING}]Transcript #{record_id} not found[/]")

    def clear_context(self) -> None:
        """Clear the current context, return to bare prompt."""
        self.last_id = None
        self.context_record = None
        self._history_cursor = -1

    def refresh_context(self) -> None:
        """Refresh the current context record from DB."""
        if self.last_id is not None:
            repo = self._get_repo()
            self.context_record = repo.get_by_id(self.last_id)

    # ── Readline history ──

    def load_history(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            readline.read_history_file(str(self._history_file))
        readline.set_history_length(1000)

    def save_history(self) -> None:
        with contextlib.suppress(OSError):
            readline.write_history_file(str(self._history_file))


# ── Dot-Command Typo Correction ─────────────────────────────


def _suggest_dot_command(typed: str) -> str | None:
    """Find closest dot-command match using difflib."""
    all_cmds = list(DOT_COMMANDS.keys())
    # Try prefix match first
    prefix_matches = [c for c in all_cmds if c.startswith(typed[:3])]
    if prefix_matches:
        return prefix_matches[0]
    # Fuzzy match
    close = difflib.get_close_matches(typed, all_cmds, n=1, cutoff=0.5)
    return close[0] if close else None


# ── Dot Commands ────────────────────────────────────────────


DOT_COMMANDS = {
    ".stats": "Word count, duration, language, model",
    ".show": "Display full transcript with timestamps",
    ".segments": "Timestamped segment breakdown",
    ".vocab": "Word frequency analysis (top 20)",
    ".info": "Full metadata for current transcript",
    ".find": 'Search within transcript: .find "keyword"',
    ".export": "Re-export: .export srt  |  .export json",
    ".ask": 'AI question: .ask "What was decided?"',
    ".chat": "Start AI chat with this transcript",
    ".summarize": "AI summary of this transcript",
    ".play": "Play audio: .play  |  .play 01:25",
    ".edit": "Edit transcript text in $EDITOR",
    ".path": "Show source audio file path",
    ".open": "Open source audio in default player",
    ".use": "Switch context: .use 42",
    ".clear": "Clear context (return to bare prompt)",
    ".next": "Jump to next transcript in history",
    ".prev": "Jump to previous transcript in history",
    ".recent": "Show 5 most recent transcriptions",
    ".search": 'Search all transcripts: .search "keyword"',
    ".help": "Show this dot-command list",
}


def _handle_dot_command(cmd: str, session: ReplSession) -> None:
    """Handle a dot-command that operates on the current context."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # Strip surrounding quotes from arg (fixes double-quote bug)
    if arg and len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ('"', "'"):
        arg = arg[1:-1]

    # ── Context-free dot-commands ──

    if command == ".use":
        if not arg or not arg.isdigit():
            console.print(f"  [{DIM}]Usage: .use <transcript_id>[/]")
            return
        session.set_context(int(arg))
        if session.context_record:
            _print_context_summary(session)
        return

    if command == ".clear":
        session.clear_context()
        console.print(f"  [{SUCCESS}]✓[/] Context cleared")
        return

    if command == ".next":
        _navigate_context(session, direction=1)
        return

    if command == ".prev":
        _navigate_context(session, direction=-1)
        return

    if command == ".recent":
        _dispatch_command(session, ["history", "--tail", "5"])
        return

    if command == ".search":
        if not arg:
            console.print(f'  [{DIM}]Usage: .search "keyword"[/]')
            return
        _dispatch_command(session, ["search", arg])
        return

    if command in (".help", ".commands", ".?"):
        _print_dot_help()
        return

    if command in (".exit", ".quit"):
        # Redirect to /exit — user naturally types .exit
        console.print(f"  [{DIM}]Tip: Use /exit or exit to quit[/]")
        return

    # ── Context-required dot-commands ──

    if session.last_id is None or session.context_record is None:
        console.print(
            f"\n  [{WARNING}]No active context.[/] Set one first:\n"
            f"    [{ACCENT}].use <ID>[/]            Switch to a transcript\n"
            f"    [{ACCENT}].recent[/]              See recent transcriptions\n"
            f"    [{ACCENT}]transcribe file.mp3[/]  Transcribe and auto-set\n"
        )
        return

    rec = session.context_record

    if command == ".stats":
        _dot_stats(rec)
    elif command == ".show":
        _dot_show(rec)
    elif command == ".segments":
        _dot_segments(rec)
    elif command == ".info":
        _dot_info(rec)
    elif command == ".find":
        _dot_find(rec, arg)
    elif command == ".vocab":
        extra = arg.split() if arg else []
        _dispatch_command(session, ["vocab", str(session.last_id)] + extra)
    elif command == ".export":
        fmt = arg if arg else "srt"
        _dispatch_command(session, ["export", str(session.last_id), "-f", fmt])
    elif command == ".ask":
        if not arg:
            console.print(f'  [{DIM}]Usage: .ask "question"[/]')
            return
        _dispatch_command(session, ["ask", str(session.last_id), arg])
    elif command == ".chat":
        _dispatch_command(session, ["chat", str(session.last_id)])
    elif command == ".summarize":
        _dispatch_command(session, ["summarize", str(session.last_id)])
    elif command == ".play":
        _dot_play(session, arg)
    elif command == ".edit":
        _dot_edit(session)
    elif command == ".path":
        _dot_path(rec)
    elif command == ".open":
        _dot_open(rec)
    else:
        suggestion = _suggest_dot_command(command)
        if suggestion:
            console.print(
                f"  [{WARNING}]Unknown: {command}[/]\n"
                f"  [{DIM}]Did you mean [{ACCENT}]{suggestion}[/]?[/]"
            )
        else:
            console.print(
                f"  [{DIM}]Unknown: {command}  —  Type .help for available dot-commands[/]"
            )


# ── Dot-Command Implementations ─────────────────────────────


def _dot_stats(rec: dict) -> None:
    duration = rec.get("duration", 0) or 0
    mins = int(duration // 60)
    secs = int(duration % 60)
    console.print(f"\n  [{BOLD}][{ACCENT}]#{rec['id']}[/] — {rec.get('file_name', '?')}[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Words:    [{ACCENT}]{rec.get('word_count', 0):,}[/]")
    console.print(f"    Duration: [{ACCENT}]{mins}m {secs}s[/]")
    console.print(f"    Segments: {rec.get('segment_count', 0)}")
    console.print(f"    Language: {rec.get('language') or 'auto-detected'}")
    console.print(f"    Model:    {rec.get('model') or '?'}")
    console.print(f"    Engine:   {rec.get('engine') or '?'}")
    console.print(f"    Created:  {rec.get('created_at', '?')}")
    console.print()


def _dot_show(rec: dict) -> None:
    console.print(f"\n  [{BOLD}][{ACCENT}]#{rec['id']}[/] — {rec.get('file_name', '?')}[/]\n")
    segments = rec.get("segments", [])
    if segments:
        for seg in segments:
            start = seg.get("start", 0) or 0
            ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
            speaker = f" ({seg['speaker']})" if seg.get("speaker") else ""
            console.print(f"  [{DIM}]{ts}{speaker}[/] {seg.get('text', '').strip()}")
    else:
        text = rec.get("full_text", "")
        if text:
            console.print(f"  {text}")
        else:
            console.print(f"  [{DIM}]No text available[/]")
    console.print()


def _dot_segments(rec: dict) -> None:
    segments = rec.get("segments", [])
    if not segments:
        console.print(f"  [{DIM}]No segments available[/]")
        return
    table = make_table(
        f"Segments — #{rec['id']} ({len(segments)} total)",
        [
            ("Time", {"style": DIM, "width": 14}),
            ("Speaker", {"width": 10}),
            ("Text", {}),
        ],
    )
    for seg in segments:
        start = seg.get("start", 0) or 0
        end = seg.get("end", 0) or 0
        ts = (
            f"{int(start // 60):02d}:{int(start % 60):02d}→{int(end // 60):02d}:{int(end % 60):02d}"
        )
        speaker = seg.get("speaker") or "—"
        text = (seg.get("text", "") or "").strip()
        if len(text) > 80:
            text = text[:77] + "..."
        table.add_row(ts, speaker, text)
    console.print(table)


def _dot_info(rec: dict) -> None:
    table = make_table(
        f"Metadata — #{rec['id']}",
        [("Field", {"style": BOLD}), ("Value", {})],
    )
    for key in [
        "id",
        "file_name",
        "file_path",
        "source",
        "language",
        "language_probability",
        "engine",
        "model",
        "duration",
        "word_count",
        "segment_count",
        "status",
        "created_at",
    ]:
        val = rec.get(key, "—")
        if val is None:
            val = "—"
        if key == "duration" and val:
            val = format_duration(val)
        table.add_row(key, str(val))
    console.print(table)


def _dot_find(rec: dict, query: str) -> None:
    """Search within the current transcript's text."""
    if not query:
        console.print(f'  [{DIM}]Usage: .find "keyword"[/]')
        return

    segments = rec.get("segments", [])
    full_text = rec.get("full_text", "")
    query_lower = query.lower()

    if not full_text or query_lower not in full_text.lower():
        console.print(f'  [{DIM}]No matches for "{query}" in #{rec["id"]}[/]')
        return

    # Count occurrences
    count = full_text.lower().count(query_lower)
    console.print(
        f'\n  [{ACCENT}]{count}[/] match(es) for "[{ACCENT}]{query}[/]" in #{rec["id"]}:\n'
    )

    if segments:
        for seg in segments:
            text = (seg.get("text", "") or "").strip()
            if query_lower in text.lower():
                start = seg.get("start", 0) or 0
                ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
                # Highlight the match
                highlighted = re.sub(
                    re.escape(query),
                    f"[bold {ACCENT}]{query}[/]",
                    text,
                    flags=re.IGNORECASE,
                )
                console.print(f"  [{DIM}]{ts}[/] {highlighted}")
    else:
        # No segments: highlight in full text
        highlighted = re.sub(
            re.escape(query),
            f"[bold {ACCENT}]{query}[/]",
            full_text,
            flags=re.IGNORECASE,
        )
        console.print(f"  {highlighted}")
    console.print()


def _dot_play(session: ReplSession, arg: str) -> None:
    """Play the source audio file using ffplay."""
    rec = session.context_record
    file_path = rec.get("file_path")
    if not file_path or not Path(file_path).exists():
        console.print(
            f"  [{WARNING}]Source audio file not found[/]\n"
            f"  [{DIM}]Path: {file_path or 'unknown'}[/]"
        )
        return

    # Parse optional start time
    start_seconds = 0.0
    if arg:
        # Handle "segment N" syntax
        seg_match = re.match(r"segment\s+(\d+)", arg, re.IGNORECASE)
        if seg_match:
            seg_idx = int(seg_match.group(1))
            segments = rec.get("segments", [])
            matched = [s for s in segments if s.get("index") == seg_idx]
            if matched:
                start_seconds = matched[0].get("start", 0) or 0
                end_seconds = matched[0].get("end", 0) or 0
                console.print(
                    f"  [{DIM}]Playing segment {seg_idx}: "
                    f"{int(start_seconds // 60):02d}:"
                    f"{int(start_seconds % 60):02d} → "
                    f"{int(end_seconds // 60):02d}:"
                    f"{int(end_seconds % 60):02d}[/]"
                )
            else:
                console.print(
                    f"  [{WARNING}]Segment {seg_idx} not found. Use .segments to see available.[/]"
                )
                return
        else:
            # Parse MM:SS or HH:MM:SS
            time_match = re.match(r"(?:(\d+):)?(\d+):(\d+)", arg)
            if time_match:
                hours = int(time_match.group(1) or 0)
                mins = int(time_match.group(2))
                secs = int(time_match.group(3))
                start_seconds = hours * 3600 + mins * 60 + secs
            else:
                console.print(f"  [{DIM}]Usage: .play  |  .play 01:25  |  .play segment 3[/]")
                return

    # Build ffplay command
    cmd = ["ffplay", "-nodisp", "-autoexit", str(file_path)]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])

    # Duration limit for segment playback
    if arg and arg.lower().startswith("segment"):
        segments = rec.get("segments", [])
        seg_idx = int(re.match(r"segment\s+(\d+)", arg).group(1))
        matched = [s for s in segments if s.get("index") == seg_idx]
        if matched:
            duration = (matched[0].get("end", 0) or 0) - start_seconds
            if duration > 0:
                cmd.extend(["-t", str(duration)])

    console.print(
        f"  [{ACCENT}]▶[/] Playing: {rec.get('file_name', '?')}"
        + (
            f" from {int(start_seconds // 60):02d}:{int(start_seconds % 60):02d}"
            if start_seconds > 0
            else ""
        )
    )
    console.print(f"  [{DIM}]Press 'q' to stop[/]")

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        console.print(f"  [{WARNING}]ffplay not found. Install ffmpeg to use playback.[/]")
    except KeyboardInterrupt:
        console.print(f"  [{DIM}]Playback stopped[/]")


def _dot_edit(session: ReplSession) -> None:
    """Open transcript text in $EDITOR, save changes back to DB."""
    rec = session.context_record
    full_text = rec.get("full_text", "")

    editor = os.environ.get("EDITOR", "nano")

    # Write current text to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_transcript_{rec['id']}.txt",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(full_text)
        tmp_path = tmp.name

    try:
        # Open in editor
        console.print(f"  [{DIM}]Opening #{rec['id']} in {editor}...[/]")
        result = subprocess.run([editor, tmp_path])

        if result.returncode != 0:
            console.print(f"  [{WARNING}]Editor exited with code {result.returncode}[/]")
            return

        # Read back edited text
        with open(tmp_path, encoding="utf-8") as f:
            new_text = f.read()

        # Compare
        if new_text == full_text:
            console.print(f"  [{DIM}]No changes made[/]")
            return

        # Word count diff
        old_wc = len(full_text.split())
        new_wc = len(new_text.split())
        diff = new_wc - old_wc

        # Save to DB
        repo = session._get_repo()
        ok = repo.update_text(rec["id"], new_text)
        if ok:
            console.print(
                f"  [{SUCCESS}]✓[/] Transcript #{rec['id']} updated ({new_wc} words, {diff:+d})"
            )
            session.refresh_context()
        else:
            console.print(f"  [{WARNING}]Failed to save changes[/]")
    finally:
        # Clean up temp file
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _dot_path(rec: dict) -> None:
    """Show the source audio file path."""
    file_path = rec.get("file_path")
    if file_path:
        exists = Path(file_path).exists()
        status = f"[{SUCCESS}]exists[/]" if exists else f"[{WARNING}]not found[/]"
        console.print(f"  [{ACCENT}]{file_path}[/]  ({status})")
    else:
        console.print(f"  [{DIM}]No source file path (live session?)[/]")


def _dot_open(rec: dict) -> None:
    """Open the source audio file in the default system player."""
    file_path = rec.get("file_path")
    if not file_path or not Path(file_path).exists():
        console.print(f"  [{WARNING}]Source audio not found: {file_path or 'unknown'}[/]")
        return

    console.print(f"  [{ACCENT}]Opening:[/] {file_path}")
    try:
        # Linux: xdg-open, macOS: open, Windows: start
        if sys.platform == "darwin":
            subprocess.Popen(["open", file_path])
        elif sys.platform == "win32":
            os.startfile(file_path)  # noqa: S606
        else:
            subprocess.Popen(
                ["xdg-open", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as e:
        console.print(f"  [{WARNING}]{e}[/]")


def _navigate_context(session: ReplSession, direction: int) -> None:
    """Navigate to .next or .prev transcript in history."""
    if not session._history_ids:
        session._load_history_ids()

    if not session._history_ids:
        console.print(f"  [{DIM}]No transcription history[/]")
        return

    if session._history_cursor < 0:
        # Not in navigation yet — start from the newest or oldest
        if direction > 0:
            new_cursor = 0
        else:
            new_cursor = len(session._history_ids) - 1
    else:
        new_cursor = session._history_cursor + direction

    if new_cursor < 0 or new_cursor >= len(session._history_ids):
        label = "newest" if direction > 0 else "oldest"
        console.print(f"  [{DIM}]Already at the {label} transcript[/]")
        return

    session._history_cursor = new_cursor
    new_id = session._history_ids[new_cursor]
    session.set_context(new_id)
    if session.context_record:
        pos = f"{new_cursor + 1}/{len(session._history_ids)}"
        console.print(
            f"  [{SUCCESS}]✓[/] [{ACCENT}]#{new_id}[/] — "
            f"{session.context_record.get('file_name', '?')} "
            f"[{DIM}]({pos})[/]"
        )


def _print_dot_help() -> None:
    """Print dot-command reference, grouped by function."""
    groups = {
        "View & Analyze": [
            ".stats",
            ".show",
            ".segments",
            ".vocab",
            ".info",
            ".find",
        ],
        "Audio": [".play", ".open", ".path"],
        "AI": [".ask", ".chat", ".summarize"],
        "Actions": [".export", ".edit"],
        "Navigation": [
            ".use",
            ".clear",
            ".next",
            ".prev",
            ".recent",
        ],
        "Search": [".search"],
        "Meta": [".help"],
    }

    console.print(f"\n  [{BOLD}][{ACCENT}]Dot Commands[/][/]")
    console.print(f"  [{DIM}]Operate on the current context transcript[/]\n")
    for group_name, group_cmds in groups.items():
        console.print(f"  [{BOLD}]{group_name}[/]")
        for cmd_name in group_cmds:
            desc = DOT_COMMANDS.get(cmd_name, "")
            console.print(f"    [{ACCENT}]{cmd_name:<14}[/] {desc}")
        console.print()


def _print_context_summary(session: ReplSession) -> None:
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


# ── Slash / Meta Commands ───────────────────────────────────


ALIASES = {
    "help": "/help",
    "exit": "/exit",
    "quit": "/exit",
    "q": "/exit",
    "clear": "/clear",
    "commands": "/commands",
}


def _handle_slash_command(cmd: str, session: ReplSession) -> bool:
    """Handle a meta command. Returns True to exit REPL."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()

    if command in ("/exit", "/quit", "/q"):
        return True

    elif command in ("/help", "/h", "/?"):
        _print_full_help(session)

    elif command == "/clear":
        os.system("clear" if os.name != "nt" else "cls")

    elif command == "/commands":
        cmds = sorted(session.cli_group.commands.keys())
        console.print(f"\n  [{BOLD}][{ACCENT}]Commands ({len(cmds)})[/][/]\n")
        categories = {
            "Core": [
                "transcribe",
                "subtitle",
                "listen",
                "speak",
                "download-voice",
            ],
            "AI": ["summarize", "ask", "chat"],
            "Data": [
                "history",
                "search",
                "show",
                "export",
                "delete",
            ],
            "Config": ["preset", "install-completion"],
            "Analytics": ["vocab"],
            "System": [
                "download",
                "info",
                "doctor",
                "status",
                "cleanup",
            ],
            "Interactive": ["repl"],
        }
        for cat, cat_cmds in categories.items():
            available = [c for c in cat_cmds if c in cmds]
            if available:
                cmd_str = "  ".join(f"[{ACCENT}]{c}[/]" for c in available)
                console.print(f"    [{BOLD}]{cat:<12}[/] {cmd_str}")
        console.print(f"\n  [{DIM}]Type <command> --help for details[/]\n")

    elif command == "/context":
        if session.context_record:
            _print_context_summary(session)
        else:
            console.print(f"  [{DIM}]No context set. Use .use <ID> or transcribe a file.[/]")

    else:
        console.print(f"  [{DIM}]Unknown: {command} — type /help or help[/]")

    return False


# ── Command Dispatch ────────────────────────────────────────


def _dispatch_command(session: ReplSession, args: list[str]) -> None:
    """Dispatch a command to the Click CLI group."""
    args = session.expand_vars(args)
    args = session.auto_inject_id(args)

    try:
        session.cli_group(args, standalone_mode=False)

        # After transcribe, auto-capture the new ID
        if args and args[0] == "transcribe":
            _try_capture_last_id(session)

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


def _try_capture_last_id(session: ReplSession) -> None:
    """After a transcribe command, grab the newest ID."""
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=1)
        if records:
            new_id = records[0]["id"]
            session.set_context(new_id)
            _print_context_summary(session)
            # Refresh navigation IDs
            session._load_history_ids()
    except Exception:
        pass


# ── Help ────────────────────────────────────────────────────


def _print_full_help(session: ReplSession) -> None:
    """Print comprehensive REPL help."""
    ctx_label = f"#{session.last_id}" if session.last_id else "none"
    ctx_name = ""
    if session.context_record:
        fname = session.context_record.get("file_name", "")
        ctx_name = f" ({fname})" if fname else ""

    console.print(f"""
  [{BOLD}][{ACCENT}]{APP_NAME} REPL — Interactive Shell[/][/]
  [{DIM}]{"─" * 50}[/]

  [{BOLD}]1. Run any command[/] [{DIM}](without 'audiobench' prefix)[/]
     [{ACCENT}]transcribe interview.mp3[/]     Transcribe a file
     [{ACCENT}]history --tail 5[/]            Recent transcriptions
     [{ACCENT}]search "meeting"[/]            Search transcripts
     [{ACCENT}]doctor[/]                       Check system health

  [{BOLD}]2. Context-aware[/] [{DIM}](current: {ctx_label}{ctx_name})[/]
     When context is set, these auto-fill the transcript ID:
     [{ACCENT}]show[/]   [{ACCENT}]ask "..."[/]   [{ACCENT}]summarize[/]   \
[{ACCENT}]vocab[/]   [{ACCENT}]export -f srt[/]

  [{BOLD}]3. Dot commands[/] [{DIM}](quick actions on context)[/]
     [{ACCENT}].stats[/]  [{ACCENT}].show[/]  [{ACCENT}].segments[/]  \
[{ACCENT}].vocab[/]  [{ACCENT}].info[/]  [{ACCENT}].find "..."[/]
     [{ACCENT}].play[/]  [{ACCENT}].play 01:25[/]  [{ACCENT}].play segment 3[/]  \
[{ACCENT}].open[/]  [{ACCENT}].path[/]
     [{ACCENT}].ask "..."[/]  [{ACCENT}].chat[/]  [{ACCENT}].summarize[/]  \
[{ACCENT}].export srt[/]  [{ACCENT}].edit[/]
     [{ACCENT}].use <ID>[/]  [{ACCENT}].clear[/]  [{ACCENT}].next[/]  \
[{ACCENT}].prev[/]  [{ACCENT}].recent[/]  [{ACCENT}].search "..."[/]

  [{BOLD}]4. Shortcuts[/]
     [{ACCENT}]$last[/]  Expands to context ID ({ctx_label})
     [{ACCENT}]? ...[/]  AI question shorthand: \
[{ACCENT}]? what are the key points?[/]

  [{BOLD}]5. Meta[/]
     [{ACCENT}]help[/] [{ACCENT}]/help[/]  This help       \
[{ACCENT}]/commands[/]  All commands
     [{ACCENT}]/clear[/]  Clear screen     \
[{ACCENT}]/context[/]  Show context
     [{ACCENT}]/exit[/]  Quit             \
[{ACCENT}]!<cmd>[/]    Shell escape
""")


# ── Onboarding ──────────────────────────────────────────────


def _print_onboarding(session: ReplSession) -> None:
    """Show a helpful onboarding for new or returning users."""
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=5)
        if records:
            console.print(f"  [{BOLD}]Recent transcriptions:[/]")
            for r in records:
                duration = r.get("duration", 0) or 0
                dur_str = format_duration(duration) if duration else "?"
                console.print(
                    f"    [{ACCENT}]#{r['id']:<4}[/] "
                    f"{r.get('file_name', '?'):<30} "
                    f"[{DIM}]{r.get('word_count', 0):>5,} words "
                    f" {dur_str:>8}[/]"
                )
            console.print(
                f"\n  [{DIM}]Tip:[/] "
                f"[{ACCENT}].use {records[0]['id']}[/] "
                f"[{DIM}]to set context, or[/] "
                f"[{ACCENT}]transcribe <file>[/] "
                f"[{DIM}]for a new one. Type[/] "
                f"[{ACCENT}]help[/] [{DIM}]for guide.[/]"
            )
        else:
            console.print(
                f"  [{DIM}]No transcriptions yet. "
                f"Start with:[/] "
                f"[{ACCENT}]transcribe <audio_file>[/]"
            )
    except Exception:
        console.print(
            f"  [{DIM}]Type[/] [{ACCENT}]help[/] "
            f"[{DIM}]for commands, or[/] "
            f"[{ACCENT}]transcribe <file>[/] "
            f"[{DIM}]to begin.[/]"
        )
    console.print()


# ── Tab Completion ──────────────────────────────────────────


def _setup_completion(session: ReplSession) -> None:
    """Set up readline tab-completion."""
    commands = sorted(session.cli_group.commands.keys())
    dot_cmds = sorted(DOT_COMMANDS.keys())
    meta_words = ["help", "exit", "quit", "clear", "commands"]
    slash_cmds = [
        "/help",
        "/commands",
        "/clear",
        "/exit",
        "/context",
    ]
    all_completions = commands + dot_cmds + slash_cmds + meta_words

    def completer(text: str, state: int) -> str | None:
        matches = [c for c in all_completions if c.startswith(text)]
        if not matches:
            import glob

            matches = glob.glob(text + "*")
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")


# ── Banner ──────────────────────────────────────────────────


def _print_banner(session: ReplSession) -> None:
    cmd_count = len(session.cli_group.commands)
    dot_count = len(DOT_COMMANDS)

    console.print(f"""
  [{BOLD}][{ACCENT}]╭────────────────────────────────────────────╮[/][/]
  [{BOLD}][{ACCENT}]│  {APP_NAME} REPL{" " * 31}│[/][/]
  [{BOLD}][{ACCENT}]│  v{APP_VERSION}  •  {cmd_count} commands  \
•  {dot_count} dot-commands{" " * 6}│[/][/]
  [{BOLD}][{ACCENT}]╰────────────────────────────────────────────╯[/][/]
""")


# ── Main REPL ───────────────────────────────────────────────


@click.command("repl")
@click.argument("transcript_id", required=False, type=int, default=None)
def repl(transcript_id: int | None) -> None:
    """Start the interactive AudioBench shell.

    \b
    Start fresh:
      audiobench repl

    \b
    Start with a transcript loaded:
      audiobench repl 42

    \b
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
    session.load_history()
    session._load_history_ids()
    _setup_completion(session)
    _print_banner(session)

    # Pre-load context if transcript ID given
    if transcript_id is not None:
        session.set_context(transcript_id)
        if session.context_record:
            _print_context_summary(session)
            console.print()

    # Onboarding: show recent transcripts
    _print_onboarding(session)

    # ── Input loop ──
    while True:
        try:
            user_input = input(session.prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            session.save_history()
            _print_goodbye(session)
            break

        if not user_input:
            continue

        # ── AI shorthand: ? what are the key points? ──
        if user_input.startswith("?"):
            question = user_input[1:].strip()
            if question and session.last_id is not None:
                _dispatch_command(
                    session,
                    ["ask", str(session.last_id), question],
                )
            elif question:
                console.print(f"  [{WARNING}]Set context first (.use <ID>) to use ? shorthand[/]")
            else:
                console.print(f"  [{DIM}]Usage: ? what are the key points?[/]")
            continue

        # Shell escape: !command
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                try:
                    subprocess.run(shell_cmd, shell=True)
                except Exception as e:
                    console.print(f"  [{WARNING}]{e}[/]")
            continue

        # Dot commands: .stats, .show, etc.
        if user_input.startswith("."):
            _handle_dot_command(user_input, session)
            continue

        # Slash commands: /help, /exit, etc.
        if user_input.startswith("/"):
            should_exit = _handle_slash_command(user_input, session)
            if should_exit:
                session.save_history()
                _print_goodbye(session)
                break
            continue

        # Bare aliases: help, exit, quit, clear, commands
        if user_input.lower() in ALIASES:
            mapped = ALIASES[user_input.lower()]
            if mapped == "/exit":
                session.save_history()
                _print_goodbye(session)
                break
            _handle_slash_command(mapped, session)
            continue

        # Parse command line
        try:
            args = shlex.split(user_input)
        except ValueError as e:
            console.print(f"  [{WARNING}]Parse error: {e}[/]")
            continue

        if not args:
            continue

        # Strip 'audiobench' prefix if user types it out of habit
        cmd_name = args[0]
        if cmd_name == "audiobench" and len(args) > 1:
            args = args[1:]
            cmd_name = args[0]

        # Block nested REPL
        if cmd_name == "repl":
            console.print(f"  [{DIM}]Already in REPL. Type help for usage or /exit to quit.[/]")
            continue

        # Check if it's a known command
        if cmd_name not in session.cli_group.commands:
            close = difflib.get_close_matches(
                cmd_name,
                list(session.cli_group.commands.keys()),
                n=1,
                cutoff=0.5,
            )
            suggestion = f"  Did you mean: [{ACCENT}]{close[0]}[/]?" if close else ""
            console.print(
                f"  [{WARNING}]Unknown command:[/] {cmd_name}\n"
                f"  [{DIM}]Type[/] [{ACCENT}]help[/] "
                f"[{DIM}]for usage or[/] "
                f"[{ACCENT}]/commands[/] "
                f"[{DIM}]to see all commands[/]"
            )
            if suggestion:
                console.print(suggestion)
            continue

        _dispatch_command(session, args)


def _print_goodbye(session: ReplSession) -> None:
    if session._command_count > 0:
        console.print(f"  [{DIM}]Session: {session._command_count} command(s) run[/]")
    console.print(f"  [{DIM}]Goodbye![/]\n")
