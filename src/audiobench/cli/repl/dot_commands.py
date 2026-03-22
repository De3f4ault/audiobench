"""REPL dot-commands — quick actions on the current transcript context.

Provides:
    - DOT_COMMANDS: Registry of available dot-commands with descriptions
    - handle_dot_command(): Router for all dot-command input
    - Individual implementations: _dot_stats, _dot_show, _dot_play, etc.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    format_duration,
    make_table,
)
from audiobench.cli.repl.dispatch import dispatch_command, print_context_summary
from audiobench.cli.repl.session import ReplSession

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


# ── Typo Correction ─────────────────────────────────────────


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


# ── Main Router ─────────────────────────────────────────────


def handle_dot_command(cmd: str, session: ReplSession) -> None:
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
            print_context_summary(session)
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
        dispatch_command(session, ["history", "--tail", "5"])
        return

    if command == ".search":
        if not arg:
            console.print(f'  [{DIM}]Usage: .search "keyword"[/]')
            return
        dispatch_command(session, ["search", arg])
        return

    if command in (".help", ".commands", ".?"):
        print_dot_help()
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
        dispatch_command(session, ["vocab", str(session.last_id)] + extra)
    elif command == ".export":
        fmt = arg if arg else "srt"
        dispatch_command(session, ["export", str(session.last_id), "-f", fmt])
    elif command == ".ask":
        if not arg:
            console.print(f'  [{DIM}]Usage: .ask "question"[/]')
            return
        dispatch_command(session, ["ask", str(session.last_id), arg])
    elif command == ".chat":
        dispatch_command(session, ["chat", str(session.last_id)])
    elif command == ".summarize":
        dispatch_command(session, ["summarize", str(session.last_id)])
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
        new_cursor = 0 if direction > 0 else len(session._history_ids) - 1
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


def print_dot_help() -> None:
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
