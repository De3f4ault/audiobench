"""Interactive Reader and Viewer REPL for AudioBench Transcripts.

Provides an interactive console powered by prompt_toolkit:
- 2-column slash-command autocomplete and auto-suggestions
- Live format switching (/format txt|srt|vtt|json|tsv)
- In-transcript full-text search (/find <query>)
- Synchronized audio playback (/play [MM:SS])
- Pager reading (/page)
- Metadata inspection (/info, /chapters)
- Instant clipboard copy (/copy) and file export (/export)
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    BOX_STYLE,
    DIM,
    SUCCESS,
    WARNING,
    console as default_console,
    error_panel,
    format_duration,
)
from audiobench.cli.shared.repl_shell import make_prompt_session
from audiobench.cli.shared.transcript_resolver import (
    list_recent_transcripts,
    resolve_transcript_target,
)
from audiobench.cli.shared.transcript_views import (
    copy_to_clipboard,
    export_transcript_to_file,
    format_transcript,
    render_transcript_header,
    search_transcript_segments,
)
from audiobench.core.settings import get_settings
from audiobench.playback.controls import is_bg_exit_requested, pause_playback_if_owned
from audiobench.storage.repository import TranscriptionRepository


# ── Slash Command Metadata ───────────────────────────────────────────────────

SHOW_COMMANDS: dict[str, tuple[str, str]] = {
    "/open": ("<id|title>", "Switch to a different transcript"),
    "/format": ("<txt|srt|vtt|json|tsv>", "Change output display format"),
    "/find": ("<query>", "Search text within the current transcript"),
    "/play": ("[MM:SS]", "Play audio from start or timestamp"),
    "/pause": ("", "Pause audio playback and save bookmark"),
    "/resume": ("", "Resume playback at saved bookmark"),
    "/timestamps": ("[on|off]", "Toggle timestamp prefixes"),
    "/speakers": ("[on|off]", "Toggle speaker labels"),
    "/page": ("", "Open full transcript in scrollable terminal pager"),
    "/info": ("", "Display detailed audio and model metadata"),
    "/chapters": ("", "List audio chapters and jump offsets"),
    "/raw": ("", "Toggle between raw Whisper and refined text"),
    "/copy": ("", "Copy current formatted transcript to clipboard"),
    "/export": ("[format] [path]", "Export transcript to file on disk"),
    "/clear": ("", "Clear screen and reprint header"),
    "/help": ("", "Show interactive command guide"),
    "/exit": ("", "Exit the interactive viewer"),
    "/quit": ("", "Exit the interactive viewer"),
}

FORMAT_CHOICES = ["txt", "srt", "vtt", "json", "tsv", "csv", "raw"]


def _pad_meta(desc: str) -> str:
    return f"   {desc}" if desc else ""


class ShowSlashCompleter(Completer):
    """Context-aware autocomplete for the interactive Show REPL."""

    def __init__(self) -> None:
        self._cached_recent: list[dict[str, Any]] = []

    def refresh_recent(self) -> None:
        try:
            self._cached_recent = list_recent_transcripts(limit=15, session=None)
        except Exception:
            self._cached_recent = []

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        tokens = text_before.split()
        word = document.get_word_before_cursor(WORD=True)

        # ── 1. Typing the slash command name ────────────────────────────────
        if text_before.startswith("/") and (
            len(tokens) == 0 or (len(tokens) == 1 and not text_before.endswith(" "))
        ):
            for cmd, (syntax, desc) in SHOW_COMMANDS.items():
                if cmd.startswith(word):
                    meta = f"{syntax} · {desc}" if syntax else desc
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display=cmd,
                        display_meta=_pad_meta(meta),
                    )
            return

        if not text_before.startswith("/"):
            # If user hasn't typed a slash, suggest quick commands when line starts
            if len(tokens) <= 1 and not text_before.endswith(" "):
                for cmd in ("/open", "/format", "/find", "/play", "/page", "/help"):
                    if cmd.startswith(word):
                        syntax, desc = SHOW_COMMANDS[cmd]
                        meta = f"{syntax} · {desc}" if syntax else desc
                        yield Completion(
                            cmd,
                            start_position=-len(word),
                            display=cmd,
                            display_meta=_pad_meta(meta),
                        )
            return

        # ── 2. Completing arguments after a slash command ────────────────────
        cmd_name = tokens[0].lower()

        # /format [txt|srt|vtt|json|tsv|csv|raw]
        if cmd_name == "/format":
            prefix = word if not text_before.endswith(" ") else ""
            for fmt in FORMAT_CHOICES:
                if not prefix or fmt.startswith(prefix):
                    yield Completion(
                        fmt,
                        start_position=-len(prefix) if prefix else 0,
                        display=fmt,
                        display_meta=_pad_meta(f"{fmt.upper()} format"),
                    )
            return

        # /timestamps [on|off|toggle]
        if cmd_name == "/timestamps":
            prefix = word if not text_before.endswith(" ") else ""
            for opt, desc in [("on", "Show timestamps"), ("off", "Hide timestamps"), ("toggle", "Toggle state")]:
                if not prefix or opt.startswith(prefix):
                    yield Completion(
                        opt,
                        start_position=-len(prefix) if prefix else 0,
                        display=opt,
                        display_meta=_pad_meta(desc),
                    )
            return

        # /speakers [on|off|toggle]
        if cmd_name == "/speakers":
            prefix = word if not text_before.endswith(" ") else ""
            for opt, desc in [("on", "Show speakers"), ("off", "Hide speakers"), ("toggle", "Toggle state")]:
                if not prefix or opt.startswith(prefix):
                    yield Completion(
                        opt,
                        start_position=-len(prefix) if prefix else 0,
                        display=opt,
                        display_meta=_pad_meta(desc),
                    )
            return

        # /open <target> — complete recent transcripts
        if cmd_name == "/open":
            if not self._cached_recent:
                self.refresh_recent()

            prefix = word if not text_before.endswith(" ") else ""
            # Keyword completions
            for kw in ("latest", "last", "-1", "-2"):
                if not prefix or kw.startswith(prefix):
                    yield Completion(
                        kw,
                        start_position=-len(prefix) if prefix else 0,
                        display=kw,
                        display_meta=_pad_meta("Relative target"),
                    )

            # Recent transcript IDs and titles
            for r in self._cached_recent:
                t_id = str(r["id"])
                t_name = r.get("file_name", "untitled")
                if not prefix or t_id.startswith(prefix.lstrip("#")):
                    yield Completion(
                        t_id,
                        start_position=-len(prefix) if prefix else 0,
                        display=f"#{t_id}",
                        display_meta=_pad_meta(t_name[:35]),
                    )
            return

        # /export <format> <path>
        if cmd_name == "/export" and (len(tokens) == 1 or (len(tokens) == 2 and not text_before.endswith(" "))):
            prefix = word if not text_before.endswith(" ") else ""
            for fmt in FORMAT_CHOICES:
                if not prefix or fmt.startswith(prefix):
                    yield Completion(
                        fmt,
                        start_position=-len(prefix) if prefix else 0,
                        display=fmt,
                        display_meta=_pad_meta("Export format"),
                    )
            return


class ShowAutoSuggest(AutoSuggest):
    """Contextual ghost suggestions for show command."""

    def get_suggestion(self, buffer, document) -> Suggestion | None:
        text = document.text
        if not text:
            return Suggestion("/page")
        if text == "/":
            return Suggestion("page")
        if text.startswith("/fo"):
            return Suggestion("rmat srt"[len(text) - 3:])
        if text.startswith("/fi"):
            return Suggestion("nd "[len(text) - 3:])
        if text.startswith("/pl"):
            return Suggestion("ay"[len(text) - 3:])
        return None


# ── Interactive REPL Execution ───────────────────────────────────────────────


def start_show_repl(
    initial_target: str | int | None = None,
    *,
    initial_format: str = "txt",
    timestamps: bool = True,
    speakers: bool = True,
    console: Console | None = None,
) -> None:
    """Launch the interactive Show REPL console."""
    c = console or default_console
    repo = TranscriptionRepository()

    # Resolve initial transcript
    current_record = resolve_transcript_target(initial_target, interactive=True, console=c, session=None)
    if not current_record:
        # Prompt user with picker
        recent = list_recent_transcripts(limit=10, session=None)
        if not recent:
            c.print(error_panel("No Transcriptions", "No completed transcriptions available in database."))
            return
        from audiobench.cli.shared.transcript_resolver import _prompt_transcript_picker
        current_record = _prompt_transcript_picker(recent, console=c, repo=repo, session=None)
        if not current_record:
            return

    # View state
    active_format = initial_format
    show_timestamps = timestamps
    show_speakers = speakers
    show_raw_whisper = False

    settings = get_settings()
    hist_path = settings.data_dir / "show_history.txt"

    completer = ShowSlashCompleter()
    completer.refresh_recent()
    auto_suggest = ShowAutoSuggest()

    pt_session = make_prompt_session(
        history_path=hist_path,
        slash_completer=completer,
        auto_suggest=auto_suggest,
    )

    # Initial Welcome Banner
    c.print()
    c.print(f"  [{BOLD} {ACCENT}]AudioBench Transcript Viewer[/]  [{DIM}]· Type [bold]/help[/bold] for commands or [bold]/exit[/bold] to quit[/]")
    _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

    # REPL Loop
    while True:
        try:
            prompt_label = ANSI(
                f"\033[1;36mshow\033[0m \033[2m[#{current_record['id']}|{active_format}]\033[0m \033[1;34m›\033[0m "
            )
            raw_input_line = pt_session.prompt(prompt_label).strip()
        except (KeyboardInterrupt, EOFError):
            _pause_transcript_playback(current_record, console=c, force_bg=False, reason="exit")
            c.print("\n  [dim]Exiting viewer.[/dim]")
            break

        if not raw_input_line:
            continue

        parts = shlex.split(raw_input_line)
        if not parts:
            continue

        cmd = parts[0].lower()
        args = parts[1:]

        # ── Commands ────────────────────────────────────────────────────────
        if cmd in ("/exit", "/quit", ":q", "exit", "quit"):
            bg = is_bg_exit_requested(args)
            _pause_transcript_playback(current_record, console=c, force_bg=bg, reason="exit")
            c.print("  [dim]Goodbye.[/dim]")
            break

        elif cmd in ("/help", "/?"):
            _render_help_screen(console=c)

        elif cmd in ("/clear", "/cls"):
            c.clear()
            _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

        elif cmd == "/open":
            if not args:
                # Open interactive picker
                recent = list_recent_transcripts(limit=10, session=None)
                from audiobench.cli.shared.transcript_resolver import _prompt_transcript_picker
                picked = _prompt_transcript_picker(recent, console=c, repo=repo, session=None)
                if picked:
                    if picked.get("id") != current_record.get("id"):
                        _pause_transcript_playback(current_record, console=c, force_bg=False, reason="switch")
                    current_record = picked
                    completer.refresh_recent()
                    _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)
            else:
                target_arg = " ".join(args)
                found = resolve_transcript_target(target_arg, interactive=True, console=c, session=None)
                if found:
                    if found.get("id") != current_record.get("id"):
                        _pause_transcript_playback(current_record, console=c, force_bg=False, reason="switch")
                    current_record = found
                    completer.refresh_recent()
                    _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

        elif cmd == "/format":
            if not args:
                c.print(f"  [{DIM}]Current format:[/] [{ACCENT}]{active_format}[/]")
                c.print(f"  [{DIM}]Available:[/] {', '.join(FORMAT_CHOICES)}")
            else:
                new_fmt = args[0].lower()
                if new_fmt in FORMAT_CHOICES or new_fmt in ("text",):
                    active_format = "txt" if new_fmt == "text" else new_fmt
                    c.print(f"  [{SUCCESS}]Format set to:[/] [{ACCENT}]{active_format}[/]")
                    _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)
                else:
                    c.print(error_panel("Invalid Format", f"'{new_fmt}' is not supported. Use one of: {', '.join(FORMAT_CHOICES)}"))

        elif cmd == "/timestamps":
            if args:
                opt = args[0].lower()
                show_timestamps = (opt in ("on", "true", "1", "yes")) if opt != "toggle" else not show_timestamps
            else:
                show_timestamps = not show_timestamps
            status = "ON" if show_timestamps else "OFF"
            c.print(f"  [{SUCCESS}]Timestamps:[/] [{ACCENT}]{status}[/]")
            _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

        elif cmd == "/speakers":
            if args:
                opt = args[0].lower()
                show_speakers = (opt in ("on", "true", "1", "yes")) if opt != "toggle" else not show_speakers
            else:
                show_speakers = not show_speakers
            status = "ON" if show_speakers else "OFF"
            c.print(f"  [{SUCCESS}]Speaker labels:[/] [{ACCENT}]{status}[/]")
            _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

        elif cmd == "/raw":
            show_raw_whisper = not show_raw_whisper
            status = "RAW (Pre-Refinement)" if show_raw_whisper else "REFINED"
            c.print(f"  [{SUCCESS}]Display mode:[/] [{ACCENT}]{status}[/]")
            _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)

        elif cmd in ("/page", "/less", "/view"):
            content = format_transcript(
                current_record,
                format_type=active_format,
                timestamps=show_timestamps,
                speakers=show_speakers,
                show_raw_whisper=show_raw_whisper,
            )
            with c.pager(styles=True):
                c.print(content, highlight=False)

        elif cmd == "/find":
            if not args:
                c.print(f"  [{DIM}]Usage: /find <search query>[/]")
            else:
                q = " ".join(args)
                _run_in_transcript_search(current_record, q, console=c)

        elif cmd == "/play":
            seek_seconds = None
            if args:
                seek_str = args[0]
                m = re.match(r"(?:(\d+):)?(\d+):(\d+)", seek_str)
                if m:
                    h = int(m.group(1) or 0)
                    mins = int(m.group(2))
                    s = int(m.group(3))
                    seek_seconds = float(h * 3600 + mins * 60 + s)
                elif seek_str.replace(".", "", 1).isdigit():
                    seek_seconds = float(seek_str)
            _play_transcript_audio(current_record, start_seconds=seek_seconds, console=c)

        elif cmd in ("/pause", "/stop"):
            _pause_transcript_playback(current_record, console=c, force_bg=False, reason="manual")

        elif cmd == "/resume":
            _play_transcript_audio(current_record, start_seconds=None, console=c)

        elif cmd == "/info":
            _render_info_panel(current_record, console=c)

        elif cmd == "/chapters":
            _render_chapters_panel(current_record, console=c)

        elif cmd == "/copy":
            content = format_transcript(
                current_record,
                format_type=active_format,
                timestamps=show_timestamps,
                speakers=show_speakers,
                show_raw_whisper=show_raw_whisper,
            )
            ok = copy_to_clipboard(content)
            if ok:
                c.print(f"  [{SUCCESS}]✓ Copied {len(content):,} characters ({active_format}) to clipboard[/]")
            else:
                c.print(f"  [{WARNING}][!] Clipboard utility not available (install wl-copy or xclip).[/]")

        elif cmd == "/export":
            export_fmt = args[0] if args else active_format
            export_path = args[1] if len(args) > 1 else None

            if not export_path:
                ext = "txt" if export_fmt in ("text", "txt") else export_fmt
                sanitized_title = re.sub(r"[^\w\-.]", "_", current_record.get("file_name", "transcript"))
                default_name = f"{sanitized_title}_#{current_record['id']}.{ext}"
                export_path = Path.cwd() / default_name

            try:
                saved = export_transcript_to_file(
                    current_record,
                    format_type=export_fmt,
                    output_path=export_path,
                    timestamps=show_timestamps,
                    speakers=show_speakers,
                )
                c.print(f"  [{SUCCESS}]✓ Exported {export_fmt.upper()} to:[/] [{ACCENT}]{saved}[/]")
            except Exception as exc:
                c.print(error_panel("Export Failed", str(exc)))

        else:
            # Check if input is a direct transcript ID (e.g., "84" or "#84")
            raw_target = raw_input_line.lstrip("#")
            if raw_target.isdigit():
                found = resolve_transcript_target(int(raw_target), interactive=True, console=c, session=None)
                if found:
                    if found.get("id") != current_record.get("id"):
                        _pause_transcript_playback(current_record, console=c, force_bg=False, reason="switch")
                    current_record = found
                    completer.refresh_recent()
                    _render_transcript_card(current_record, active_format, show_timestamps, show_speakers, show_raw_whisper, console=c)
                    continue

            # Fallback: treat plain text without slash as an instant in-transcript search!
            _run_in_transcript_search(current_record, raw_input_line, console=c)


# ── Render Helpers ───────────────────────────────────────────────────────────


def _render_transcript_card(
    data: dict[str, Any],
    format_type: str,
    timestamps: bool,
    speakers: bool,
    show_raw_whisper: bool,
    console: Console,
) -> None:
    """Render the top transcript summary card with a quick preview."""
    console.print()
    header = render_transcript_header(data)
    console.print(header)
    console.print(f"[{DIM}]{'─' * 60}[/]")

    # Quick preview (up to 8 lines)
    content = format_transcript(
        data,
        format_type=format_type,
        timestamps=timestamps,
        speakers=speakers,
        show_raw_whisper=show_raw_whisper,
    )

    lines = content.splitlines()
    preview_count = 8
    preview = "\n".join(lines[:preview_count])

    console.print(preview, highlight=False)
    if len(lines) > preview_count:
        console.print(f"  [{DIM}]... ({len(lines) - preview_count} more lines · Type [bold]/page[/bold] to read full transcript)[/]")
    console.print()


def _render_help_screen(console: Console) -> None:
    """Display the interactive command guide table."""
    table = Table(
        title="Interactive Transcript Viewer Commands",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("Command", style=BOLD, width=15)
    table.add_column("Arguments", style=ACCENT, width=22)
    table.add_column("Description")

    for cmd, (args, desc) in SHOW_COMMANDS.items():
        table.add_row(cmd, args, desc)

    console.print()
    console.print(table)
    console.print(f"  [{DIM}]Tip: Typing plain text without a slash instantly searches inside the active transcript.[/]\n")


def _run_in_transcript_search(data: dict[str, Any], query: str, console: Console) -> None:
    """Search within active transcript segments and display matching hits."""
    hits = search_transcript_segments(data, query)
    if not hits:
        console.print(f"  [{WARNING}]No matches found for '{query}' in this transcript.[/]\n")
        return

    table = Table(
        title=f"Matches for '{query}' ({len(hits)} found)",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("Time", style=BOLD, width=10)
    table.add_column("Speaker", style=DIM, width=12)
    table.add_column("Snippet")

    for i, h in enumerate(hits[:25], 1):
        spk = f"<{h['speaker']}>" if h.get("speaker") else "—"
        text = h["text"].strip()
        # Highlight match
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(f"[{BOLD} {ACCENT}]\\g<0>[/]", text)
        table.add_row(str(i), h["time_str"], spk, highlighted)

    console.print()
    console.print(table)
    if len(hits) > 25:
        console.print(f"  [{DIM}]Showing first 25 of {len(hits)} matches.[/]")
    console.print(f"  [{DIM}]Tip: Run [bold]/play {hits[0]['time_str']}[/bold] to listen from the first match.[/]\n")


def _pause_transcript_playback(
    record: dict[str, Any] | None,
    console: Console | None = None,
    force_bg: bool = False,
    reason: str = "exit",
) -> None:
    """Pause playback and save bookmark for record if currently playing in daemon."""
    try:
        from audiobench.daemon.factory import get_daemon_client
        client = get_daemon_client()
    except Exception:
        return

    tid = record.get("id") if record else None
    pause_playback_if_owned(
        client,
        tid,
        console or default_console,
        reason=reason,
        force_bg=force_bg,
    )


def _play_transcript_audio(record: dict[str, Any], start_seconds: float | None = None, console: Console | None = None) -> None:
    """Play source audio via daemon client or fallback to local mpv."""
    c = console or default_console
    source = record.get("file_path")
    if not source or not Path(source).exists():
        c.print(error_panel("Audio File Missing", f"Path '{source or 'unknown'}' not found on disk."))
        return

    tid = record.get("id")
    afid = record.get("audio_file_id")

    # 1. Try daemon client
    try:
        from audiobench.daemon.factory import get_daemon_client
        client = get_daemon_client()
        st = client.playback_status()

        # If already playing/paused this transcript and no timestamp requested
        if start_seconds is None:
            if st.get("transcription_id") == tid:
                is_paused = bool(st.get("paused") or st.get("state") == "paused")
                is_playing = bool((st.get("playing") or st.get("state") == "playing") and not is_paused)
                if is_paused:
                    client.playback_resume()
                    pos = float(st.get("position", 0.0))
                    c.print(f"  [{SUCCESS}]> Resumed {record.get('file_name')} @ {format_duration(pos)}[/]\n")
                    return
                elif is_playing:
                    pos = float(st.get("position", 0.0))
                    c.print(f"  [{DIM}]Already playing {record.get('file_name')} @ {format_duration(pos)}[/]\n")
                    return

            # Determine start position from saved resume bookmark
            from audiobench.storage.bookmark_repository import BookmarkRepository
            saved = BookmarkRepository().get_resume_position(tid, apply_courtesy_rollback=True)
            actual_start = saved if saved is not None else 0.0
            source_desc = f"{format_duration(actual_start)} (resumed)" if saved is not None else format_duration(actual_start)
        else:
            actual_start = start_seconds
            source_desc = format_duration(actual_start)

        client.playback_play(
            str(source),
            start_pos=actual_start,
            audio_file_id=afid,
            transcription_id=tid,
        )
        c.print(f"  [{SUCCESS}]> Playing {record.get('file_name')} from {source_desc} via background daemon[/]\n")
        return
    except Exception:
        pass

    # 2. Fallback to direct mpv
    fallback_start = start_seconds if start_seconds is not None else 0.0
    try:
        from audiobench.playback import MpvController
        mpv = MpvController()
        mpv.start()
        mpv.load_file(str(source), start_pos=fallback_start)
        c.print(f"  [{SUCCESS}]> Playing {record.get('file_name')} from {format_duration(fallback_start)} via mpv[/]\n")
    except Exception as exc:
        c.print(error_panel("Playback Unavailable", f"Could not launch audio playback: {exc}"))


def _render_info_panel(data: dict[str, Any], console: Console) -> None:
    """Display comprehensive metadata table for the transcript."""
    table = Table(
        title=f"Metadata: Transcript #{data['id']}",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("Property", style=BOLD, width=20)
    table.add_column("Value")

    table.add_row("ID", str(data["id"]))
    table.add_row("File Name", data.get("file_name", "untitled"))
    table.add_row("Source Audio", data.get("file_path") or "—")
    table.add_row("Engine", data.get("engine", "unknown"))
    table.add_row("Model", data.get("model", "unknown"))
    table.add_row("Language", f"{data.get('language', '?')} (prob: {data.get('language_probability', 0.0):.2f})")
    table.add_row("Duration", format_duration(data.get("duration", 0)))
    table.add_row("Word Count", f"{data.get('word_count', 0):,}")
    table.add_row("Segments", str(len(data.get("segments", []))))
    table.add_row("Chapters", str(len(data.get("chapters", []))))
    table.add_row("Refined At", data.get("refined_at") or "— (raw Whisper)")
    table.add_row("Created At", data.get("created_at") or "—")

    console.print()
    console.print(table)
    console.print()


def _render_chapters_panel(data: dict[str, Any], console: Console) -> None:
    """Display chapter list with offsets."""
    chapters = data.get("chapters", [])
    if not chapters:
        console.print(f"  [{DIM}]No chapters recorded for this transcript.[/]\n")
        return

    table = Table(
        title="Chapters & Markers",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("Title", style=ACCENT)
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Duration", justify="right")

    for idx, chap in enumerate(chapters, 1):
        start = format_duration(chap.get("start_time", 0))
        end = format_duration(chap.get("end_time", 0))
        dur = format_duration(max(0, chap.get("end_time", 0) - chap.get("start_time", 0)))
        table.add_row(str(idx), chap.get("title", f"Chapter {idx}"), start, end, dur)

    console.print()
    console.print(table)
    console.print(f"  [{DIM}]Tip: Use [bold]/play <start_time>[/bold] to jump to any chapter.[/]\n")
