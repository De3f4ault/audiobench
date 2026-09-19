"""Shared playback controls and utilities for AudioBench interactive modes.

Provides:
  - parse_timestamp() — parse "6:30", "1:02:30", etc. to float seconds
  - fmt_timestamp() — format float seconds to "MM:SS" or "HH:MM:SS"
  - fmt_timestamp_slug() — format float seconds to "06m30s" for clean filenames
  - extract_timestamps() — regex-scan text for cited timestamps, sorted chronologically
  - render_jump_bar() — numbered jump bar with single-key navigation
  - handle_playback_command() — dispatch /play, /seek, /pause, /speed, etc.
  - render_playback_status_line() — format compact status for prompt injection
  - PLAYBACK_COMMANDS — frozenset of supported playback slash commands
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

from rich.console import Console

from audiobench.cli.display.theme import ACCENT, DIM, ERROR, SUCCESS, WARNING
from audiobench.core.settings import get_settings

# ── Supported Playback Commands ─────────────────────────────

PLAYBACK_COMMANDS: frozenset[str] = frozenset({
    "/play",
    "/seek",
    "/pause",
    "/resume",
    "/toggle",
    "/speed",
    "/stop",
    "/np",
    "/nowplaying",
    "/status",
    "/explain",
    "/clip",
    "/next",
    "/n",
    "/prev",
    "/p",
    "/back",
    "/b",
    "/lyrics",
    "/lyric",
    "/karaoke",
    "/follow",
})


# ── Timestamp parsing & formatting ──────────────────────────


def parse_timestamp(ts: str) -> float | None:
    """Parse MM:SS or HH:MM:SS (or float seconds) into float seconds."""
    if not ts:
        return None
    s = ts.strip()
    try:
        val = float(s)
        return max(0.0, val)
    except ValueError:
        pass

    match = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$", s)
    if match:
        h = int(match.group(1) or 0)
        m = int(match.group(2))
        sec = int(match.group(3))
        ms = float(f"0.{match.group(4)}") if match.group(4) else 0.0
        return float(h * 3600 + m * 60 + sec + ms)
    return None


def fmt_timestamp(s: float) -> str:
    """Format float seconds into MM:SS or HH:MM:SS."""
    s = max(0.0, s)
    total_sec = int(s)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    sec = total_sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def fmt_timestamp_slug(s: float) -> str:
    """Format float seconds into clean slug like 06m30s or 01h02m30s."""
    s = max(0.0, s)
    total_sec = int(s)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    sec = total_sec % 60
    if h > 0:
        return f"{h:02d}h{m:02d}m{sec:02d}s"
    return f"{m:02d}m{sec:02d}s"


# ── Timestamp extraction from text ───────────────────────────

# Matches timestamp ranges like (03:19-03:28) or 03:19 - 03:28 or 03:19 → 03:28
_RANGE_PATTERN = re.compile(
    r"\b((?:\d{1,2}:)?\d{1,2}:[0-5]\d(?:\.\d+)?)\s*(?:[-–—→~]|to)\s*((?:\d{1,2}:)?\d{1,2}:[0-5]\d(?:\.\d+)?)\b",
    re.IGNORECASE,
)
# Matches timestamps like 03:30, 0:33, 1:05:00, (03:30), [03:30], Timestamp 03:30, etc.
# Ignores ISO dates (e.g. 2026-09-19 19:37:04), aspect ratios (16:9), and ports (localhost:8080).
_GENERIC_TS_PATTERN = re.compile(
    r"(?<!\d{4}-\d{2}-\d{2}[ T])(?<![:\w])((?:\d{1,2}:)?[0-5]?\d:[0-5]\d(?:\.\d+)?)(?![:\w])"
)


def extract_timestamps(text: str) -> list[tuple[str, float]]:
    """Extract cited timestamps from markdown text, sorted chronologically.

    Deduplicates citations that fall within a 3.0-second window.
    For ranges like (03:19-03:28), extracts the start timestamp as the seek anchor.
    Returns list of (display_string, seconds) tuples sorted by seconds ascending.
    """
    if not text:
        return []

    # Identify trailing ends of ranges so we jump to section/quote start anchors
    end_spans = {m.span(2) for m in _RANGE_PATTERN.finditer(text)}

    parsed: list[tuple[str, float]] = []
    for match in _GENERIC_TS_PATTERN.finditer(text):
        if match.span(1) in end_spans:
            continue
        ts_str = match.group(1)
        secs = parse_timestamp(ts_str)
        if secs is not None:
            parsed.append((ts_str, secs))

    if not parsed:
        return []

    # Sort chronologically (ascending time)
    parsed.sort(key=lambda item: item[1])

    # Deduplicate citations within 3.0 seconds
    deduped: list[tuple[str, float]] = []
    for item in parsed:
        if not deduped:
            deduped.append(item)
        else:
            last_sec = deduped[-1][1]
            if abs(item[1] - last_sec) > 3.0:
                deduped.append(item)

    return deduped


# ── Smart Timestamp Navigation & Progress Bar ────────────────


def find_next_timestamp(
    timestamps: list[tuple[str, float]],
    current_pos: float,
    tolerance: float = 1.0,
) -> tuple[int, str, float] | None:
    """Find the next upcoming timestamp after current_pos.

    Returns:
        (1-based index, display_string, seconds) or None if already at/past last citation.
    """
    for i, (ts_str, sec) in enumerate(timestamps, 1):
        if sec > current_pos + tolerance:
            return i, ts_str, sec
    return None


def find_prev_timestamp(
    timestamps: list[tuple[str, float]],
    current_pos: float,
    rewind_threshold: float = 3.0,
) -> tuple[int, str, float] | None:
    """Find the previous timestamp before current_pos with standard media player rewind behavior.

    If current_pos is > rewind_threshold (e.g. 3s) into a citation section, rewinds to the
    start of that citation. If <= rewind_threshold, jumps to the previous citation before it.

    Returns:
        (1-based index, display_string, seconds) or None if timestamps is empty.
    """
    if not timestamps:
        return None

    if current_pos < timestamps[0][1]:
        return 1, timestamps[0][0], timestamps[0][1]

    active_idx = 0
    for i, (_, sec) in enumerate(timestamps):
        if current_pos >= sec - 0.5:
            active_idx = i
        else:
            break

    curr_sec = timestamps[active_idx][1]
    if current_pos - curr_sec > rewind_threshold:
        return active_idx + 1, timestamps[active_idx][0], curr_sec

    if active_idx > 0:
        prev_idx = active_idx - 1
        return prev_idx + 1, timestamps[prev_idx][0], timestamps[prev_idx][1]

    return 1, timestamps[0][0], timestamps[0][1]


def render_progress_bar(pos: float, dur: float, width: int = 30) -> str:
    """Render a colored unicode audio progress bar.

    Example: [green]━━━━━━━━━━━━[/green][dim]░░░░░░░░░░░░░░░░░░[/dim]
    """
    if dur <= 0.0:
        return f"[{DIM}]{'─' * width}[/]"
    frac = max(0.0, min(1.0, pos / dur))
    filled_len = int(frac * width)
    empty_len = width - filled_len
    filled = "━" * filled_len
    empty = "░" * empty_len
    return f"[{SUCCESS}]{filled}[/][{DIM}]{empty}[/]"


def render_citations_banner(
    target_console: Console,
    timestamps: list[tuple[str, float]],
) -> None:
    """Render a permanent, non-erased citation list into the terminal scrollback."""
    if not timestamps:
        return
    display_ts = timestamps[:9]
    parts = [f"[{i}] {ts_str}" for i, (ts_str, _) in enumerate(display_ts, 1)]
    bar_line = f"  [{ACCENT}][>] Timestamps:[/]  " + "  ".join(parts)
    target_console.print(bar_line)
    if len(timestamps) > 9:
        target_console.print(f"  [{DIM}]+{len(timestamps) - 9} more citations — use /seek <ts> to navigate[/]")


# ── Post-response jump bar ───────────────────────────────────


def render_jump_bar(
    target_console: Console,
    timestamps: list[tuple[str, float]],
    client: Any,
    *,
    auto_bookmark: bool = False,
    bookmark_name: str | None = None,
    audio_file_id: int | None = None,
    default_record: dict | None = None,
) -> float | None:
    """Render numbered timestamp options, prompt for jump, seek audio.

    Supports single-keypress input (1-9 to seek, Space to toggle pause, Enter/Esc to skip).
    Erases the prompt after selection so scrollback remains clean.
    Returns the target seconds if jumped, None otherwise.
    """
    if not timestamps:
        return None

    # Cap display to top 9 for single-digit hotkeys
    display_ts = timestamps[:9]
    parts = [f"[{i}] {ts_str}" for i, (ts_str, _) in enumerate(display_ts, 1)]
    bar_line = f"  [{ACCENT}][>] Timestamps:[/]  " + "  ".join(parts)

    target_console.print(bar_line)
    has_overflow = len(timestamps) > 9
    if has_overflow:
        overflow_count = len(timestamps) - 9
        target_console.print(f"  [{DIM}]+{overflow_count} more — use /seek <ts> to navigate manually[/]")

    prompt_msg = f"  [{DIM}][>] Jump to (1-{len(display_ts)}, Space pause, Enter skip): [/]"

    raw = ""
    if sys.stdin.isatty():
        try:
            import termios
            import tty

            target_console.print(prompt_msg, end="")
            target_console.file.flush()

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                raw = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            try:
                raw = target_console.input(prompt_msg).strip()
            except (KeyboardInterrupt, EOFError):
                raw = ""
    else:
        try:
            raw = target_console.input(prompt_msg).strip()
        except (KeyboardInterrupt, EOFError):
            raw = ""

    # Erase the jump bar lines
    erase_lines = 3 if has_overflow else 2
    target_console.file.write(f"\033[{erase_lines}A\033[J")
    target_console.file.flush()

    # Handle spacebar toggle
    if raw == " ":
        st = client.playback_toggle()
        state = "Paused" if st.get("paused") else "Playing"
        target_console.print(f"  [{SUCCESS}][DONE] {state}[/]")
        return None

    if not raw or not raw.isdigit():
        return None

    idx = int(raw) - 1
    if 0 <= idx < len(display_ts):
        ts_str, seconds = display_ts[idx]
        st = client.playback_status()
        if not st.get("playing") and default_record and default_record.get("file_path"):
            client.playback_play(
                default_record["file_path"],
                start_pos=seconds,
                audio_file_id=default_record.get("audio_file_id"),
                transcription_id=default_record.get("id"),
            )
        else:
            client.playback_seek(seconds)

        target_console.print(f"  [{SUCCESS}][DONE] Playing from {ts_str}[/]")

        if auto_bookmark and audio_file_id:
            _create_bookmark(audio_file_id, seconds, bookmark_name or f"AI citation ({ts_str})")

        return seconds

    return None


# ── Persistence Helpers ─────────────────────────────────────


def _create_bookmark(audio_file_id: int, timestamp: float, name: str) -> None:
    """Helper to add a point bookmark."""
    try:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        repo = BookmarkRepository()
        repo.add(audio_file_id, timestamp, name=name[:80], bookmark_type="highlight")
    except Exception:  # noqa: BLE001
        pass


def _register_clip(
    audio_file_id: int,
    start: float,
    end: float,
    clip_name: str,
    clip_path: str,
    transcription_id: int | None = None,
) -> None:
    """Helper to register a rendered audio clip as a region bookmark."""
    try:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        repo = BookmarkRepository()
        repo.add_region(
            audio_file_id,
            start,
            end,
            name=clip_name[:80],
            bookmark_type="clip",
            notes=f"File: {clip_path}",
            transcription_id=transcription_id,
        )
    except Exception:  # noqa: BLE001
        pass


def _register_explain(
    audio_file_id: int,
    start: float,
    end: float,
    topic: str,
    notes: str,
    transcription_id: int | None = None,
) -> None:
    """Helper to register an explanation as a region bookmark."""
    try:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        repo = BookmarkRepository()
        repo.add_region(
            audio_file_id,
            start,
            end,
            name=topic[:80],
            bookmark_type="explain",
            notes=notes,
            transcription_id=transcription_id,
        )
    except Exception:  # noqa: BLE001
        pass


# ── Playback Slash Commands Dispatcher ───────────────────────


def handle_playback_command(
    command: str,
    arg: str,
    client: Any,
    target_console: Console,
    *,
    transcript_resolver: Callable[[int], dict | None] | None = None,
    loaded_transcript_ids: list[int] | None = None,
    on_explain_context: Callable[[str], None] | None = None,
    active_citations: list[tuple[str, float]] | None = None,
    default_record: dict | None = None,
) -> bool:
    """Handle universal playback slash commands."""
    cmd = command.lower()

    try:
        if cmd == "/play":
            return _cmd_play(arg, client, target_console, transcript_resolver, loaded_transcript_ids)
        elif cmd == "/seek":
            return _cmd_seek(arg, client, target_console)
        elif cmd in ("/pause", "/resume", "/toggle"):
            return _cmd_toggle(cmd, client, target_console)
        elif cmd == "/speed":
            return _cmd_speed(arg, client, target_console)
        elif cmd == "/stop":
            client.playback_stop()
            target_console.print(f"  [{SUCCESS}][DONE] Playback stopped[/]")
            return True
        elif cmd in ("/np", "/nowplaying", "/status"):
            return _cmd_status(client, target_console)
        elif cmd in ("/next", "/n"):
            return _cmd_next(client, active_citations, default_record, target_console)
        elif cmd in ("/prev", "/p", "/back", "/b"):
            return _cmd_prev(client, active_citations, default_record, target_console)
        elif cmd.startswith("/") and cmd[1:].isdigit():
            idx = int(cmd[1:])
            return _cmd_jump_index(idx, client, active_citations, default_record, target_console)
        elif cmd == "/explain":
            return _cmd_explain(arg, client, target_console, on_explain_context)
        elif cmd == "/clip":
            return _cmd_clip(arg, client, target_console)
        elif cmd in ("/lyrics", "/lyric", "/karaoke", "/follow"):
            return _cmd_lyrics(client, target_console, transcript_resolver)
        return False
    except Exception as exc:
        target_console.print(f"  [{WARNING}]Playback error: {exc}[/]")
        return True


def _cmd_next(
    client: Any,
    citations: list[tuple[str, float]] | None,
    default_record: dict | None,
    target_console: Console,
) -> bool:
    if not citations:
        target_console.print(f"  [{DIM}]No active citations from recent response[/]")
        return True

    st = client.playback_status()
    pos = st.get("position", 0.0)
    hit = find_next_timestamp(citations, pos)
    if not hit:
        last_ts = citations[-1][0]
        target_console.print(f"  [{DIM}]Already at or past final citation [{len(citations)}/{len(citations)}] ({last_ts})[/]")
        return True

    idx, ts_str, sec = hit
    if not st.get("playing") and default_record and default_record.get("file_path"):
        client.playback_play(
            default_record["file_path"],
            start_pos=sec,
            audio_file_id=default_record.get("audio_file_id"),
            transcription_id=default_record.get("id"),
        )
    else:
        client.playback_seek(sec)
        if st.get("paused"):
            client.playback_resume()

    target_console.print(f"  [{SUCCESS}][DONE] Playing [{idx}/{len(citations)}] from {ts_str}[/]")
    return True


def _cmd_prev(
    client: Any,
    citations: list[tuple[str, float]] | None,
    default_record: dict | None,
    target_console: Console,
) -> bool:
    if not citations:
        target_console.print(f"  [{DIM}]No active citations from recent response[/]")
        return True

    st = client.playback_status()
    pos = st.get("position", 0.0)
    hit = find_prev_timestamp(citations, pos)
    if not hit:
        first_ts = citations[0][0]
        target_console.print(f"  [{DIM}]Already at first citation [1/{len(citations)}] ({first_ts})[/]")
        return True

    idx, ts_str, sec = hit
    if not st.get("playing") and default_record and default_record.get("file_path"):
        client.playback_play(
            default_record["file_path"],
            start_pos=sec,
            audio_file_id=default_record.get("audio_file_id"),
            transcription_id=default_record.get("id"),
        )
    else:
        client.playback_seek(sec)
        if st.get("paused"):
            client.playback_resume()

    target_console.print(f"  [{SUCCESS}][DONE] Playing [{idx}/{len(citations)}] from {ts_str}[/]")
    return True


def _cmd_jump_index(
    idx: int,
    client: Any,
    citations: list[tuple[str, float]] | None,
    default_record: dict | None,
    target_console: Console,
) -> bool:
    if not citations:
        target_console.print(f"  [{DIM}]No active citations from recent response[/]")
        return True

    if not (1 <= idx <= len(citations)):
        target_console.print(f"  [{WARNING}]Citation index [{idx}] out of range (1–{len(citations)})[/]")
        return True

    ts_str, sec = citations[idx - 1]
    st = client.playback_status()
    if not st.get("playing") and default_record and default_record.get("file_path"):
        client.playback_play(
            default_record["file_path"],
            start_pos=sec,
            audio_file_id=default_record.get("audio_file_id"),
            transcription_id=default_record.get("id"),
        )
    else:
        client.playback_seek(sec)
        if st.get("paused"):
            client.playback_resume()

    target_console.print(f"  [{SUCCESS}][DONE] Playing [{idx}/{len(citations)}] from {ts_str}[/]")
    return True


def _play_record(
    client: Any,
    rec: dict[str, Any],
    start_pos: float,
    target_console: Console,
) -> bool:
    """Helper to dispatch playback for a resolved transcript/audio record."""
    client.playback_play(
        rec["file_path"],
        start_pos=start_pos,
        audio_file_id=rec.get("audio_file_id"),
        transcription_id=rec.get("id"),
    )
    name = rec.get("file_name") or os.path.basename(rec["file_path"])
    pos_str = f" from {fmt_timestamp(start_pos)}" if start_pos > 0.0 else ""
    target_console.print(f"  [{SUCCESS}][DONE] Playing: {name}{pos_str}[/]")
    return True


def _cmd_play(
    arg: str,
    client: Any,
    target_console: Console,
    resolver: Callable[[int], dict | None] | None,
    loaded_ids: list[int] | None,
) -> bool:
    raw = arg.strip()

    # Case 1: /play <timestamp> (e.g. /play 6:30 or /play 01:15:00)
    ts = parse_timestamp(raw) if (raw and ":" in raw) else None
    if ts is not None:
        st = client.playback_status()
        if st.get("playing"):
            client.playback_seek(ts)
            target_console.print(f"  [{SUCCESS}][DONE] Seeked to {fmt_timestamp(ts)}[/]")
            return True
        elif loaded_ids and resolver:
            rec = resolver(loaded_ids[0])
            if rec and rec.get("file_path"):
                return _play_record(client, rec, ts, target_console)

    # Case 2: /play <transcript_id> (e.g. /play 1430)
    if raw.isdigit() and resolver:
        tid = int(raw)
        rec = resolver(tid)
        if rec and rec.get("file_path"):
            return _play_record(client, rec, 0.0, target_console)
        else:
            target_console.print(f"  [{WARNING}]Transcript #{tid} has no playable audio file.[/]")
            return True

    # Case 3: /play <fuzzy_text> (e.g. /play interview)
    if raw and not raw.isdigit() and ":" not in raw:
        try:
            from audiobench.storage.repository import TranscriptionRepository

            repo = TranscriptionRepository()
            matches = repo.search(raw, limit=5)
            if len(matches) == 1:
                return _play_record(client, matches[0], 0.0, target_console)
            elif len(matches) > 1:
                target_console.print(f"  [{ACCENT}]Found {len(matches)} matching transcripts:[/]")
                for m in matches:
                    name = m.get("file_name") or f"Transcript #{m.get('id')}"
                    target_console.print(f"    [{DIM}]#{m.get('id')}[/] {name}")
                target_console.print(f"  [{DIM}]Use /play <ID> to choose[/]")
                return True
        except Exception:  # noqa: BLE001
            pass

    # Case 4: /play with no args — resume if paused, or play first loaded transcript
    st = client.playback_status()
    if st.get("playing") and st.get("paused"):
        client.playback_resume()
        target_console.print(f"  [{SUCCESS}][DONE] Resumed playback[/]")
        return True

    if loaded_ids and resolver:
        rec = resolver(loaded_ids[0])
        if rec and rec.get("file_path"):
            return _play_record(client, rec, 0.0, target_console)

    target_console.print(f"  [{DIM}]Nothing to play — load a transcript first (/load <ID>)[/]")
    return True


def _cmd_seek(arg: str, client: Any, target_console: Console) -> bool:
    raw = arg.strip()
    if not raw:
        target_console.print(f"  [{DIM}]Usage: /seek <timestamp> (e.g. /seek 6:30 or /seek +15)[/]")
        return True

    # Check relative seek
    if raw.startswith(("+", "-")):
        try:
            offset = float(raw)
            client.playback_seek_relative(offset)
            target_console.print(f"  [{SUCCESS}][DONE] Seeked {raw}s[/]")
            return True
        except ValueError:
            pass

    ts = parse_timestamp(raw)
    if ts is None:
        target_console.print(f"  [{WARNING}]Invalid timestamp: {raw}[/]")
        return True

    client.playback_seek(ts)
    target_console.print(f"  [{SUCCESS}][DONE] Seeked to {fmt_timestamp(ts)}[/]")
    return True


def _cmd_toggle(cmd: str, client: Any, target_console: Console) -> bool:
    if cmd == "/pause":
        client.playback_pause()
        target_console.print(f"  [{SUCCESS}][DONE] Paused[/]")
    elif cmd == "/resume":
        client.playback_resume()
        target_console.print(f"  [{SUCCESS}][DONE] Resumed[/]")
    else:
        st = client.playback_toggle()
        state = "Paused" if st.get("paused") else "Playing"
        target_console.print(f"  [{SUCCESS}][DONE] {state}[/]")
    return True


def _cmd_speed(arg: str, client: Any, target_console: Console) -> bool:
    raw = arg.strip()
    try:
        spd = float(raw)
    except ValueError:
        target_console.print(f"  [{DIM}]Usage: /speed <multiplier> (e.g. /speed 1.5)[/]")
        return True

    client.playback_speed(spd)
    target_console.print(f"  [{SUCCESS}][DONE] Playback speed set to {spd:.2f}x[/]")
    return True


def _parse_status_fields(st: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Extract formatted status components (icon, pos, dur, spd_str, file_name)."""
    icon = "[PAUSE]" if st.get("paused") else "[PLAY]"
    pos = fmt_timestamp(st.get("position", 0.0))
    dur = fmt_timestamp(st.get("duration", 0.0))
    spd = st.get("speed", 1.0)
    spd_str = f" [{spd:.1f}x]" if spd != 1.0 else ""
    file_name = os.path.basename(st.get("file", "Audio")) if st.get("file") else "Audio"
    return icon, pos, dur, spd_str, file_name


def _cmd_status(client: Any, target_console: Console) -> bool:
    st = client.playback_status()
    if not st.get("playing"):
        target_console.print(f"  [{DIM}]No audio currently playing[/]")
        return True

    icon, pos, dur, spd_str, file_name = _parse_status_fields(st)
    pos_sec = st.get("position", 0.0)
    dur_sec = st.get("duration", 0.0)
    bar = render_progress_bar(pos_sec, dur_sec, width=28)
    target_console.print(f"  [{ACCENT}]{icon}[/]  {pos} {bar} {dur}{spd_str} · [{DIM}]{file_name}[/]")
    return True


def _cmd_explain(
    arg: str,
    client: Any,
    target_console: Console,
    on_explain_context: Callable[[str], None] | None,
) -> bool:
    """Fetch transcript context around the current/given playback position and explain."""
    ts = parse_timestamp(arg) if arg.strip() else None
    ctx = client.playback_context(position=ts, window=3)
    segs = ctx.get("segments", [])
    if not segs:
        target_console.print(f"  [{DIM}]No transcript context available for current playback position[/]")
        return True

    cur_idx = ctx.get("current_idx", 0)
    pos = ctx.get("position", 0.0)

    lines = []
    for i, s in enumerate(segs):
        prefix = ">> " if i == cur_idx else "   "
        spk = f"[{s['speaker']}]: " if s.get("speaker") else ""
        lines.append(f"{prefix}[{fmt_timestamp(s['start'])}] {spk}{s['text']}")

    context_block = "\n".join(lines)
    prompt_text = (
        f"Explain what is being said and the significance around timestamp {fmt_timestamp(pos)}:\n\n"
        f"{context_block}"
    )

    # Persist explanation anchor as bookmark if audio_file_id is available
    st = client.playback_status()
    audio_id = st.get("audio_file_id")
    if audio_id and segs:
        topic = f"Explain at {fmt_timestamp(pos)}"
        _register_explain(
            audio_id,
            segs[0]["start"],
            segs[-1]["end"],
            topic,
            context_block,
            transcription_id=st.get("transcription_id"),
        )

    if on_explain_context:
        on_explain_context(prompt_text)
    else:
        target_console.print(f"\n  [{ACCENT}]Context at {fmt_timestamp(pos)}:[/]")
        target_console.print(f"  [dim]{context_block}[/]\n")

    return True


def _cmd_clip(arg: str, client: Any, target_console: Console) -> bool:
    """Extract an audio clip between two timestamps using ffmpeg and store in data/clips/."""
    parts = arg.strip().split()
    if len(parts) < 2:
        target_console.print(f"  [{DIM}]Usage: /clip <start> <end> [filename] (e.g. /clip 6:30 7:00 clip.mp3)[/]")
        return True

    start = parse_timestamp(parts[0])
    end = parse_timestamp(parts[1])
    if start is None or end is None:
        target_console.print(f"  [{WARNING}]Invalid start/end timestamps[/]")
        return True

    if end <= start:
        target_console.print(f"  [{WARNING}]End timestamp must be greater than start timestamp[/]")
        return True

    st = client.playback_status()
    src_file = st.get("file")
    if not src_file or not os.path.exists(src_file):
        target_console.print(f"  [{WARNING}]No active audio file to clip from[/]")
        return True

    audio_file_id = st.get("audio_file_id") or 0
    src_stem = Path(src_file).stem
    safe_stem = re.sub(r"[^\w\-]", "_", src_stem)[:30]
    slug = f"{audio_file_id}_{safe_stem}"

    # Resolve output directory and filename
    settings = get_settings()
    if len(parts) > 2:
        user_path = Path(parts[2]).expanduser()
        if user_path.parent == Path("."):
            out_dir = settings.data_dir / "clips" / slug
            out_path = out_dir / user_path.name
        else:
            out_path = user_path.resolve()
    else:
        out_dir = settings.data_dir / "clips" / slug
        start_s = fmt_timestamp_slug(start)
        end_s = fmt_timestamp_slug(end)
        out_path = out_dir / f"clip_{start_s}_{end_s}.mp3"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        src_file,
        "-t",
        str(duration),
        "-c",
        "copy",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if audio_file_id:
            _register_clip(
                audio_file_id,
                start,
                end,
                out_path.name,
                str(out_path),
                transcription_id=st.get("transcription_id"),
            )
        target_console.print(
            f"  [{SUCCESS}][DONE] Clipped {fmt_timestamp(start)}–{fmt_timestamp(end)} to {out_path}[/]"
        )
    except Exception as exc:
        target_console.print(f"  [{ERROR}]ffmpeg clip failed: {exc}[/]")

    return True


def _cmd_lyrics(
    client: Any, target_console: Console, resolver: Any = None
) -> bool:
    """Launch universal interactive follow / lyrics HUD for daemon playback."""
    try:
        from audiobench.playback.lyrics_hud import show_daemon_lyrics

        show_daemon_lyrics(client, target_console, tx_repo=resolver)
    except Exception as exc:
        target_console.print(f"  [{WARNING}]Lyrics follow error: {exc}[/]")
    return True


# ── Prompt Status Line ──────────────────────────────────────


def render_playback_status_line(
    client: Any,
    citations: list[tuple[str, float]] | None = None,
) -> str | None:
    """Return a compact status line for prompt injection. Returns None if inactive.

    When active citations are passed, includes the current citation index [i/N]
    and a short hint for available empty-buffer hotkeys.
    """
    try:
        st = client.playback_status()
    except Exception:
        return None

    if not st.get("playing"):
        return None

    icon, pos, dur, spd_str, _ = _parse_status_fields(st)
    base_line = f"{icon} {pos}/{dur}{spd_str}"

    if not citations:
        return base_line

    # Determine which citation section the playhead is currently in
    pos_sec = st.get("position", 0.0)
    curr_idx = 1
    for i, (_, sec) in enumerate(citations, 1):
        if pos_sec >= sec - 0.5:
            curr_idx = i
        else:
            break

    total = len(citations)
    cap_idx = min(9, total)
    jump_hint = f"1-{cap_idx}:jump" if cap_idx > 1 else "1:jump"
    hint = f"n:next · b:back · {jump_hint} · Space:pause"
    return f"{base_line}  ·  [{curr_idx}/{total}]  ({hint})"
