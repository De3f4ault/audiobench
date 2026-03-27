"""Play command — play audio files with optional transcript sync.

Modes:
  audiobench play <file|ID>              Plain audio playback
  audiobench play <file|ID> --lyrics     Terminal follow mode (karaoke)
  audiobench play <file|ID> --watch      MPV with subtitle overlay
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    error_panel,

)

# ── Engine priority for smart transcript selection ──
# Higher = preferred. Google transcripts are generally more accurate.
ENGINE_PRIORITY = {
    "google": 100,
    "google-genai": 100,
    "faster-whisper": 50,
    "vosk": 30,
    "live": 10,
}

MODEL_PRIORITY = {
    "gemini-2.5-pro": 95,
    "gemini-2.0-flash": 90,
    "gemini-1.5-pro": 85,
    "gemini-1.5-flash": 80,
    "large-v3-turbo": 90,
    "large-v3": 85,
    "large-v2": 80,
    "large": 75,
    "medium": 60,
    "small": 40,
    "base": 20,
    "tiny": 10,
}


def _score_transcript(record: dict) -> int:
    """Score a transcript for quality ranking.

    Higher score = better transcript.
    """
    engine = (record.get("engine") or "").lower()
    model = (record.get("model") or "").lower()

    engine_score = ENGINE_PRIORITY.get(engine, 25)
    model_score = MODEL_PRIORITY.get(model, 50)

    # More words generally means more complete transcription
    word_bonus = min(record.get("word_count", 0) // 100, 20)

    return engine_score + model_score + word_bonus


def _find_best_transcript_for_file(
    repo, file_path: str
) -> dict | None:
    """Find the best transcript for a given audio file path.

    Searches all transcripts that reference the same audio file
    and returns the highest-scored one.
    """
    from sqlalchemy import desc

    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    resolved = str(Path(file_path).resolve())

    with get_session() as session:
        # Find audio records matching this file path
        audio_records = (
            session.query(AudioFileRecord)
            .filter(AudioFileRecord.file_path == resolved)
            .all()
        )

        if not audio_records:
            return None

        audio_ids = [a.id for a in audio_records]

        # Get all transcriptions for these audio files
        transcriptions = (
            session.query(TranscriptionRecord)
            .filter(TranscriptionRecord.audio_file_id.in_(audio_ids))
            .order_by(desc(TranscriptionRecord.created_at))
            .all()
        )

        if not transcriptions:
            return None

        # Score and pick the best
        best_rec = None
        best_score = -1

        for rec in transcriptions:
            rec_dict = {
                "id": rec.id,
                "engine": rec.engine,
                "model": rec.model_name,
                "word_count": rec.word_count,
            }
            score = _score_transcript(rec_dict)
            if score > best_score:
                best_score = score
                best_rec = rec

        if best_rec is None:
            return None

        return best_rec.id  # Return the ID, caller will use get_by_id()


def _parse_timestamp(ts: str) -> float | None:
    """Parse MM:SS or HH:MM:SS to seconds."""
    match = re.match(r"(?:(\d+):)?(\d+):(\d+)", ts)
    if match:
        hours = int(match.group(1) or 0)
        mins = int(match.group(2))
        secs = int(match.group(3))
        return hours * 3600 + mins * 60 + secs
    return None


def _format_srt(segments: list[dict]) -> str:
    """Generate SRT subtitle content from transcript segments."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        lines.append(str(i))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Lyrics mode (interactive terminal follow) ──────────────


def _play_with_lyrics(
    file_path: Path,
    segments: list[dict],
    record_name: str,
    start_seconds: float,
    speed: float | None,
    total_duration: float,
    resume: bool = False,
    audio_file_id: int | None = None,
    transcription_id: int | None = None,
) -> None:
    """Play audio with interactive controls and synchronized transcript."""
    import select

    import termios
    import tty

    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from audiobench.playback import MpvController

    mpv = MpvController()

    try:
        mpv.start(
            str(file_path),
            start_pos=start_seconds,
            speed=speed or 1.0,
            save_position=resume,  # Only save/restore position with --resume
        )
    except FileNotFoundError:
        console.print(
            error_panel(
                "mpv not found",
                "Install mpv for interactive lyrics mode:\n"
                "  Arch: sudo pacman -S mpv\n"
                "  Ubuntu: sudo apt install mpv\n"
                "  macOS: brew install mpv",
            )
        )
        sys.exit(1)
    except RuntimeError as e:
        console.print(error_panel("mpv failed to start", str(e)))
        sys.exit(1)

    # Wait for mpv to report duration (definitive source of truth)
    time.sleep(0.3)
    for _ in range(10):
        mpv_duration = mpv.get_duration()
        if mpv_duration > 0:
            total_duration = mpv_duration
            break
        time.sleep(0.1)

    # Track listening for stats
    listen_start = time.monotonic()
    listen_start_pos = mpv.get_position()

    # ── Bookmark support ──────────────────────────────────
    bookmark_repo = None
    bookmark_flash: str | None = None
    bookmark_flash_until: float = 0.0
    region_start: float | None = None
    last_bookmark_id: int | None = None

    if audio_file_id is not None:
        from audiobench.storage.bookmark_repository import (
            BOOKMARK_TYPES,
            BookmarkRepository,
            _format_timestamp as _bfmt,
        )
        bookmark_repo = BookmarkRepository()

    def _set_flash(msg: str) -> None:
        nonlocal bookmark_flash, bookmark_flash_until
        bookmark_flash = msg
        bookmark_flash_until = time.monotonic() + 2.0  # show for 2s

    # cbreak mode: read individual keypresses while preserving output processing
    # (setraw breaks ANSI escape handling needed by Rich Live)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        with Live(console=console, refresh_per_second=4, transient=True) as live:
            current_idx = -1  # track segment index across loop iterations
            while mpv.is_running():
                # ── Handle keypresses (non-blocking) ──
                # Keybindings follow mpv defaults
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)

                    if key == "q":
                        mpv.quit()
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        console.print(f"\n  [{DIM}]Playback stopped[/]")
                        console.print()
                        return

                    elif key == " ":
                        mpv.toggle_pause()

                    # Speed: [ ] = ±10%, { } = half/double
                    elif key == "[":
                        mpv.set_speed(max(0.25, mpv.get_speed() * 0.9))
                    elif key == "]":
                        mpv.set_speed(min(4.0, mpv.get_speed() * 1.1))
                    elif key == "{":
                        mpv.set_speed(max(0.25, mpv.get_speed() * 0.5))
                    elif key == "}":
                        mpv.set_speed(min(4.0, mpv.get_speed() * 2.0))

                    # Reset speed
                    elif key == "\x7f" or key == "\x08":  # Backspace
                        mpv.set_speed(1.0)

                    # Volume: 9/0
                    elif key == "9":
                        vol = mpv.get_property("volume")
                        mpv.set_property("volume", max(0, (float(vol) if vol else 100) - 10))
                    elif key == "0":
                        vol = mpv.get_property("volume")
                        mpv.set_property("volume", min(150, (float(vol) if vol else 100) + 10))

                    # Mute
                    elif key == "m":
                        muted = mpv.get_property("mute")
                        mpv.set_property("mute", not bool(muted))

                    # Arrow keys (escape sequences)
                    elif key == "\x1b":
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            seq = sys.stdin.read(1)
                            if seq == "[":
                                arrow = sys.stdin.read(1)
                                if arrow == "C":     # → seek +5s
                                    mpv.seek(5)
                                elif arrow == "D":   # ← seek -5s
                                    mpv.seek(-5)
                                elif arrow == "A":   # ↑ seek +60s
                                    mpv.seek(60)
                                elif arrow == "B":   # ↓ seek -60s
                                    mpv.seek(-60)

                    # ── Bookmark keybindings ──
                    elif key == "b" and bookmark_repo and audio_file_id:
                        # Instant point bookmark — auto-name from transcript
                        pos = mpv.get_position()
                        auto_name = f"Bookmark @ {_bfmt(pos)}"
                        if segments and current_idx >= 0:
                            seg_text = (segments[current_idx].get("text", "") or "").strip()
                            if seg_text:
                                auto_name = seg_text[:80]
                        bid = bookmark_repo.add(
                            audio_file_id, pos,
                            name=auto_name,
                            transcription_id=transcription_id,
                        )
                        last_bookmark_id = bid
                        _set_flash(f"🔖 {_bfmt(pos)} \"{auto_name[:40]}\"")

                    elif key == "B" and bookmark_repo and audio_file_id:
                        # Region start/end toggle
                        pos = mpv.get_position()
                        if region_start is None:
                            region_start = pos
                            _set_flash(f"▶ Region start @ {_bfmt(pos)}")
                        else:
                            auto_name = f"Region {_bfmt(region_start)}→{_bfmt(pos)}"
                            bid = bookmark_repo.add_region(
                                audio_file_id, region_start, pos,
                                name=auto_name,
                                transcription_id=transcription_id,
                            )
                            last_bookmark_id = bid
                            _set_flash(f"■ Region saved: {_bfmt(region_start)}→{_bfmt(pos)}")
                            region_start = None

                    elif key == "n" and bookmark_repo and audio_file_id:
                        # Jump to next bookmark
                        pos = mpv.get_position()
                        nxt = bookmark_repo.get_nearest(audio_file_id, pos, "next")
                        if nxt:
                            mpv.seek_absolute(nxt["timestamp"])
                            emoji = BOOKMARK_TYPES.get(nxt["bookmark_type"], "🔖")
                            _set_flash(f"→ {emoji} {_bfmt(nxt['timestamp'])} {nxt['name'][:30]}")
                        else:
                            _set_flash("→ No more bookmarks")

                    elif key == "p" and bookmark_repo and audio_file_id:
                        # Jump to previous bookmark
                        pos = mpv.get_position()
                        prev = bookmark_repo.get_nearest(audio_file_id, pos, "prev")
                        if prev:
                            mpv.seek_absolute(prev["timestamp"])
                            emoji = BOOKMARK_TYPES.get(prev["bookmark_type"], "🔖")
                            _set_flash(f"← {emoji} {_bfmt(prev['timestamp'])} {prev['name'][:30]}")
                        else:
                            _set_flash("← No earlier bookmarks")

                    elif key == "l" and bookmark_repo and last_bookmark_id:
                        # Cycle type of last-created bookmark
                        new_type = bookmark_repo.cycle_type(last_bookmark_id)
                        if new_type:
                            emoji = BOOKMARK_TYPES.get(new_type, "🔖")
                            _set_flash(f"{emoji} Type → {new_type}")

                # ── Read state from mpv (batch for low latency) ──
                current_time, current_speed, paused = mpv.get_playback_state()

                # Find current segment
                current_idx = -1
                for i, seg in enumerate(segments):
                    if seg.get("start", 0) <= current_time <= seg.get("end", 0):
                        current_idx = i
                        break
                    if seg.get("start", 0) > current_time:
                        current_idx = max(0, i - 1)
                        break

                if current_idx == -1 and segments:
                    current_idx = len(segments) - 1

                # ── Build display ──
                from rich.align import Align
                from rich.console import Group

                parts = []  # renderable parts for Group

                # Helper for consistent MM:SS / HH:MM:SS formatting
                def _fmt(secs: float) -> str:
                    s = int(secs)
                    if s >= 3600:
                        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
                    return f"{s // 60:02d}:{s % 60:02d}"

                # Progress bar (centered)
                pause_icon = "⏸" if paused else "▶"
                speed_str = f"  {current_speed:.2g}x" if current_speed != 1.0 else ""

                if total_duration > 0:
                    pct = min(current_time / total_duration, 1.0)
                    bar_width = 30
                    filled = int(pct * bar_width)
                    bar = "━" * filled + "░" * (bar_width - filled)
                    progress_line = (
                        f"{pause_icon}  {_fmt(current_time)} / {_fmt(total_duration)}"
                        f"  {bar}{speed_str}"
                    )
                    progress_text = Text(progress_line)
                    progress_text.stylize(
                        "bold cyan" if paused else "dim"
                    )
                    parts.append(Text(""))
                    parts.append(Align.center(progress_text))
                    parts.append(Text(""))

                # Segment window — subtitle-style (no timestamps)
                win_start = max(0, current_idx - 3)
                win_end = min(len(segments), current_idx + 4)

                segment_text = Text()

                if win_start > 0:
                    segment_text.append("⋮\n", style="dim")

                for i in range(win_start, win_end):
                    seg = segments[i]
                    text = (seg.get("text", "") or "").strip()
                    if not text:
                        continue

                    if i == current_idx:
                        segment_text.append(f"▸ {text}\n", style="bold")
                    elif i < current_idx:
                        segment_text.append(f"  {text}\n", style="dim")
                    else:
                        segment_text.append(f"  {text}\n", style="dim italic")

                if win_end < len(segments):
                    segment_text.append("⋮\n", style="dim")

                parts.append(segment_text)

                # ── Bookmark flash / region indicator ──
                flash_text = None
                if region_start is not None:
                    elapsed = mpv.get_position() - region_start
                    flash_text = Text(
                        f"  ● Recording region ({_bfmt(elapsed)})...",
                        style="bold red",
                    )
                elif bookmark_flash and time.monotonic() < bookmark_flash_until:
                    flash_text = Text(f"  {bookmark_flash}", style="bold green")

                if flash_text:
                    parts.append(Text(""))
                    parts.append(Align.center(flash_text))

                # ── Bookmark indicator bar ──
                if bookmark_repo and audio_file_id and total_duration > 0:
                    file_bookmarks = bookmark_repo.list_for_file(audio_file_id)
                    if file_bookmarks:
                        bar_width = 30
                        indicator = list("─" * bar_width)
                        for bm in file_bookmarks:
                            pos_frac = min(bm["timestamp"] / total_duration, 1.0)
                            idx = min(int(pos_frac * bar_width), bar_width - 1)
                            emoji = BOOKMARK_TYPES.get(bm["bookmark_type"], "│")
                            # Use a simple marker character for the bar
                            indicator[idx] = "┃"
                        marker_line = Text("  " + "".join(indicator), style="dim cyan")
                        parts.append(Align.center(marker_line))

                # Controls hint (centered)
                if bookmark_repo:
                    ctrl_str = (
                        "␣ pause  ←→ ±5s  ↑↓ ±60s  b mark  B region  "
                        "n/p jump  l type  [ ] speed  q quit"
                    )
                else:
                    ctrl_str = (
                        "␣ pause  ←→ ±5s  ↑↓ ±60s  [ ] speed  "
                        "9/0 vol  m mute  ⌫ reset  q quit"
                    )
                controls = Text(ctrl_str, style="dim")
                parts.append(Text(""))
                parts.append(Align.center(controls))

                live.update(
                    Panel(
                        Group(*parts),
                        border_style="dim",
                        expand=True,
                        padding=(0, 1),
                    )
                )
                time.sleep(0.1)  # 10fps for smooth sync

        # Playback finished naturally
        _show_listen_stats(
            listen_start, mpv.get_position(), listen_start_pos, total_duration
        )

    except KeyboardInterrupt:
        mpv.terminate()
        _show_listen_stats(
            listen_start, current_time, listen_start_pos, total_duration
        )
    finally:
        # Restore terminal
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _show_listen_stats(
    listen_start: float,
    end_pos: float,
    start_pos: float,
    total_duration: float,
) -> None:
    """Show listening stats after playback ends."""
    wall_time = time.monotonic() - listen_start
    listened = end_pos - start_pos

    def _sfmt(secs: float) -> str:
        s = int(abs(secs))
        if s >= 3600:
            return f"{s // 3600}h {(s % 3600) // 60:02d}m"
        if s >= 60:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s}s"

    console.print()
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    if total_duration > 0:
        pct = min(end_pos / total_duration * 100, 100)
        console.print(
            f"  [{DIM}]Listened:  {_sfmt(listened)} "
            f"({pct:.0f}% of {_sfmt(total_duration)})[/]"
        )
    else:
        console.print(f"  [{DIM}]Listened:  {_sfmt(listened)}[/]")
    console.print(f"  [{DIM}]Session:   {_sfmt(wall_time)}[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()


# ── Watch mode (mpv with subtitles) ────────────────────────


def _play_with_subtitles(
    file_path: Path,
    segments: list[dict],
    record_name: str,
    start_seconds: float,
    speed: float | None,
) -> None:
    """Play audio in mpv with subtitle overlay."""
    # Generate SRT to temp file
    srt_content = _format_srt(segments)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".srt",
        prefix="audiobench_",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(srt_content)
    tmp.close()

    cmd = [
        "mpv",
        "--no-video",
        "--force-window=yes",
        f"--sub-file={tmp.name}",
        "--sub-visibility=yes",
        "--sub-font-size=36",
        "--osd-level=2",
        f"--title=AudioBench: {record_name}",
        str(file_path),
    ]
    if start_seconds > 0:
        cmd.append(f"--start={start_seconds}")
    if speed and speed != 1.0:
        cmd.append(f"--speed={speed}")

    console.print(f"  [{ACCENT}]▶[/] Opening in mpv: {record_name}")
    console.print(f"  [{DIM}]Controls: space=pause, ←→=seek, []=speed, q=quit[/]")
    console.print()

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"  [{SUCCESS}]✓ Playback complete[/]")
        console.print()
    except FileNotFoundError:
        console.print(
            error_panel(
                "mpv not found",
                "Install mpv for subtitle playback:\n"
                "  Arch: sudo pacman -S mpv\n"
                "  Ubuntu: sudo apt install mpv\n"
                "  macOS: brew install mpv",
            )
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}]Playback stopped[/]")
        console.print()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── Main Command ───────────────────────────────────────────


@click.command()
@click.argument("target")
@click.option(
    "--from",
    "seek_to",
    default=None,
    help="Start playback from timestamp (e.g., 01:25, 1:02:30)",
)
@click.option(
    "--speed",
    default=None,
    type=float,
    help="Playback speed (e.g., 1.5 for 50%% faster)",
)
@click.option(
    "--lyrics",
    is_flag=True,
    help="Follow transcript in terminal while playing",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Open in mpv with subtitle overlay",
)
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from last saved position",
)
@click.option(
    "--bookmark",
    "from_bookmark",
    default=None,
    help="Start from bookmark (by ID or name)",
)
@click.option(
    "--bookmarks",
    "show_bookmarks",
    is_flag=True,
    help="List bookmarks for this file before playing",
)
def play(
    target: str,
    seek_to: str | None,
    speed: float | None,
    lyrics: bool,
    watch: bool,
    quiet: bool,
    resume: bool,
    from_bookmark: str | None,
    show_bookmarks: bool,
) -> None:
    """Play an audio file or a transcript's source audio.

    \b
    TARGET can be:
      • A file path:       audiobench play meeting.m4a
      • A transcript ID:   audiobench play 66

    \b
    Modes:
      --lyrics     Scroll transcript in terminal (karaoke-style)
      --watch      Open in mpv with subtitle overlay

    \b
    Examples:
      audiobench play meeting.m4a                   Play a file
      audiobench play 66                            Play transcript #66's audio
      audiobench play 66 --lyrics                   Terminal follow mode
      audiobench play 66 --watch                    MPV with subtitles
      audiobench play 66 --from 01:25               Start at 1m 25s
      audiobench play meeting.m4a --lyrics          Auto-find best transcript
    """
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()

    # ── Resolve target to file_path + optional record ──
    file_path: Path | None = None
    record: dict | None = None
    record_name: str | None = None

    if target.isdigit():
        # Treat as transcript ID
        record = repo.get_by_id(int(target))
        if not record:
            console.print(error_panel("Not found", f"Transcript #{target} not found"))
            sys.exit(1)

        source = record.get("file_path")
        if not source or not Path(source).exists():
            console.print(
                error_panel(
                    "Source audio not found",
                    f"Path: {source or 'unknown'}\n"
                    "The original audio file may have been moved or deleted.",
                )
            )
            sys.exit(1)

        file_path = Path(source)
        record_name = record.get("file_name", file_path.name)
    else:
        # Treat as file path
        candidate = Path(target).expanduser().resolve()
        if not candidate.exists():
            console.print(error_panel("File not found", str(candidate)))
            sys.exit(1)
        file_path = candidate
        record_name = file_path.name

        # If --lyrics or --watch, auto-find best transcript for this file
        if lyrics or watch:
            best_id = _find_best_transcript_for_file(repo, str(file_path))
            if best_id:
                record = repo.get_by_id(best_id)
                if record:
                    console.print(
                        f"  [{DIM}]Using transcript #{record['id']} "
                        f"({record.get('engine', '?')}/{record.get('model', '?')}, "
                        f"{record.get('word_count', 0):,} words)[/]"
                    )
            if record is None:
                console.print(
                    f"  [{WARNING}]No transcript found for this file.[/]\n"
                    f"  [{DIM}]Transcribe first: audiobench transcribe {file_path.name}[/]"
                )
                console.print(f"  [{DIM}]Falling back to plain playback.[/]")
                lyrics = False
                watch = False

    # ── Parse seek timestamp ──
    start_seconds = 0.0
    if seek_to:
        parsed = _parse_timestamp(seek_to)
        if parsed is not None:
            start_seconds = parsed
        else:
            console.print(
                error_panel(
                    "Invalid timestamp",
                    f"'{seek_to}' — expected MM:SS or HH:MM:SS",
                )
            )
            sys.exit(1)

    # ── Get segments for lyrics/watch ──
    segments = []
    total_duration = 0.0
    if record and (lyrics or watch):
        segments = record.get("segments", [])
        total_duration = record.get("duration", 0) or 0

        if not segments:
            console.print(
                f"  [{WARNING}]Transcript has no segments — "
                f"falling back to plain playback.[/]"
            )
            lyrics = False
            watch = False

    # ── Header ──
    if not quiet:
        mode_label = " (lyrics)" if lyrics else " (watch)" if watch else ""
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Play{mode_label}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    File:   [{ACCENT}]{record_name}[/]")
        if record:
            console.print(
                f"    Using:  [{DIM}]#{record['id']} "
                f"({record.get('engine', '?')}/{record.get('model', '?')})[/]"
            )
        if start_seconds > 0:
            console.print(
                f"    From:   {int(start_seconds // 60):02d}:"
                f"{int(start_seconds % 60):02d}"
            )
        if speed:
            console.print(f"    Speed:  {speed:.1f}x")
        if segments:
            console.print(f"    Segs:   {len(segments)}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")

    # ── Handle --bookmark ──
    audio_file_id: int | None = None
    transcription_id: int | None = None

    if record:
        audio_file_id = record.get("audio_file_id")
        transcription_id = record.get("id")  # transcription record ID

    # If no audio_file_id yet (playing by file path), try to resolve from DB
    if audio_file_id is None and file_path:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        _bm_repo = BookmarkRepository()
        audio_file_id = _bm_repo._get_audio_file_id_by_path(str(file_path))

    # ── Show bookmarks before playback (--bookmarks) ──
    if show_bookmarks and audio_file_id and not quiet:
        from audiobench.storage.bookmark_repository import (
            BOOKMARK_TYPES as _BM_TYPES,
            BookmarkRepository as _BmRepo,
            _format_timestamp as _bfmt,
        )

        _sb_repo = _BmRepo()
        file_bms = _sb_repo.list_for_file(audio_file_id)
        if file_bms:
            console.print(f"\n  [{ACCENT}]Bookmarks ({len(file_bms)})[/]")
            for bm in file_bms:
                emoji = _BM_TYPES.get(bm["bookmark_type"], "🔖")
                t = _bfmt(bm["timestamp"])
                if bm.get("is_region") and bm.get("end_timestamp"):
                    t += f"→{_bfmt(bm['end_timestamp'])}"
                console.print(
                    f"    [{DIM}]#{bm['id']}[/] {emoji} {t}  {bm['name'][:40]}"
                )
            console.print()
        else:
            console.print(f"  [{DIM}]No bookmarks yet (press b during playback)[/]")

    if from_bookmark and audio_file_id:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        bm_repo = BookmarkRepository()
        bm = None

        # Try as bookmark ID
        if from_bookmark.isdigit():
            bm = bm_repo.get_by_id(int(from_bookmark))

        # Fallback: fuzzy name match
        if bm is None:
            candidates = bm_repo.search(from_bookmark, audio_file_id=audio_file_id)
            if candidates:
                bm = candidates[0]

        if bm:
            start_seconds = bm["timestamp"]
            if not quiet:
                from audiobench.storage.bookmark_repository import (
                    BOOKMARK_TYPES,
                    _format_timestamp as _bfmt,
                )
                emoji = BOOKMARK_TYPES.get(bm["bookmark_type"], "🔖")
                console.print(
                    f"    From:   {emoji} #{bm['id']} {_bfmt(start_seconds)} "
                    f"\"{bm['name'][:40]}\""
                )
        else:
            console.print(
                error_panel("Bookmark not found", f"'{from_bookmark}'")
            )
            sys.exit(1)

    # ── Route to mode ──
    if lyrics and segments:
        _play_with_lyrics(
            file_path, segments, record_name,
            start_seconds, speed, total_duration,
            resume=resume,
            audio_file_id=audio_file_id,
            transcription_id=transcription_id,
        )
    elif watch and segments:
        _play_with_subtitles(
            file_path, segments, record_name,
            start_seconds, speed,
        )
    else:
        # Plain playback
        cmd = [
            "ffplay", "-autoexit",
            "-window_title", f"AudioBench: {record_name}",
            "-x", "1", "-y", "1",
            "-loglevel", "quiet",
            str(file_path),
        ]
        if start_seconds > 0:
            cmd.extend(["-ss", str(start_seconds)])
        if speed and speed != 1.0:
            tempo = max(0.5, min(speed, 2.0))
            cmd.extend(["-af", f"atempo={tempo}"])

        if not quiet:
            console.print(f"  [{ACCENT}]▶[/] Playing: {record_name}")
            console.print(
                f"  [{DIM}]Press 'q' in the popup window to stop, "
                f"or Ctrl+C here[/]"
            )
            console.print()

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not quiet:
                console.print(f"  [{SUCCESS}]✓ Playback complete[/]")
                console.print()
        except FileNotFoundError:
            console.print(
                error_panel(
                    "ffplay not found",
                    "Install ffmpeg to use playback:\n"
                    "  Arch: sudo pacman -S ffmpeg\n"
                    "  Ubuntu: sudo apt install ffmpeg\n"
                    "  macOS: brew install ffmpeg",
                )
            )
            sys.exit(1)
        except KeyboardInterrupt:
            if not quiet:
                console.print(f"\n  [{DIM}]Playback stopped[/]")
                console.print()
