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


def _find_best_transcript_for_file(repo, file_path: str) -> dict | None:
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
            session.query(AudioFileRecord).filter(AudioFileRecord.file_path == resolved).all()
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


def _build_atempo_filter(speed: float) -> str:
    """Build an ffplay -af atempo filter string for the given speed.

    FFmpeg's atempo filter accepts values in [0.5, 2.0] per instance.
    For speeds outside that range, filters are chained:
      - Speed < 0.5 → clamped to 0.5 (ffplay lower bound)
      - Speed > 2.0 → atempo=2.0 repeated, remainder as final filter
        e.g. 3.0x → "atempo=2.0,atempo=1.500000"
             4.0x → "atempo=2.0,atempo=2.000000"
             5.0x → "atempo=2.0,atempo=2.000000,atempo=1.250000"

    Returns the complete comma-joined filter string ready for -af.
    """
    clamped = max(0.5, speed)
    filters: list[str] = []
    remaining = clamped
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)



# ── Lyrics mode → ChapterReaderApp ────────────────────────


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
    """Launch ChapterReaderApp for an audio file with synchronized transcript.

    Routes through the daemon if available; falls back to an in-process
    PlaybackService otherwise.  Audio keeps playing after the reader exits.
    """
    service = None

    # ── Try daemon first ──
    try:
        from audiobench.daemon.factory import get_daemon_client

        service = get_daemon_client()
        service.playback_play(  # type: ignore[union-attr]
            str(file_path),
            start_pos=start_seconds,
            speed=speed or 1.0,
            audio_file_id=audio_file_id,
            transcription_id=transcription_id,
        )
    except Exception:
        service = None

    # ── Fall back to in-process PlaybackService ──
    if service is None:
        from audiobench.playback.service import PlaybackService

        svc = PlaybackService()
        svc.play(
            str(file_path),
            start_pos=start_seconds,
            speed=speed or 1.0,
            audio_file_id=audio_file_id,
            transcription_id=transcription_id,
        )
        service = svc

    # ── Launch reader ──
    listen_start = time.monotonic()

    from audiobench.cli.tui.chapter_reader import open_reader

    open_reader(
        service,
        title=record_name,
        transcription_id=transcription_id,
        audio_file_id=audio_file_id,
        raw_segments=segments,
    )

    # ── Post-exit listen stats ──
    try:
        st = service.status()
        end_pos = float(st.get("position", start_seconds))
        dur = float(st.get("duration", total_duration)) or total_duration
    except Exception:
        end_pos = start_seconds
        dur = total_duration

    wall_time = time.monotonic() - listen_start
    listened = end_pos - start_seconds

    def _sfmt(secs: float) -> str:
        s = int(abs(secs))
        if s >= 3600:
            return f"{s // 3600}h {(s % 3600) // 60:02d}m"
        if s >= 60:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s}s"

    console.print()
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    if dur > 0:
        pct = min(end_pos / dur * 100, 100)
        console.print(
            f"  [{DIM}]Listened:  {_sfmt(listened)} ({pct:.0f}% of {_sfmt(dur)})[/]"
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

    console.print(f"  [{ACCENT}]>[/] Opening in mpv: {record_name}")
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
@click.option(
    "--chapter",
    type=int,
    default=None,
    help="Start playback from a specific chapter",
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
    chapter: int | None,
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
        record = repo.get_by_id(int(target), session=None)
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
                record = repo.get_by_id(best_id, session=None)
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

    # ── Parse seek timestamp or chapter ──
    start_seconds = 0.0
    if chapter is not None:
        if record:
            audio_id = record.get("audio_file_id")
            if audio_id:
                from audiobench.storage.chapter_repository import get_chapter_repo

                chap = get_chapter_repo().get_chapter_by_index(audio_id, chapter)
                if chap:
                    start_seconds = chap.start_time
                    if not quiet:
                        console.print(f"  [{DIM}]Starting from Chapter {chapter}: {chap.title}[/]")
                else:
                    console.print(
                        error_panel("Not found", f"Chapter {chapter} does not exist for this file.")
                    )
                    sys.exit(1)
        else:
            console.print(
                error_panel("No Record", "Cannot use --chapter without a database record.")
            )
            sys.exit(1)
    elif seek_to:
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
                f"  [{WARNING}]Transcript has no segments — falling back to plain playback.[/]"
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
                f"    From:   {int(start_seconds // 60):02d}:{int(start_seconds % 60):02d}"
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
    if not audio_file_id and file_path:
        from audiobench.storage.bookmark_repository import BookmarkRepository

        _bm_repo = BookmarkRepository()
        audio_file_id = _bm_repo._get_audio_file_id_by_path(str(file_path))

    # ── Show bookmarks before playback (--bookmarks) ──
    if show_bookmarks and audio_file_id and not quiet:
        from audiobench.storage.bookmark_repository import (
            BOOKMARK_TYPES as _BM_TYPES,
        )
        from audiobench.storage.bookmark_repository import (
            BookmarkRepository as _BmRepo,
        )
        from audiobench.storage.bookmark_repository import (
            _format_timestamp as _bfmt,
        )

        _sb_repo = _BmRepo()
        file_bms = _sb_repo.list_for_file(audio_file_id)
        if file_bms:
            console.print(f"\n  [{ACCENT}]Bookmarks ({len(file_bms)})[/]")
            for bm in file_bms:
                emoji = _BM_TYPES.get(bm["bookmark_type"], "[BM]")
                t = _bfmt(bm["timestamp"])
                if bm.get("is_region") and bm.get("end_timestamp"):
                    t += f"→{_bfmt(bm['end_timestamp'])}"
                console.print(f"    [{DIM}]#{bm['id']}[/] {emoji} {t}  {bm['name'][:40]}")
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
                )
                from audiobench.storage.bookmark_repository import (
                    _format_timestamp as _bfmt,
                )

                emoji = BOOKMARK_TYPES.get(bm["bookmark_type"], "[BM]")
                console.print(
                    f'    From:   {emoji} #{bm["id"]} {_bfmt(start_seconds)} "{bm["name"][:40]}"'
                )
        else:
            console.print(error_panel("Bookmark not found", f"'{from_bookmark}'"))
            sys.exit(1)

    # ── Route to mode ──
    if lyrics and segments:
        _play_with_lyrics(
            file_path,
            segments,
            record_name,
            start_seconds,
            speed,
            total_duration,
            resume=resume,
            audio_file_id=audio_file_id,
            transcription_id=transcription_id,
        )
    elif watch and segments:
        _play_with_subtitles(
            file_path,
            segments,
            record_name,
            start_seconds,
            speed,
        )
    else:
        # Plain playback: route to daemon if available
        try:
            from audiobench.daemon.factory import get_daemon_client
            client = get_daemon_client()
            client.playback_play(
                str(file_path),
                start_pos=start_seconds,
                speed=speed or 1.0,
                audio_file_id=audio_file_id,
                transcription_id=transcription_id,
            )
            if not quiet:
                console.print(f"  [{ACCENT}]>[/] Playing: {record_name}")
                console.print(f"  [{DIM}]Controls available in Chat (/pause, /seek) or REPL (\\pause, \\seek)[/]\n")
            return
        except Exception:
            pass

        # Fallback to ffplay
        cmd = [
            "ffplay",
            "-autoexit",
            "-window_title",
            f"AudioBench: {record_name}",
            "-x",
            "1",
            "-y",
            "1",
            "-loglevel",
            "quiet",
            str(file_path),
        ]
        if start_seconds > 0:
            cmd.extend(["-ss", str(start_seconds)])
        if speed and speed != 1.0:
            if speed < 0.5 and not quiet:
                console.print(
                    f"  [{WARNING}]Speed {speed}x is below ffplay minimum (0.5x); using 0.5x[/]"
                )
            cmd.extend(["-af", _build_atempo_filter(speed)])

        if not quiet:
            console.print(f"  [{ACCENT}]>[/] Playing: {record_name}")
            console.print(f"  [{DIM}]Press 'q' in the popup window to stop, or Ctrl+C here[/]")
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
