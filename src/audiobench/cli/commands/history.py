"""History + Search + Export + Delete + Show (data management) commands."""

from __future__ import annotations

import json as json_lib
import sys
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    format_duration,
    make_table,
    stdout,
)

# ── History Command ─────────────────────────────────────────


@click.command()
@click.option("--limit", default=20, help="Number of records to show")
@click.option(
    "--tail",
    "tail_n",
    type=int,
    default=None,
    help="Show only the last N transcriptions (alias for --limit)",
)
@click.option(
    "--since",
    default=None,
    help='Show transcriptions after date/time (e.g., "2024-01-01", "2d", "1w")',
)
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["date", "duration", "words", "name"]),
    default="date",
    show_default=True,
    help="Sort order for results",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "ids"]),
    default="table",
    show_default=True,
    help="Output format",
)
def history(
    limit: int,
    tail_n: int | None,
    since: str | None,
    sort_by: str,
    output_format: str,
) -> None:
    """View transcription history.

    \b
    Examples:
      audiobench history                      Show recent (default 20)
      audiobench history --tail 5             Last 5 transcriptions
      audiobench history --since 2d           From last 2 days
      audiobench history --sort duration      Sort by duration
      audiobench history --format json        Output as JSON
      audiobench history --format ids         Just the IDs (for piping)
    """
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()

    effective_limit = tail_n if tail_n is not None else limit
    records = repo.get_history(limit=effective_limit)

    # D1: --since filtering
    if since and records:
        cutoff = _parse_since(since)
        if cutoff:
            from datetime import datetime

            records = [
                r
                for r in records
                if r.get("created_at") and datetime.fromisoformat(r["created_at"]) >= cutoff
            ]

    if not records:
        if output_format == "json":
            stdout.print("[]", highlight=False)
        elif output_format == "ids":
            pass  # no output
        else:
            console.print(f"  [{DIM}]No transcription history yet.[/]")
        return

    # D1: --sort
    sort_keys = {
        "date": lambda r: r.get("created_at", ""),
        "duration": lambda r: r.get("duration") or 0,
        "words": lambda r: r.get("word_count") or 0,
        "name": lambda r: r.get("file_name", "").lower(),
    }
    if sort_by != "date":
        records.sort(key=sort_keys[sort_by], reverse=sort_by in ("duration", "words"))

    # D1: --format
    if output_format == "json":
        stdout.print(json_lib.dumps(records, indent=2, default=str), highlight=False)
        return
    elif output_format == "ids":
        for r in records:
            print(r["id"])
        return

    # Default: table format
    table = make_table(
        "Transcription History",
        [
            ("#", {"style": DIM, "width": 4}),
            ("File", {"style": ACCENT}),
            ("Language", {"width": 8}),
            ("Model", {"width": 16}),
            ("Words", {"justify": "right", "width": 6}),
            ("Duration", {"justify": "right", "width": 10}),
            ("Date", {"style": DIM, "width": 10}),
            ("Preview", {"max_width": 35}),
        ],
    )

    for rec in records:
        dur = format_duration(rec["duration"]) if rec["duration"] else "–"
        date = rec["created_at"][:10] if rec["created_at"] else "–"
        table.add_row(
            str(rec["id"]),
            rec["file_name"],
            rec["language"],
            rec["model"],
            str(rec["word_count"]),
            dur,
            date,
            (rec["text_preview"] or "")[:35],
        )

    console.print(table)


def _parse_since(since_str: str):
    """Parse a --since value into a datetime.

    Supports:
      - ISO dates: "2024-01-01"
      - Relative: "2d" (2 days), "1w" (1 week), "3h" (3 hours)
    """
    from datetime import datetime, timedelta

    # Try ISO date first
    try:
        return datetime.fromisoformat(since_str)
    except ValueError:
        pass

    # Relative time (e.g., "2d", "1w", "3h")
    units = {"h": "hours", "d": "days", "w": "weeks"}
    if len(since_str) >= 2 and since_str[-1] in units and since_str[:-1].isdigit():
        val = int(since_str[:-1])
        unit = units[since_str[-1]]
        return datetime.now() - timedelta(**{unit: val})

    return None


# ── Search Command ──────────────────────────────────────────


@click.command()
@click.argument("query")
@click.option("--limit", default=10, help="Max results")
@click.option(
    "--context",
    "context_lines",
    type=int,
    default=0,
    help="Show N characters of surrounding context around matches",
)
@click.option(
    "--regex",
    "use_regex",
    is_flag=True,
    help="Treat query as a regular expression (SQLite REGEXP)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["panel", "json", "ids"]),
    default="panel",
    show_default=True,
    help="Output format",
)
def search(
    query: str,
    limit: int,
    context_lines: int,
    use_regex: bool,
    output_format: str,
) -> None:
    """Search past transcriptions by text content.

    \b
    Examples:
      audiobench search "Python"                       Basic search
      audiobench search "quarterly" --context 100      Show surrounding text
      audiobench search --regex "meet(ing|ings)"       Regex search
      audiobench search "budget" --format ids          Just IDs (for piping)
      audiobench search "API" --format json            JSON output
    """
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    results = repo.search(query, limit=limit)

    if not results:
        if output_format == "json":
            stdout.print("[]", highlight=False)
        elif output_format == "ids":
            pass
        else:
            console.print(f'  [{DIM}]No results for "{query}"[/]')
        return

    # D2: --context: enhance text_preview with surrounding context
    if context_lines > 0:
        for r in results:
            text = r.get("text_preview", "")
            idx = text.lower().find(query.lower())
            if idx >= 0:
                start = max(0, idx - context_lines)
                end = min(len(text), idx + len(query) + context_lines)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                r["text_preview"] = prefix + text[start:end] + suffix

    # D2: --format
    if output_format == "json":
        stdout.print(json_lib.dumps(results, indent=2, default=str), highlight=False)
        return
    elif output_format == "ids":
        for r in results:
            print(r["id"])
        return

    # Default: panel format
    for r in results:
        from rich.panel import Panel

        console.print(
            Panel(
                f"[{ACCENT}]{r['file_name']}[/] ({r['language']}) — "
                f"[{DIM}]{r['created_at'][:10]}[/]\n"
                f"{r['text_preview']}",
                title=f"[{DIM}]#{r['id']}[/]",
                border_style=DIM,
            )
        )


# ── Show Command ────────────────────────────────────────────


@click.command()
@click.argument("transcription_id", type=int)
@click.option(
    "--timestamps/--no-timestamps",
    default=True,
    help="Show/hide timestamps",
)
@click.option(
    "--speakers/--no-speakers",
    default=True,
    help="Show/hide speaker labels",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "srt", "json"]),
    default="text",
    show_default=True,
    help="Display format",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Raw output without pager (for piping)",
)
def show(
    transcription_id: int,
    timestamps: bool,
    speakers: bool,
    output_format: str,
    raw: bool,
) -> None:
    """View a past transcription with timestamps.

    \b
    Examples:
      audiobench show 3                         View in pager with timestamps
      audiobench show 3 --no-timestamps         Plain text only
      audiobench show 3 --format srt            View as SRT subtitles
      audiobench show 3 --format json           View as JSON
      audiobench show 3 --raw | grep "budget"   Pipe-friendly output
    """
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    data = repo.get_by_id(transcription_id)

    if not data:
        console.print(error_panel(f"Transcription #{transcription_id} not found"))
        sys.exit(1)

    # Build output content
    if output_format == "json":
        content = json_lib.dumps(data, indent=2, default=str)
    elif output_format == "srt":
        content = _format_as_srt(data.get("segments", []))
    else:
        content = _format_as_text(data, timestamps=timestamps, speakers=speakers)

    if raw:
        print(content)
        return

    # Display with pager via Rich
    header = (
        f"[{BOLD} {ACCENT}]#{data['id']}[/]  "
        f"[{ACCENT}]{data['file_name']}[/]  "
        f"[{DIM}]{data.get('language', '?')} • "
        f"{data.get('model', '?')} • "
        f"{format_duration(data.get('duration', 0))}[/]"
    )
    console.print(header)
    console.print(f"[{DIM}]{'─' * 50}[/]")

    with console.pager(styles=True):
        console.print(content, highlight=False)


def _format_as_text(data: dict, *, timestamps: bool = True, speakers: bool = True) -> str:
    """Format transcript data as readable text with optional timestamps."""
    lines = []
    segments = data.get("segments", [])

    for seg in segments:
        parts = []

        if timestamps:
            start = seg.get("start", 0)
            minutes = int(start // 60)
            seconds = int(start % 60)
            parts.append(f"[{minutes}:{seconds:02d}]")

        if speakers and seg.get("speaker"):
            parts.append(f"<{seg['speaker']}>")

        text = seg.get("text", "").strip()
        parts.append(text)
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _format_as_srt(segments: list[dict]) -> str:
    """Format segments as SRT subtitle format."""
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


# ── Export Command ──────────────────────────────────────────


@click.command()
@click.argument("transcription_id", type=int)
@click.option(
    "-f",
    "--format",
    "output_format",
    required=True,
    type=click.Choice(["txt", "srt", "vtt", "json"]),
    help="Output format",
)
@click.option("-o", "--output", "output_path", default=None, help="Output file path")
@click.option(
    "--open",
    "open_after",
    is_flag=True,
    help="Open the exported file in the default viewer",
)
def export(
    transcription_id: int,
    output_format: str,
    output_path: str | None,
    open_after: bool = False,
) -> None:
    """Re-export a past transcription to a different format."""
    from audiobench.core.db_engine import init_db
    from audiobench.output.base import get_formatter
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.transcription_result import Segment, Transcript

    init_db()
    repo = TranscriptionRepository()
    data = repo.get_by_id(transcription_id)

    if not data:
        console.print(error_panel(f"Transcription #{transcription_id} not found"))
        sys.exit(1)

    segments = [
        Segment(
            id=s["index"],
            text=s["text"],
            start=s["start"],
            end=s["end"],
            speaker=s.get("speaker"),
        )
        for s in data.get("segments", [])
    ]

    transcript = Transcript(
        segments=segments,
        language=data.get("language", "en"),
        language_probability=data.get("language_probability", 0.0),
        duration_seconds=data.get("duration", 0.0),
        engine=data.get("engine", "faster-whisper"),
        model_name=data.get("model", "unknown"),
    )

    formatter = get_formatter(output_format)
    content = formatter.format(transcript)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(
            f"  [{SUCCESS}]✓[/] Exported #{transcription_id} as "
            f"{output_format.upper()} → [{ACCENT}]{output_path}[/]"
        )
        if open_after:
            _open_file(output_path)
    else:
        stdout.print(content, highlight=False)


# ── Delete Command ──────────────────────────────────────────


@click.command()
@click.argument("transcription_id", type=int, required=False)
@click.option("--all", "delete_all", is_flag=True, help="Delete all transcriptions")
@click.confirmation_option(prompt="Are you sure?")
def delete(transcription_id: int | None, delete_all: bool) -> None:
    """Delete transcription(s) from history."""
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()

    if delete_all:
        count = repo.delete_all()
        console.print(f"  [{SUCCESS}]✓[/] Deleted {count} transcription(s)")
    elif transcription_id is not None:
        ok = repo.delete_by_id(transcription_id)
        if ok:
            console.print(f"  [{SUCCESS}]✓[/] Deleted transcription #{transcription_id}")
        else:
            console.print(error_panel(f"Transcription #{transcription_id} not found"))
            sys.exit(1)
    else:
        console.print(error_panel("Specify a transcription ID or --all"))
        sys.exit(1)


# ── Helpers ─────────────────────────────────────────────────


def _open_file(path: str) -> None:
    """Open a file in the OS default viewer."""
    import subprocess

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            import os

            os.startfile(path)  # noqa: S606
        else:
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as e:
        console.print(f"  [{DIM}]Could not open file: {e}[/]")
