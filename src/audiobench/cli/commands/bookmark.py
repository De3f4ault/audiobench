"""Bookmark management commands — list, add, rename, search, export/import.

Provides the ``audiobench bookmark`` CLI group for managing timestamp
markers and region annotations attached to audio files.
"""

from __future__ import annotations

import json as json_lib
import sys
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    DIM,
    SUCCESS,
    console,
    error_panel,
    make_table,
    stdout,
)
from audiobench.storage.bookmark_repository import BOOKMARK_TYPES, _format_timestamp


# ── Helpers ─────────────────────────────────────────────────


def _resolve_audio_file_id(target: str) -> int | None:
    """Resolve a file path or transcript ID to an audio_file_id."""
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    with get_session() as session:
        # Try as transcript ID
        if target.isdigit():
            rec = (
                session.query(TranscriptionRecord)
                .filter_by(id=int(target))
                .first()
            )
            if rec and rec.audio_file_id:
                return rec.audio_file_id

        # Try as file path
        resolved = str(Path(target).expanduser().resolve())
        audio = (
            session.query(AudioFileRecord)
            .filter(AudioFileRecord.file_path == resolved)
            .first()
        )
        if audio:
            return audio.id

    return None


def _get_audio_name(audio_file_id: int) -> str:
    """Get a human-readable name for an audio file."""
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord

    with get_session() as session:
        rec = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
        return rec.file_name if rec else f"audio#{audio_file_id}"


def _parse_time_or_range(time_str: str) -> tuple[float, float | None]:
    """Parse a timestamp or time range (e.g. '12:35' or '12:35-14:20').

    Returns:
        (start_seconds, end_seconds_or_None)
    """
    import re

    def _parse_ts(ts: str) -> float | None:
        match = re.match(r"(?:(\d+):)?(\d+):(\d+)", ts.strip())
        if match:
            hours = int(match.group(1) or 0)
            mins = int(match.group(2))
            secs = int(match.group(3))
            return hours * 3600 + mins * 60 + secs
        return None

    if "-" in time_str and not time_str.startswith("-"):
        parts = time_str.split("-", 1)
        start = _parse_ts(parts[0])
        end = _parse_ts(parts[1])
        if start is not None:
            return (start, end)

    parsed = _parse_ts(time_str)
    if parsed is not None:
        return (parsed, None)

    return (0.0, None)


# ── Bookmark Group ──────────────────────────────────────────


@click.group()
def bookmark() -> None:
    """Manage audio bookmarks and region markers.

    \b
    Examples:
      audiobench bookmark list                          All bookmarks
      audiobench bookmark list meeting.m4a              For specific file
      audiobench bookmark add 66 12:35                  Add at timestamp
      audiobench bookmark add 66 12:35-14:20            Add region
      audiobench bookmark rename 1 "Key Insight"        Rename
      audiobench bookmark rm 1                          Delete
      audiobench bookmark export 66 -o marks.json       Export to JSON
    """
    from audiobench.core.db_engine import init_db

    init_db()


# ── List ────────────────────────────────────────────────────


@bookmark.command(name="list")
@click.argument("target", required=False, default=None)
@click.option(
    "--type", "type_filter", default=None,
    type=click.Choice(list(BOOKMARK_TYPES.keys())),
    help="Filter by bookmark type",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["table", "json", "ids"]),
    default="table", show_default=True,
    help="Output format",
)
def list_cmd(
    target: str | None,
    type_filter: str | None,
    output_format: str,
) -> None:
    """List bookmarks, optionally for a specific file or transcript ID."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()

    if target:
        audio_id = _resolve_audio_file_id(target)
        if audio_id is None:
            console.print(
                error_panel("Not found", f"No audio file found for '{target}'")
            )
            sys.exit(1)
        bookmarks = repo.list_for_file(audio_id, type_filter=type_filter)
        file_label = _get_audio_name(audio_id)
    else:
        bookmarks = repo.list_all()
        file_label = "All Files"

    if not bookmarks:
        if output_format == "json":
            stdout.print("[]", highlight=False)
        elif output_format != "ids":
            console.print(f"  [{DIM}]No bookmarks found.[/]")
        return

    if output_format == "json":
        stdout.print(json_lib.dumps(bookmarks, indent=2, default=str), highlight=False)
        return
    elif output_format == "ids":
        for b in bookmarks:
            print(b["id"])
        return

    # Table output
    table = make_table(
        f"Bookmarks — {file_label}",
        [
            ("#", {"style": DIM, "width": 4}),
            ("Type", {"width": 4}),
            ("Time", {"width": 12}),
            ("Name", {"style": ACCENT, "max_width": 45}),
            ("Notes", {"style": DIM, "max_width": 25}),
            ("Date", {"style": DIM, "width": 10}),
        ],
    )

    for b in bookmarks:
        emoji = BOOKMARK_TYPES.get(b["bookmark_type"], "🔖")
        if b["is_region"]:
            time_str = (
                f"{_format_timestamp(b['timestamp'])}→"
                f"{_format_timestamp(b['end_timestamp'])}"
            )
        else:
            time_str = _format_timestamp(b["timestamp"])

        notes_preview = (b["notes"] or "")[:25]
        date = b["created_at"][:10] if b["created_at"] else "–"

        table.add_row(
            str(b["id"]),
            emoji,
            time_str,
            b["name"][:45],
            notes_preview,
            date,
        )

    console.print(table)


# ── Add ─────────────────────────────────────────────────────


@bookmark.command()
@click.argument("target")
@click.argument("time_str")
@click.argument("name", required=False, default=None)
@click.option(
    "--type", "bookmark_type", default="bookmark",
    type=click.Choice(list(BOOKMARK_TYPES.keys())),
    help="Bookmark type",
)
@click.option("--notes", default=None, help="Optional notes")
def add(
    target: str,
    time_str: str,
    name: str | None,
    bookmark_type: str,
    notes: str | None,
) -> None:
    """Add a bookmark at a timestamp (e.g. 12:35) or region (12:35-14:20)."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    audio_id = _resolve_audio_file_id(target)
    if audio_id is None:
        console.print(
            error_panel("Not found", f"No audio file found for '{target}'")
        )
        sys.exit(1)

    start, end = _parse_time_or_range(time_str)
    if start == 0.0 and end is None and time_str != "00:00":
        console.print(
            error_panel("Invalid time", f"'{time_str}' — expected MM:SS or MM:SS-MM:SS")
        )
        sys.exit(1)

    repo = BookmarkRepository()
    default_name = name or f"Bookmark @ {_format_timestamp(start)}"

    if end is not None:
        bid = repo.add_region(
            audio_id, start, end,
            name=default_name, bookmark_type=bookmark_type, notes=notes,
        )
        console.print(
            f"  [{SUCCESS}]✓[/] Region #{bid}: "
            f"{_format_timestamp(start)}→{_format_timestamp(end)} "
            f"{BOOKMARK_TYPES.get(bookmark_type, '🔖')} {default_name}"
        )
    else:
        bid = repo.add(
            audio_id, start,
            name=default_name, bookmark_type=bookmark_type, notes=notes,
        )
        console.print(
            f"  [{SUCCESS}]✓[/] Bookmark #{bid}: "
            f"{_format_timestamp(start)} "
            f"{BOOKMARK_TYPES.get(bookmark_type, '🔖')} {default_name}"
        )


# ── Rename ──────────────────────────────────────────────────


@bookmark.command()
@click.argument("bookmark_id", type=int)
@click.argument("new_name")
def rename(bookmark_id: int, new_name: str) -> None:
    """Rename a bookmark."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()
    if repo.update(bookmark_id, name=new_name):
        console.print(f"  [{SUCCESS}]✓[/] Renamed #{bookmark_id} → \"{new_name}\"")
    else:
        console.print(error_panel(f"Bookmark #{bookmark_id} not found"))
        sys.exit(1)


# ── Note ────────────────────────────────────────────────────


@bookmark.command()
@click.argument("bookmark_id", type=int)
@click.argument("text")
def note(bookmark_id: int, text: str) -> None:
    """Add or update notes on a bookmark."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()
    if repo.update(bookmark_id, notes=text):
        console.print(f"  [{SUCCESS}]✓[/] Updated notes on #{bookmark_id}")
    else:
        console.print(error_panel(f"Bookmark #{bookmark_id} not found"))
        sys.exit(1)


# ── Type ────────────────────────────────────────────────────


@bookmark.command(name="type")
@click.argument("bookmark_id", type=int)
@click.argument(
    "new_type",
    type=click.Choice(list(BOOKMARK_TYPES.keys())),
)
def set_type(bookmark_id: int, new_type: str) -> None:
    """Change a bookmark's type."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()
    if repo.update(bookmark_id, bookmark_type=new_type):
        emoji = BOOKMARK_TYPES.get(new_type, "🔖")
        console.print(f"  [{SUCCESS}]✓[/] #{bookmark_id} type → {emoji} {new_type}")
    else:
        console.print(error_panel(f"Bookmark #{bookmark_id} not found"))
        sys.exit(1)


# ── Remove ──────────────────────────────────────────────────


@bookmark.command()
@click.argument("bookmark_id", type=int, required=False, default=None)
@click.option("--all", "clear_all", is_flag=True, help="Delete all bookmarks for a file")
@click.option("--file", "target_file", default=None, help="Target file (with --all)")
@click.confirmation_option(prompt="Are you sure?")
def rm(bookmark_id: int | None, clear_all: bool, target_file: str | None) -> None:
    """Delete bookmark(s)."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()

    if clear_all:
        if not target_file:
            console.print(error_panel("Specify --file with --all"))
            sys.exit(1)
        audio_id = _resolve_audio_file_id(target_file)
        if audio_id is None:
            console.print(error_panel("Not found", f"No audio file for '{target_file}'"))
            sys.exit(1)
        count = repo.delete_for_file(audio_id)
        console.print(f"  [{SUCCESS}]✓[/] Deleted {count} bookmark(s)")
    elif bookmark_id is not None:
        if repo.delete(bookmark_id):
            console.print(f"  [{SUCCESS}]✓[/] Deleted bookmark #{bookmark_id}")
        else:
            console.print(error_panel(f"Bookmark #{bookmark_id} not found"))
            sys.exit(1)
    else:
        console.print(error_panel("Specify a bookmark ID or --all --file <file>"))
        sys.exit(1)


# ── Search ──────────────────────────────────────────────────


@bookmark.command()
@click.argument("query")
@click.option("--limit", default=20, help="Max results")
def search(query: str, limit: int) -> None:
    """Search bookmarks by name or notes content."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    repo = BookmarkRepository()
    results = repo.search(query, limit=limit)

    if not results:
        console.print(f'  [{DIM}]No bookmarks matching "{query}"[/]')
        return

    for b in results:
        emoji = BOOKMARK_TYPES.get(b["bookmark_type"], "🔖")
        time_str = _format_timestamp(b["timestamp"])
        console.print(
            f"    [{ACCENT}]#{b['id']}[/] {emoji} {time_str}  {b['name']}"
        )
        if b.get("notes"):
            console.print(f"         [{DIM}]{b['notes'][:60]}[/]")


# ── Export ──────────────────────────────────────────────────


@bookmark.command(name="export")
@click.argument("target")
@click.option("-o", "--output", "output_path", default=None, help="Output file")
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "audacity"]),
    default="json", show_default=True,
    help="Export format (json or audacity label track)",
)
def export_cmd(target: str, output_path: str | None, fmt: str) -> None:
    """Export bookmarks for a file as JSON or Audacity labels."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    audio_id = _resolve_audio_file_id(target)
    if audio_id is None:
        console.print(error_panel("Not found", f"No audio file for '{target}'"))
        sys.exit(1)

    repo = BookmarkRepository()
    if fmt == "audacity":
        data = repo.export_audacity(audio_id)
    else:
        data = repo.export_json(audio_id)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(data, encoding="utf-8")
        console.print(
            f"  [{SUCCESS}]✓[/] Exported ({fmt}) → [{ACCENT}]{output_path}[/]"
        )
    else:
        stdout.print(data, highlight=False)


# ── Import ──────────────────────────────────────────────────


@bookmark.command(name="import")
@click.argument("target")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "audacity", "auto"]),
    default="auto", show_default=True,
    help="Import format (auto-detects from content)",
)
def import_cmd(target: str, input_file: str, fmt: str) -> None:
    """Import bookmarks from a JSON or Audacity label file."""
    from audiobench.storage.bookmark_repository import BookmarkRepository

    audio_id = _resolve_audio_file_id(target)
    if audio_id is None:
        console.print(error_panel("Not found", f"No audio file for '{target}'"))
        sys.exit(1)

    data = Path(input_file).read_text(encoding="utf-8")
    repo = BookmarkRepository()

    # Auto-detect format
    if fmt == "auto":
        stripped = data.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            fmt = "json"
        else:
            fmt = "audacity"

    if fmt == "audacity":
        count = repo.import_audacity(audio_id, data)
    else:
        count = repo.import_json(audio_id, data)

    console.print(f"  [{SUCCESS}]✓[/] Imported {count} bookmark(s) ({fmt} format)")


# ── AI Auto-Bookmarking ────────────────────────────────────


@bookmark.command(name="auto")
@click.argument("target")
@click.option("--model", default=None, help="Override AI model (default: from settings)")
@click.option(
    "--focus",
    default=None,
    help="Focus extraction (e.g. 'action items only', 'key decisions')",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without saving",
)
def auto_cmd(target: str, model: str | None, focus: str | None, dry_run: bool) -> None:
    """Use AI to automatically identify and bookmark key moments.

    \b
    Examples:
      audiobench bookmark auto 66                       Auto-bookmark transcript #66
      audiobench bookmark auto 66 --focus "action items" Focus on action items
      audiobench bookmark auto 66 --model deepseek-v3.2:cloud  Use a specific model
      audiobench bookmark auto 66 --dry-run              Preview without saving
    """
    import re

    from audiobench.chat.context_builder import AUTO_BOOKMARK_SYSTEM, auto_bookmark
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.core.settings import get_settings
    from audiobench.storage.bookmark_repository import BookmarkRepository
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.bookmark_model

    # ── Resolve transcript ──
    init_db()
    repo = TranscriptionRepository()

    # Try as transcript ID
    record = None
    try:
        tid = int(target)
        record = repo.get_by_id(tid)
    except ValueError:
        pass

    if record is None:
        console.print(error_panel("Not found", f"Transcript '{target}' not found"))
        sys.exit(1)

    audio_file_id = record.get("audio_file_id")
    if audio_file_id is None:
        console.print(error_panel("No audio link", "Transcript has no linked audio file"))
        sys.exit(1)

    segments = record.get("segments", [])
    if not segments:
        console.print(error_panel("No segments", "Transcript has no timed segments"))
        sys.exit(1)

    # ── Build timestamped transcript ──
    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        mins, secs = int(start // 60), int(start % 60)
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{mins:02d}:{secs:02d}] {text}")

    timestamped_text = "\n".join(lines)

    # ── Header ──
    console.print()
    console.print(f"  [{ACCENT}]🤖 AI Auto-Bookmark[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:  [{ACCENT}]#{record['id']} {record['file_name']}[/]")
    console.print(f"    Model:   {model_name}")
    console.print(f"    Segments: {len(segments)}")
    if focus:
        console.print(f"    Focus:   {focus}")
    if dry_run:
        console.print(f"    Mode:    [yellow]DRY RUN[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # ── Call AI ──
    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model_name,
        )

        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not running",
                    f"Start with: ollama serve\nThen pull: ollama pull {model_name}",
                )
            )
            sys.exit(1)

        prompt = auto_bookmark(timestamped_text, focus=focus)

        console.print(f"  [{DIM}]Analyzing transcript...[/]")

        response = client.generate(
            prompt,
            model=model_name,
            system_prompt=AUTO_BOOKMARK_SYSTEM,
            temperature=0.2,
        )

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))
        sys.exit(1)

    # ── Parse JSON response ──
    # Strip markdown fences if the model wraps output
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Extract <think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    try:
        bookmarks_data = json_lib.loads(cleaned)
    except json_lib.JSONDecodeError as e:
        console.print(error_panel("Parse error", f"AI returned invalid JSON:\n{e}\n\nRaw:\n{cleaned[:500]}"))
        sys.exit(1)

    if not isinstance(bookmarks_data, list):
        console.print(error_panel("Parse error", "AI response is not a JSON array"))
        sys.exit(1)

    if not bookmarks_data:
        console.print(f"  [{DIM}]AI found no notable moments in this transcript.[/]")
        return

    # ── Validate and display ──
    valid_types = {"bookmark", "highlight", "todo", "note", "edit"}
    bm_repo = BookmarkRepository()
    created = 0

    table = make_table(
        "AI Bookmarks",
        columns=[
            ("Type", {"justify": "center", "width": 6}),
            ("Time", {"justify": "right", "width": 14}),
            ("Name", {"justify": "left", "width": 50}),
            ("Notes", {"justify": "left", "width": 30}),
        ],
    )

    for item in bookmarks_data:
        ts = item.get("timestamp")
        if ts is None:
            continue

        try:
            ts = float(ts)
        except (ValueError, TypeError):
            continue

        end_ts = item.get("end_timestamp")
        if end_ts is not None:
            try:
                end_ts = float(end_ts)
            except (ValueError, TypeError):
                end_ts = None

        name = str(item.get("name", "Untitled"))[:80]
        bm_type = str(item.get("type", "bookmark"))
        if bm_type not in valid_types:
            bm_type = "bookmark"
        notes = item.get("notes")
        if notes:
            notes = str(notes)[:200]

        emoji = BOOKMARK_TYPES.get(bm_type, "🔖")
        time_str = _format_timestamp(ts)
        if end_ts:
            time_str += f"→{_format_timestamp(end_ts)}"

        table.add_row(
            emoji,
            time_str,
            name,
            (notes or "")[:30],
        )

        if not dry_run:
            if end_ts:
                bm_repo.add_region(
                    audio_file_id=audio_file_id,
                    start_timestamp=ts,
                    end_timestamp=end_ts,
                    name=name,
                    bookmark_type=bm_type,
                    notes=notes,
                    transcription_id=record.get("id"),
                )
            else:
                bm_repo.add(
                    audio_file_id=audio_file_id,
                    timestamp=ts,
                    name=name,
                    bookmark_type=bm_type,
                    notes=notes,
                    transcription_id=record.get("id"),
                )
            created += 1

    console.print(table)
    console.print()

    if dry_run:
        console.print(f"  [yellow]⚠ Dry run — {len(bookmarks_data)} bookmark(s) NOT saved[/]")
    else:
        console.print(f"  [{SUCCESS}]✓[/] Created {created} bookmark(s) via AI")
    console.print()

