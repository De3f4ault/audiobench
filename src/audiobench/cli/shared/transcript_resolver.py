"""Transcript target resolver.

Resolves user-supplied identifiers (integer ID, #ID, relative keywords like 'latest',
audio file paths, or fuzzy titles) into a concrete transcription record from the database.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from audiobench.cli.repl.session import ReplSession

from rich.console import Console
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
from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository


def list_recent_transcripts(
    limit: int = 15,
    completed_only: bool = True,
    session: ReplSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent completed transcripts for picker and autocomplete menus."""
    with get_session() as db_session:
        query = db_session.query(TranscriptionRecord)
        if completed_only:
            query = query.filter(
                (TranscriptionRecord.pipeline_phase.in_(["complete", "aligned", "diarized", "degraded"]))
                | (TranscriptionRecord.word_count > 0)
            )
        records = (
            query.order_by(TranscriptionRecord.id.desc())
            .limit(limit)
            .all()
        )
        repo = TranscriptionRepository()
        results: list[dict[str, Any]] = []
        for r in records:
            data = repo.get_by_id(r.id, session=session)
            if data:
                results.append(data)
        return results


def resolve_transcript_target(
    target: str | int | None = None,
    *,
    interactive: bool = False,
    console: Console | None = None,
    session: ReplSession | None = None,
) -> dict[str, Any] | None:
    """Resolve a user-provided target string or ID into a transcription dictionary.

    Args:
        target: The target identifier:
            - None or "": latest transcript (or interactive picker if interactive=True)
            - int or numeric string (e.g. 42, "42", "#42", "t42", "tx42")
            - relative keyword: "latest", "last", "recent", "-1", "-2"
            - audio path: e.g. "/path/to/file.mp3" or "file.mp3"
            - query/substring: e.g. "Taoism", "Zen CD1"
        interactive: If True, prompt for disambiguation or selection in TTY.
        console: Optional Rich Console instance for user output.
        session: Optional ReplSession to check privacy_tier unlock state.

    Returns:
        The transcription dictionary from TranscriptionRepository.get_by_id(),
        or None if not found or aborted.
    """
    c = console or default_console
    repo = TranscriptionRepository()

    # ── 1. Target is None or empty ──────────────────────────────────────────
    if target is None or str(target).strip() == "":
        recent = list_recent_transcripts(limit=10, session=session)
        if not recent:
            c.print(error_panel("No Transcriptions Found", "Your library has no transcriptions yet."))
            return None

        if interactive:
            return _prompt_transcript_picker(recent, console=c, repo=repo, session=session)
        else:
            # Default to latest
            return repo.get_by_id(recent[0]["id"], session=session)

    target_str = str(target).strip()

    # ── 2. Numeric ID with optional prefixes (#42, tx42, t42, 42) ───────────
    m_id = re.match(r"^(?:#|tx|t|tx#)?(\d+)$", target_str, re.IGNORECASE)
    if m_id:
        tx_id = int(m_id.group(1))
        record = repo.get_by_id(tx_id, session=session)
        if record:
            return record
        c.print(error_panel("Not Found", f"Transcription #{tx_id} not found in database."))
        return None

    # ── 3. Relative keywords (latest, last, recent, -1, -2, ...) ───────────
    lower_target = target_str.lower()
    if lower_target in ("latest", "last", "recent"):
        recent = list_recent_transcripts(limit=1, session=session)
        if not recent:
            c.print(error_panel("Not Found", "No transcriptions available."))
            return None
        return repo.get_by_id(recent[0]["id"], session=session)

    # Negative offset index: -1 = latest, -2 = 2nd latest, etc.
    m_neg = re.match(r"^-(\d+)$", target_str)
    if m_neg:
        offset = int(m_neg.group(1))  # 1 means index 0, 2 means index 1
        if offset < 1:
            offset = 1
        recent = list_recent_transcripts(limit=offset, session=session)
        if len(recent) >= offset:
            return repo.get_by_id(recent[offset - 1]["id"], session=session)
        c.print(error_panel("Out of Range", f"Only {len(recent)} recent transcription(s) available."))
        return None

    # ── 4. File path on disk ────────────────────────────────────────────────
    path_candidate = Path(target_str).expanduser()
    if path_candidate.exists() and path_candidate.is_file():
        resolved_abs = str(path_candidate.resolve())
        # Try matching by audio_file path
        file_dict = repo.get_file_by_path(resolved_abs)
        if file_dict:
            latest_tx = repo.get_latest_transcript_for_file(file_dict["id"], session=session)
            if latest_tx:
                return repo.get_by_id(latest_tx["id"], session=session)

        # Also search by exact filename in transcriptions table
        with get_session() as db_session:
            tx = (
                db_session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.file_name == path_candidate.name)
                .order_by(TranscriptionRecord.id.desc())
                .first()
            )
            if tx:
                return repo.get_by_id(tx.id, session=session)

    # ── 5. Search by substring / title / path ────────────────────────────────
    with get_session() as db_session:
        query_pattern = f"%{target_str}%"
        matches = (
            db_session.query(TranscriptionRecord)
            .outerjoin(AudioFileRecord, TranscriptionRecord.audio_file_id == AudioFileRecord.id)
            .filter(
                (TranscriptionRecord.file_name.ilike(query_pattern))
                | (AudioFileRecord.file_name.ilike(query_pattern))
                | (AudioFileRecord.file_path.ilike(query_pattern))
            )
            .order_by(TranscriptionRecord.id.desc())
            .limit(20)
            .all()
        )

        match_ids = [m.id for m in matches]

    # If no filename match, check full-text search snippet match
    if not match_ids:
        fts_matches = repo.search(target_str, limit=5, session=session)
        match_ids = [m["id"] for m in fts_matches]

    if len(match_ids) == 1:
        return repo.get_by_id(match_ids[0], session=session)

    if len(match_ids) > 1:
        # Multiple matches — disambiguation required
        candidate_records = [repo.get_by_id(mid, session=session) for mid in match_ids if mid]
        valid_candidates = [r for r in candidate_records if r is not None]

        if interactive:
            return _disambiguate_candidates(
                valid_candidates,
                query=target_str,
                console=c,
                repo=repo,
            )
        else:
            _render_disambiguation_table(valid_candidates, query=target_str, console=c)
            c.print(f"  [{DIM}]Tip: Specify the exact transcription ID (e.g. audiobench show {valid_candidates[0]['id']})[/]\n")
            return None

    # ── 6. Zero matches — Suggest closest matches ───────────────────────────
    all_recent = list_recent_transcripts(limit=30)
    names = [r["file_name"] for r in all_recent if r.get("file_name")]
    close = difflib.get_close_matches(target_str, names, n=3, cutoff=0.4)

    err_msg = f"No transcription matching '{target_str}' found."
    if close:
        suggestions = ", ".join(f"'{c_name}'" for c_name in close)
        err_msg += f"\n\nDid you mean: {suggestions}?"
    c.print(error_panel("Transcript Not Found", err_msg))
    return None


def _render_disambiguation_table(
    candidates: list[dict[str, Any]],
    query: str,
    console: Console,
) -> None:
    """Render a clean Rich table of matching candidates."""
    table = Table(
        title=f"Multiple Transcripts Match '{query}'",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("ID", style=BOLD, width=6)
    table.add_column("File / Title", style=ACCENT)
    table.add_column("Engine / Model", style=DIM)
    table.add_column("Duration", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Created", style=DIM)

    for idx, r in enumerate(candidates, 1):
        dt = r.get("created_at", "")[:10] if r.get("created_at") else "—"
        dur = format_duration(r.get("duration", 0))
        words = f"{r.get('word_count', 0):,}"
        engine_model = f"{r.get('engine', '?')} · {r.get('model', '?')}"
        table.add_row(
            str(idx),
            f"#{r['id']}",
            r.get("file_name", "untitled"),
            engine_model,
            dur,
            words,
            dt,
        )

    console.print()
    console.print(table)


def _disambiguate_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    console: Console,
    repo: TranscriptionRepository,
) -> dict[str, Any] | None:
    """Interactive prompt to select from multiple candidates."""
    _render_disambiguation_table(candidates, query, console)

    console.print(
        f"\n  [{DIM}]Select 1-{len(candidates)}, enter a #[ID], or press Enter for #1 (or 'q' to cancel):[/]"
    )
    try:
        raw = input("  › ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]")
        return None

    if not raw:
        return candidates[0]

    if raw.lower() in ("q", "quit", "exit"):
        return None

    # Check for direct ID (#84 or 84)
    raw_id = raw.lstrip("#")
    if raw_id.isdigit():
        val = int(raw_id)
        # Direct match in candidate IDs
        for c in candidates:
            if c["id"] == val:
                return c
        # If user picked by 1-indexed table index
        if 1 <= val <= len(candidates):
            return candidates[val - 1]

    console.print(f"  [{WARNING}]Invalid selection.[/]")
    return None


def _prompt_transcript_picker(
    recent: list[dict[str, Any]],
    console: Console,
    repo: TranscriptionRepository,
    session: ReplSession | None = None,
) -> dict[str, Any] | None:
    """Prompt the user with recent transcripts when invoked without arguments."""
    table = Table(
        title="Recent Transcriptions",
        box=BOX_STYLE,
        show_header=True,
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("ID", style=BOLD, width=6)
    table.add_column("File / Title", style=ACCENT)
    table.add_column("Engine / Model", style=DIM)
    table.add_column("Duration", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Created", style=DIM)

    for idx, r in enumerate(recent, 1):
        dt = r.get("created_at", "")[:10] if r.get("created_at") else "—"
        dur = format_duration(r.get("duration", 0))
        words = f"{r.get('word_count', 0):,}"
        engine_model = f"{r.get('model', '?')}"
        table.add_row(
            str(idx),
            f"#{r['id']}",
            r.get("file_name", "untitled"),
            engine_model,
            dur,
            words,
            dt,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n  [{DIM}]Select 1-{len(recent)} or press Enter for latest (#[{recent[0]['id']}]) (or 'q' to quit):[/]"
    )

    try:
        ans = input("  › ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]")
        return None

    if not ans:
        return repo.get_by_id(recent[0]["id"], session=session)

    if ans.lower() in ("q", "quit", "exit"):
        return None

    raw = ans.lstrip("#")
    if raw.isdigit():
        val = int(raw)
        for r in recent:
            if r["id"] == val:
                return repo.get_by_id(val, session=session)
        if 1 <= val <= len(recent):
            return repo.get_by_id(recent[val - 1]["id"], session=session)

    return None
