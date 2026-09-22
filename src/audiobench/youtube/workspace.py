"""YouTube channel workspace — Phase 3.

Replaces `youtube/repl.py` as the entry point for `audiobench youtube`.

Two views:
  1. Home — ordered channel list (engagement weight, computed at render time)
  2. Channel workspace — whiteboard + catalog browse/search + slash commands

Denominator decision (explicit):
  The progress fraction "89/312" uses video_count_available as denominator.
  video_count_total (which includes private/removed placeholders) appears only
  as a footnote when it differs from available_count. See implementation_plan.md.

Draft sentence decision:
  Never persisted. Generated fresh on each workspace entry from the playlist diff.
  Never written to whiteboard_text or any other field.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.table import Table

from audiobench.cli.display.theme import (
    ACCENT, DIM, ERROR, SUCCESS, WARNING, console, error_panel
)
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.youtube.channel_store import (
    compute_engagement_weight,
    get_all_channels,
    get_channel,
    get_or_create_channel,
    get_whiteboard,
    mark_visited,
    update_whiteboard,
    get_library_count,
)
from audiobench.youtube.playlist import (
    cache_is_stale,
    compute_gap,
    on_workspace_entry,
    refresh_channel_cache,
)
from audiobench.youtube.search import (
    SearchResultSet,
    VideoResult,
    YouTubeAPIError,
    parse_selection,
    resolve_channel,
    search_videos,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger("youtube.workspace")

PAGE_SIZE = 15
RULE = "─" * 56


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _ChannelState:
    """Per-channel REPL state. Replaced entirely on /switch."""
    channel_id: str
    channel_title: str
    # Catalog browse (playlist cache, chronological)
    browse_page: int = 1
    browse_entries: list[dict] = dataclasses.field(default_factory=list)
    browse_total: int = 0
    # Search (search.list, relevance-ranked)
    search_query: str | None = None
    search_results: list[VideoResult] = dataclasses.field(default_factory=list)
    search_next_token: str | None = None
    search_prev_token: str | None = None
    search_page: int = 1
    # Active display — which result list is live for fetch/info
    mode: str = "browse"   # "browse" | "search" | "gap"
    # gap view (subset of browse_entries)
    gap_entries: list[dict] = dataclasses.field(default_factory=list)
    # Ephemeral draft sentence — never persisted
    _draft_sentence: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rule() -> None:
    console.print(f"[{DIM}]{RULE}[/]")


def _format_age(dt: datetime | None) -> str:
    if dt is None:
        return "never"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - dt
    days = delta.days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return f"{months} month{'s' if months > 1 else ''} ago"


def _render_channel_header(state: _ChannelState, session: Session) -> None:
    """Print the per-channel workspace header."""
    from audiobench.storage.models import YouTubePlaylistCache
    cache = session.query(YouTubePlaylistCache).filter_by(
        channel_id=state.channel_id
    ).first()

    library = get_library_count(state.channel_id, session)

    if cache:
        available = cache.video_count_available
        total = cache.video_count_total
        private_count = total - available
        progress = f"{library}/{available} in library"
        private_note = (
            f"  [{DIM}](+ {private_count} private/unavailable)[/]"
            if private_count > 0 else ""
        )
    else:
        progress = f"{library}/? in library  [{DIM}](cache not loaded)[/]"
        private_note = ""

    console.print()
    console.print(
        f"[bold {ACCENT}]{state.channel_title.upper()}[/]"
        f"  [{DIM}]—[/]  {progress}{private_note}"
    )
    _rule()


def _render_whiteboard(channel_id: str, draft: str | None, session: Session) -> None:
    """Print whiteboard note (user-written) and draft sentence (ephemeral)."""
    from audiobench.storage.models import YouTubeChannelNode
    node = session.query(YouTubeChannelNode).filter_by(channel_id=channel_id).first()

    if node and node.whiteboard_text:
        age = _format_age(node.whiteboard_updated_at)
        console.print(f"  [{DIM}]\"{node.whiteboard_text}\"[/]")
        console.print(f"  [{DIM}][{age} · /note to update][/]")
    else:
        console.print(f"  [{DIM}](No whiteboard note. /note to add one.)[/]")

    # Draft sentence — ephemeral, never persisted, shown below the note
    if draft:
        console.print(f"  [{DIM}]↳ {draft}[/]")

    console.print()


def _render_browse_page(state: _ChannelState, session: Session) -> None:
    """Render one page of the uploads playlist (chronological, newest-first)."""
    from audiobench.storage.models import YouTubePlaylistCache, AudioFileRecord

    cache = session.query(YouTubePlaylistCache).filter_by(
        channel_id=state.channel_id
    ).first()
    if not cache:
        console.print(f"[{WARNING}]Playlist cache not loaded. Run /refresh.[/]")
        return

    entries: list[dict] = json.loads(cache.videos_json)
    available = [e for e in entries if e["availability"] != "private_or_removed"]
    state.browse_total = len(available)
    state.browse_entries = available

    start = (state.browse_page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_entries = available[start:end]

    if not page_entries:
        console.print(f"[{DIM}]No videos on this page.[/]")
        return

    # Which video IDs are in library?
    in_library = {
        row.youtube_video_id
        for row in session.query(AudioFileRecord)
        .filter(
            AudioFileRecord.youtube_channel_id == state.channel_id,
            AudioFileRecord.youtube_video_id.isnot(None),
        )
        .all()
    }

    term_width = shutil.get_terminal_size().columns
    title_max = max(24, term_width - 28)

    total_pages = (state.browse_total + PAGE_SIZE - 1) // PAGE_SIZE
    page_label = f"Page {state.browse_page}/{total_pages}" if total_pages > 1 else ""
    tbl_title = (
        f"[bold {ACCENT}]{state.channel_title}[/]"
        + (f"  [{DIM}]{page_label}[/]" if page_label else "")
    )

    table = Table(
        title=tbl_title, show_header=True,
        header_style=f"bold {ACCENT}", box=None,
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("", width=2, no_wrap=True)   # status badge
    table.add_column("Title")
    table.add_column("Published", justify="right", no_wrap=True, style="dim")

    global_base = start + 1
    for i, entry in enumerate(page_entries):
        n = global_base + i
        badge = "✓" if entry["video_id"] in in_library else " "
        badge_style = SUCCESS if badge == "✓" else "dim"
        t = entry["title"]
        if len(t) > title_max:
            t = t[:title_max - 1] + "…"
        table.add_row(
            str(n), f"[{badge_style}]{badge}[/]", t, entry["published_at"]
        )

    console.print()
    console.print(table)

    hints = [f"[cyan]fetch <1-3 | 1,3 | all>[/]", "[cyan]/gap[/] for unwatched"]
    if end < state.browse_total:
        hints.append("[cyan]next[/] for older")
    if state.browse_page > 1:
        hints.append("[cyan]prev[/] for newer")
    hints.append("[cyan]?[/] for help")
    console.print(f"\n[{DIM}]" + "  ·  ".join(hints) + "[/]")


def _render_gap_view(state: _ChannelState, session: Session) -> None:
    """Render videos not yet in library (gap view)."""
    gap = compute_gap(state.channel_id, session)
    state.mode = "gap"
    state.gap_entries = gap["gap_videos"]

    if not gap["gap_videos"]:
        console.print(
            f"\n[{SUCCESS}]Library complete —[/] all {gap['available_count']} "
            f"available videos are in library."
        )
        return

    term_width = shutil.get_terminal_size().columns
    title_max = max(24, term_width - 28)
    private_note = (
        f"  [{DIM}]({gap['total_count'] - gap['available_count']} private/unavailable not shown)[/]"
        if gap["total_count"] != gap["available_count"] else ""
    )

    tbl_title = (
        f"[bold {ACCENT}]{state.channel_title} — gap[/]  "
        f"[{DIM}]{gap['gap_count']}/{gap['available_count']} remaining[/]"
        + private_note
    )

    table = Table(
        title=tbl_title, show_header=True,
        header_style=f"bold {ACCENT}", box=None,
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Title")
    table.add_column("Published", justify="right", no_wrap=True, style="dim")

    for i, entry in enumerate(gap["gap_videos"][:50], 1):
        t = entry["title"]
        if len(t) > title_max:
            t = t[:title_max - 1] + "…"
        table.add_row(str(i), t, entry["published_at"])

    if len(gap["gap_videos"]) > 50:
        table.add_row("…", f"[{DIM}]and {len(gap['gap_videos']) - 50} more[/]", "")

    console.print()
    console.print(table)
    console.print(
        f"\n[{DIM}]fetch <1-5 | all>[/] to queue · "
        f"[{DIM}]browse[/] to return to full catalog"
    )


def _render_search_results(state: _ChannelState) -> None:
    """Render search results for the current channel."""
    if not state.search_results:
        console.print(f"[{DIM}]No results.[/]")
        return

    term_width = shutil.get_terminal_size().columns
    title_max = max(24, term_width - 36)
    total_pages_label = f" [{DIM}](page {state.search_page})[/]" if state.search_page > 1 else ""

    table = Table(
        title=f"[bold {ACCENT}]{state.channel_title}[/] — \"{state.search_query}\"{total_pages_label}",
        show_header=True, header_style=f"bold {ACCENT}", box=None,
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Title")
    table.add_column("Duration", justify="right", no_wrap=True, style="dim")
    table.add_column("Published", justify="right", no_wrap=True, style="dim")

    for r in state.search_results:
        t = r.title if len(r.title) <= title_max else r.title[:title_max - 1] + "…"
        table.add_row(str(r.n), t, r.duration_str or "", r.published_at)

    console.print()
    console.print(table)

    hints = ["[cyan]fetch <N>[/]", "[cyan]info <N>[/]"]
    if state.search_next_token:
        hints.append("[cyan]next[/]")
    if state.search_prev_token or state.search_page > 1:
        hints.append("[cyan]prev[/]")
    hints.append("[cyan]browse[/] for catalog")
    console.print(f"\n[{DIM}]" + "  ·  ".join(hints) + "[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────────────────────────────────────

def _active_results(state: _ChannelState) -> list:
    """Return whichever result list is currently active for fetch/info."""
    if state.mode == "search":
        return state.search_results
    if state.mode == "gap":
        return state.gap_entries
    # browse mode — return current page entries with global numbering
    start = (state.browse_page - 1) * PAGE_SIZE
    return state.browse_entries[start: start + PAGE_SIZE]


def _handle_fetch(parts: list[str], state: _ChannelState) -> None:
    """Queue selected videos for download."""
    from audiobench.jobs.scheduler import enqueue, ensure_worker
    from audiobench.storage.models import AudioFileRecord

    if len(parts) < 2:
        console.print(f"[{WARNING}]Usage:[/] fetch <1-5 | 1,3,7 | all>")
        return

    active = _active_results(state)
    if not active:
        console.print(f"[{WARNING}]No results to select from. Browse or search first.[/]")
        return

    # Build an id→title map. Entries can be VideoResult (search) or dict (browse/gap).
    def _vid(entry) -> str:
        return entry.video_id if isinstance(entry, VideoResult) else entry["video_id"]

    def _ttl(entry) -> str:
        return entry.title if isinstance(entry, VideoResult) else entry["title"]

    raw = " ".join(parts[1:])
    try:
        global_base = (state.browse_page - 1) * PAGE_SIZE + 1 if state.mode == "browse" else 1
        indices = parse_selection(raw, max_count=len(active), start_n=global_base)
    except TypeError:
        # parse_selection may not support start_n — fall back
        try:
            indices = parse_selection(raw, max_count=len(active))
        except ValueError as e:
            console.print(f"[{WARNING}]Selection error:[/] {e}")
            return
    except ValueError as e:
        console.print(f"[{WARNING}]Selection error:[/] {e}")
        return

    # Map indices to entries — indices are 1-based global page numbers in browse
    if state.mode == "browse":
        base = (state.browse_page - 1) * PAGE_SIZE
        local_map = {base + i + 1: entry for i, entry in enumerate(active)}
        targets = [local_map[idx] for idx in indices if idx in local_map]
    else:
        local_map = {i + 1: entry for i, entry in enumerate(active)}
        targets = [local_map[idx] for idx in indices if idx in local_map]

    if not targets:
        console.print(f"[{WARNING}]No matching videos for selection.[/]")
        return

    queued, skipped = [], []
    with get_session() as db:
        for entry in targets:
            vid = _vid(entry)
            existing = db.query(AudioFileRecord).filter_by(youtube_video_id=vid).first()
            if existing:
                skipped.append((vid, _ttl(entry)))
            else:
                job_id = enqueue(
                    job_type="youtube_fetch",
                    slot="network",
                    args=["youtube", "_fetch_internal", vid],
                    file_label=_ttl(entry),
                    command_display=f"youtube download {vid}",
                )
                queued.append((vid, _ttl(entry), job_id))

    if queued:
        ensure_worker()

    console.print()
    if queued:
        if len(queued) == 1:
            vid, title, job_id = queued[0]
            console.print(f"[{SUCCESS}]Queued[/] · Job #{job_id} · {title}")
        else:
            console.print(f"[{SUCCESS}]Queued {len(queued)} download(s):[/]")
            for vid, title, job_id in queued:
                console.print(f"  • Job [bold]#{job_id}[/bold]: {title}")
        console.print(
            f"[{DIM}]Run[/] [bold]audiobench jobs list[/bold] [{DIM}]to check status.[/]"
        )
    if skipped:
        console.print(f"[{DIM}]Already in library ({len(skipped)} skipped)[/]")


def _handle_info(parts: list[str], state: _ChannelState) -> None:
    """Show metadata for a video by selection index."""
    if len(parts) < 2:
        console.print(f"[{WARNING}]Usage:[/] info <N>")
        return

    active = _active_results(state)
    try:
        idx = int(parts[1])
    except ValueError:
        console.print(f"[{WARNING}]info takes a single number.[/]")
        return

    if state.mode == "browse":
        base = (state.browse_page - 1) * PAGE_SIZE + 1
        entry = next((e for i, e in enumerate(active) if base + i == idx), None)
    else:
        entry = active[idx - 1] if 1 <= idx <= len(active) else None

    if entry is None:
        console.print(f"[{WARNING}]No entry #{idx} on screen.[/]")
        return

    if isinstance(entry, VideoResult):
        console.print(f"\n[{ACCENT}]#{idx} · {entry.title}[/]")
        console.print(f"[{DIM}]ID:[/] {entry.video_id}  [{DIM}]Duration:[/] {entry.duration_str or '?'}  [{DIM}]Published:[/] {entry.published_at}")
        if entry.description:
            console.print(f"\n[{DIM}]Description:[/]\n{entry.description}\n")
    else:
        console.print(f"\n[{ACCENT}]#{idx} · {entry['title']}[/]")
        url = f"https://www.youtube.com/watch?v={entry['video_id']}"
        console.print(f"[{DIM}]ID:[/] {entry['video_id']}  [{DIM}]Published:[/] {entry['published_at']}  [{DIM}]Availability:[/] {entry['availability']}")
        console.print(f"[{DIM}]URL:[/] {url}")


def _handle_channel_search(query: str, state: _ChannelState, page_token: str | None = None) -> None:
    """Run search.list restricted to this channel. Switches to search mode."""
    start_n = (state.search_page - 1) * PAGE_SIZE + 1 if page_token else 1
    if not page_token:
        state.search_page = 1
        start_n = 1

    try:
        with console.status(f"[{DIM}]Searching…[/]"):
            results = search_videos(
                query=query,
                channel_id=state.channel_id,
                max_results=PAGE_SIZE,
                sort="relevance",
                page_token=page_token,
                start_n=start_n,
            )
    except YouTubeAPIError as e:
        console.print(error_panel("Search failed", str(e)))
        return
    except Exception as e:
        console.print(error_panel("Error", str(e)))
        return

    state.mode = "search"
    state.search_query = query
    state.search_results = results.results
    state.search_next_token = results.next_page_token
    state.search_prev_token = results.prev_page_token
    _render_search_results(state)


# ─────────────────────────────────────────────────────────────────────────────
# Slash command handlers (channel workspace level)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_slash_workspace(
    user_input: str, state: _ChannelState, session: Session
) -> str:
    """
    Handle slash commands inside a channel workspace.

    Returns:
      "continue" — keep looping
      "switch"   — return to home view
      "exit"     — leave workspace entirely
    """
    import shlex
    try:
        parts = shlex.split(user_input)
    except ValueError:
        parts = user_input.split()
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        return "exit"

    if cmd in ("/switch", "/home", "/back"):
        return "switch"

    if cmd == "/gap":
        _render_gap_view(state, session)
        return "continue"

    if cmd in ("/browse", "/catalog"):
        state.mode = "browse"
        state.browse_page = 1
        _render_browse_page(state, session)
        return "continue"

    if cmd == "/refresh":
        console.print(f"[{DIM}]Refreshing playlist cache…[/]")
        with get_session() as fresh_session:
            refresh_channel_cache(state.channel_id, fresh_session)
            fresh_session.commit()
        state.mode = "browse"
        state.browse_page = 1
        with get_session() as fresh_session:
            _render_browse_page(state, fresh_session)
        return "continue"

    if cmd == "/note":
        _handle_note_edit(state, session)
        return "continue"

    if cmd in ("/help", "/commands"):
        _print_workspace_help()
        return "continue"

    console.print(f"[{WARNING}]Unknown command:[/] {cmd}  (type /help for commands)")
    return "continue"


def _handle_note_edit(state: _ChannelState, session: Session) -> None:
    """Inline whiteboard editor — single line for now."""
    from prompt_toolkit import prompt as pt_prompt
    existing = get_whiteboard(state.channel_id, session)
    console.print(f"[{DIM}]Current note:[/] {existing or '(empty)'}")
    console.print(f"[{DIM}]Enter new note (blank to clear, Ctrl-C to cancel):[/]")
    try:
        new_text = pt_prompt("> ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"[{DIM}]Cancelled.[/]")
        return
    update_whiteboard(state.channel_id, new_text or None, session)
    session.commit()
    if new_text:
        console.print(f"[{SUCCESS}]Note saved.[/]")
    else:
        console.print(f"[{DIM}]Note cleared.[/]")


def _print_workspace_help() -> None:
    console.print(f"\n[{ACCENT}]Browse & Search:[/]")
    console.print(f"  [cyan]<empty>[/]           Browse uploads (chronological, newest-first)")
    console.print(f"  [cyan]<query>[/]            Search this channel by relevance")
    console.print(f"  [cyan]next / n[/]           Next page")
    console.print(f"  [cyan]prev / p[/]           Previous page")
    console.print(f"\n[{ACCENT}]Actions:[/]")
    console.print(f"  [cyan]fetch <1-5 | all>[/]  Queue for download")
    console.print(f"  [cyan]info <N>[/]           Show video metadata + URL")
    console.print(f"\n[{ACCENT}]Channel:[/]")
    console.print(f"  [cyan]/gap[/]               Videos not yet in library")
    console.print(f"  [cyan]/note[/]              Edit whiteboard note")
    console.print(f"  [cyan]/refresh[/]           Force playlist cache refresh")
    console.print(f"  [cyan]/switch[/]            Return to channel home")
    console.print(f"  [cyan]/exit[/]              Leave YouTube workspace\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel workspace loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_channel_workspace(channel_id: str) -> str:
    """
    Run the per-channel workspace REPL.

    Returns "switch" to return to home, "exit" to leave entirely.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    with get_session() as init_session:
        node = get_channel(channel_id, init_session)
        if node is None:
            # Should not happen — caller (home view or /add) ensures the node exists
            logger.error("_run_channel_workspace: channel node missing for %s", channel_id)
            console.print(f"[{WARNING}]Channel not found in database. Try /add <name> again.[/]")
            return "switch"
        mark_visited(channel_id, init_session)
        init_session.commit()
        channel_title = node.title

    # On workspace entry: refresh if stale, else check for new uploads (1–2 units)
    with get_session() as ws_session:
        entry_result = on_workspace_entry(channel_id, ws_session)
        ws_session.commit()

    # Build ephemeral draft sentence — never persisted
    draft: str | None = None
    api_error = entry_result.get("api_error", False)
    if not entry_result["refreshed"] and entry_result["new_videos"]:
        n = len(entry_result["new_videos"])
        draft = f"{n} new upload{'s' if n > 1 else ''} since last visit."

    # Fallback: if title is somehow still empty (shouldn't happen after the guard fix),
    # use the channel ID so the prompt is never blank.
    display_title = channel_title or channel_id

    state = _ChannelState(
        channel_id=channel_id,
        channel_title=display_title,
        _draft_sentence=draft,
    )

    with get_session() as render_session:
        _render_channel_header(state, render_session)
        if api_error:
            console.print(
                f"  [{DIM}]⚠ YouTube API unavailable — showing cached catalog. "
                f"Set AUDIOBENCH_YOUTUBE_API_KEY to check for new uploads.[/]"
            )
            console.print()
        _render_whiteboard(channel_id, draft, render_session)
        # Default view: catalog browse
        _render_browse_page(state, render_session)

    pt_style = Style.from_dict({"prompt": "ansicyan bold"})
    pt_session = PromptSession(style=pt_style)

    while True:
        try:
            prompt_text = [("class:prompt", f"{state.channel_title.lower()[:16]}> ")]
            user_input = pt_session.prompt(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "exit"

        if not user_input:
            # Empty input in browse mode → show current page; in search → re-show
            with get_session() as db:
                if state.mode in ("browse", "gap"):
                    _render_browse_page(state, db)
                else:
                    state.mode = "browse"
                    state.browse_page = 1
                    _render_browse_page(state, db)
            continue

        if user_input.startswith("/"):
            with get_session() as db:
                result = _handle_slash_workspace(user_input, state, db)
            if result in ("switch", "exit"):
                return result
            continue

        parts = user_input.split()
        first = parts[0].lower()

        if first in ("next", "n"):
            with get_session() as db:
                if state.mode == "search" and state.search_next_token:
                    state.search_page += 1
                    _handle_channel_search(
                        state.search_query or "", state, page_token=state.search_next_token
                    )
                elif state.mode == "browse":
                    total_pages = (state.browse_total + PAGE_SIZE - 1) // PAGE_SIZE
                    if state.browse_page < total_pages:
                        state.browse_page += 1
                        _render_browse_page(state, db)
                    else:
                        console.print(f"[{DIM}]Last page.[/]")
                else:
                    console.print(f"[{DIM}]No next page.[/]")
            continue

        if first in ("prev", "p"):
            with get_session() as db:
                if state.mode == "search" and state.search_prev_token:
                    state.search_page = max(1, state.search_page - 1)
                    _handle_channel_search(
                        state.search_query or "", state, page_token=state.search_prev_token
                    )
                elif state.mode == "browse" and state.browse_page > 1:
                    state.browse_page -= 1
                    _render_browse_page(state, db)
                else:
                    console.print(f"[{DIM}]No previous page.[/]")
            continue

        if first in ("browse", "catalog"):
            state.mode = "browse"
            state.browse_page = 1
            with get_session() as db:
                _render_browse_page(state, db)
            continue

        if first == "fetch":
            _handle_fetch(parts, state)
            continue

        if first == "info":
            _handle_info(parts, state)
            continue

        if user_input in ("?", "help"):
            _print_workspace_help()
            continue

        # Everything else is a search query
        state.search_query = user_input
        state.search_page = 1
        _handle_channel_search(user_input, state)


# ─────────────────────────────────────────────────────────────────────────────
# Home view
# ─────────────────────────────────────────────────────────────────────────────

def _run_home(session: Session) -> tuple[str, str | None]:
    """
    Render the channel home view and get user selection.

    Returns (action, channel_id):
      ("open", channel_id) — user selected a channel
      ("add", None)        — user typed /add <name> (handled by caller)
      ("exit", None)       — user exited
    The caller sees the raw user input for /add handling.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    channels = get_all_channels(session)
    weights = [compute_engagement_weight(c) for c in channels]

    if not channels:
        console.print(f"\n[{ACCENT}]YouTube channel workspace[/]")
        _rule()
        console.print(f"  [{DIM}]No channels yet.[/]  /add <name or URL> to begin.")
        console.print()
    else:
        console.print(f"\n[{ACCENT}]Your channels[/]  [{DIM}](ordered by engagement)[/]")
        _rule()

        term_width = shutil.get_terminal_size().columns
        for ch, w in zip(channels, weights):
            with get_session() as db:
                library = get_library_count(ch.channel_id, db)
                from audiobench.storage.models import YouTubePlaylistCache
                cache = db.query(YouTubePlaylistCache).filter_by(
                    channel_id=ch.channel_id
                ).first()
            available = cache.video_count_available if cache else None
            progress = (
                f"{library}/{available}" if available is not None else f"{library}/?"
            )
            last = _format_age(ch.last_visited_at)
            console.print(
                f"  [bold]{ch.title}[/bold]"
                f"  [{DIM}]{progress} in library  ·  {last}[/]"
            )

        _rule()
        console.print(
            f"  [{DIM}]Type a channel name to open  ·  /add <name> to add  ·  /exit to quit[/]"
        )
        console.print()

    pt_style = Style.from_dict({"prompt": "ansicyan bold"})
    pt_session = PromptSession(style=pt_style)

    while True:
        try:
            user_input = pt_session.prompt([("class:prompt", "youtube> ")]).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "exit", None

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            return "exit", None

        if user_input.lower().startswith("/add "):
            return "add", user_input[5:].strip()

        if user_input.lower() in ("/help", "/commands"):
            console.print(
                f"[{DIM}]Type a channel name to open it, "
                "/add <name or URL> to add, /exit to quit.[/]"
            )
            continue

        # Try to match against known channels
        query = user_input.lower()
        matches = [c for c in channels if query in c.title.lower()]
        if len(matches) == 1:
            return "open", matches[0].channel_id
        if len(matches) > 1:
            console.print(
                f"[{WARNING}]Ambiguous:[/] "
                + ", ".join(m.title for m in matches)
                + " — be more specific."
            )
            continue

        # No match — try resolving as a new channel name
        console.print(
            f"[{DIM}]'{user_input}' not in your channels. "
            f"Use /add {user_input} to add it.[/]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_channel_workspace() -> None:
    """Main entry point. Called by `audiobench youtube` with no subcommand."""
    with get_session() as session:
        channels = get_all_channels(session)

    # If only one channel, jump straight into it
    if len(channels) == 1:
        active_channel_id = channels[0].channel_id
    else:
        active_channel_id = None

    while True:
        if active_channel_id is None:
            with get_session() as session:
                action, payload = _run_home(session)

            if action == "exit":
                break

            if action == "add":
                # Resolve and create the new channel node
                name_or_url = payload or ""
                with get_session() as session:
                    try:
                        with console.status(f"[{DIM}]Resolving channel…[/]"):
                            channel_id, channel_title = resolve_channel(name_or_url, session)
                        node, created = get_or_create_channel(channel_id, channel_title, session)
                        session.commit()
                        console.print(
                            f"[{SUCCESS}]Added:[/] {channel_title} ({channel_id})"
                            + (" (already known)" if not created else "")
                        )
                        active_channel_id = channel_id
                    except Exception as e:
                        console.print(error_panel("Could not resolve channel", str(e)))
                continue

            # action == "open"
            active_channel_id = payload

        # Run the per-channel workspace
        result = _run_channel_workspace(active_channel_id)

        if result == "exit":
            break
        if result == "switch":
            active_channel_id = None
            continue
