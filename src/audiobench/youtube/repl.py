"""Interactive REPL loop for YouTube integration."""

import dataclasses
import shlex
import shutil
from typing import Any

from rich.console import Console
from rich.table import Table

from audiobench.cli.display.theme import console, SUCCESS, WARNING, ERROR, ACCENT, DIM, error_panel
from audiobench.core.db_session import get_session
from audiobench.youtube.search import (
    search_videos, 
    resolve_channel, 
    parse_selection, 
    VideoResult, 
    SearchResultSet,
    YouTubeAPIError
)

@dataclasses.dataclass
class YouTubeSessionState:
    channel_id: str | None = None
    channel_title: str | None = None
    sort: str = "relevance"
    limit: int = 15
    after: str | None = None
    before: str | None = None
    last_query: str | None = None
    is_browse_mode: bool = False
    last_results: list[VideoResult] = dataclasses.field(default_factory=list)
    next_page_token: str | None = None
    prev_page_token: str | None = None
    current_page: int = 1


def _display_results_table(result_set: SearchResultSet, state: YouTubeSessionState, title_prefix: str = "") -> None:
    """Render a formatted table for search or browse results."""
    state.last_results = result_set.results
    state.next_page_token = result_set.next_page_token
    state.prev_page_token = result_set.prev_page_token

    if not result_set.results:
        console.print(f"[{DIM}]No videos found.[/]")
        return

    term_width = shutil.get_terminal_size().columns
    title_max = max(20, term_width - 40)

    tbl_title = title_prefix if title_prefix else (
        f"Videos in {state.channel_title}" if state.is_browse_mode and state.channel_title
        else f"Results for '{state.last_query}'" if state.last_query
        else "Search Results"
    )
    if state.current_page > 1:
        tbl_title += f" [dim](Page {state.current_page})[/dim]"

    table = Table(title=tbl_title, title_style=f"bold {ACCENT}", show_header=True, header_style=f"bold {ACCENT}", box=None)
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Title")
    table.add_column("Duration", justify="right", no_wrap=True)
    table.add_column("Published", justify="right", no_wrap=True)

    for r in result_set.results:
        t = r.title if len(r.title) <= title_max else r.title[:title_max-1] + "…"
        table.add_row(str(r.n), t, r.duration_str, r.published_at)

    console.print()
    console.print(table)

    hints = [
        "[cyan]fetch <1-5 | 1,3,7 | all>[/] to download",
        "[cyan]info <N>[/] for details"
    ]
    if state.next_page_token:
        hints.append("[cyan]next[/] for more")
    if state.prev_page_token or state.current_page > 1:
        hints.append("[cyan]prev[/] for previous")

    console.print(f"\n[{DIM}]Actions:[/] " + "  ·  ".join(hints))


def _run_search_or_browse(
    state: YouTubeSessionState, 
    query: str | None = None, 
    page_token: str | None = None,
    page_delta: int = 0
) -> None:
    """Execute search or channel browsing with active filters and pagination."""
    try:
        sort_order = state.sort
        if query is None and state.is_browse_mode and sort_order == "relevance":
            sort_order = "date"

        with console.status(f"[{DIM}]Contacting YouTube...[/]") as status:
            def progress(msg: str):
                status.update(f"[{DIM}]{msg}[/]")

            start_n = 1
            result_set = search_videos(
                query=query,
                channel_id=state.channel_id,
                max_results=state.limit,
                progress_callback=progress,
                sort=sort_order,
                after=state.after,
                before=state.before,
                page_token=page_token,
                start_n=start_n,
            )

        state.current_page = max(1, state.current_page + page_delta)
        _display_results_table(result_set, state)

    except YouTubeAPIError as e:
        console.print(error_panel("Search failed", str(e)))
    except Exception as e:
        console.print(error_panel("Error", str(e)))


def _handle_slash_command(user_input: str, state: YouTubeSessionState) -> bool:
    """Handle /slash commands. Returns True if REPL should exit."""
    parts = shlex.split(user_input)
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        return True

    if cmd == "/clear":
        state.channel_id = None
        state.channel_title = None
        state.sort = "relevance"
        state.limit = 15
        state.after = None
        state.before = None
        state.last_query = None
        state.is_browse_mode = False
        state.last_results = []
        state.next_page_token = None
        state.prev_page_token = None
        state.current_page = 1
        console.print(f"[{SUCCESS}]Filters cleared.[/]")
        return False

    if cmd == "/channel":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /channel <name or url>")
            return False
            
        channel_arg = parts[1]
        with get_session() as session:
            try:
                from audiobench.cli.commands.youtube_cmd import _resolve_fetch_target
                if "youtube.com" in channel_arg or channel_arg.startswith("UC"):
                    channel_id = _resolve_fetch_target(channel_arg)[0]
                    channel_title = channel_arg
                else:
                    channel_id, channel_title = resolve_channel(channel_arg, session)
                state.channel_id = channel_id
                state.channel_title = channel_title
                console.print(f"[{SUCCESS}]Channel filter set:[/] {channel_title} ({channel_id})")
            except Exception as e:
                console.print(error_panel("Channel resolution failed", str(e)))
        return False

    if cmd in ("/browse", "/videos"):
        if not state.channel_id:
            console.print(f"[{WARNING}]Set a channel first with:[/] /channel <name>")
            return False
        state.is_browse_mode = True
        state.last_query = None
        state.current_page = 1
        _run_search_or_browse(state, query=None)
        return False

    if cmd == "/limit":
        if len(parts) < 2 or not parts[1].isdigit():
            console.print(f"[{WARNING}]Usage:[/] /limit <1-50> (currently {state.limit})")
            return False
        val = int(parts[1])
        if not (1 <= val <= 50):
            console.print(f"[{WARNING}]Limit must be between 1 and 50.[/]")
            return False
        state.limit = val
        console.print(f"[{SUCCESS}]Results per page set to:[/] {state.limit}")
        return False

    if cmd == "/sort":
        if len(parts) < 2 or parts[1] not in ["relevance", "date", "viewCount", "rating", "title"]:
            console.print(f"[{WARNING}]Usage:[/] /sort [relevance|date|viewCount|rating|title]")
            return False
        state.sort = parts[1]
        console.print(f"[{SUCCESS}]Sort order set to:[/] {state.sort}")
        return False

    if cmd == "/after":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /after YYYY-MM-DD")
            return False
        state.after = parts[1]
        console.print(f"[{SUCCESS}]Date filter set:[/] published after {state.after}")
        return False

    if cmd == "/before":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /before YYYY-MM-DD")
            return False
        state.before = parts[1]
        console.print(f"[{SUCCESS}]Date filter set:[/] published before {state.before}")
        return False

    if cmd in ("/help", "/commands"):
        console.print(f"\n[{ACCENT}]Search & Browse:[/]")
        console.print(f"  [cyan]<query>[/]                 Search YouTube")
        console.print(f"  [cyan]/browse[/] or [cyan]browse[/]       Browse latest channel videos")
        console.print(f"  [cyan]next[/] or [cyan]n[/]              Load next page of results")
        console.print(f"  [cyan]prev[/] or [cyan]p[/]              Load previous page of results\n")
        console.print(f"[{ACCENT}]Batch & Selection Actions:[/]")
        console.print(f"  [cyan]fetch <1-5>[/]            Download range 1 through 5")
        console.print(f"  [cyan]fetch <1,3,7>[/]          Download subset items 1, 3, and 7")
        console.print(f"  [cyan]fetch all[//]              Download all videos on screen")
        console.print(f"  [cyan]info <1-3>[/]             Show metadata for item(s)\n")
        console.print(f"[{ACCENT}]Filters & Options:[/]")
        console.print(f"  [cyan]/channel <name>[/]         Restrict searches to channel")
        console.print(f"  [cyan]/limit <N>[/]              Set items per page (1-50, default {state.limit})")
        console.print(f"  [cyan]/sort <order>[/]           Set sort (relevance, date, viewCount, etc.)")
        console.print(f"  [cyan]/after <YYYY-MM-DD>[/]     Published after date")
        console.print(f"  [cyan]/before <YYYY-MM-DD>[/]    Published before date")
        console.print(f"  [cyan]/clear[/]                  Clear all active filters")
        console.print(f"  [cyan]/exit[/]                   Leave YouTube REPL\n")
        return False

    console.print(f"[{WARNING}]Unknown command:[/] {cmd}")
    return False


def _handle_action(parts: list[str], state: YouTubeSessionState) -> None:
    """Handle action verbs: fetch, download, dl, info, next, prev."""
    action = parts[0].lower()
    
    if action in ("next", "n"):
        if not state.next_page_token:
            console.print(f"[{DIM}]No further pages available.[/]")
            return
        _run_search_or_browse(
            state, 
            query=state.last_query if not state.is_browse_mode else None,
            page_token=state.next_page_token,
            page_delta=1
        )
        return

    if action in ("prev", "p"):
        if not state.prev_page_token:
            console.print(f"[{DIM}]No previous page available.[/]")
            return
        _run_search_or_browse(
            state, 
            query=state.last_query if not state.is_browse_mode else None,
            page_token=state.prev_page_token,
            page_delta=-1
        )
        return

    if action in ("browse", "videos"):
        if not state.channel_id:
            console.print(f"[{WARNING}]Set a channel first with:[/] /channel <name>")
            return
        state.is_browse_mode = True
        state.last_query = None
        state.current_page = 1
        _run_search_or_browse(state, query=None)
        return

    if len(parts) < 2:
        console.print(f"[{WARNING}]Usage:[/] {action} <1-5 | 1,3,7 | all>")
        return

    raw_selection = " ".join(parts[1:])
    try:
        selected_indices = parse_selection(raw_selection, max_count=len(state.last_results))
    except ValueError as e:
        console.print(f"[{WARNING}]Selection error:[/] {e}")
        return

    target_results = [r for r in state.last_results if r.n in selected_indices]
    if not target_results:
        console.print(f"[{WARNING}]No matching results found for selection.[/]")
        return

    if action in ("fetch", "download", "dl"):
        from audiobench.jobs.scheduler import enqueue, ensure_worker
        from audiobench.storage.models import AudioFileRecord
        
        queued_jobs = []
        already_in_lib = []

        with get_session() as session:
            for item in target_results:
                existing = session.query(AudioFileRecord).filter_by(youtube_video_id=item.video_id).first()
                if existing:
                    already_in_lib.append((item, existing))
                else:
                    job_id = enqueue(
                        job_type="youtube_fetch",
                        slot="network",
                        args=["youtube", "_fetch_internal", item.video_id],
                        file_label=item.title,
                        command_display=f"youtube download {item.video_id}",
                    )
                    queued_jobs.append((item, job_id))

        if queued_jobs:
            ensure_worker()

        console.print()
        if queued_jobs:
            if len(queued_jobs) == 1:
                item, job_id = queued_jobs[0]
                console.print(f"[{SUCCESS}]Download queued[/] · Job #{job_id} · {item.title}")
            else:
                console.print(f"[{SUCCESS}]Queued {len(queued_jobs)} download(s):[/]")
                for item, job_id in queued_jobs:
                    console.print(f"  • Job [bold]#{job_id}[/bold]: {item.title} [{DIM}]({item.duration_str})[/]")

        if already_in_lib:
            console.print(f"[{DIM}]Already in library ({len(already_in_lib)} skipped):[/]")
            for item, rec in already_in_lib:
                console.print(f"  • #{rec.id} — {item.title}")

        if queued_jobs:
            console.print(f"\n[{DIM}]Jobs queue automatically. Run[/] [bold]audiobench jobs list[/bold] [{DIM}]to check status.[/]")

    elif action == "info":
        for target_result in target_results:
            console.print(f"\n[{ACCENT}]#{target_result.n} · {target_result.title}[/]")
            console.print(f"[{DIM}]Video ID:[/] {target_result.video_id}  ·  [{DIM}]Duration:[/] {target_result.duration_str}  ·  [{DIM}]Published:[/] {target_result.published_at}")
            if target_result.description:
                console.print(f"\n[{DIM}]Description:[/]\n{target_result.description}\n")
            console.print("─" * 40)


def run_youtube_repl():
    """Main entrypoint for the YouTube interactive loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    
    state = YouTubeSessionState()
    
    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })
    session = PromptSession(style=style)
    
    console.print(f"[{ACCENT}]YouTube Interactive Mode[/]")
    console.print(f"[{DIM}]Type a search query, 'browse' for channel uploads, or '/commands' for help.[/]\n")
    
    while True:
        try:
            prefix = f"[{state.channel_title}] " if state.channel_title else ""
            prompt_text = [("class:prompt", f"{prefix}youtube> ")]
            user_input = session.prompt(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
            
        if not user_input:
            continue
            
        if user_input.startswith("/"):
            if _handle_slash_command(user_input, state):
                break
            continue

        parts = user_input.split()
        first_word = parts[0].lower()
        if first_word in ("fetch", "download", "dl", "info", "next", "n", "prev", "p", "browse", "videos"):
            _handle_action(parts, state)
            continue
            
        state.last_query = user_input
        state.is_browse_mode = False
        state.current_page = 1
        _run_search_or_browse(state, query=user_input)
