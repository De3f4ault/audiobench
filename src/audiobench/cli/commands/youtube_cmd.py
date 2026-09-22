"""YouTube CLI commands."""

import click
from rich.console import Console

from audiobench.core.db_session import get_session
from audiobench.cli.display.theme import console, SUCCESS, WARNING, ERROR, ACCENT, DIM, error_panel
from audiobench.youtube.fetcher import extract_video_id, fetch_and_register
from audiobench.youtube.search import (
    resolve_channel,
    search_videos,
    write_last_search,
    load_search_result,
    SearchStateExpiredError,
    YouTubeAPIError
)
from audiobench.storage.models import YouTubeChannel

@click.group(name="youtube", invoke_without_command=True)
@click.option("--job-id", type=int, hidden=True)
@click.pass_context
def youtube_group(ctx, job_id):
    """YouTube integration: search and fetch videos."""
    ctx.ensure_object(dict)
    ctx.obj["job_id"] = job_id
    
    if ctx.invoked_subcommand is None:
        from audiobench.youtube.workspace import run_channel_workspace
        run_channel_workspace()

def _resolve_fetch_target(arg: str) -> list[str]:
    """Return a list of canonical YouTube URLs or IDs from numbers (search results), ranges, or URLs."""
    from audiobench.youtube.search import parse_selection
    
    # Check if it's a selection expression like "1-5", "1,3,7", "all"
    if any(c in arg for c in (",", "-")) and not ("youtube.com" in arg or "youtu.be" in arg or "http" in arg):
        try:
            indices = parse_selection(arg)
            return [load_search_result(idx) for idx in indices]
        except Exception as e:
            raise click.BadParameter(f"Failed to resolve range '{arg}': {e}")

    if arg.isdigit():
        try:
            return [load_search_result(int(arg))]
        except SearchStateExpiredError as e:
            raise click.BadParameter(str(e))
    
    if "youtube.com" in arg or "youtu.be" in arg or len(arg) == 11:
        return [arg]
        
    raise click.BadParameter(f"'{arg}' is not a valid YouTube URL, video ID, or result index.")


@youtube_group.command("_fetch_internal", hidden=True)
@click.argument("video_id", type=str)
@click.pass_context
def fetch_internal_cmd(ctx, video_id: str):
    job_id = ctx.obj.get("job_id")
    with get_session() as session:
        try:
            audio_record, queue_record = fetch_and_register(video_id, session)
            if audio_record:
                console.print(f"[{SUCCESS}]Saved[/]  {audio_record.file_path}")
                console.print(f"[{DIM}]Audio file #{audio_record.id} · Queue Job #{queue_record.id} queued for transcription[/]")
            else:
                console.print(f"[{DIM}]Already in library, skipping.[/]")
        except Exception as e:
            console.print(error_panel("Fetch failed", str(e)))
            import sys
            sys.exit(1)


@youtube_group.command("fetch")
@click.argument("targets", nargs=-1, required=True)
def fetch_cmd(targets: tuple[str, ...]):
    """Fetch one or more YouTube videos and queue them for transcription."""
    from audiobench.jobs.scheduler import enqueue, ensure_worker
    from audiobench.storage.models import AudioFileRecord
    
    all_video_ids: list[str] = []
    for target in targets:
        try:
            resolved_urls = _resolve_fetch_target(target)
            for url in resolved_urls:
                vid = extract_video_id(url)
                if vid not in all_video_ids:
                    all_video_ids.append(vid)
        except Exception as e:
            console.print(error_panel("Error", str(e)))
            return

    if not all_video_ids:
        console.print(f"[{WARNING}]No valid video targets specified.[/]")
        return

    queued_jobs = []
    already_in_lib = []

    with get_session() as session:
        for vid in all_video_ids:
            existing = session.query(AudioFileRecord).filter_by(youtube_video_id=vid).first()
            if existing:
                already_in_lib.append((vid, existing))
            else:
                job_id = enqueue(
                    job_type="youtube_fetch",
                    slot="network",
                    args=["youtube", "_fetch_internal", vid],
                    file_label=vid,
                    command_display=f"youtube download {vid}",
                )
                queued_jobs.append((vid, job_id))

    if queued_jobs:
        ensure_worker()

    if queued_jobs:
        if len(queued_jobs) == 1:
            vid, job_id = queued_jobs[0]
            console.print(f"[{SUCCESS}]Download queued[/] · Job #{job_id} ({vid})")
        else:
            console.print(f"[{SUCCESS}]Queued {len(queued_jobs)} download(s):[/]")
            for vid, job_id in queued_jobs:
                console.print(f"  • Job [bold]#{job_id}[/bold]: {vid}")

    if already_in_lib:
        console.print(f"[{DIM}]Already in library ({len(already_in_lib)} skipped):[/]")
        for vid, rec in already_in_lib:
            console.print(f"  • #{rec.id} — {rec.file_name}")

    if queued_jobs:
        console.print(f"\n[{DIM}]Run[/] [bold]audiobench jobs list[/bold] [{DIM}]or[/] [bold]audiobench jobs fg <id>[/bold] [{DIM}]to follow progress.[/]")


@youtube_group.command("search")
@click.argument("query", type=str)
@click.option("--channel", type=str, help="Restrict to channel name or URL")
@click.option("--limit", type=int, default=15, help="Max results (default 15)")
@click.option("--sort", type=click.Choice(["relevance", "date", "viewCount", "rating", "title"]), default="relevance", help="Sort order")
@click.option("--after", type=str, help="Published after date (YYYY-MM-DD)")
@click.option("--before", type=str, help="Published before date (YYYY-MM-DD)")
@click.option("--no-cache", is_flag=True, help="Bypass channel resolution cache")
def search_cmd(
    query: str, 
    channel: str | None, 
    limit: int,
    sort: str,
    after: str | None,
    before: str | None,
    no_cache: bool
):
    """Search YouTube for videos."""
    from rich.table import Table
    
    channel_id = None
    with get_session() as session:
        if channel:
            try:
                if "youtube.com" in channel or channel.startswith("UC"):
                    channel_id = _resolve_fetch_target(channel)[0]
                    channel_title = channel
                else:
                    channel_id, channel_title = resolve_channel(channel, session, no_cache=no_cache)
                console.print(f"[{DIM}]Resolving channel...[/]  {channel_title}  ({channel_id})")
            except Exception as e:
                console.print(error_panel("Error resolving channel", str(e)))
                return
                
        try:
            with console.status(f"[{DIM}]Searching YouTube...[/]") as status:
                def progress(msg):
                    status.update(f"[{DIM}]{msg}[/]")
                results = search_videos(
                    query, 
                    channel_id, 
                    limit, 
                    progress_callback=progress,
                    sort=sort,
                    after=after,
                    before=before
                )
                
            if not results:
                console.print(f"[{DIM}]No results found for '{query}'[/]")
                return
                
            write_last_search(results)
            
            import shutil
            term_width = shutil.get_terminal_size().columns
            title_max = max(20, term_width - 40)
            
            table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
            table.add_column("#", style="dim", width=4, no_wrap=True)
            table.add_column("Title")
            table.add_column("Duration", justify="right", no_wrap=True)
            table.add_column("Published", justify="right", no_wrap=True)
            
            for r in results:
                title = r.title if len(r.title) <= title_max else r.title[:title_max-1] + "…"
                table.add_row(str(r.n), title, r.duration_str, r.published_at)
                
            console.print()
            console.print(table)
            console.print(f"\n[{DIM}]fetch a result:[/]  audiobench youtube fetch <number>")
            console.print(f"[{DIM}]results expire in 1 hour[/]")
            
        except YouTubeAPIError as e:
            console.print(error_panel("Search failed", str(e)))
        except ValueError as e:
            console.print(error_panel("Error", str(e)))


@youtube_group.command("channels")
@click.option("--refresh", type=str, help="Force re-resolve a specific channel query.")
def channels_cmd(refresh: str | None):
    """List cached channel resolutions."""
    with get_session() as session:
        if refresh:
            normalized = refresh.strip().lower()
            deleted = session.query(YouTubeChannel).filter_by(query=normalized).delete()
            session.commit()
            if deleted:
                console.print(f"[{SUCCESS}]Cleared cache[/] for channel query '{refresh}'")
            else:
                console.print(f"[{DIM}]No cached entry found[/] for '{refresh}'")
            return
            
        channels = session.query(YouTubeChannel).order_by(YouTubeChannel.query).all()
        if not channels:
            console.print(f"[{DIM}]No cached channel resolutions yet.[/]")
            return
            
        from rich.table import Table
        table = Table(title="Cached channel resolutions", title_style=f"bold {ACCENT}", show_header=True, header_style=f"bold {ACCENT}", box=None)
        table.add_column("Query", style="cyan")
        table.add_column("Title")
        table.add_column("Channel ID", style="dim")
        
        for c in channels:
            table.add_row(c.query, c.title, c.channel_id)
            
        console.print()
        console.print(table)
        console.print(f"\n[{DIM}]--refresh <name>  to force re-resolve[/]")

