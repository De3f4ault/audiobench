"""Management CLI commands for the semantic memory daemon."""

from __future__ import annotations

import click
from rich.console import Console

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.client import DaemonClient

logger = get_logger("cli.daemon")
console = Console()


@click.group(name="daemon")
def daemon_group() -> None:
    """Manage the background semantic memory server."""
    pass


@daemon_group.command(name="start")
def start_daemon() -> None:
    """Start the daemon in the foreground (supervised)."""
    import audiobench.supervisor as supervisor

    console.print("[dim]Starting supervised audiobench memory daemon...[/dim]")
    try:
        from audiobench.events import get_bus
        from audiobench.observatory.db import init_journal_db
        from audiobench.observatory.subscriber import get_subscriber

        init_journal_db()
        get_bus().on("*", get_subscriber().record)

        supervisor.start("daemon")
        from audiobench.supervisor.commands import wait_all
        wait_all()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Shutting down...[/yellow]")
        from audiobench.supervisor.watcher import _SHUTDOWN_EVENT
        _SHUTDOWN_EVENT.set()
        supervisor.stop("daemon")
    except Exception as e:
        logger.exception("Daemon crashed")
        console.print(f"[bold red]Fatal error:[/bold red] {e}")


@daemon_group.command(name="stop")
def stop_daemon() -> None:
    """Stop the running daemon gracefully via supervisor."""
    import audiobench.supervisor as supervisor
    console.print("Sending SIGTERM to daemon via supervisor...")
    supervisor.stop("daemon")
    console.print("[bold green]Stop requested.[/bold green]")


@daemon_group.command(name="status")
def daemon_status() -> None:
    """Check if daemon is running and view its statistics."""
    settings = get_settings()
    client = DaemonClient()

    with console.status("Pinging daemon..."):
        if not client.ping():
            console.print("[yellow]Daemon is not running or socket is unreachable.[/yellow]")
            try:
                import psutil

                from audiobench.supervisor.registry import get_process
                p = get_process("daemon")
                if p:
                    state = p["state"].lower()
                    if state == "fatal":
                        console.print("[bold red]Supervisor reports daemon is in FATAL state (restarts exhausted).[/bold red]")
                    elif state == "backoff":
                        console.print("[bold yellow]Supervisor reports daemon is in BACKOFF (restarting shortly).[/bold yellow]")
                    elif state == "stopped":
                        console.print("[dim]Supervisor reports daemon is gracefully STOPPED.[/dim]")
                    elif state == "running":
                        if p["pid"] and not psutil.pid_exists(p["pid"]):
                            console.print(f"[bold red]Supervisor reports RUNNING but PID {p['pid']} is dead (STALE state).[/bold red]")
                        else:
                            console.print("[bold red]Supervisor reports RUNNING but socket ping failed.[/bold red]")
            except ImportError:
                pass
            return

    try:
        stats = client.status()
        console.print("\n[bold green]Daemon is ONLINE[/bold green]")
        console.print(f"[bold]Socket:[/bold] {settings.daemon_socket_path}")
        console.print(f"[bold]Uptime:[/bold] {stats.get('uptime_seconds', 0)}s")
        console.print(f"[bold]Model:[/bold] {stats.get('embedding_model_version', 'unknown')}")
        console.print(f"[bold]Nodes Stored:[/bold] {stats.get('total_nodes', 0)}")
        console.print(f"[bold]Queue Depth:[/bold] {stats.get('queue_depth', 0)}")
        pending_rev = stats.get('pending_privacy_reviews', 0)
        if pending_rev > 0:
            console.print(f"[bold]Pending Privacy Reviews:[/bold] [bold yellow]{pending_rev}[/bold yellow] (run 'audiobench review audiobooks')")
        else:
            console.print("[bold]Pending Privacy Reviews:[/bold] [dim]0[/dim]")

        pb = stats.get("playback", {})
        if pb and pb.get("playing"):
            import os
            st_icon = "[||] Paused" if pb.get("paused") else "[>] Playing"
            f_name = os.path.basename(pb.get("file", "Audio"))
            console.print(f"[bold]Playback:[/bold] {st_icon} · {f_name} ({pb.get('position', 0)}s / {pb.get('duration', 0)}s)")
        else:
            console.print("[bold]Playback:[/bold] [dim]Idle[/dim]")

        sweep_failures = stats.get("sweep_consecutive_failures", 0)
        if sweep_failures == 0:
            console.print("[bold]Sweep Health:[/bold] [dim]OK (0 consecutive failures)[/dim]")
        else:
            console.print(
                f"[bold]Sweep Health:[/bold] [bold red][!]  {sweep_failures} consecutive failure(s)[/bold red]"
                f" — RAG indexing loop is degraded. Check logs for details."
            )
    except Exception as e:
        console.print(f"[bold red]Failed to retrieve status:[/bold red] {e}")


@daemon_group.command(name="reindex")
def reindex_daemon() -> None:
    """Force the daemon to re-embed all nodes with the current model.

    Works with or without the daemon running — if the daemon is unavailable,
    models are loaded in-process for the duration of the reindex.
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    from audiobench.core.db_session import get_session
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.memory.enums import SourceType
    from audiobench.storage.models import ExpressionRecord

    client = get_daemon_client()

    with get_session() as session:
        expressions = session.query(ExpressionRecord).all()

    if not expressions:
        console.print("[yellow]No expressions found in the database to reindex.[/yellow]")
        return

    console.print(f"[bold]Reindexing {len(expressions)} expression(s) into LanceDB…[/bold]")
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(expressions))

        for expr in expressions:
            try:
                source_type = SourceType(expr.source_type)
                client.embed(
                    expression_id=expr.id,
                    content=expr.content,
                    source_type=source_type,
                    speaker=expr.speaker,
                )
            except Exception as e:
                errors.append(f"Expression {expr.id}: {e}")
                logger.warning("Failed to embed expression %d: %s", expr.id, e)
            finally:
                progress.advance(task)

    if errors:
        console.print(f"[yellow]Completed with {len(errors)} error(s):[/yellow]")
        for err in errors[:5]:
            console.print(f"  [dim red]{err}[/dim red]")
        if len(errors) > 5:
            console.print(f"  [dim]… and {len(errors) - 5} more[/dim]")
    else:
        console.print(
            f"[bold green]✓ Reindexed {len(expressions)} expression(s) successfully.[/bold green]"
        )


@daemon_group.command(name="optimize")
@click.option(
    "--vacuum",
    is_flag=True,
    help="Aggressively vacuum all historical versions down to the latest snapshot and delete orphaned fragments.",
)
@click.option(
    "--retention-hours",
    type=float,
    default=None,
    help="Historical versions older than this many hours will be purged (defaults to 2 hours, or 0 if --vacuum is set).",
)
def optimize_lancedb(vacuum: bool, retention_hours: float | None) -> None:
    """Optimize LanceDB tables: compact fragments and remove old versions.

    Sends the request to the running daemon if available (safe, coordinated).
    Falls back to in-process optimization if the daemon is not running.
    """
    client = DaemonClient()

    if vacuum:
        effective_retention = 0.0 if retention_hours is None else retention_hours
        delete_unverified = True
    else:
        effective_retention = retention_hours
        delete_unverified = False

    def _format_bytes(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def _print_result(result: dict) -> None:
        for table in result.get("tables_optimized", []):
            console.print(f"  [green]✓[/green] Optimized {table}")

        bytes_freed = result.get("bytes_freed", 0)
        freed_str = f" | Freed {_format_bytes(bytes_freed)}" if bytes_freed > 0 else ""

        cleared_writes = result.get("cleared_writes", 0)
        writes_str = f"Cleared {cleared_writes} fragmented writes" if cleared_writes > 0 else "No fragmented writes"

        console.print(f"[bold green]✓ Optimization complete in {result.get('duration_seconds', 0.0):.2f}s ({writes_str}{freed_str}).[/bold green]")

    if client.ping():
        mode_str = " (vacuum mode)" if vacuum else ""
        console.print(f"Daemon is ONLINE — sending optimize request{mode_str}...")
        try:
            result = client.optimize(
                retention_hours=effective_retention,
                delete_unverified=delete_unverified,
            )
            _print_result(result)
        except Exception as e:
            console.print(f"[bold red]Failed to optimize via daemon:[/bold red] {e}")
    else:
        mode_str = " (vacuum mode)" if vacuum else ""
        console.print(f"Daemon is OFFLINE — running in-process optimization{mode_str}...")
        try:
            from audiobench.daemon.lancedb_optimizer import _do_optimize_all_tables
            result = _do_optimize_all_tables(
                triggered_by="cli_fallback" if not vacuum else "cli_vacuum_fallback",
                retention_hours=effective_retention,
                delete_unverified=delete_unverified,
            )
            _print_result(result)
        except Exception as e:
            console.print(f"[bold red]Failed to optimize:[/bold red] {e}")
