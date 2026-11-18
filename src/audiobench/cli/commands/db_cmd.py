"""CLI commands for database management and manual ingest triggers."""

from __future__ import annotations

import click

from audiobench.cli.display.theme import DIM, SUCCESS, console, error_panel


@click.group()
def db() -> None:
    """Manage the local SQLite database and search indices."""


@db.command("ingest")
@click.argument("audio_file_id", type=int)
@click.option("--force", is_flag=True, help="Force re-ingest even if already ingested")
def db_ingest(audio_file_id: int, force: bool) -> None:
    """Manually trigger knowledge graph ingestion for an audio file.
    
    This breaks the transcript into semantic segments and inserts them into the
    local knowledge graph (nodes and edges) for research querying.
    """
    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.memory.graph_builder import GraphBuilder
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    init_db()

    with get_session() as session:
        audio = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
        if not audio:
            console.print(error_panel("Not found", f"Audio file #{audio_file_id} not found"))
            return

        tx = session.query(TranscriptionRecord).filter_by(audio_file_id=audio_file_id).first()
        if not tx:
            console.print(error_panel("No transcript", f"Audio file #{audio_file_id} has no transcript to ingest"))
            return

    console.print(f"  [{DIM}]Starting graph ingest for file #{audio_file_id}...[/]")
    builder = GraphBuilder()

    with console.status("Extracting semantic concepts and relationships..."):
        try:
            builder.ingest_transcript(tx.id)
            console.print(f"  [{SUCCESS}]✓ Ingest complete[/]")
        except Exception as e:
            console.print(error_panel("Ingest failed", str(e)))


@db.command("reindex")
def db_reindex() -> None:
    """Rebuild the Full-Text Search (FTS5) indices."""
    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session

    init_db()

    console.print(f"  [{DIM}]Rebuilding FTS5 indices...[/]")

    with console.status("Executing FTS rebuild..."):
        try:
            with get_session() as db:
                db.execute("INSERT INTO expressions_fts(expressions_fts) VALUES('rebuild')")
                db.execute("INSERT INTO fts_segments(fts_segments) VALUES('rebuild')")
                db.commit()
            console.print(f"  [{SUCCESS}]✓ FTS indices rebuilt successfully[/]")
        except Exception as e:
            console.print(error_panel("Reindex failed", str(e)))


@db.command("embed-segments")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-embed all segments, even those already vectorized.",
)
def db_embed_segments(force: bool) -> None:
    """Embed audio segments into the LanceDB vector index.

    Reads segments from SQLite, embeds each one through the daemon's warm
    Nomic model, and writes the result to the segment_vectors LanceDB table.

    This is idempotent — already-embedded segments are skipped unless --force
    is given. Run this once after updating to enable Dense and ColBERT search.
    """
    from sqlalchemy import text as sql_text

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.daemon.factory import get_daemon_client

    init_db()

    # Fetch segments to embed
    where_clause = "" if force else "WHERE s.vector_indexed = 0"
    query_sql = (
        "SELECT s.id, s.text, s.start_time, s.end_time, af.file_path "
        "FROM segments s "
        "JOIN transcriptions t ON s.transcription_id = t.id "
        "LEFT JOIN audio_files af ON t.audio_file_id = af.id "
        f"{where_clause} "
        "ORDER BY s.id"
    )

    with get_session() as session:
        rows = session.execute(sql_text(query_sql)).mappings().all()
        rows = list(rows)  # materialise before session closes

    total = len(rows)
    if total == 0:
        console.print(
            f"  [{SUCCESS}]✓ All segments already embedded.[/] "
            f"[{DIM}]Use --force to re-embed.[/]"
        )
        return

    action = "Re-embedding" if force else "Embedding"
    console.print(
        f"  [{DIM}]{action} [bold]{total}[/bold] segments via daemon...[/]"
    )

    daemon = get_daemon_client()
    success = 0
    failed = 0

    for row in rows:
        try:
            daemon.embed_segment(
                segment_id=int(row["id"]),
                text=row["text"],
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                source_file=row["file_path"] or "",
            )
            # Mark vectorized in SQLite
            with get_session() as upd:
                upd.execute(
                    sql_text("UPDATE segments SET vector_indexed=1 WHERE id=:id"),
                    {"id": int(row["id"])},
                )
                upd.commit()
            success += 1

            # Live progress every 10 segments
            if success % 10 == 0 or success == total:
                console.print(
                    f"  [{DIM}]{success}/{total} embedded...[/]",
                    end="\r",
                )

        except Exception as e:
            failed += 1
            console.print(
                f"\n  [red]✗ Segment #{row['id']} failed: {e}[/red]"
            )

    console.print()  # clear the \r line
    if failed == 0:
        console.print(
            f"  [{SUCCESS}]✓ All {success} segments embedded successfully.[/]\n"
            f"  [{DIM}]Dense and ColBERT search are now active.[/]"
        )
    else:
        console.print(
            f"  [yellow][!] {success} succeeded, {failed} failed.[/yellow]\n"
            f"  [{DIM}]Re-run to retry failed segments.[/]"
        )


@db.command("prune-orphans")
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Permanently delete orphaned vector rows from LanceDB. Without this flag, runs in dry-run mode.",
)
def db_prune_orphans(apply: bool) -> None:
    """Find and purge orphaned vector records from LanceDB.

    Scans both 'segment_vectors' and 'expressions' in LanceDB and identifies
    records that point to segment or expression IDs that no longer exist in SQLite.
    """
    from rich.table import Table
    from sqlalchemy import text as sql_text

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.memory.singletons import get_memory_store, get_segment_store

    init_db()

    with console.status("Scanning SQLite and LanceDB for orphaned vectors..."):
        with get_session() as session:
            live_seg_ids = set(
                r[0] for r in session.execute(sql_text("SELECT id FROM segments")).fetchall()
            )
            live_expr_ids = set(
                r[0] for r in session.execute(sql_text("SELECT id FROM expressions")).fetchall()
            )

        seg_store = get_segment_store()
        df_seg = seg_store.table.search().select(["segment_id"]).limit(1_000_000).to_pandas()
        lancedb_seg_ids = set(df_seg["segment_id"].tolist())
        orphaned_seg_ids = list(lancedb_seg_ids - live_seg_ids)

        mem_store = get_memory_store()
        df_expr = mem_store.table.search().select(["expression_id"]).limit(1_000_000).to_pandas()
        lancedb_expr_ids = set(df_expr["expression_id"].tolist())
        orphaned_expr_ids = list(lancedb_expr_ids - live_expr_ids)

    table = Table(title="LanceDB Vector Orphan Reconciliation", show_header=True)
    table.add_column("Vector Table", style="cyan")
    table.add_column("LanceDB Rows", justify="right")
    table.add_column("SQLite Parent Rows", justify="right")
    table.add_column("Orphaned Vectors", justify="right", style="bold red" if (orphaned_seg_ids or orphaned_expr_ids) else "green")
    table.add_column("Status", style="yellow")

    table.add_row(
        "segment_vectors",
        str(len(lancedb_seg_ids)),
        str(len(live_seg_ids)),
        str(len(orphaned_seg_ids)),
        "Needs Pruning" if orphaned_seg_ids else "Clean",
    )
    table.add_row(
        "expressions",
        str(len(lancedb_expr_ids)),
        str(len(live_expr_ids)),
        str(len(orphaned_expr_ids)),
        "Needs Pruning" if orphaned_expr_ids else "Clean",
    )

    console.print()
    console.print(table)
    console.print()

    if not orphaned_seg_ids and not orphaned_expr_ids:
        console.print(f"  [{SUCCESS}]✓ All LanceDB vector tables are 100% consistent with SQLite.[/]\n")
        return

    if not apply:
        console.print(
            f"  [{DIM}]Dry run complete. Found [bold yellow]{len(orphaned_seg_ids)}[/bold yellow] orphaned segment vectors "
            f"and [bold yellow]{len(orphaned_expr_ids)}[/bold yellow] orphaned expression vectors.[/]\n"
            f"  [{DIM}]Run with [bold cyan]audiobench db prune-orphans --apply[/bold cyan] to permanently purge these rows from LanceDB.[/]\n"
        )
        return

    # Apply pruning
    if orphaned_seg_ids:
        with console.status(f"Purging {len(orphaned_seg_ids)} orphaned segment vectors..."):
            seg_store.delete_segments(orphaned_seg_ids)
        console.print(f"  [{SUCCESS}]✓ Purged {len(orphaned_seg_ids)} orphaned segment vectors from LanceDB.[/]")

    if orphaned_expr_ids:
        with console.status(f"Purging {len(orphaned_expr_ids)} orphaned expression vectors..."):
            mem_store.delete_node_batch(orphaned_expr_ids)
        console.print(f"  [{SUCCESS}]✓ Purged {len(orphaned_expr_ids)} orphaned expression vectors from LanceDB.[/]")
    console.print()

