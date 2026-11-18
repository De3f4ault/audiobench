from pathlib import Path

import click

from audiobench.cli.display.theme import ACCENT, DIM, SUCCESS, console, error_panel, make_table
from audiobench.core.db_session import get_session
from audiobench.events import get_bus
from audiobench.memory.enums import SourceType
from audiobench.storage.models import (
    AudioFileRecord,
    ExpressionRecord,
    TranscriptionRecord,
    WorkRecord,
)


@click.group(name="work", invoke_without_command=True)
@click.pass_context
def work_group(ctx: click.Context):
    """Manage semantic works and assignments."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@work_group.command(name="create")
@click.option("--title", required=True, help="Title of the work")
@click.option("--author", help="Author of the work")
def work_create(title: str, author: str | None):
    """Create a new work record."""
    from audiobench.core.db_engine import init_db
    init_db()

    with get_session() as session:
        work = WorkRecord(title=title, author=author)
        session.add(work)
        session.commit()
        author_str = f" by {author}" if author else ""
        console.print(f"[{SUCCESS}]✓ Created work #{work.id}[/]: '{title}'{author_str}")


@work_group.command(name="list")
def work_list():
    """List all semantic works."""
    from audiobench.core.db_engine import init_db
    init_db()

    with get_session() as session:
        works = session.query(WorkRecord).order_by(WorkRecord.id).all()

    if not works:
        console.print(f"[{DIM}]No works found.[/]")
        return

    table = make_table("Semantic Works", [
        ("ID", {"justify": "right", "style": DIM, "width": 4}),
        ("Title", {"style": ACCENT, "max_width": 50}),
        ("Author", {"style": "green", "max_width": 30}),
    ])

    for work in works:
        table.add_row(str(work.id), work.title, work.author or "")

    console.print(table)


@work_group.command(name="assign")
@click.argument("args", nargs=-1, required=True)
def work_assign(args):
    """Assign audio files to a work.
    
    Usage: audiobench work assign <file_id_or_path> [file_id_or_path...] <work_id>
    """
    from audiobench.core.db_engine import init_db
    init_db()

    if len(args) < 2:
        console.print(error_panel("Invalid usage", "Expected: audiobench work assign <targets...> <work_id>"))
        return

    work_id_str = args[-1]
    if not work_id_str.isdigit():
        console.print(error_panel("Invalid work_id", f"'{work_id_str}' is not a valid integer Work ID."))
        return

    work_id = int(work_id_str)
    targets = args[:-1]

    with get_session() as session:
        # Verify work exists
        work = session.query(WorkRecord).filter_by(id=work_id).first()
        if not work:
            console.print(error_panel("Not found", f"Work #{work_id} does not exist."))
            return

        audio_ids = set()

        for target in targets:
            if target.isdigit():
                # Direct ID
                aid = int(target)
                af = session.query(AudioFileRecord).filter_by(id=aid).first()
                if af:
                    audio_ids.add(af.id)
            else:
                # Path or glob - shell expands globs, so this is likely a path
                resolved = str(Path(target).expanduser().resolve())
                af = session.query(AudioFileRecord).filter(AudioFileRecord.file_path == resolved).first()
                if af:
                    audio_ids.add(af.id)
                else:
                    # Try LIKE match for simple globs if not expanded by shell (e.g., quoted)
                    if "*" in target or "?" in target:
                        import glob
                        for p in glob.glob(target):
                            resolved_p = str(Path(p).expanduser().resolve())
                            af_match = session.query(AudioFileRecord).filter(AudioFileRecord.file_path == resolved_p).first()
                            if af_match:
                                audio_ids.add(af_match.id)

        if not audio_ids:
            console.print(error_panel("Not found", "No matching audio files found."))
            return

        # Update AudioFiles
        updated_count = 0
        for aid in audio_ids:
            af = session.query(AudioFileRecord).filter_by(id=aid).first()
            if af and af.work_id != work_id:
                af.work_id = work_id

                # Also cascade directly to related expressions in SQLite
                # (LanceDB will be reconciled later via events/queue)
                session.query(ExpressionRecord).filter(
                    ExpressionRecord.source_id.in_(
                        session.query(TranscriptionRecord.id)
                        .filter_by(audio_file_id=aid)
                    ),
                    ExpressionRecord.source_type == SourceType.AUDIO_TRANSCRIPT.value
                ).update({"work_id": work_id}, synchronize_session=False)

                updated_count += 1

                # Emit event for the reconciliation queue
                get_bus().emit(
                    "work_assigned",
                    audio_file_id=aid,
                    work_id=work_id
                )

        session.commit()

        console.print(f"[{SUCCESS}]✓ Assigned {updated_count} audio file(s) to Work #{work_id}[/]")


@work_group.command(name="unassigned")
def work_unassigned():
    """List audio files that have no work assigned."""
    from audiobench.core.db_engine import init_db
    init_db()

    with get_session() as session:
        files = session.query(AudioFileRecord).filter(AudioFileRecord.work_id == None).all()

    if not files:
        console.print(f"[{SUCCESS}]All audio files are assigned to a work![/]")
        return

    table = make_table("Unassigned Audio Files", [
        ("ID", {"justify": "right", "style": DIM, "width": 4}),
        ("File Name", {"style": ACCENT, "max_width": 60}),
    ])

    for f in files:
        table.add_row(str(f.id), f.file_name)

    console.print(table)
