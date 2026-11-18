"""CLI commands for Study Mode — project-based deep-dive sessions."""

from __future__ import annotations

import click

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, SUCCESS, console, error_panel


@click.group()
def study() -> None:
    """Manage study projects — focused multi-session deep dives on a file."""


# ── study new ────────────────────────────────────────────────────────────────


@study.command("new")
@click.argument("file_id", type=int)
@click.option("--name", default=None, help="Optional project name")
def study_new(file_id: int, name: str | None) -> None:
    """Start a new study project for an audio file.

    \b
    Examples:
      audiobench study new 83
      audiobench study new 83 --name "Goggins Deep Dive"
    """
    import json

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, StudyProject, StudySession

    init_db()

    with get_session() as session:
        audio_file = session.query(AudioFileRecord).filter_by(id=file_id).first()
        if audio_file is None:
            console.print(error_panel("Not found", f"Audio file #{file_id} not found"))
            return

        file_name = audio_file.file_name

    # Auto-trigger chapter picker if file is large
    picked_chapters: list[int] | None = None
    from audiobench.cli.commands.chat import _maybe_pick_chapters
    picked_chapters = _maybe_pick_chapters(file_id)

    # Create study project
    project_name = name or f"Study: {file_name[:40]}"
    with get_session() as session:
        project = StudyProject(
            audio_file_id=file_id,
            name=project_name,
        )
        session.add(project)
        session.flush()  # get project.id

        # Create the first session immediately
        first_session = StudySession(
            project_id=project.id,
            session_number=1,
            chapter_ids=json.dumps(picked_chapters or []),
        )
        session.add(first_session)
        session.commit()

        project_id = project.id
        session_id = first_session.id

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]Study Project Created[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Project:  #{project_id} {project_name}")
    console.print(f"    File:     #{file_id} {file_name}")
    if picked_chapters:
        console.print(f"    Chapters: {len(picked_chapters)} selected")
    else:
        console.print("    Chapters: All (no chapter filter)")
    console.print("    Session:  #1 (just created)")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()
    console.print(f"  [{DIM}]Start chatting: audiobench chat --project {project_id}[/]")
    console.print()


# ── study close ──────────────────────────────────────────────────────────────


@study.command("close")
@click.argument("project_id", type=int)
@click.option("--session", "session_id", type=int, default=None, help="Close a specific session (default: latest open)")
@click.option("--no-memoir", is_flag=True, default=False, help="Skip memoir generation")
def study_close(project_id: int, session_id: int | None, no_memoir: bool) -> None:
    """Close the current session in a study project and generate a memoir.

    \b
    Examples:
      audiobench study close 1
      audiobench study close 1 --no-memoir
    """
    import sys
    from datetime import UTC, datetime

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import StudyProject, StudySession

    init_db()

    with get_session() as db:
        project = db.query(StudyProject).filter_by(id=project_id).first()
        if project is None:
            console.print(error_panel("Not found", f"Study project #{project_id} not found"))
            sys.exit(1)

        if session_id is not None:
            sess = db.query(StudySession).filter_by(id=session_id, project_id=project_id).first()
        else:
            # Find the latest open session
            sess = (
                db.query(StudySession)
                .filter_by(project_id=project_id)
                .filter(StudySession.closed_at.is_(None))
                .order_by(StudySession.id.desc())
                .first()
            )

        if sess is None:
            console.print(error_panel("Not found", "No open session found for this project"))
            sys.exit(1)

        sess.closed_at = datetime.now(UTC)
        db.commit()

        sess_id = sess.id
        conv_id = sess.conversation_id

    console.print(f"  [{SUCCESS}]✓ Session #{sess_id} closed[/]")

    # Generate memoir if conversation exists and not suppressed
    if not no_memoir and conv_id is not None:
        console.print(f"  [{DIM}]Generating memoir for session #{sess_id}...[/]")
        try:
            from audiobench.core.db_session import get_session
            from audiobench.memory.memoir_writer import MemoirWriter
            from audiobench.storage.models import ChatConversation

            with get_session() as db:
                conv = db.query(ChatConversation).filter_by(id=conv_id).first()
                if conv:
                    db.expunge(conv)

            if conv:
                # Re-fetch session as detached object
                with get_session() as db:
                    sess_obj = db.query(StudySession).filter_by(id=sess_id).first()
                    if sess_obj:
                        db.expunge(sess_obj)

                memoir = MemoirWriter().generate(conv, sess_obj)
                console.print(f"  [{SUCCESS}]✓ Memoir generated[/]")
                console.print(f"    [{DIM}]{memoir.narrative[:120]}...[/]")
            else:
                console.print(f"  [{DIM}]No conversation found for memoir generation[/]")
        except Exception as e:
            console.print(f"  [{DIM}]Memoir generation failed: {e}[/]")
    elif no_memoir:
        console.print(f"  [{DIM}]Memoir skipped (--no-memoir)[/]")

    console.print()

    # Show next session prompt
    with get_session() as db:
        session_count = db.query(StudySession).filter_by(project_id=project_id).count()

    console.print(f"  [{DIM}]Start next session: audiobench study resume {project_id}[/]")
    console.print(f"  [{DIM}]Total sessions completed: {session_count}[/]")
    console.print()


# ── study resume ─────────────────────────────────────────────────────────────


@study.command("resume")
@click.argument("project_id", type=int)
def study_resume(project_id: int) -> None:
    """Resume a study project by creating a new session.

    \b
    Examples:
      audiobench study resume 1
    """
    import sys

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import StudyProject, StudySession

    init_db()

    with get_session() as db:
        project = db.query(StudyProject).filter_by(id=project_id).first()
        if project is None:
            console.print(error_panel("Not found", f"Study project #{project_id} not found"))
            sys.exit(1)

        # Check no open sessions
        open_sess = (
            db.query(StudySession)
            .filter_by(project_id=project_id)
            .filter(StudySession.closed_at.is_(None))
            .first()
        )
        if open_sess is not None:
            console.print(f"  [{DIM}]Session #{open_sess.id} is already open — close it first:[/]")
            console.print(f"  [{DIM}]  audiobench study close {project_id}[/]")
            return

        session_count = db.query(StudySession).filter_by(project_id=project_id).count()
        new_session_number = session_count + 1

        new_session = StudySession(
            project_id=project_id,
            session_number=new_session_number,
            chapter_ids="[]",
        )
        db.add(new_session)
        db.commit()
        new_id = new_session.id

    console.print(f"  [{SUCCESS}]✓ Session #{new_id} (session {new_session_number}) created[/]")
    console.print(f"  [{DIM}]Chat: audiobench chat --project {project_id}[/]")
    console.print()


# ── study list ───────────────────────────────────────────────────────────────


@study.command("list")
def study_list() -> None:
    """List all study projects."""
    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import StudyProject, StudySession

    init_db()

    with get_session() as db:
        projects = db.query(StudyProject).order_by(StudyProject.id).all()
        if not projects:
            console.print(f"  [{DIM}]No study projects yet. Create one: audiobench study new <file_id>[/]")
            return

        console.print()
        console.print(f"  [{BOLD} {ACCENT}]Study Projects[/]")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        for p in projects:
            sess_count = db.query(StudySession).filter_by(project_id=p.id).count()
            open_count = (
                db.query(StudySession)
                .filter_by(project_id=p.id)
                .filter(StudySession.closed_at.is_(None))
                .count()
            )
            status = f"[{ACCENT}]{open_count} open[/]" if open_count > 0 else f"[{DIM}]all closed[/]"
            console.print(
                f"    [{ACCENT}]#{p.id}[/] {(p.name or 'Unnamed Project')[:40]} "
                f"[{DIM}]({sess_count} sessions, {status})[/]"
            )
        console.print()
