"""Backslash meta-commands for the AudioBench REPL.

These are local, instant commands that run against the database without
requiring any AI or external service.

Registration is side-effect-based: importing this module populates the
_BACKSLASH_HANDLERS registry in dispatch.py via @register_backslash.

To add a new \\command, decorate a function here:

    @register_backslash("mycommand")
    def cmd_mycommand(session: ReplSession, args: str) -> None:
        ...
"""

from __future__ import annotations

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    format_duration,
    make_table,
)
from audiobench.cli.repl.dispatch import register_backslash, register_resume
from audiobench.cli.repl.session import ReplSession

# ── Security helpers ──────────────────────────────────────────────────────


def require_tier(minimum_tier: int):
    """Decorator factory for sudo-style command authorization.

    Wraps a backslash command function to require a minimum session tier
    before executing. If the session tier is insufficient, the command is
    blocked and the user is prompted to run \\unlock.

    Usage::

        @require_tier(2)
        @register_backslash("delete")
        def cmd_delete(session: ReplSession, args: str) -> None:
            ...

    Design: The decorator checks ``session.effective_tier()`` which respects
    the 10-minute TTL for Tier 1 and never caches Tier 2+. This means that
    if someone walks away and the session auto-locks, destructive commands
    become inaccessible without re-authentication.
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(session: ReplSession, args: str) -> None:
            if session.effective_tier() < minimum_tier:
                tier_labels = {1: "Relational", 2: "Intimate", 3: "Override"}
                label = tier_labels.get(minimum_tier, f"Tier {minimum_tier}")
                console.print(
                    f"  [{WARNING}]Access denied.[/] "
                    f"[{DIM}]{label} ({minimum_tier}) required.[/]  "
                    f"Run [{ACCENT}]\\unlock {minimum_tier if minimum_tier > 2 else ''}[/]"
                    .rstrip()
                )
                return
            fn(session, args)
        return wrapper
    return decorator


# ── Navigation ───────────────────────────────────────────────


@register_backslash("back")
def cmd_back(session: ReplSession, args: str) -> None:
    """Return to the previous navigation context."""
    frame = session.pop_frame()
    if frame is None:
        console.print(f"  [{DIM}]Already at top level.[/]")
        return

    console.print(f"  [{SUCCESS}]← Back to {frame.context}[/]")

    # Try to resume via registry if a handler exists
    from audiobench.cli.repl.dispatch import _RESUME_HANDLERS
    if frame.context in _RESUME_HANDLERS:
        handler = _RESUME_HANDLERS[frame.context]
        handler(session, frame)


@register_backslash("home")
def cmd_home(session: ReplSession, args: str) -> None:
    """Clear the entire navigation stack and return to the top level."""
    count = len(session.navigation_stack)
    if count == 0:
        console.print(f"  [{DIM}]Already at top level.[/]")
        return
    session.navigation_stack.clear()
    session._persist_stack()
    console.print(f"  [{SUCCESS}]↑ Cleared {count} level(s). Back at top.[/]")


@register_backslash("stack")
def cmd_stack(session: ReplSession, args: str) -> None:
    """Show the current navigation stack depth and contents."""
    if not session.navigation_stack:
        console.print(f"  [{DIM}]Stack is empty — you are at the top level.[/]")
        return

    depth = len(session.navigation_stack)
    console.print(f"\n  [{BOLD}]Navigation stack[/] [{DIM}]({depth} level(s))[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    for i, frame in enumerate(session.navigation_stack, 1):
        resumed_tag = f" [{DIM}][resumed][/]" if frame.resumed else ""
        state_hint = ""
        if frame.state:
            keys = list(frame.state.keys())[:3]
            state_hint = f"  [{DIM}]{{{', '.join(keys)}}}[/]"
        intent_str = f" — {frame.intent}" if frame.intent else ""
        console.print(f"    {i}. [{ACCENT}]{frame.context}[/]{intent_str}{state_hint}{resumed_tag}")
    console.print()


# ── Focus ────────────────────────────────────────────────────


@register_backslash("focus")
def cmd_focus(session: ReplSession, args: str) -> None:
    """Focus on an audio file by ID, or show the current focus."""
    from audiobench.cli.repl.dispatch import print_context_summary

    if not args.strip():
        if session.focus:
            print_context_summary(session)
        else:
            console.print(
                f"  [{DIM}]No focus set.[/]  "
                f"Use [{ACCENT}]\\focus <id>[/] or [{ACCENT}]\\ls[/] to pick a file."
            )
        return

    arg = args.strip()

    from audiobench.core.db_engine import init_db
    from audiobench.core.db_session import get_session
    from audiobench.core.focused_entity import FocusedEntity
    from audiobench.storage.models import WorkRecord
    from audiobench.storage.repository import TranscriptionRepository

    init_db()

    # Support \focus w12
    if arg.lower().startswith("w") and arg[1:].isdigit():
        work_id = int(arg[1:])
        with get_session() as session_db:
            work = session_db.query(WorkRecord).filter_by(id=work_id).first()
            if not work:
                console.print(f"  [{WARNING}]Work #{work_id} not found.[/]")
                return
            session.focus = FocusedEntity(
                type="work",
                id=work.id,
                label=f"Work: {work.title}"
            )
            print_context_summary(session)
        return

    if not arg.isdigit():
        # Text string: focus by author or title
        search_str = arg.strip("\"'")
        with get_session() as session_db:
            works = session_db.query(WorkRecord).filter(
                (WorkRecord.title.ilike(f"%{search_str}%")) |
                (WorkRecord.author.ilike(f"%{search_str}%"))
            ).all()
            if not works:
                console.print(f"  [{WARNING}]Usage: \\focus <audio_file_id|w<work_id>|\"author/title\">[/]")
                return
            if len(works) > 1:
                console.print(f"  [{WARNING}]Multiple works match '{search_str}'. Please be more specific.[/]")
                return
            work = works[0]
            session.focus = FocusedEntity(
                type="work",
                id=work.id,
                label=f"Work: {work.title}"
            )
            print_context_summary(session)
        return

    file_id = int(arg)
    try:
        repo = TranscriptionRepository()
        audio_file = repo.get_audio_file(file_id)
        if not audio_file:
            console.print(f"  [{WARNING}]Audio file #{file_id} not found.[/]")
            return

        session.focus = FocusedEntity(
            type="file",
            id=file_id,
            label=audio_file["file_name"],
        )
        print_context_summary(session)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not focus on #{file_id}: {e}[/]")


@register_backslash("unfocus")
def cmd_unfocus(session: ReplSession, args: str) -> None:
    """Clear the current focus and return to the bare prompt."""
    if not session.focus:
        console.print(f"  [{DIM}]No focus to clear.[/]")
        return
    label = session.focus.display_label
    session.focus = None
    console.print(f"  [{DIM}]Focus cleared ({label}).[/]")


# ── View ─────────────────────────────────────────────────────


@register_backslash("ls")
def cmd_ls(session: ReplSession, args: str) -> None:
    """List recent audio files. Optionally pass a count: \\ls 20"""
    from audiobench.core.db_engine import init_db

    try:
        init_db()
        repo = session._get_repo()
        limit = int(args.strip()) if args.strip().isdigit() else 10
        records = repo.get_history(limit=limit)

        if not records:
            console.print(f"  [{DIM}]No transcriptions yet.[/]")
            return

        table = make_table(
            f"Library — {len(records)} recent",
            [
                ("#", {"style": ACCENT, "justify": "right", "width": 5}),
                ("File", {"no_wrap": True}),
                ("Words", {"justify": "right", "width": 8}),
                ("Duration", {"justify": "right", "width": 10}),
                ("Model", {"width": 14}),
            ],
        )
        for rec in records:
            dur = format_duration(rec.get("duration") or 0)
            table.add_row(
                str(rec["id"]),
                rec.get("file_name", "?"),
                f"{rec.get('word_count', 0):,}",
                dur,
                rec.get("model", "?") or "?",
            )
        console.print(table)
        console.print(
            f"  [{DIM}]Use [{ACCENT}]\\focus <id>[/] to work on a file.[/]"
        )
    except Exception as e:
        console.print(f"  [{WARNING}]Could not list files: {e}[/]")


@register_backslash("show")
def cmd_show(session: ReplSession, args: str) -> None:
    """Show a transcript by ID, or the currently focused one."""
    from audiobench.cli.repl.dispatch import dispatch_command

    if args.strip().isdigit():
        dispatch_command(session, ["show", args.strip()])
    elif session.last_id:
        dispatch_command(session, ["show", str(session.last_id)])
    else:
        console.print(
            f"  [{DIM}]Usage: [{ACCENT}]\\show <id>[/] or focus on a file first.[/]"
        )


@register_backslash("jobs")
def cmd_jobs(session: ReplSession, args: str) -> None:
    """List background jobs and their status."""
    from audiobench.cli.repl.dispatch import dispatch_command

    dispatch_command(session, ["jobs"])


@register_backslash("search")
def cmd_search(session: ReplSession, args: str) -> None:
    """Full-text search transcripts — SQLite FTS5, instant, exact text match.

    Usage:  \\search "quarterly budget"
            \\search meeting --limit 20

    For meaning-based cross-source search, use: .search "concept"
    """
    import shlex

    from audiobench.cli.repl.dispatch import print_context_summary
    from audiobench.cli.repl.session import NavigationFrame
    from audiobench.core.db_engine import init_db
    from audiobench.core.focused_entity import FocusedEntity
    from audiobench.storage.repository import TranscriptionRepository

    if not args.strip():
        console.print(f"  [{DIM}]Usage: [{ACCENT}]\\search \"query\"[/][/]")
        console.print(f"  [{DIM}]For semantic search across all sources: [{ACCENT}].search \"concept\"[/][/]")
        return

    # Parse: first token is query, optional --limit N
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()

    query = parts[0]
    limit = 10

    i = 1
    while i < len(parts):
        if parts[i] == "--limit" and i + 1 < len(parts):
            try:
                limit = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    init_db()
    repo = TranscriptionRepository()
    results = repo.search(query, limit=limit)

    if not results:
        console.print(f"  [{DIM}]No results for \"{query}\"[/]")
        console.print(f"  [{DIM}]Try [{ACCENT}].search \"{query}\"[/] for semantic search.[/]")
        return

    # Display results
    console.print(f"\n  [{BOLD}]Text Search — SQLite FTS5[/]  [{DIM}]\"{query}\"[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")
    for i, r in enumerate(results, 1):
        file_name = r.get("file_name", f"Transcript #{r.get('id', '?')}")
        tx_id = r.get("id", "?")
        snippet = (r.get("text_preview") or "").strip().replace("\n", " ")
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        console.print(f"  [{ACCENT}][{i}][/] [{ACCENT}]◎[/] {file_name}  [{DIM}](#{tx_id})[/]")
        if snippet:
            console.print(f"       [{DIM}]\"{snippet}\"[/]")
        console.print()
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    # Interactive selection
    while True:
        console.print(
            f"  [{DIM}]Select to focus [1-{len(results)}], "
            f"[{ACCENT}]n <num>[/] to capture, "
            f"[{ACCENT}]q[/] to quit, or [{ACCENT}].search \"{query}\"[/] for semantic results[/]"
        )
        try:
            choice = input("  → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("q", "quit", ""):
            return

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                # Search returns transcript records — focus on the audio file
                audio_file_id = r.get("audio_file_id")
                tx_id = r.get("id")
                file_name = r.get("file_name", f"#{tx_id}")

                if audio_file_id:
                    session.push_frame(
                        NavigationFrame(context="\\search", state={}, intent=f'focused from search "{query}"')
                    )
                    session.focus = FocusedEntity(type="file", id=audio_file_id, label=file_name)
                    print_context_summary(session)
                    return
                elif tx_id:
                    session.set_context(tx_id)
                    print_context_summary(session)
                    return
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(results)}.[/]")
        elif choice.startswith("n ") and choice[2:].isdigit():
            idx = int(choice[2:]) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                tx_id = r.get("id")
                if tx_id:
                    from audiobench.core.db_session import get_session as _gs
                    from audiobench.memory.enums import SourceType
                    from audiobench.storage.models import ExpressionRecord
                    with _gs() as db:
                        expr = db.query(ExpressionRecord).filter_by(
                            source_id=tx_id,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value
                        ).first()
                    if expr:
                        cmd_capture(session, f"expression:{expr.id}")
                        return
                    else:
                        console.print(f"  [{WARNING}]Could not find expression for transcript #{tx_id}[/]")
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(results)}.[/]")
        else:
            console.print(f"  [{WARNING}]Enter a number, n <num>, or q.[/]")


@register_backslash("history")
def cmd_history(session: ReplSession, args: str) -> None:
    """Show transcript history. Usage: \\history [--limit 5]"""
    import shlex

    from audiobench.cli.repl.dispatch import dispatch_command

    parsed_args = shlex.split(args)
    dispatch_command(session, ["history"] + parsed_args)


@register_backslash("stats")
def cmd_stats(session: ReplSession, args: str) -> None:
    """Show detailed stats for the current transcript."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console, format_duration

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return

    console.print(f"\n  [{BOLD}]Transcription #{rec['id']}[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    console.print(f"  [{DIM}]File:[/]       [{ACCENT}]{rec.get('file_name', 'Unknown')}[/]")

    duration = format_duration(rec.get("duration_seconds") or 0)
    console.print(f"  [{DIM}]Duration:[/]   {duration}")
    console.print(f"  [{DIM}]Words:[/]      {rec.get('word_count', 0):,}")

    metadata = rec.get("metadata") or {}
    console.print(f"  [{DIM}]Model:[/]      {metadata.get('model', 'Unknown')}")
    console.print(f"  [{DIM}]Engine:[/]     {metadata.get('engine', 'Unknown')}")
    console.print(f"  [{DIM}]Preset:[/]     {metadata.get('speed_preset', 'balanced')}")
    console.print()


@register_backslash("chat")
def cmd_chat(session: ReplSession, args: str) -> None:
    """Start an interactive chat with the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    dispatch_command(session, ["chat", str(session.last_id)])


@register_backslash("ask")
def cmd_ask(session: ReplSession, args: str) -> None:
    """Ask a question about the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    if not args.strip():
        dispatch_command(session, ["ask", str(session.last_id), "--interactive"])
    else:
        dispatch_command(session, ["ask", str(session.last_id), args.strip()])


@register_backslash("summarize")
def cmd_summarize(session: ReplSession, args: str) -> None:
    """Generate an AI summary of the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    if not args.strip():
        dispatch_command(session, ["summarize", str(session.last_id), "--interactive"])
    else:
        dispatch_command(session, ["summarize", str(session.last_id)])

@register_backslash("import")
@require_tier(2)
def cmd_import(session: ReplSession, args: str) -> None:
    """Import audio files into the internal library. [Tier 2 required]"""
    import shlex

    from audiobench.cli.repl.dispatch import dispatch_command
    from audiobench.cli.repl.session import NavigationFrame

    session.push_frame(NavigationFrame(context="import", state={}, intent="import files"))

    if not args.strip():
        dispatch_command(session, ["import"])
    else:
        parsed_args = shlex.split(args)
        dispatch_command(session, ["import"] + parsed_args)

@register_backslash("transcribe")
@require_tier(2)
def cmd_transcribe(session: ReplSession, args: str) -> None:
    """Transcribe audio files (interactive wizard by default). [Tier 2 required]"""
    import shlex

    from audiobench.cli.repl.dispatch import dispatch_command
    from audiobench.cli.repl.session import NavigationFrame

    session.push_frame(NavigationFrame(context="transcribe", state={}, intent="transcribe audio"))

    if not args.strip():
        dispatch_command(session, ["transcribe", "--interactive"])
    else:
        parsed_args = shlex.split(args)
        dispatch_command(session, ["transcribe"] + parsed_args)

@register_backslash("config")
@require_tier(2)
def cmd_config(session: ReplSession, args: str) -> None:
    """Run configuration wizard. [Tier 2 required]"""
    from audiobench.cli.repl.dispatch import dispatch_command
    from audiobench.cli.repl.session import NavigationFrame
    session.push_frame(NavigationFrame(context="config", state={}, intent="configure app"))
    dispatch_command(session, ["config", "--interactive"])

# ── Notes & Capture ──────────────────────────────────────────

def _resolve_capture_destination(session: ReplSession, repo) -> int:
    """Tier 1: active note collection, Tier 2: context note collection, Tier 3: global inbox"""
    # Tier 1: User explicitly navigated to a note collection (if supported later)
    active_id = session.navigation_stack[-1].state.get("collection_id") if session.navigation_stack else None
    if active_id:
        return active_id

    # Tier 2: Focus is on an audio file
    if session.focus and session.focus.type == "file":
        col = repo.find_or_create_collection(session.focus.id, f"Notes: {session.focus.label}")
        return col.id

    # Tier 3: No focus, global inbox
    col = repo.find_or_create_collection(None, "Global Inbox")
    return col.id


@register_backslash("capture")
def cmd_capture(session: ReplSession, args: str) -> None:
    """Capture a thought or expression to the current context collection or inbox."""
    from audiobench.cli.display.theme import SUCCESS, console
    from audiobench.storage.note_repository import NoteRepository

    arg = args.strip()
    if not arg:
        try:
            arg = input("  Capture text > ").strip()
            if not arg: return
        except (EOFError, KeyboardInterrupt):
            return

    repo = NoteRepository()
    collection_id = _resolve_capture_destination(session, repo)

    expression_id = None
    if arg.startswith("expression:"):
        from audiobench.storage.expression_repository import ExpressionRepository
        expr_repo = ExpressionRepository()
        eid_str = arg.replace("expression:", "").strip()
        if eid_str.isdigit():
            expression_id = int(eid_str)
            expr = expr_repo.get_by_id(expression_id)
            if expr:
                arg = f"Captured expression: {expr.content[:100]}..."

    cols = repo.list_collections(limit=100)
    target_col = next((c for c in cols if c.id == collection_id), None)

    cap = repo.create_capture(
        collection_id=collection_id,
        body=arg,
        segment_id=None,
        transcript_expression_id=expression_id,
        collection_expression_id=target_col.expression_id if target_col else None
    )
    console.print(f"  [{SUCCESS}]→ Captured to: {target_col.title if target_col else 'Inbox'} (#capture:{cap.id})[/]")


# ── Help ─────────────────────────────────────────────────────


@register_backslash("help")
def cmd_help(session: ReplSession, args: str) -> None:
    """Show available backslash commands."""
    console.print(f"\n  [{BOLD}]AudioBench Shell[/]\n")
    console.print(f"  [{DIM}]Three execution contexts, one prompt:[/]\n")

    console.print(f"  [{ACCENT}]\\[/][{DIM}]command[/]   — local, instant (SQLite)")
    console.print(f"  [{DIM}]?[/] question  — AI/semantic (requires focus)")
    console.print(f"  [yellow]![/][{DIM}]command[/]   — OS shell passthrough")
    console.print(f"  [{DIM}]bare text  — implicit ask (requires focus)[/]\n")

    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"  [{BOLD}]Navigation[/]")
    console.print(f"    [{ACCENT}]\\back[/]              Return to previous context")
    console.print(f"    [{ACCENT}]\\home[/]              Clear stack, return to top")
    console.print(f"    [{ACCENT}]\\stack[/]             Show navigation stack\n")

    console.print(f"  [{BOLD}]Focus & View[/]")
    console.print(f"    [{ACCENT}]\\focus <id>[/]        Focus on audio file by ID")
    console.print(f"    [{ACCENT}]\\unfocus[/]           Clear current focus")
    console.print(f"    [{ACCENT}]\\ls [n][/]            List recent files (default 10)")
    console.print(f"    [{ACCENT}]\\show [id][/]         Show transcript\n")

    console.print(f"  [{BOLD}]Workflows[/]")
    console.print(f"    [{ACCENT}]\\workflow save <name>[/]   Capture last N commands")
    console.print(f"    [{ACCENT}]\\workflow list[/]          Show all saved workflows")
    console.print(f"    [{ACCENT}]\\workflow run <name>[/]    Replay a workflow")
    console.print(f"    [{ACCENT}]\\workflow delete <name>[/] Delete a workflow\n")

    console.print(f"  [{BOLD}]Intelligence[/]")
    console.print(f"    [{ACCENT}]\\related[/]           Show related fragments from other files")
    console.print(f"    [{ACCENT}]\\name[/]              Rename a speaker globally (e.g. \\name \"Speaker 1\" \"John\")")
    console.print(f"    [{ACCENT}]\\graph[/]             Command frequency stats")
    console.print(f"    [{ACCENT}]\\suggest[/]           What to do next (based on history)\n")

    console.print(f"  [{BOLD}]Interactive Flows[/]")
    console.print(f"    [{ACCENT}]\\library[/]  \\import[/]  \\transcribe[/]  \\chat[/]  \\ask[/]  \\summarize[/]\n")
    console.print(f"  [{DIM}]Shell passthrough: [/]  !ls  !ffmpeg  !play  ...\n")

# ── Workflow ──────────────────────────────────────────────────

@register_backslash("workflow")
def cmd_workflow(session: ReplSession, args: str) -> None:
    """Manage named workflows. Usage: \\workflow save|list|run|delete [name]"""
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    name = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list" or not subcmd:
        _workflow_list(session)
    elif subcmd == "save":
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow save <name>[/][/]")
            return
        _workflow_save(session, name)
    elif subcmd == "run":
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow run <name>[/][/]")
            return
        _workflow_run(session, name)
    elif subcmd in ("delete", "del", "rm"):
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow delete <name>[/][/]")
            return
        _workflow_delete(session, name)
    elif subcmd == "show":
        _workflow_show(name) if name else console.print(
            f"  [{WARNING}]Usage: [{ACCENT}]\\workflow show <name>[/][/]"
        )
    else:
        console.print(f"  [{WARNING}]Unknown subcommand: {subcmd}[/]")
        console.print(f"  [{DIM}]Use: save | list | run | delete | show[/]")


def _workflow_list(session: ReplSession) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        workflows = repo.list_workflows()
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflows: {e}[/]")
        return

    if not workflows:
        console.print(
            f"  [{DIM}]No workflows yet. Run [{ACCENT}]\\workflow save <name>[/] "
            f"to capture the last few commands.[/]"
        )
        return

    table = make_table(
        "Saved Workflows",
        [
            ("Name", {"style": ACCENT}),
            ("Steps", {"width": 6, "justify": "right"}),
            ("Description", {}),
            ("Updated", {"width": 20}),
        ],
    )
    for wf in workflows:
        table.add_row(
            wf["name"],
            str(len(wf["steps"])),
            wf.get("description", ""),
            str(wf.get("updated_at", ""))[:19],
        )
    console.print(table)
    console.print(f"  [{DIM}]Run with [{ACCENT}]\\workflow run <name>[/][/]")


def _workflow_save(session: ReplSession, name: str) -> None:
    """Capture the last N commands from command_events as a named workflow."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        recent = repo.get_recent_commands(limit=10)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command history: {e}[/]")
        return

    if not recent:
        console.print(f"  [{DIM}]No commands in history yet. Run some commands first.[/]")
        return

    import json as _json
    steps = [
        {"command": r["command"], "args": _json.loads(r.get("args_json", "[]"))}
        for r in reversed(recent)
    ]

    console.print(f"\n  [{BOLD}]Last {len(steps)} commands:[/]")
    for i, step in enumerate(steps, 1):
        args_str = " ".join(str(a) for a in step["args"]) if step["args"] else ""
        console.print(f"    [{ACCENT}]{i}.[/] {step['command']} [{DIM}]{args_str}[/]")

    try:
        keep_raw = input(
            f"\n  Keep all {len(steps)}? Enter indices to keep (e.g. 1,3,5) or Enter for all: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"\n  [{DIM}]Cancelled.[/]")
        return

    if keep_raw:
        try:
            indices = [int(x.strip()) - 1 for x in keep_raw.split(",")]
            steps = [steps[i] for i in indices if 0 <= i < len(steps)]
        except (ValueError, IndexError):
            console.print(f"  [{WARNING}]Invalid indices — saving all steps.[/]")

    try:
        desc_raw = input("  Description (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        desc_raw = ""

    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        repo.save_workflow(name, steps, description=desc_raw)
        console.print(f"\n  [{SUCCESS}]✓ Workflow [{ACCENT}]{name}[/] saved ({len(steps)} steps)[/]")
    except Exception as e:
        console.print(f"  [{WARNING}]Failed to save workflow: {e}[/]")


def _workflow_run(session: ReplSession, name: str) -> None:
    """Replay a named workflow step-by-step."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        wf = repo.get_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflow: {e}[/]")
        return

    if not wf:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")
        return

    steps = wf["steps"]
    if not steps:
        console.print(f"  [{DIM}]Workflow is empty.[/]")
        return

    console.print(f"\n  [{BOLD}]Running workflow:[/] [{ACCENT}]{name}[/] ({len(steps)} steps)")
    console.print(f"  [{DIM}]{'─' * 40}[/]")

    from audiobench.cli.repl.dispatch import dispatch_command
    for i, step in enumerate(steps, 1):
        cmd = step["command"]
        args = step.get("args", [])
        full_args = [cmd] + [str(a) for a in args]
        args_display = " ".join(str(a) for a in args)
        console.print(f"  [{DIM}][{i}/{len(steps)}][/] [{ACCENT}]{cmd}[/] {args_display}")
        dispatch_command(session, full_args)

    console.print(f"\n  [{SUCCESS}]✓ Workflow [{ACCENT}]{name}[/] complete.[/]")


def _workflow_delete(session: ReplSession, name: str) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        deleted = repo.delete_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not delete: {e}[/]")
        return
    if deleted:
        console.print(f"  [{SUCCESS}]✓ Deleted workflow [{ACCENT}]{name}[/][/]")
    else:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")


def _workflow_show(name: str) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        wf = repo.get_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflow: {e}[/]")
        return
    if not wf:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")
        return
    console.print(f"\n  [{BOLD}]Workflow:[/] [{ACCENT}]{name}[/]")
    if wf.get("description"):
        console.print(f"  [{DIM}]{wf['description']}[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    for i, step in enumerate(wf["steps"], 1):
        args_str = " ".join(str(a) for a in step.get("args", []))
        console.print(f"    [{ACCENT}]{i}.[/] {step['command']} [{DIM}]{args_str}[/]")
    console.print()


# ── Intelligence ──────────────────────────────────────────────

@register_backslash("related")
def cmd_related(session: ReplSession, args: str) -> None:
    """Show semantically related fragments from other transcripts."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return

    # Get text to search. Summary if available, otherwise first 500 chars.
    text_to_search = rec.get("summary", "")
    if not text_to_search:
        text_to_search = rec.get("full_text", "")[:500]

    if not text_to_search.strip():
        console.print(f"  [{DIM}]Transcript is empty, cannot find related content.[/]")
        return

    console.print(f"  [{DIM}]Searching for related fragments across the library...[/]")

    from audiobench.daemon.factory import get_daemon_client
    daemon = get_daemon_client()

    try:
        # Ask daemon for search
        results = daemon.search(
            query=text_to_search,
            top_k=20,
            use_bm25=False,  # pure semantic relation
            use_dense=True,
            use_colbert=True,
        )
    except Exception as e:
        console.print(f"  [{WARNING}]Daemon search failed: {e}[/]")
        return

    if not results:
        console.print(f"  [{DIM}]No related fragments found.[/]")
        return

    from audiobench.storage.expression_repository import ExpressionRepository
    expr_repo = ExpressionRepository()

    # Filter out current source_id
    related_exprs = []
    seen_ids = set()
    for r in results:
        expr_id = r.get("expression_id")
        if expr_id:
            expr = expr_repo.get_by_id(expr_id)
            if expr and expr.source_id != session.last_id and expr.source_id not in seen_ids:
                seen_ids.add(expr.source_id)  # One fragment per related transcript
                related_exprs.append((expr, r.get("score", 0.0)))
                if len(related_exprs) >= 5:
                    break

    if not related_exprs:
        console.print(f"  [{DIM}]No related fragments found outside of this transcript.[/]")
        return

    console.print(f"\n  [{BOLD}]Related Thoughts[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")

    for i, (expr, score) in enumerate(related_exprs, 1):
        # get original audio file name
        audio_file = repo.get_by_id(expr.source_id)
        name = audio_file.get("file_name", f"Transcript #{expr.source_id}") if audio_file else f"Source #{expr.source_id}"

        console.print(f"  [{ACCENT}]{i}. {name}[/] [{DIM}](score: {score:.2f})[/]")
        content = expr.content.strip().replace('\n', ' ')
        if len(content) > 150:
            content = content[:147] + "..."
        console.print(f"     \"{content}\"")
    console.print()

@register_backslash("name")
def cmd_name(session: ReplSession, args: str) -> None:
    """Rename a speaker in the focused transcript and globally."""
    import shlex

    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    try:
        parts = shlex.split(args)
    except ValueError as e:
        console.print(f"  [{WARNING}]Failed to parse arguments: {e}[/]")
        return

    if len(parts) != 2:
        console.print(f"  [{WARNING}]Usage: \\name \"Old Name\" \"New Name\"[/]")
        return

    old_name, new_name = parts[0], parts[1]
    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return

    # Update local segments
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import SegmentRecord

    with get_session() as db:
        segments = db.query(SegmentRecord).filter(
            SegmentRecord.transcription_id == session.last_id,
            SegmentRecord.speaker == old_name
        ).all()

        if not segments:
            console.print(f"  [{WARNING}]Speaker '{old_name}' not found in this transcript.[/]")
            return

        for seg in segments:
            seg.speaker = new_name

        # Update speaker_map on the transcription record
        import json
        try:
            transcription = db.query(TranscriptionRecord).filter_by(id=session.last_id).first()
            if transcription and transcription.speaker_map:
                s_map = json.loads(transcription.speaker_map)
                # Reverse lookup the original pyannote label (e.g. SPEAKER_00)
                pyannote_labels = [k for k, v in s_map.items() if v == old_name]
                for p_label in pyannote_labels:
                    s_map[p_label] = new_name
                transcription.speaker_map = json.dumps(s_map)
        except Exception as e:
            console.print(f"  [{WARNING}]Failed to update speaker_map: {e}[/]")

        db.commit()

    # Update Global Speaker Profile
    try:
        from audiobench.memory.memory_store import SpeakerProfileStore
        profile_store = SpeakerProfileStore()

        # We need to find the profile with old_name.
        # But LanceDB search is vector-based. We can filter by name.
        results = profile_store.table.search().where(f"name = '{old_name}'").limit(1).to_list()
        if results:
            profile_id = results[0]["profile_id"]
            vector = results[0]["vector"]
            profile_store.save_speaker(profile_id, new_name, vector)
            console.print(f"  [{ACCENT}]Global speaker profile updated to '{new_name}'[/]")
        else:
            console.print(f"  [{DIM}]No global voice print found for '{old_name}' to rename.[/]")
    except Exception as e:
        console.print(f"  [{WARNING}]Failed to update global speaker profile: {e}[/]")

    console.print(f"  [{BOLD}]Renamed {len(segments)} segments to '{new_name}' in transcript #{session.last_id}[/]")


@register_backslash("graph")
def cmd_graph(session: ReplSession, args: str) -> None:
    """Show command frequency stats from the command graph."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        days = int(args.strip()) if args.strip().isdigit() else 30
        stats = repo.get_command_stats(days=days)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command graph: {e}[/]")
        return

    if not stats:
        console.print(
            f"  [{DIM}]No command history yet. Run commands inside the REPL to build it.[/]"
        )
        return

    table = make_table(
        f"Command Frequency — last {days} days",
        [
            ("Command", {"style": ACCENT}),
            ("Count", {"width": 8, "justify": "right"}),
            ("Avg ms", {"width": 10, "justify": "right"}),
        ],
    )
    for row in stats:
        avg = f"{row['avg_ms']:.0f}" if row["avg_ms"] else "—"
        table.add_row(row["command"], str(row["count"]), avg)
    console.print(table)


@register_backslash("suggest")
def cmd_suggest(session: ReplSession, args: str) -> None:
    """Suggest next actions based on your command history."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        recent = repo.get_recent_commands(limit=5)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command history: {e}[/]")
        return

    if not recent:
        console.print(f"  [{DIM}]No history yet. Run some commands first.[/]")
        return

    last_cmd = recent[0]["command"]
    suggestions = []
    try:
        suggestions = repo.get_next_command_suggestions(after_command=last_cmd, limit=5)
    except Exception:
        pass

    console.print(f"\n  [{BOLD}]Last command:[/] [{ACCENT}]{last_cmd}[/]")
    if suggestions:
        console.print(f"  [{DIM}]What you usually do next:[/]")
        for s in suggestions:
            console.print(f"    [{ACCENT}]\\{s['command']}[/] [{DIM}]({s['count']}×)[/]")
    else:
        console.print(f"  [{DIM}]Not enough history to suggest yet. Keep using AudioBench![/]")

    try:
        workflows = repo.list_workflows()
        if workflows:
            console.print(f"\n  [{DIM}]Saved workflows:[/]")
            for wf in workflows[:3]:
                console.print(
                    f"    [{ACCENT}]\\workflow run {wf['name']}[/] "
                    f"[{DIM}]({len(wf['steps'])} steps)[/]"
                )
    except Exception:
        pass
    console.print()


# ── Library ──────────────────────────────────────────────────

@register_backslash("library")
def cmd_library(session: ReplSession, args: str) -> None:
    """Launch the interactive Library Command Center."""
    from audiobench.cli.commands.library_cmd import run_library
    run_library(session)

@register_resume("library")
def resume_library(session: ReplSession, frame) -> None:
    """Resume the library TUI with its previous state."""
    from audiobench.cli.commands.library_cmd import run_library
    run_library(session, restore_state=frame.state)

@register_resume("chat")
def resume_chat(session: ReplSession, frame) -> None:
    """Resume a previous chat conversation."""
    conversation_id = frame.state.get("conversation_id")
    if conversation_id:
        from audiobench.cli.repl.dispatch import dispatch_command
        dispatch_command(session, ["chat", "--resume", str(conversation_id)])

@register_resume("import")
def resume_import(session: ReplSession, frame) -> None:
    """Resume the import flow."""
    from audiobench.cli.commands.import_cmd import run_import_flow
    run_import_flow(session=session, restore_state=frame.state)

@register_resume("transcribe")
def resume_transcribe(session: ReplSession, frame) -> None:
    """Resume the transcribe flow."""
    from audiobench.cli.repl.dispatch import dispatch_command
    dispatch_command(session, ["transcribe", "--interactive"])

@register_resume("config")
def resume_config(session: ReplSession, frame) -> None:
    """Resume the config flow."""
    from audiobench.cli.repl.dispatch import dispatch_command
    dispatch_command(session, ["config", "--interactive"])


# ── Observatory ──────────────────────────────────────────────


@register_backslash("obs")
def cmd_obs(session: ReplSession, args: str) -> None:
    """Launch the Observatory live TUI. Usage: \\obs [--subsystem X] [--level L]"""
    import shlex

    from audiobench.events import get_bus
    from audiobench.observatory.db import init_journal_db
    from audiobench.observatory.subscriber import get_subscriber

    init_journal_db()
    get_bus().on("*", get_subscriber().record)

    subsystem = None
    level = None
    try:
        parts = shlex.split(args)
        for i, p in enumerate(parts):
            if p in ("--subsystem", "-s") and i + 1 < len(parts):
                subsystem = parts[i + 1]
            elif p in ("--level", "-l") and i + 1 < len(parts):
                level = parts[i + 1]
    except Exception:
        pass

    from audiobench.cli.tui.observatory_app import ObservatoryApp
    ObservatoryApp(subsystem=subsystem, level=level).run()


@register_backslash("logs")
def cmd_logs(session: ReplSession, args: str) -> None:
    r"""Print recent Observatory events. Usage: \logs [daemon|errors|--entity ID]

    \b
    Examples:
      \logs              — last 50 events
      \logs daemon       — subsystem=daemon
      \logs errors       — level in WARN/ERROR/CRITICAL
      \logs --entity 42  — entity_id=42
      \logs --follow     — live tail mode
    """
    import shlex

    from audiobench.observatory.db import init_journal_db, query_events

    init_journal_db()

    parts = shlex.split(args) if args.strip() else []

    subsystem = None
    level = None
    entity_id = None
    follow = False

    if parts:
        first = parts[0].lower()
        if first == "daemon":
            subsystem = "daemon"
        elif first in ("errors", "error"):
            level = "ERROR"
        elif first == "--follow":
            follow = True
        elif first == "--entity" and len(parts) > 1:
            try:
                entity_id = int(parts[1])
            except ValueError:
                pass

    if follow:
        import time
        console.print("[dim]Following Observatory events. Ctrl+C to stop.[/]\n")
        last_id = 0
        try:
            while True:
                events = query_events(
                    subsystem=subsystem, level=level,
                    entity_id=entity_id, id_gt=last_id, limit=50
                )
                for ev in events:
                    ts = (ev.get("ts") or "")[:19]
                    lvl = ev.get("level", "INFO")
                    sub = ev.get("subsystem", "?")
                    etype = ev.get("event_type", "")
                    msg = (ev.get("message") or "")[:120]
                    lc = {"INFO": "cyan", "WARN": "yellow", "ERROR": "red", "CRITICAL": "bold red"}.get(lvl, "white")
                    console.print(f"[dim]{ts}[/] [{lc}]{lvl:5}[/] [bold]{sub:12}[/] {etype:30} {msg}")
                    last_id = max(last_id, ev.get("id", 0))
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    else:
        events = query_events(
            subsystem=subsystem, level=level,
            entity_id=entity_id, limit=50
        )
        if not events:
            console.print(f"[{DIM}]No events found.[/]")
            return
        for ev in reversed(events):
            ts = (ev.get("ts") or "")[:19]
            lvl = ev.get("level", "INFO")
            sub = ev.get("subsystem", "?")
            etype = ev.get("event_type", "")
            msg = (ev.get("message") or "")[:120]
            lc = {"INFO": "cyan", "WARN": "yellow", "ERROR": "red", "CRITICAL": "bold red"}.get(lvl, "white")
            console.print(f"[dim]{ts}[/] [{lc}]{lvl:5}[/] [bold]{sub:12}[/] {etype:30} {msg}")
        console.print(f"\n[{DIM}]{len(events)} event(s)[/]")


# ── Security ──────────────────────────────────────────────────────────────


@register_backslash("lock")
def cmd_lock(session: ReplSession, args: str) -> None:
    """Immediately drop ALL unlock tiers (per-query AND session).

    Usage: \\lock
    """
    if session.unlocked_tier == 0 and session.session_unlock_tier == 0:
        console.print(f"  [{DIM}]Already locked.[/]")
        return
    had_session = session.session_unlock_tier > 0
    session.unlocked_tier = 0
    session.tier1_unlocked_at = None
    session.session_unlock_tier = 0
    extra = " (session unlock cleared too)" if had_session else ""
    console.print(f"  [{DIM}]● Locked{extra}.[/]")

@register_backslash("unlock")
def cmd_unlock(session: ReplSession, args: str) -> None:
    """Elevate session tier with passphrase verification.

    Usage:
        \\unlock            → Tier 2 per-query (re-auth each time)
        \\unlock 1          → Tier 1 per-query (10-min TTL)
        \\unlock 2          → Tier 2 per-query
        \\unlock 3          → Tier 3 per-query (never gets --session)
        \\unlock --session  → Tier 2 for the entire REPL session
        \\unlock 1 --session → Tier 1 for the entire REPL session
        \\unlock 2 --session → Tier 2 for the entire REPL session

    Session mode (--session):
        The tier stays active until \\lock or idle-lock fires.
        The ★N badge appears in the prompt as a permanent reminder.
        Tier 3 cannot use --session — it is always per-query.
    """
    import getpass
    from datetime import UTC, datetime

    from audiobench.security.auth import verify_passphrase

    # Parse args: support "2", "--session", "2 --session", "--session 2"
    parts = args.strip().split()
    session_mode = "--session" in parts
    tier_parts = [p for p in parts if p != "--session"]

    if tier_parts and tier_parts[0].isdigit():
        tier = int(tier_parts[0])
    else:
        tier = 2  # default to Intimate

    if tier < 1 or tier > 3:
        console.print(f"  [{WARNING}]Invalid tier. Use 1, 2, or 3.[/]")
        return

    # Tier 3 cannot be session-unlocked
    if session_mode and tier == 3:
        console.print(
            f"  [{WARNING}]Tier 3 (Override) cannot use --session.[/] "
            f"[{DIM}]It is always per-query for maximum safety.[/]"
        )
        return

    # No-op if already at or above the requested tier via session unlock
    if session_mode and session.session_unlock_tier >= tier:
        tier_labels = {1: "Relational", 2: "Intimate"}
        console.print(
            f"  [{DIM}]Session already unlocked at Tier {session.session_unlock_tier} "
            f"({tier_labels.get(session.session_unlock_tier, '')}).[/]"
        )
        return

    # No-op if already at or above via per-query unlock (and not requesting session)
    if not session_mode and session.effective_tier() >= tier:
        tier_labels = {1: "Relational", 2: "Intimate", 3: "Override"}
        console.print(
            f"  [{DIM}]Already at Tier {tier} ({tier_labels.get(tier, '')}).[/] "
            f"[{DIM}]Add --session to make it persistent.[/]"
        )
        return

    try:
        passphrase = getpass.getpass("  Passphrase: ")
    except (KeyboardInterrupt, EOFError):
        console.print(f"\n  [{DIM}]Cancelled.[/]")
        return

    if not verify_passphrase(passphrase):
        console.print(f"  [{WARNING}]Incorrect passphrase.[/]")
        return

    tier_labels = {1: "Relational", 2: "Intimate", 3: "Override"}
    tl = tier_labels.get(tier, f"Tier {tier}")

    if session_mode:
        # Session unlock — persists until \lock or idle-lock
        session.session_unlock_tier = tier
        session.unlocked_tier = tier  # also set per-query so effective_tier() is consistent
        if tier == 1:
            session.tier1_unlocked_at = datetime.now(UTC)
        console.print(
            f"  [{SUCCESS}]✓ Tier {tier} — {tl} unlocked for session.[/] "
            f"[{DIM}](★{tier} badge active — \\lock to clear)[/]"
        )
    else:
        # Per-query unlock
        session.unlocked_tier = tier
        if tier == 1:
            session.tier1_unlocked_at = datetime.now(UTC)
            console.print(
                f"  [{SUCCESS}]✓ Tier 1 — {tl} unlocked.[/] "
                f"[{DIM}](expires in 10 min — use --session to persist)[/]"
            )
        elif tier == 2:
            console.print(
                f"  [{SUCCESS}]✓ Tier 2 — {tl} unlocked.[/] "
                f"[{DIM}](active for this query — use --session to persist)[/]"
            )
        else:
            console.print(
                f"  [{SUCCESS}]✓ Tier 3 — {tl} unlocked.[/] [{DIM}](per-query only)[/]"
            )

@register_backslash("security")
def cmd_security(session: ReplSession, args: str) -> None:
    """Show the security status dashboard.

    Displays: current session tier, session unlock mode, enrolled voiceprint,
    tier map, idle-lock setting, and available security commands.

    Usage: \\security
    """
    from audiobench.cli.display.theme import BOLD
    from audiobench.security.voiceprint import get_enrollment_summary

    # ── Session Status panel ──
    tier = session.effective_tier()
    tier_labels = {0: "Public Mode", 1: "Relational", 2: "Intimate", 3: "Override"}
    tier_color = {0: DIM, 1: "yellow", 2: "green", 3: "cyan"}
    tier_label = tier_labels.get(tier, f"Tier {tier}")
    tc = tier_color.get(tier, DIM)

    vp_summary = get_enrollment_summary() or "Not enrolled"
    vp_color = SUCCESS if "enrolled" in vp_summary.lower() or "(" in vp_summary else WARNING

    from audiobench.core.settings import get_settings
    settings = get_settings()
    idle_str = f"{settings.idle_lock_seconds}s" if settings.idle_lock_seconds > 0 else "disabled"

    # Session unlock mode label
    if session.session_unlock_tier > 0:
        stl = tier_labels.get(session.session_unlock_tier, f"Tier {session.session_unlock_tier}")
        session_mode_str = f"★{session.session_unlock_tier} — {stl} (session)"
        sm_color = "green"
    else:
        session_mode_str = "per-query only"
        sm_color = DIM

    console.print(f"\n  [{BOLD}]● Security Status[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")
    console.print(f"  [{DIM}]Effective Tier:[/] [{tc}]{tier} — {tier_label}[/]")
    console.print(f"  [{DIM}]Unlock Mode:  [/] [{sm_color}]{session_mode_str}[/]")
    console.print(f"  [{DIM}]Voiceprint:   [/] [{vp_color}]{vp_summary}[/]")
    console.print(f"  [{DIM}]Auto-lock:    [/] {idle_str} idle")

    # ── Tier Map panel ──
    console.print(f"\n  [{BOLD}]Tier Map[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")
    tiers_info = [
        (0, "Public",     "audiobooks, podcasts, technical notes",  tier >= 0),
        (1, "Relational", "conversations with other speakers",       tier >= 1),
        (2, "Intimate",   "your voice segments",                    tier >= 2),
        (3, "Override",   "manually flagged (--sensitive)",          tier >= 3),
    ]
    for n, name, desc, unlocked in tiers_info:
        lock_icon = "✓" if unlocked else "🔒"
        color = SUCCESS if unlocked else "red"
        # Mark which tier is the active session unlock
        star = f" [{ACCENT}]★[/]" if n == session.session_unlock_tier and n > 0 else ""
        console.print(
            f"  [{color}]{lock_icon}[/] Tier {n} [{ACCENT}]{name:<12}[/] [{DIM}]{desc}[/]{star}"
        )

    # ── Available Commands panel ──
    console.print(f"\n  [{BOLD}]Commands[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")
    cmds = [
        ("\\unlock [1|2] --session", "Unlock tier for whole session (★ badge)"),
        ("\\unlock [1|2|3]",         "Per-query unlock"),
        ("\\lock",                   "Drop all tiers immediately"),
        ("\\enroll [audio]",          "Enroll voiceprint (SpeechBrain)"),
        ("\\voices",                 "Manage enrolled speaker voices"),
    ]
    for cmd, desc in cmds:
        console.print(f"  [{ACCENT}]{cmd:<28}[/] [{DIM}]{desc}[/]")
    console.print()

@register_backslash("enroll")
@require_tier(2)
def cmd_enroll(session: ReplSession, args: str) -> None:
    r"""Enroll a voiceprint using SpeechBrain ECAPA-TDNN.

    When called with no arguments, launches an interactive file picker so you
    can browse your filesystem and select a voice sample. When called with a
    path, skips straight to the confirmation step.

    Usage:
        \enroll                           Interactive wizard (file picker)
        \enroll ~/recordings/voice.opus   Direct path, prompts for name
        \enroll ~/voice.opus "My Name"    Direct path + name, confirm only
    """
    from pathlib import Path

    from rich.panel import Panel


    # ── Resolve audio path ───────────────────────────────────────────────────
    parts = args.strip().split(None, 1) if args.strip() else []

    if not parts:
        # ── Interactive file picker ──────────────────────────────────────
        console.print(
            Panel(
                f"  [{DIM}]↑↓/k j: Move  →/Enter: Enter dir  ←/h: Go up[/]\n"
                f"  [{DIM}]s: Confirm highlighted file  q: Cancel[/]",
                title=f"[bold][{ACCENT}]🎙 Voice Enrollment — Select Audio File[/][/]",
                title_align="left",
                border_style=ACCENT,
                expand=False,
            )
        )
        try:
            import curses

            from audiobench.cli.tui.import_tui import ImportFileManager

            AUDIO_EXTS = {
                "mp3", "m4a", "m4b", "aac", "ogg", "opus", "flac",
                "wav", "wma", "mp4", "webm", "mkv", "mov", "avi",
            }

            class EnrollFileManager(ImportFileManager):
                def __init__(self):
                    super().__init__()
                    self.navigator.allowed_extensions = AUDIO_EXTS

                def draw(self, stdscr):
                    curses.curs_set(0)
                    stdscr.nodelay(0)
                    stdscr.timeout(100)
                    try:
                        curses.start_color()
                        curses.use_default_colors()
                        curses.init_pair(1, curses.COLOR_CYAN,    -1)
                        curses.init_pair(2, curses.COLOR_WHITE,   -1)
                        curses.init_pair(3, curses.COLOR_GREEN,   -1)
                        curses.init_pair(4, curses.COLOR_YELLOW,  -1)
                        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
                        curses.init_pair(6, curses.COLOR_CYAN,    -1)
                    except Exception:
                        pass

                    while True:
                        stdscr.clear()
                        height, width = stdscr.getmaxyx()
                        header = " 🎙 Voice Enrollment — Pick a sample "
                        path_str = f" {self.navigator.current_path}"
                        if len(path_str) > width - len(header) - 3:
                            path_str = "..." + path_str[-(width - len(header) - 6):]
                        stdscr.addstr(0, 0, header, curses.color_pair(1) | curses.A_BOLD)
                        stdscr.addstr(0, len(header), path_str.ljust(width - len(header)), curses.color_pair(1))

                        items = self.navigator.list_items()
                        info = f" Audio files only  |  {self.navigator.selected + 1}/{len(items)}" if items else " (empty)"
                        stdscr.addstr(1, 0, info.ljust(width), curses.color_pair(5))
                        if width > 1:
                            stdscr.addstr(2, 0, "─" * (width - 1), curses.color_pair(6))

                        max_vis = height - 5
                        if self.navigator.selected < self.ui_offset:
                            self.ui_offset = self.navigator.selected
                        elif self.navigator.selected >= self.ui_offset + max_vis:
                            self.ui_offset = self.navigator.selected - max_vis + 1

                        for off, item in enumerate(items[self.ui_offset:self.ui_offset + max_vis]):
                            idx = self.ui_offset + off
                            y = off + 3
                            item_path = self.navigator.current_path / item
                            icon = "📁" if item_path.is_dir() else "🎵"
                            marker = "►" if idx == self.navigator.selected else " "
                            attr = curses.color_pair(3) | curses.A_BOLD if idx == self.navigator.selected else curses.color_pair(2)
                            max_len = width - 10
                            display = item[:max_len] + ".." if len(item) > max_len else item
                            try:
                                stdscr.addstr(y, 0, f"{marker} {icon}  {display}", attr)
                            except curses.error:
                                pass

                        footer = " ↑↓/k j: Move | →/Enter/l: Enter dir | ←/h: Go up | s: Select | q: Cancel "
                        if height > 1:
                            try:
                                stdscr.addstr(height - 1, 0, footer.ljust(width - 1)[:width - 1], curses.color_pair(1))
                            except curses.error:
                                pass

                        key = stdscr.getch()
                        if key == ord("q"):
                            self.cancelled = True
                            break
                        elif key == ord("s"):
                            current = self.navigator.get_current_item_path()
                            if current and current.is_file():
                                self.confirmed_files = [current]
                                break
                            else:
                                try:
                                    stdscr.addstr(height - 2, 0, " Please highlight a file, not a folder ".ljust(width - 1), curses.color_pair(4))
                                    stdscr.refresh()
                                    curses.napms(1200)
                                except curses.error:
                                    pass
                        elif key in (curses.KEY_RIGHT, 10, 13, ord("l")):
                            self.navigator.enter()
                        elif key in (curses.KEY_LEFT, ord("h")):
                            self.navigator.go_up()
                        elif key in (curses.KEY_UP, ord("k")):
                            self.navigator.select_prev()
                        elif key in (curses.KEY_DOWN, ord("j")):
                            self.navigator.select_next()

            fm = EnrollFileManager()
            curses.wrapper(fm.draw)

            if fm.cancelled or not fm.confirmed_files:
                console.print(f"\n  [{DIM}]Enrollment cancelled.[/]")
                return

            audio_path = fm.confirmed_files[0]

        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Cancelled.[/]")
            return

    else:
        # Direct path provided — backward-compatible
        audio_path = Path(parts[0].strip('"').strip("'"))
        if not audio_path.exists():
            console.print(f"  [{WARNING}]File not found: {audio_path}[/]")
            return

    # ── Duration gate ────────────────────────────────────────────────────────
    try:
        from audiobench.transcribe.audio_converter import probe
        info = probe(str(audio_path))
        duration_s = info.duration
        duration_str = f"{int(duration_s // 60)}m {int(duration_s % 60)}s"
        size_mb = audio_path.stat().st_size / (1024 * 1024)
    except Exception:
        duration_s = 0
        duration_str = "unknown"
        size_mb = audio_path.stat().st_size / (1024 * 1024)

    if 0 < duration_s < 20:
        console.print(
            f"  [{WARNING}]Audio too short ({duration_str}). "
            f"Enrollment needs at least 20 seconds of your voice.[/]"
        )
        return

    # ── Name prompt ─────────────────────────────────────────────────────────
    name_from_args = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else None

    if name_from_args:
        name = name_from_args
    else:
        try:
            import questionary
            name = questionary.text("  Name for this voice:", default="Owner").ask()
            if name is None:
                console.print(f"\n  [{DIM}]Cancelled.[/]")
                return
            name = name.strip() or "Owner"
        except ImportError:
            name = input("  Name for this voice [Owner]: ").strip() or "Owner"
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{DIM}]Cancelled.[/]")
            return

    # ── Confirmation summary ─────────────────────────────────────────────────
    from audiobench.core.settings import get_settings
    threshold = get_settings().voiceprint_threshold

    console.print(
        Panel(
            f"  [bold]File:[/]       {audio_path.name}  [{DIM}]({duration_str} · {size_mb:.1f} MB)[/]\n"
            f"  [bold]Name:[/]       {name}\n"
            f"  [bold]Model:[/]      ECAPA-TDNN  [{DIM}](speechbrain/spkrec-ecapa-voxceleb)[/]\n"
            f"  [bold]Threshold:[/]  {threshold}  [{DIM}](cosine similarity)[/]",
            title=f"[bold][{ACCENT}]🎙 Voice Enrollment — Confirm[/][/]",
            title_align="left",
            border_style=ACCENT,
            expand=False,
        )
    )

    try:
        import questionary
        go = questionary.confirm("  Start enrollment?", default=True).ask()
        if not go:
            console.print(f"  [{DIM}]Cancelled.[/]")
            return
    except ImportError:
        confirm_input = input("  Start enrollment? [Y/n] ").strip().lower()
        if confirm_input not in ("", "y", "yes"):
            console.print(f"  [{DIM}]Cancelled.[/]")
            return
    except (KeyboardInterrupt, EOFError):
        console.print(f"\n  [{DIM}]Cancelled.[/]")
        return

    # ── Enroll ───────────────────────────────────────────────────────────────
    console.print(f"\n  [{DIM}]Loading SpeechBrain ECAPA-TDNN…[/]")
    try:
        from audiobench.security.voiceprint import enroll

        result = enroll(audio_path=audio_path, name=name)

        console.print(f"\n  [{SUCCESS}]✓ Voiceprint enrolled as '{result['name']}'[/]")
        console.print(f"  [{SUCCESS}]✓ All future transcriptions will auto-tag on biometric pass[/]")
        console.print(f"  [{DIM}]To retroactively tag existing recordings in your database, run [{ACCENT}]\\voices retag[/].[/]")

    except ImportError as e:
        console.print(f"  [{WARNING}]{e}[/]")
    except ValueError as e:
        console.print(f"  [{WARNING}]{e}[/]")
    except Exception as e:
        console.print(f"  [{WARNING}]Enrollment failed: {e}[/]")
    console.print()



@register_backslash("voices")
def cmd_voices(session: ReplSession, args: str) -> None:
    r"""Manage enrolled speaker voices.

    Shows the enrolled speaker table. Management sub-commands require Tier 2.

    Usage:
        \voices                   Show enrolled speakers
        \voices tag <id> <name>   Assign a name (Tier 2)
        \voices remove <id>       Remove enrollment (Tier 2)
        \voices threshold <id> <val>  Adjust threshold 0.0-1.0 (Tier 2)
        \voices retag             Re-run biometric pass on corpus (Tier 2)
    """
    import json

    from audiobench.core.settings import get_settings
    from audiobench.security.voiceprint import is_enrolled

    parts = args.strip().split()
    sub = parts[0].lower() if parts else ""

    # ── Sub-commands that require Tier 2 ──
    if sub in ("tag", "remove", "threshold", "retag"):
        if session.effective_tier() < 2:
            console.print(
                f"  [{WARNING}]Access denied.[/] [{DIM}]Tier 2 required. "
                f"Run [{ACCENT}]\\unlock[/].[/]"
            )
            return

    if sub == "retag":
        console.print(f"  [{DIM}]Re-running biometric pass on entire corpus...[/]")
        try:
            import numpy as np

            from audiobench.security.voiceprint import (
                _load_ecapa,
                _retroactive_tag,
                _voiceprint_path,
            )
            model = _load_ecapa()
            enrolled_vec = np.load(str(_voiceprint_path())).astype("float32")
            n = _retroactive_tag(model, enrolled_vec)
            console.print(f"  [{SUCCESS}]✓ Retroactive pass complete — {n:,} segments tagged.[/]")
        except Exception as e:
            console.print(f"  [{WARNING}]Retag failed: {e}[/]")
        return

    # ── Default: show status table ──
    console.print("\n  [bold]Voice Management[/]")
    console.print(f"  [{DIM}]{'─' * 60}[/]")

    if not is_enrolled():
        console.print(f"  [{WARNING}]No voiceprint enrolled.[/]")
        console.print(f"  [{DIM}]Run [{ACCENT}]\\enroll <audio>[/] to enroll your voice.[/]\n")
        return

    settings = get_settings()
    meta_path = settings.security_dir / "enrollment.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    name = meta.get("name", "Unknown")
    n_tagged = meta.get("segments_tagged", 0)
    enrolled_at = (meta.get("enrolled_at", "")[:10]) or "—"
    threshold = settings.voiceprint_threshold

    console.print(f"  [{DIM}]{'ID':<4} {'Name':<14} {'Type':<8} {'Threshold':<10} {'Segments':<10} {'Status':<10} Enrolled[/]")
    console.print(f"  [{DIM}]{'─' * 65}[/]")
    console.print(
        f"  [{ACCENT}]{'1':<4}[/] [{ACCENT}]{name:<14}[/] {'Owner':<8} "
        f"{threshold:<10.2f} [{SUCCESS}]{n_tagged:,<9}[/] [{SUCCESS}]✓ Active  [/] {enrolled_at}"
    )
    console.print()

    console.print(f"  [{DIM}]Voice → Tier mapping:[/]")
    console.print(f"  [{DIM}]  Owner voices    → Tier 2 (Intimate)[/]")
    console.print(f"  [{DIM}]  Peer voices     → Tier 1 (Relational)[/]")
    console.print(f"  [{DIM}]  Foreign voices  → Tier 0 (Public)[/]")
    console.print()

    console.print(f"  [{DIM}]Sub-commands (Tier 2 required):[/]")
    sub_cmds = [
        ("\\voices retag",            "Re-run biometric pass on entire corpus"),
    ]
    for sc, desc in sub_cmds:
        console.print(f"  [{ACCENT}]{sc:<28}[/] [{DIM}]{desc}[/]")
    console.print()
