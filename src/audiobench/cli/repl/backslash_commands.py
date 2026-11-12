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

import logging
from typing import Any, Callable

logger = logging.getLogger("audiobench.cli.repl.backslash_commands")


from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    BOX_STYLE,
    DIM,
    ERROR,
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


def _check_bridge_staleness(session: ReplSession, bridge: Any, b_idx: int) -> None:
    """Emit a staleness warning if bridge b_idx no longer matches current synthesis citations.

    Uses citation-EID comparison only — no snapshot timestamps, which were shown to produce
    false positives when cite 1 coincidentally pointed to the same expression across syntheses.

    A stale bridge occurs when:
    - ``session.last_synthesis_citations[for_cite] != bridge.target_expression_id``: the
      citation the bookmark was built from now points to a different expression, meaning a
      ``\\synthesize`` ran between ``\\bookmark`` and ``\\play``/``\\focus``.
    - Fallback: ``bridge.target_expression_id`` not in any current citation's EID set, and
      ``for_cite`` is unknown.
    """
    bridge_eid = getattr(bridge, "target_expression_id", None)
    citations: dict[int, int] = getattr(session, "last_synthesis_citations", {})
    for_cite: int | None = getattr(session, "last_bookmark_bridges_for_cite", None)

    is_stale = False
    if citations and for_cite and for_cite in citations:
        if citations[for_cite] != bridge_eid:
            is_stale = True
    elif citations and bridge_eid and bridge_eid not in citations.values():
        is_stale = True

    if is_stale and bridge_eid:
        if for_cite:
            refresh_hint = f"\\bookmark cite {for_cite}"
        elif citations:
            refresh_hint = f"\\bookmark cite {max(citations.keys())}"
        else:
            refresh_hint = f"\\bookmark {bridge_eid}"
        console.print(
            f"  [{WARNING}]⚠ Bridge b{b_idx} is stale — built from expression #{bridge_eid}. "
            f"Run [{ACCENT}]{refresh_hint}[/] to refresh.[/]"
        )


def _focus_bridge(session: ReplSession, bridge: Any, idx: int) -> None:
    """Switch focus to the work or audio file represented by a semantic bridge."""
    from audiobench.cli.repl.dispatch import print_context_summary
    from audiobench.core.db_session import get_session
    from audiobench.core.focused_entity import FocusedEntity
    from sqlalchemy import text as _sql

    with get_session() as session_db:
        row = session_db.execute(
            _sql("""
                SELECT e.id, e.work_id, w.title, w.author,
                       COALESCE(t.audio_file_id, c.audio_file_id, c2.audio_file_id) as audio_file_id,
                       COALESCE(af_t.file_name, af_c.file_name, af_c2.file_name) as file_name
                FROM expressions e
                LEFT JOIN transcriptions t ON (e.source_type IN ('audio_transcript', 'transcription_chunk') AND e.source_id = t.id)
                LEFT JOIN audio_files af_t ON t.audio_file_id = af_t.id
                LEFT JOIN chapters c ON (e.source_type = 'chapter_summary' AND e.source_id = c.id)
                LEFT JOIN audio_files af_c ON c.audio_file_id = af_c.id
                LEFT JOIN chapters c2 ON (e.chapter_id IS NOT NULL AND e.chapter_id = c2.id)
                LEFT JOIN audio_files af_c2 ON c2.audio_file_id = af_c2.id
                LEFT JOIN works w ON e.work_id = w.id
                WHERE e.id = :eid
            """),
            {"eid": bridge.target_expression_id},
        ).fetchone()

        if row and row[4]:
            af_id, f_name = row[4], row[5] or bridge.work_title
            session.focus = FocusedEntity(
                type="file",
                id=af_id,
                label=f_name,
            )
        elif row and row[1]:
            wid, wtitle = row[1], row[2] or bridge.work_title
            session.focus = FocusedEntity(
                type="work",
                id=wid,
                label=f"Work: {wtitle}",
            )
        else:
            session.focus = FocusedEntity(
                type="expression",
                id=bridge.target_expression_id,
                label=f"{bridge.author} — {bridge.work_title}",
            )

    console.print(
        f"  [{SUCCESS}]Focus switched to semantic bridge [B{idx}]:[/] [{ACCENT}]{bridge.author} — {bridge.work_title}[/] "
        f"[{DIM}](Expression #{bridge.target_expression_id})[/]"
    )
    print_context_summary(session)


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

    # Support \focus b1, \focus b2, \focus bridge 1 (Semantic Bridges from Bookmark card)
    lower_arg = arg.lower()
    b_num: int | None = None
    if lower_arg.startswith("b") and lower_arg[1:].isdigit():
        b_num = int(lower_arg[1:])
    elif lower_arg.startswith("bridge") and lower_arg[6:].strip().isdigit():
        b_num = int(lower_arg[6:].strip())

    if b_num is not None:
        bridges = getattr(session, "last_bookmark_bridges", {})
        if b_num not in bridges:
            console.print(f"  [{WARNING}]Semantic bridge [B{b_num}] not found. Run \\bookmark to view bridges.[/]")
            return
        bridge = bridges[b_num]
        _check_bridge_staleness(session, bridge, b_num)
        _focus_bridge(session, bridge, b_num)
        return

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

    # Support \focus s12 or \focus school:12
    school_id: int | None = None
    if lower_arg.startswith("school:") and lower_arg[7:].isdigit():
        school_id = int(lower_arg[7:])
    elif lower_arg.startswith("s") and lower_arg[1:].isdigit():
        school_id = int(lower_arg[1:])

    if school_id is not None:
        from sqlalchemy import text as _sql
        with get_session() as session_db:
            row = session_db.execute(
                _sql("SELECT name FROM expression_communities WHERE community_id = :cid"),
                {"cid": school_id},
            ).fetchone()
            if not row:
                console.print(f"  [{WARNING}]School #{school_id} not found.[/]")
                return
            school_name = row[0]
            short_name = school_name if len(school_name) <= 35 else school_name[:32] + "..."
            session.focus = FocusedEntity(
                type="school",
                id=school_id,
                label=f"School #{school_id}: {short_name}",
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

    if arg.isdigit():
        idx = int(arg)
        if getattr(session, "last_related", None) and 1 <= idx <= len(session.last_related):
            thought = session.last_related[idx - 1]
            if thought.audio_file_id:
                session.focus = FocusedEntity(
                    type="file",
                    id=thought.audio_file_id,
                    label=thought.file_name,
                )
            else:
                session.focus = FocusedEntity(
                    type="transcript",
                    id=thought.transcript_id,
                    label=thought.file_name,
                )
            console.print(
                f"  [{SUCCESS}]Focus switched to thought #{idx}:[/] [{ACCENT}]{thought.file_name}[/] "
                f"[{DIM}](Transcript #{thought.transcript_id})[/]"
            )
            print_context_summary(session)
            return
        elif getattr(session, "last_bookmark_bridges", None) and idx in session.last_bookmark_bridges:
            bridge = session.last_bookmark_bridges[idx]
            _check_bridge_staleness(session, bridge, idx)
            _focus_bridge(session, bridge, idx)
            return

    file_id = int(arg)
    try:
        repo = TranscriptionRepository()
        audio_file = repo.get_audio_file(file_id)
        if not audio_file:
            # Fallback: check if the argument matches a transcript ID (e.g. from Recent transcriptions banner)
            tx = repo.get_by_id(file_id, session=session)
            if tx:
                session.set_context(tx["id"])
                af_id = tx.get("audio_file_id")
                if af_id:
                    audio_file = repo.get_audio_file(af_id)
                    file_id = af_id
                else:
                    session.focus = FocusedEntity(
                        type="file",
                        id=file_id,
                        label=tx.get("file_name", f"Transcript #{file_id}"),
                    )
                    print_context_summary(session)
                    return

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


@register_backslash("context")
def cmd_context(session: ReplSession, args: str) -> None:
    """Display active entity focus, navigation frame stack, and playback context."""
    from audiobench.cli.repl.dispatch import print_context_summary

    if session.focus:
        print_context_summary(session)
    else:
        console.print(f"  [{DIM}]No context set. Use \\focus <id|file> or transcribe a file.[/]")


@register_backslash("scope")
def cmd_scope(session: ReplSession, args: str) -> None:
    """Scope REPL search, synthesis, and playback to an intellectual school or work.

    Usage:
      \\scope school:<id>   Scope to School #<id> (e.g. \\scope school:12 or \\scope s12)
      \\scope work:<id>     Scope to Work #<id>   (e.g. \\scope work:5 or \\scope w5)
      \\scope clear         Clear current scope
      \\scope               Display active scope summary
    """
    from audiobench.cli.repl.dispatch import print_context_summary

    arg = args.strip()
    if not arg:
        if session.focus:
            print_context_summary(session)
        else:
            console.print(
                f"  [{DIM}]No active scope.[/]  "
                f"Use [{ACCENT}]\\scope school:<id>[/] or [{ACCENT}]\\scope work:<id>[/] to scope context."
            )
        return

    lower = arg.lower()
    if lower in ("clear", "reset", "none", "off"):
        cmd_unfocus(session, "")
        return

    if lower.startswith("school:") or lower.startswith("s"):
        raw_id = lower.removeprefix("school:").removeprefix("s")
        if raw_id.isdigit():
            cmd_focus(session, f"s{raw_id}")
            return

    if lower.startswith("work:") or lower.startswith("w"):
        raw_id = lower.removeprefix("work:").removeprefix("w")
        if raw_id.isdigit():
            cmd_focus(session, f"w{raw_id}")
            return

    console.print(
        f"  [{WARNING}]Unknown scope target: '{arg}'[/]\n"
        f"  [{DIM}]Usage: \\scope <school:<id>|s<id>|work:<id>|w<id>|clear>[/]"
    )



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
    cid = session.focus.id if (session.focus and session.focus.type == "school") else None

    # Synthesis router: hard-filter only for FTS5 (Option C — no soft boost).
    # Rationale: Step 0 fixed FTS5's actual failure (ILIKE → BM25). There is
    # no demonstrated FTS5 failure that soft-scope would address.
    from audiobench.cli.repl.synthesis_router import get_synthesis_scope
    router = get_synthesis_scope(session)
    fts_work_ids = router.work_ids if router.mode == "hard" else None

    results = repo.search(query, limit=limit, session=session, community_id=cid, work_ids=fts_work_ids)

    # EC-13: Zero-results fallback for hard mode (FTS5).
    # Same invariant as ANN path: unambiguous empty → expand, deferred for 1-3.
    if not results and router.mode == "hard":
        import logging as _logging
        _logging.getLogger("audiobench.synthesis_router").info(
            "EC-13 FTS5: hard-scope zero-results fallback for query=%r work_ids=%s",
            query, router.work_ids,
        )
        console.print(f"  [{DIM}]No results in synthesis scope — expanding to full library.[/]")
        results = repo.search(query, limit=limit, session=session, community_id=cid)

    if not results:
        scope_note = f" in {session.focus.label}" if cid else ""
        console.print(f"  [{DIM}]No results for \"{query}\"{scope_note}[/]")
        console.print(f"  [{DIM}]Try [{ACCENT}].search \"{query}\"[/] for semantic search.[/]")
        return

    # Display results
    router_badge = f"  [{ACCENT}]‹Scoped: {len(router.work_ids)} works›[/]" if router.mode == "hard" else ""
    scope_badge = (f"  [{ACCENT}]‹Scoped to {session.focus.label}›[/]" if cid else "") + router_badge
    console.print(f"\n  [{BOLD}]Text Search — SQLite FTS5[/]  [{DIM}]\"{query}\"[/]{scope_badge}")
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
    rec = repo.get_by_id(session.last_id, session=session)
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
    """Start an interactive chat with current and/or related transcripts."""
    from audiobench.cli.repl.dispatch import dispatch_command

    target_ids: list[int] = []
    if session.last_id:
        target_ids.append(session.last_id)

    raw = args.strip().lower()

    if raw in ("related", "all"):
        if not getattr(session, "last_related", None):
            console.print(f"  [{WARNING}]No related thoughts found yet. Run [{ACCENT}]\\related[/] first.[/]")
            return
        for thought in session.last_related:
            if thought.transcript_id not in target_ids:
                target_ids.append(thought.transcript_id)

        console.print(
            f"  [{SUCCESS}]✦ Seeding chat session with {len(target_ids)} connected transcripts as context...[/]"
        )
        dispatch_command(session, ["chat", *[str(tid) for tid in target_ids]])
        return

    # Check for index references like "1", "1 2", "1, 3"
    tokens = [t.strip(",") for t in raw.split()] if raw else []
    if (
        tokens
        and getattr(session, "last_related", None)
        and all(t.isdigit() and 1 <= int(t) <= len(session.last_related) for t in tokens)
    ):
        for t in tokens:
            idx = int(t)
            tid = session.last_related[idx - 1].transcript_id
            if tid not in target_ids:
                target_ids.append(tid)

        console.print(
            f"  [{SUCCESS}]✦ Seeding chat session with {len(target_ids)} connected transcripts as context...[/]"
        )
        dispatch_command(session, ["chat", *[str(tid) for tid in target_ids]])
        return

    if not target_ids and not raw:
        if session.focus and session.focus.type == "school":
            from audiobench.core.db_session import get_session
            from sqlalchemy import text as _sql
            with get_session() as sdb:
                rows = sdb.execute(
                    _sql("""
                        SELECT DISTINCT e.source_id 
                        FROM expressions e 
                        WHERE e.community_id = :cid AND e.source_type = 'transcription_chunk' AND e.source_id IS NOT NULL
                        LIMIT 15
                    """),
                    {"cid": session.focus.id},
                ).fetchall()
                school_tids = [r[0] for r in rows if r[0] is not None]
            if school_tids:
                console.print(
                    f"  [{SUCCESS}]✦ Seeding chat session with {len(school_tids)} transcripts from {session.focus.label}...[/]"
                )
                dispatch_command(session, ["chat", *[str(tid) for tid in school_tids]])
                return

        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    if raw:
        import shlex

        dispatch_command(session, ["chat", *shlex.split(args)])
    else:
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
            expr = expr_repo.get_by_id(expression_id, session=session)
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


# ── Help ─────────────────────────────────────────────────────

_COMMAND_DOCS: dict[str, dict[str, Any]] = {
    "synthesize": {
        "title": "Synthesize (Dialectic Graph MMR)",
        "aliases": ["synth"],
        "syntax": "\\synthesize <thematic query or philosophical question>",
        "summary": "Synthesize cross-library perspectives using Graph-Augmented MMR and local LLM.",
        "description": (
            "Executes Blended-Relevance Graph MMR (α=0.60, λ=0.55) to surface non-redundant, "
            "author-diverse perspectives from across your library. Traverses cross-work semantic "
            "bridges (w ≥ 0.78) and detects Triadic Cliques (+0.10 boost). Generates structured "
            "Points of Convergence vs Divergence with inline bracketed citations [1], [2], [3]."
        ),
        "examples": [
            "\\synthesize How does guilt lead to self-deception and breakdown?",
            "\\synth What is the relationship between suffering and spiritual growth?",
        ],
        "flywheel": "After synthesis, run \\play cite 1 to listen, or \\bookmark cite 1 to view the smart card.",
    },
    "bookmark": {
        "title": "Smart Contextual Bookmarking",
        "aliases": ["bm"],
        "syntax": "\\bookmark [id | cite <n> | list]",
        "summary": "Two-tiered smart contextual bookmark cards (Narrative Arc + Semantic Bridges).",
        "description": (
            "Generates rich two-tiered contextual bookmark cards:\n"
            "  • Tier 1 (The Narrative Arc): Local premise and resolution (|t1 - t2| ≥ 90s) within the same work.\n"
            "  • Tier 2 (Cross-Library Dialectic): High-confidence semantic bridges (w ≥ 0.78) to other thinkers.\n"
            "  • Consensus Triangle: Automatically highlights 3-cliques connecting mutual authors."
        ),
        "examples": [
            "\\bookmark cite 1          (Inspect smart card for synthesis citation [1])",
            "\\bookmark 45955           (Generate card directly for Expression #45955)",
            "\\bookmark list            (List all saved bookmarks)",
        ],
        "flywheel": "Run \\play cite 1 to listen, or \\synthesize <concept> to bridge ideas.",
    },
    "related": {
        "title": "Related Fragments & Bridges",
        "aliases": [],
        "syntax": "\\related",
        "summary": "Explore cross-work semantic bridges and in-core HNSW connections for the focused work.",
        "description": (
            "Surfaces conceptual relatives for the currently active transcript in ~28ms:\n"
            "  • Priority 1: Bidirectional Knowledge Graph traversal with strict work isolation (e2.work_id != curr_wid).\n"
            "  • Priority 2: In-Core HNSW island fallback for unlinked chapters.\n"
            "Stores discovered relations in session memory for instant follow-up actions."
        ),
        "examples": [
            "\\related                  (Explore relatives for current transcript)",
            "\\play 2                  (Listen to the second surfaced relative)",
        ],
        "flywheel": "Type \\play <id> to listen, or \\chat to open a discussion.",
    },
    "suggest": {
        "title": "Next Listen & Command Suggestions",
        "aliases": [],
        "syntax": "\\suggest [commands | history]",
        "summary": "Next Listen recommendations powered by knowledge graph & triadic consensus, or command shortcuts.",
        "description": (
            "When a work or chapter is focused, analyzes outbound cross-work semantic edges to recommend "
            "what to listen to next in your library. Detects Triadic Consensus (3-cliques) and displays "
            "the ▲ Consensus badge. If no work is focused or when explicitly invoked, displays frequent "
            "command habits from the command graph."
        ),
        "examples": [
            "\\suggest                  (Next Listen recommendations for current work)",
            "\\suggest commands         (Show frequent action shortcuts)",
        ],
        "flywheel": "Type \\play <id> to start listening to a recommended work.",
    },
    "graph": {
        "title": "Knowledge Graph & Seeding",
        "aliases": [],
        "syntax": "\\graph [seed [limit] | <days>]",
        "summary": "Seed unlinked islands in the knowledge graph or view command stats.",
        "description": (
            "\\graph seed [limit] scans expressions across your library and forms cross-work semantic edges "
            "in-memory at >1,300 chunks/second. Strict work isolation prevents intra-book contamination.\n"
            "\\graph [days] displays command usage frequency over the specified window."
        ),
        "examples": [
            "\\graph seed 2500         (Seed up to 2,500 unlinked chunks into the KG)",
            "\\graph 30                (Show command stats for last 30 days)",
        ],
        "flywheel": "After seeding, run \\related or \\suggest to discover newly formed connections.",
    },
    "play": {
        "title": "Audio Playback & Seeking",
        "aliases": [],
        "syntax": "\\play [file | transcript_id | cite <n> | pause | resume | stop]",
        "summary": "Play back audio files, transcripts, or synthesis citations.",
        "description": (
            "Controls local audio playback via the background daemon:\n"
            "  • \\play cite <n>: Plays exact segment timestamp for a synthesis citation [n].\n"
            "  • \\play <id>: Plays transcript #id from start or current position.\n"
            "  • \\play pause / resume / stop: Playback state management."
        ),
        "examples": [
            "\\play cite 1             (Play synthesis citation [1] at exact timestamp)",
            "\\play 592                (Play transcript #592)",
            "\\play pause              (Pause playback)",
        ],
        "flywheel": "While listening, type \\bookmark to save a smart card for this passage.",
    },
    "loop": {
        "title": "A-B Loop Control",
        "aliases": [],
        "syntax": "\\loop [<start> <end> | clear]",
        "summary": "Set or clear an A-B repeat loop on the current audio.",
        "description": (
            "Loops the audio between two timestamps indefinitely until cleared:\\n"
            "  • \\loop 1:30 2:45  — Loop from 1m30s to 2m45s\\n"
            "  • \\loop 90 165     — Same in raw seconds\\n"
            "  • \\loop            — Show current loop bounds\\n"
            "  • \\loop clear      — Cancel active loop"
        ),
        "examples": [
            "\\loop 1:30 2:45         (Loop a 75-second passage)",
            "\\loop clear             (Cancel loop)",
        ],
        "flywheel": "Combine with \\read to navigate chapters while a passage loops.",
    },
    "filter": {
        "title": "Live DSP Audio Filter",
        "aliases": [],
        "syntax": "\\filter [<ffmpeg_filter> | clear]",
        "summary": "Apply or remove a live audio filter (equalizer, bass boost, etc.).",
        "description": (
            "Applies an FFmpeg-style audio filter to the live playback stream:\\n"
            "  • \\filter bass=g=5              — Boost bass +5dB\\n"
            "  • \\filter treble=g=-3            — Cut treble -3dB\\n"
            "  • \\filter equalizer=f=1k:g=-6   — Notch filter at 1kHz\\n"
            "  • \\filter clear                  — Remove active filter"
        ),
        "examples": [
            "\\filter bass=g=5                (Boost bass by 5dB)",
            "\\filter clear                   (Remove filter)",
        ],
        "flywheel": "Useful when listening in noisy environments or on poor speakers.",
    },
    "chapters": {
        "title": "Chapter Navigation",
        "aliases": [],
        "syntax": "\\chapters [<n>]",
        "summary": "List chapters for the current file, or jump to chapter n.",
        "description": (
            "Shows all indexed chapters for the currently playing audio file.\\n"
            "  • \\chapters       — List all chapters with timestamps\\n"
            "  • \\chapters 3     — Jump to chapter 3 immediately\\n"
            "Chapters are auto-seeded from the file's metadata on first play."
        ),
        "examples": [
            "\\chapters               (Show chapter list)",
            "\\chapters 3             (Jump to chapter 3)",
        ],
        "flywheel": "Use with \\read to navigate the synchronized transcript by chapter.",
    },
    "read": {
        "title": "Chapter Reader TUI",
        "aliases": [],
        "syntax": "\\read [<transcript_id>]",
        "summary": "Open the interactive chapter reader for the currently playing file.",
        "description": (
            "Launches the full-screen chapter reader with synchronized transcript,\\n"
            "A-B loop, bookmarking, and visual selection. Local playback mode only\\n"
            "(daemon attach — Phase 3b — not yet implemented).\\n"
            "  • \\read              — Open reader for the current track\\n"
            "  • \\read 592          — Open reader for transcript #592 (file must be playing)"
        ),
        "examples": [
            "\\read                   (Open reader for current track)",
            "\\read 592               (Open reader for transcript #592)",
        ],
        "flywheel": "Press b to bookmark, B to mark a region, s to toggle scroll lock.",
    },
    "workflow": {
        "title": "Command Workflows",
        "aliases": [],
        "syntax": "\\workflow <save|list|run|delete|show> [name]",
        "summary": "Capture and replay sequences of REPL commands.",
        "description": (
            "Enables macros and automations by saving recent interactive commands as named workflows."
        ),
        "examples": [
            "\\workflow save study     (Save recent commands as 'study')",
            "\\workflow run study      (Replay the 'study' workflow)",
            "\\workflow list           (List all saved workflows)",
        ],
        "flywheel": "Combine with \\suggest to see suggested workflow steps.",
    },
}


@register_backslash("help")
def cmd_help(session: ReplSession, args: str) -> None:
    """Show available backslash commands or detailed help for a specific command."""
    target = args.strip().lower().lstrip("\\").lstrip("/")
    if target:
        if target in ("synth",):
            target = "synthesize"
        elif target in ("bm",):
            target = "bookmark"

        doc = _COMMAND_DOCS.get(target)
        if doc:
            console.print(f"\n  [{BOLD}][{ACCENT}]\\{target}[/] — {doc['title']}[/]\n")
            console.print(f"  [{DIM}]Syntax:[/]   [{ACCENT}]{doc['syntax']}[/]")
            if doc.get("aliases"):
                alias_str = ", ".join(f"[{ACCENT}]\\{a}[/]" for a in doc["aliases"])
                console.print(f"  [{DIM}]Aliases:[/]  {alias_str}")
            console.print(f"\n  [{BOLD}]Description:[/]\n  {doc['description']}\n")
            if doc.get("examples"):
                console.print(f"  [{BOLD}]Examples:[/]")
                for ex in doc["examples"]:
                    console.print(f"    [{ACCENT}]{ex}[/]")
                console.print("")
            if doc.get("flywheel"):
                console.print(f"  [{BOLD}]Flywheel Next Steps:[/] [{DIM}]{doc['flywheel']}[/]\n")
            return

        from audiobench.cli.repl.dispatch import _BACKSLASH_HANDLERS
        if target in _BACKSLASH_HANDLERS:
            fn = _BACKSLASH_HANDLERS[target]
            fn_doc = (fn.__doc__ or "").strip()
            console.print(f"\n  [{BOLD}][{ACCENT}]\\{target}[/][/]\n")
            console.print(f"  {fn_doc or 'No detailed description available.'}\n")
            return

        import difflib
        candidates = list(_COMMAND_DOCS.keys()) + list(_BACKSLASH_HANDLERS.keys())
        matches = difflib.get_close_matches(target, candidates, n=1, cutoff=0.5)
        console.print(f"  [{WARNING}]Unknown command: \\{target}[/]")
        if matches:
            console.print(f"  [{DIM}]Did you mean:[/] [{ACCENT}]\\help {matches[0]}[/] ?\n")
        else:
            console.print(f"  [{DIM}]Type [{ACCENT}]\\help[/] to view all available commands.[/]\n")
        return

    # Top-level categorized help menu
    console.print(f"\n  [{BOLD}]AudioBench Shell — Command Reference[/]\n")
    console.print(f"  [{DIM}]Three execution contexts, one prompt:[/]")
    console.print(f"    [{ACCENT}]\\[/][{DIM}]command[/]   — local & graph commands (instant)")
    console.print(f"    [{DIM}]?[/] question  — AI/semantic reasoning (requires focus)")
    console.print(f"    [yellow]![/][{DIM}]command[/]   — OS shell passthrough\n")

    console.print(f"  [{DIM}]{'─' * 52}[/]")
    console.print(f"  [{BOLD}]Knowledge Graph & Dialectic[/]")
    console.print(f"    [{ACCENT}]\\synthesize <q>[/]     Synthesize cross-library perspectives (Graph MMR)")
    console.print(f"    [{ACCENT}]\\related[/]            Show cross-work connections & semantic bridges")
    console.print(f"    [{ACCENT}]\\bookmark [id|cite][/] Two-tiered smart cards (Narrative Arc + Dialectic)")
    console.print(f"    [{ACCENT}]\\suggest [commands][/]  Next Listen recommendations & triadic consensus")
    console.print(f"    [{ACCENT}]\\graph seed [n][/]     Seed unlinked islands into the knowledge graph\n")

    console.print(f"  [{BOLD}]Playback & Library[/]")
    console.print(f"    [{ACCENT}]\\play [id|cite <n>][/] Audio playback & citation seeking")
    console.print(f"    [{ACCENT}]\\library[/]             Launch interactive Library Command Center\n")

    console.print(f"  [{BOLD}]Exploration & Focus[/]")
    console.print(f"    [{ACCENT}]\\focus <id>[/]         Focus on audio file, transcript, or work")
    console.print(f"    [{ACCENT}]\\unfocus[/]            Clear active focus")
    console.print(f"    [{ACCENT}]\\show [id][/]          Print transcript with timestamps")
    console.print(f"    [{ACCENT}]\\ls [n][/]             List recent files (default 10)")
    console.print(f"    [{ACCENT}]\\search <q>[/]         FTS5 full-text transcript search")
    console.print(f"    [{ACCENT}]\\info[/]               Metadata and stats for active focus\n")

    console.print(f"  [{BOLD}]AI & Interaction[/]")
    console.print(f"    [{ACCENT}]\\chat[/]               Conversational AI chat over active transcript")
    console.print(f"    [{ACCENT}]\\ask <question>[/]     Ask AI about active transcript (? <q> shorthand)")
    console.print(f"    [{ACCENT}]\\summarize[/]          Structured chapter summary\n")

    console.print(f"  [{BOLD}]Capture & Workflows[/]")
    console.print(f"    [{ACCENT}]\\capture <note>[/]     Quick-capture thought or note")
    console.print(f"    [{ACCENT}]\\workflow <save|run>[/] Manage and replay command workflows")
    console.print(f"    [{ACCENT}]\\history[/]            Past transcriptions and commands")
    console.print(f"    [{ACCENT}]\\stats[/]              Transcript word count and duration stats\n")

    console.print(f"  [{BOLD}]Security & Diarization[/]")
    console.print(f"    [{ACCENT}]\\name <spk> <name>[/]  Rename speaker globally across transcript")
    console.print(f"    [{ACCENT}]\\lock / \\unlock[/]     Manage privacy tiers")
    console.print(f"    [{ACCENT}]\\enroll / \\voices[/]   Voiceprint biometric enrollment & selection\n")

    console.print(f"  [{BOLD}]Navigation[/]")
    console.print(f"    [{ACCENT}]\\back[/] / [{ACCENT}]\\home[/]       Navigate context stack")
    console.print(f"    [{ACCENT}]\\stack[/]              Show current navigation stack\n")

    console.print(f"  [{DIM}]Tip: Type [{ACCENT}]\\help <command>[/] [{DIM}]for syntax, examples, and options.[/]")
    console.print(f"  [{DIM}]Shell passthrough:  !ls  !ffmpeg  !play ...[/]\n")

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

def _resolve_thought_segment(
    session_db: Any, expr_id: int, transcription_id: int | None, expr_content: str
) -> tuple[float, float, str | None]:
    """Find start_time, end_time, and speaker for an expression."""
    from audiobench.storage.models import ExpressionSegmentMap, SegmentRecord

    # 1. Check ExpressionSegmentMap
    mapping = session_db.query(ExpressionSegmentMap).filter_by(expression_id=expr_id).first()
    if mapping:
        seg = session_db.query(SegmentRecord).filter_by(id=mapping.segment_id).first()
        if seg:
            return float(seg.start_time), float(seg.end_time), seg.speaker

    # 2. Match against segments of the transcription
    segs = []
    if transcription_id:
        segs = (
            session_db.query(SegmentRecord)
            .filter_by(transcription_id=transcription_id)
            .order_by(SegmentRecord.segment_index)
            .all()
        )
    if not segs:
        try:
            from audiobench.storage.models import ExpressionRecord, ChapterRecord
            expr = session_db.query(ExpressionRecord).filter_by(id=expr_id).first()
            if expr:
                chap_id = expr.chapter_id or (expr.source_id if expr.source_type == "chapter_summary" else None)
                if chap_id:
                    chap = session_db.query(ChapterRecord).filter_by(id=chap_id).first()
                    if chap and chap.start_time is not None:
                        return float(chap.start_time), float(chap.end_time or chap.start_time), None
        except Exception:
            pass
        return 0.0, 0.0, None

    content_clean = expr_content.strip()
    from audiobench.core.constants import ENGLISH_STOP_WORDS
    words = [w for w in content_clean.split() if len(w) >= 2 and w.lower() not in ENGLISH_STOP_WORDS]
    if not words:
        words = content_clean.split()
    prefix_words = " ".join(words[:4]) if len(words) >= 4 else content_clean[:30]

    for s in segs:
        s_text = s.text.strip()
        if prefix_words and prefix_words in s_text:
            return float(s.start_time), float(s.end_time), s.speaker
        if s_text and s_text in content_clean:
            return float(s.start_time), float(s.end_time), s.speaker

    if len(words) >= 3:
        for i in range(min(5, len(words) - 2)):
            triplet = f"{words[i]} {words[i+1]} {words[i+2]}"
            for s in segs:
                if triplet in s.text:
                    return float(s.start_time), float(s.end_time), s.speaker

    return float(segs[0].start_time), float(segs[0].end_time), segs[0].speaker


def _relevance_dots(rank: int, total_dots: int = 5) -> str:
    """Convert a 1-based retrieval rank to a filled-dot relevance bar.

    rank 1-2  → ●●●●●  (strongest)
    rank 3-4  → ●●●●○
    rank 5-6  → ●●●○○
    rank 7-8  → ●●○○○
    rank 9+   → ●○○○○  (weakest shown)
    """
    filled = max(1, total_dots - (rank - 1) // 2)
    filled = min(filled, total_dots)
    return "[cyan]●[/cyan]" * filled + "[dim]○[/dim]" * (total_dots - filled)


def _get_db_connection():
    """Get a raw sqlite3 connection to the primary transcriptions database."""
    import sqlite3
    from pathlib import Path
    from audiobench.core.settings import get_settings

    settings = get_settings()
    db_path = Path(settings.database_url.replace("sqlite:///", "")).resolve()
    return sqlite3.connect(str(db_path))


def _show_school_related(session: ReplSession, cid: int, args: str) -> None:
    """Show macro connected schools and micro bridge expressions for a focused school."""
    import re
    import textwrap
    from rich.table import Table

    from audiobench.cli.display.theme import (
        ACCENT,
        BOLD,
        DIM,
        SUCCESS,
        WARNING,
        console,
    )
    from audiobench.cli.repl.session import RelatedThought
    from audiobench.community.community_service import (
        get_cross_school_bridge_pairs,
        get_related_schools,
    )
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT community_id, name, node_count FROM expression_communities WHERE community_id = ? AND is_active = 1",
            (cid,),
        )
        comm_row = cur.fetchone()
        if not comm_row:
            console.print(f"  [{WARNING}]School #{cid} not found or inactive. Run \\schools to list available schools.[/]")
            return

        school_name = comm_row[1] or f"School #{cid}"
        related_schools = get_related_schools(conn, cid, limit=5)
        bridge_pairs = get_cross_school_bridge_pairs(conn, cid, limit=4)

    term_width = console.size.width or 80
    border_w = min(term_width - 4, 110)
    use_book = term_width >= 100

    console.print(
        f"\n  [{BOLD} {ACCENT}]School Dialectic Relations[/] [{DIM}](School #{cid:02d}: {school_name})[/]"
    )
    console.print(f"  [{DIM}]{'─' * border_w}[/]\n")

    # 1. Macro Connected Schools
    console.print(f"  [{BOLD}]Connected Schools (Macro Coupling)[/]")
    if related_schools:
        for rs in related_schools:
            console.print(
                f"    [{ACCENT}]• School #{rs['community_id']:02d}:[/] [{BOLD}]{rs['name']}[/] "
                f"[{DIM}]({rs['edge_count']} bundle edges · coupling: {rs['coupling_weight']:.1f} · avg: {rs['avg_weight']:.2f})[/]"
            )
    else:
        console.print(f"    [{DIM}](No external school connections found above threshold)[/]")
    console.print()

    # 2. Micro Semantic Bridge Pairs (Two-Column Book Mode)
    console.print(
        f"  [{BOLD}]Semantic Bridge Passages[/] "
        f"[{DIM}]({len(bridge_pairs)} cross-school bridge pairs · {'2-column book mode' if use_book else 'single column'})[/]"
    )
    console.print(f"  [{DIM}]{'─' * border_w}[/]\n")

    if not bridge_pairs:
        console.print(f"    [{DIM}](No direct semantic bridge expressions found above 0.78 similarity)[/]\n")
        console.print(
            f"  [{DIM}]Quick actions:[/]  "
            f"[{ACCENT}]\\focus school <id>[/] [{DIM}]switch school focus[/]  •  "
            f"[{ACCENT}]\\play school <id>[/] [{DIM}]listen to school[/]  •  "
            f"[{ACCENT}]\\school {cid}[/] [{DIM}]school details[/]\n"
        )
        return

    session.last_related = []
    wrap_w = max(42, (term_width - 6) // 2) if use_book else min(term_width - 6, 95)

    with get_session() as sdb:
        pair_blocks = []
        for i, bp in enumerate(bridge_pairs, 1):
            inside = bp["inside"]
            outside = bp["outside"]

            # Inside formatting
            in_attr = f"{inside['author']} — {inside['title']}" if inside["author"] and inside["author"] != "Unknown Author" else inside["title"]
            in_suffix = f"(Expr #{inside['id']})"
            in_max_attr = max(15, wrap_w - len(in_suffix) - 2)
            if len(in_attr) > in_max_attr:
                in_attr = in_attr[:in_max_attr - 1] + "…"
            in_spaces = " " * max(1, wrap_w - len(in_attr) - len(in_suffix))
            in_hdr = f"[{BOLD}]{in_attr}[/] {in_spaces}[{DIM}]{in_suffix}[/]"
            in_wrapped = textwrap.fill(f'"{inside["formatted_content"]}"', width=wrap_w - 4)
            in_lines = [in_hdr] + [f"  {w}" for w in in_wrapped.split("\n")]

            # Outside formatting & RelatedThought registration
            ext_txid = outside.get("source_id")
            af_row = None
            if ext_txid:
                tx_rec = sdb.query(TranscriptionRecord).filter_by(id=ext_txid).first()
                if tx_rec and tx_rec.audio_file_id:
                    af_row = sdb.query(AudioFileRecord).filter_by(id=tx_rec.audio_file_id).first()

            file_name = af_row.file_name if af_row else (f"Transcript #{ext_txid}" if ext_txid else "Unknown")
            file_path = af_row.file_path if af_row else None

            start_time, end_time, seg_speaker = _resolve_thought_segment(
                sdb, outside["id"], ext_txid or 0, outside["content"]
            )
            pct = min(99, max(40, int(bp["weight"] * 100)))

            thought = RelatedThought(
                index=i,
                transcript_id=ext_txid or 0,
                audio_file_id=af_row.id if af_row else None,
                file_name=file_name,
                file_path=file_path,
                start_time=start_time,
                end_time=end_time,
                speaker=seg_speaker,
                content=outside["content"],
                score=bp["weight"],
                match_percent=pct,
                work_title=outside["title"],
                work_author=outside["author"],
            )
            session.last_related.append(thought)

            ext_attr = f"{outside['author']} — {outside['title']}" if outside["author"] and outside["author"] != "Unknown Author" else outside["title"]
            ext_suffix = f"\\[{i}\\] (sim: {bp['weight']:.2f})"
            ext_suffix_vis = f"[{i}] (sim: {bp['weight']:.2f})"
            ext_max_attr = max(15, wrap_w - len(ext_suffix_vis) - 2)
            if len(ext_attr) > ext_max_attr:
                ext_attr = ext_attr[:ext_max_attr - 1] + "…"
            ext_spaces = " " * max(1, wrap_w - len(ext_attr) - len(ext_suffix_vis))
            ext_hdr = f"[{BOLD}]{ext_attr}[/] {ext_spaces}[{DIM}]{ext_suffix}[/]"
            ext_comm_label = f"[{ACCENT}]• {outside['community_name']}[/]"
            ext_wrapped = textwrap.fill(f'"{outside["formatted_content"]}"', width=wrap_w - 4)
            ext_lines = [ext_hdr, f"  {ext_comm_label}"] + [f"  {w}" for w in ext_wrapped.split("\n")]

            pair_blocks.append((in_lines, ext_lines))

    if use_book:
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(width=wrap_w, no_wrap=False)
        tbl.add_column(width=1, style=DIM)
        tbl.add_column(width=wrap_w, no_wrap=False)

        col_hdr_left = f"[{BOLD} {ACCENT}]School #{cid:02d} (Inside Core)[/]"
        col_hdr_right = f"[{BOLD} {ACCENT}]Connected Thoughts (Bridge Out)[/]"
        tbl.add_row(col_hdr_left, "│", col_hdr_right)
        tbl.add_row(f"[dim]{'─' * wrap_w}[/dim]", "│", f"[dim]{'─' * wrap_w}[/dim]")

        for in_lines, ext_lines in pair_blocks:
            max_h = max(len(in_lines), len(ext_lines))
            for r_i in range(max_h):
                l = in_lines[r_i] if r_i < len(in_lines) else ""
                r = ext_lines[r_i] if r_i < len(ext_lines) else ""
                tbl.add_row(l, "│", r)
            tbl.add_row("", "│", "")
        console.print(tbl)
    else:
        for i, (in_lines, ext_lines) in enumerate(pair_blocks, 1):
            console.print(f"  [{BOLD} {ACCENT}]Bridge Pair #{i}[/]")
            for line in in_lines:
                console.print(f"  {line}")
            console.print(f"  [{DIM}]── bridge to ──────────────────────────────────────[/dim]")
            for line in ext_lines:
                console.print(f"  {line}")
            console.print()

    console.print(
        f"\n  [{DIM}]Quick actions:[/]  "
        f"[{ACCENT}]\\focus school <id>[/] [{DIM}]switch school focus[/]  •  "
        f"[{ACCENT}]\\play school <id>[/] [{DIM}]listen to school[/]  •  "
        f"[{ACCENT}]\\play <1-{len(session.last_related)}>[/] [{DIM}]listen to bridge passage[/]  •  "
        f"[{ACCENT}]\\focus <1-{len(session.last_related)}>[/] [{DIM}]focus bridge file[/]\n"
    )


@register_backslash("related")
def cmd_related(session: ReplSession, args: str) -> None:
    """Show semantically related fragments from other transcripts or schools in Book Mode."""
    import re
    import textwrap
    from rich.table import Table

    from audiobench.cli.display.theme import (
        ACCENT,
        BOLD,
        DIM,
        SUCCESS,
        WARNING,
        console,
    )
    from audiobench.cli.repl.session import RelatedThought
    from audiobench.core.db_session import get_session
    from audiobench.memory.boilerplate import clean_audio_disclaimers
    from audiobench.memory.graph_linker import is_linkable_chunk
    from audiobench.playback.controls import fmt_timestamp
    from audiobench.storage.models import AudioFileRecord
    from sqlalchemy import text as _sql

    clean_arg = args.strip().lower()

    # School relatedness handling
    if clean_arg.startswith("school"):
        parts = clean_arg.split()
        target_cid = None
        if len(parts) > 1:
            raw_cid = parts[1].lstrip("#")
            if raw_cid.isdigit():
                target_cid = int(raw_cid)
        if target_cid is None and session.focus and session.focus.type == "school":
            target_cid = session.focus.id
        if target_cid is not None:
            _show_school_related(session, target_cid, args)
            return
        else:
            console.print("  [dim]Usage: \\related school <id> (or focus on a school first)[/]")
            return

    if session.focus and session.focus.type == "school":
        if clean_arg in ("work", "works", "transcript", "transcripts"):
            pass
        else:
            _show_school_related(session, session.focus.id, args)
            return

    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return

    console.print(f"  [{DIM}]Traversing knowledge graph & in-core semantic index...[/]")

    with get_session() as sdb:
        tx_info = sdb.execute(
            _sql("""
                SELECT t.id, t.audio_file_id, af.work_id, w.title, w.author, af.file_name
                FROM transcriptions t
                LEFT JOIN audio_files af ON t.audio_file_id = af.id
                LEFT JOIN works w ON af.work_id = w.id
                WHERE t.id = :txid
            """),
            {"txid": session.last_id},
        ).fetchone()

        if not tx_info:
            console.print(f"  [{DIM}]Transcript #{session.last_id} not found in database.[/]")
            return

        curr_wid = tx_info[2]
        if curr_wid is None:
            ew_row = sdb.execute(
                _sql("SELECT work_id FROM expressions WHERE source_type = 'transcription_chunk' AND source_id = :txid AND work_id IS NOT NULL LIMIT 1"),
                {"txid": session.last_id},
            ).fetchone()
            if ew_row:
                curr_wid = ew_row[0]

        # Find sweep_chunks belonging to this transcript
        chunk_rows = sdb.execute(
            _sql("""
                SELECT id FROM expressions
                WHERE source_type = 'transcription_chunk'
                  AND source_id = :txid
                  AND graph_role = 'sweep_chunk'
            """),
            {"txid": session.last_id},
        ).fetchall()
        chunk_ids = [r[0] for r in chunk_rows]
        if not chunk_ids:
            chunk_rows = sdb.execute(
                _sql("SELECT id FROM expressions WHERE source_type = 'transcription_chunk' AND source_id = :txid"),
                {"txid": session.last_id},
            ).fetchall()
            chunk_ids = [r[0] for r in chunk_rows]

        # Priority 1: Bidirectional Knowledge Graph Lookup
        # (cross-work semantic edges with strict work isolation)
        kg_related: list[tuple[int, int | None, int | None, str | None, str | None, float, str, str]] = []
        seen_works: set[Any] = set()

        if chunk_ids:
            placeholders = ",".join(f":c{i}" for i in range(len(chunk_ids)))
            params = {f"c{i}": cid for i, cid in enumerate(chunk_ids)}
            params["curr_wid"] = curr_wid
            params["txid"] = session.last_id

            sem_edges = sdb.execute(
                _sql(f"""
                    SELECT 
                        e2.id as target_eid,
                        e2.source_id as target_txid,
                        e2.work_id as target_wid,
                        w2.title as target_title,
                        w2.author as target_author,
                        r.weight as weight,
                        e2.content as content
                    FROM expression_relations r
                    JOIN expressions e2 ON e2.id = (
                        CASE WHEN r.from_expression_id IN ({placeholders}) THEN r.to_expression_id
                             ELSE r.from_expression_id END
                    )
                    LEFT JOIN works w2 ON e2.work_id = w2.id
                    WHERE (r.from_expression_id IN ({placeholders}) OR r.to_expression_id IN ({placeholders}))
                      AND r.relation_type = 'semantic'
                      AND (
                          CASE 
                              WHEN :curr_wid IS NOT NULL THEN (e2.work_id IS NOT NULL AND e2.work_id != :curr_wid)
                              ELSE (e2.work_id IS NOT NULL OR NOT (e2.source_type = 'transcription_chunk' AND e2.source_id = :txid))
                          END
                      )
                    ORDER BY r.weight DESC
                    LIMIT 50
                """),
                params,
            ).fetchall()

            for r in sem_edges:
                if is_linkable_chunk(r[6]):
                    wid_key = r[2] if r[2] is not None else f"tx_{r[1]}"
                    if wid_key not in seen_works:
                        seen_works.add(wid_key)
                        kg_related.append((r[0], r[1], r[2], r[3], r[4], float(r[5]), r[6], "KG"))
                        if len(kg_related) >= 30:
                            break

        # Priority 2: In-Core HNSW Island Fallback
        # If fewer than 9 connections found in the KG, query HNSW directly with work exclusion
        if len(kg_related) < 9 and chunk_ids:
            try:
                from audiobench.memory.graph_linker import ensure_hnsw_ready

                hnsw = ensure_hnsw_ready()
                step = max(1, len(chunk_ids) // 6)
                sampled_cids = chunk_ids[::step][:8]
                cand_hits: dict[int, float] = {}

                for scid in sampled_cids:
                    if hnsw.has_item(scid):
                        vec = hnsw.get_items([scid])[0]
                        hits = hnsw.query(vec, k=30)
                        for hid, dist in hits:
                            if hid not in chunk_ids:
                                sim = 1.0 - float(dist)
                                if sim >= 0.70:
                                    if hid not in cand_hits or sim > cand_hits[hid]:
                                        cand_hits[hid] = sim

                if cand_hits:
                    c_ids = list(cand_hits.keys())
                    c_placeholders = ",".join(f":hc{i}" for i in range(len(c_ids)))
                    c_params = {f"hc{i}": hid for i, hid in enumerate(c_ids)}
                    c_params["curr_wid"] = curr_wid
                    c_params["txid"] = session.last_id
                    h_rows = sdb.execute(
                        _sql(f"""
                            SELECT e.id, e.source_id, e.work_id, w.title, w.author, e.content
                            FROM expressions e
                            LEFT JOIN works w ON e.work_id = w.id
                            WHERE e.id IN ({c_placeholders})
                              AND (
                                  CASE 
                                      WHEN :curr_wid IS NOT NULL THEN (e.work_id IS NOT NULL AND e.work_id != :curr_wid)
                                      ELSE (e.work_id IS NOT NULL OR NOT (e.source_type = 'transcription_chunk' AND e.source_id = :txid))
                                  END
                              )
                        """),
                        c_params,
                    ).fetchall()

                    sorted_h = sorted(h_rows, key=lambda row: cand_hits.get(row[0], 0.0), reverse=True)
                    for r in sorted_h:
                        if is_linkable_chunk(r[5]):
                            wid_key = r[2] if r[2] is not None else f"tx_{r[1]}"
                            if wid_key not in seen_works:
                                seen_works.add(wid_key)
                                kg_related.append((r[0], r[1], r[2], r[3], r[4], cand_hits[r[0]], r[5], "ANN"))
                                if len(kg_related) >= 30:
                                    break
            except Exception as e:
                logger.debug("cmd_related: HNSW fallback failed: %s", e)

    if not kg_related:
        console.print(
            f"  [{DIM}]No cross-work dialectic connections found yet. Run [{ACCENT}]\\graph seed[/] to link this work.[/]"
        )
        return

    # Benchmark-proven hybrid score threshold: max(score_max * 0.85, 0.72)
    max_score = kg_related[0][5]
    cutoff = max(max_score * 0.85, 0.72)
    filtered_related = [r for r in kg_related if r[5] >= cutoff][:9]

    if not filtered_related:
        console.print(
            f"  [{DIM}]No cross-work dialectic connections found yet. Run [{ACCENT}]\\graph seed[/] to link this work.[/]"
        )
        return

    term_width = console.size.width or 80
    use_book = term_width >= 100 and len(filtered_related) >= 2
    wrap_w = max(42, (term_width - 6) // 2) if use_book else min(term_width - 6, 95)

    session.last_related = []
    thought_blocks: list[list[str]] = []

    with get_session() as sdb:
        for i, (expr_id, tx_id, wid, w_title, w_author, sim, content, source_tag) in enumerate(filtered_related, 1):
            af_row = None
            if tx_id:
                from audiobench.storage.models import TranscriptionRecord as _TX

                tx_record = sdb.query(_TX).filter_by(id=tx_id).first()
                if tx_record and tx_record.audio_file_id:
                    af_row = sdb.query(AudioFileRecord).filter_by(id=tx_record.audio_file_id).first()
            file_name = af_row.file_name if af_row else (f"Transcript #{tx_id}" if tx_id else "Unknown")
            file_path = af_row.file_path if af_row else None
            work_title = w_title or (af_row.work.title if af_row and af_row.work else None)
            work_author = w_author or (af_row.work.author if af_row and af_row.work else None)

            start_time, end_time, seg_speaker = _resolve_thought_segment(
                sdb, expr_id, tx_id or 0, content
            )
            pct = min(99, max(40, int(sim * 100)))

            thought = RelatedThought(
                index=i,
                transcript_id=tx_id,
                audio_file_id=af_row.id if af_row else None,
                file_name=file_name,
                file_path=file_path,
                start_time=start_time,
                end_time=end_time,
                speaker=seg_speaker,
                content=content,
                score=sim,
                match_percent=pct,
                work_title=work_title,
                work_author=work_author,
            )
            session.last_related.append(thought)

            # Build visual card block for Book Mode
            if work_author and work_title:
                source_label = f"{work_author} — {work_title}"
            elif work_title:
                source_label = f"{work_title}"
            else:
                source_label = file_name

            if len(source_label) > wrap_w - 8:
                source_label = source_label[:wrap_w - 11] + "..."
            rule_len = max(2, wrap_w - len(source_label) - 4)
            hdr = f"[dim]──[/dim] [bold]{source_label}[/bold] [dim]{'─' * rule_len}[/dim]"

            dots = _relevance_dots(i)
            tag_chip = f"[{BOLD} {SUCCESS}][{source_tag}][/]" if source_tag == "KG" else f"[{BOLD} {ACCENT}][{source_tag}][/]"
            meta_parts = [f"[{BOLD} {ACCENT}]\\[{i}][/]", dots, tag_chip]
            if start_time:
                meta_parts.append(f"[{DIM}]@{fmt_timestamp(start_time)}[/]")
            if seg_speaker:
                meta_parts.append(f"[{DIM}]• {seg_speaker}[/]")
            meta_line = "  ".join(meta_parts)

            clean_text = clean_audio_disclaimers(content).strip().replace("\r", "").replace("\n", " ")
            clean_text = re.sub(r"\s+", " ", clean_text)
            if len(clean_text) > 280:
                clean_text = clean_text[:277] + "..."
            wrapped = textwrap.fill(f'"{clean_text}"', width=wrap_w - 4)

            block = [hdr, meta_line]
            for w in wrapped.split("\n"):
                block.append(f"  {w}")
            thought_blocks.append(block)

    border_w = min(term_width - 4, 110)
    mode_label = "2-column book mode" if use_book else "single column"

    console.print(
        f"\n  [{BOLD} {ACCENT}]Related Dialectics[/] [{DIM}]({len(session.last_related)} cross-library connections · {mode_label})[/]"
    )
    console.print(f"  [{DIM}]{'─' * border_w}[/]\n")

    if use_book:
        left_blocks: list[list[str]] = []
        right_blocks: list[list[str]] = []
        left_h = 0
        right_h = 0
        for b in thought_blocks:
            h = len(b) + 1
            if left_h <= right_h:
                left_blocks.append(b)
                left_h += h
            else:
                right_blocks.append(b)
                right_h += h

        left_lines: list[str] = []
        for b in left_blocks:
            left_lines.extend(b)
            left_lines.append("")

        right_lines: list[str] = []
        for b in right_blocks:
            right_lines.extend(b)
            right_lines.append("")

        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(width=wrap_w, no_wrap=False)
        tbl.add_column(width=1, style=DIM)
        tbl.add_column(width=wrap_w, no_wrap=False)

        max_lines = max(len(left_lines), len(right_lines))
        for r_i in range(max_lines):
            l = left_lines[r_i] if r_i < len(left_lines) else ""
            r = right_lines[r_i] if r_i < len(right_lines) else ""
            tbl.add_row(l, "│", r)
        console.print(tbl)
    else:
        for b in thought_blocks:
            for line in b:
                console.print(f"  {line}")
            console.print()

    console.print(
        f"\n  [{DIM}]Quick actions:[/]  "
        f"[{ACCENT}]\\play <1-{len(session.last_related)}>[/] [{DIM}]listen at timestamp[/]  •  "
        f"[{ACCENT}]\\chat related[/] [{DIM}]AI discussion[/]  •  "
        f"[{ACCENT}]\\focus <1-{len(session.last_related)}>[/] [{DIM}]switch file[/]\n"
    )

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
    rec = repo.get_by_id(session.last_id, session=session)
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
    """Show command frequency stats or seed unlinked islands in knowledge graph."""
    parts = args.strip().split()
    if parts and parts[0].lower() == "seed":
        limit = 2500
        if len(parts) > 1 and parts[1].isdigit():
            limit = int(parts[1])
        console.print(f"  [{BOLD}]Seeding unlinked islands into knowledge graph (limit={limit})...[/]")
        try:
            from audiobench.memory.graph_linker import seed_unlinked_islands
            result = seed_unlinked_islands(limit=limit)
            console.print(
                f"  [{SUCCESS}]✓ Seeding complete:[/] "
                f"Processed [{BOLD}]{result.chunks_processed}[/] chunks, "
                f"created [{BOLD}]{result.edges_created}[/] cross-work semantic edges "
                f"across [{BOLD}]{len(result.linked_work_ids)}[/] works "
                f"in [{DIM}]{result.elapsed_ms:.1f}ms[/]."
            )
        except Exception as e:
            console.print(f"  [{WARNING}]Seeding failed: {e}[/]")
        return

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
def _render_smart_bookmark_card(
    session: ReplSession,
    card: Any,
    cite_num: int | None = None,
) -> None:
    """Render the Hermeneutic Codex Two-Tiered Smart Bookmark Card in monastic aesthetic."""
    import textwrap
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from audiobench.cli.display.theme import ACCENT, BOLD, BOX_STYLE, DIM, SUCCESS

    # Register bridges for immediate interactive playback and focus navigation
    bridges = getattr(card, "dialectic_bridges", []) or []
    session.last_bookmark_bridges = {
        i: b for i, b in enumerate(bridges[:3], 1)
    }
    # Stamp which expression and citation this bridge set was built from, so
    # cmd_play / cmd_focus can detect staleness via citation-EID comparison.
    session.last_bookmark_bridges_for_eid = getattr(card, "expression_id", None)
    session.last_bookmark_bridges_for_cite = cite_num

    term_width = (console.size.width if hasattr(console, "size") and console.size else None) or console.width or 80
    avail_w = max(40, term_width - 4)

    lines: list[str] = []
    # Title & Metadata
    title_text = f"Smart Bookmark: {card.work_title}"
    lines.append(f"[{BOLD} {ACCENT}]{title_text}[/]")

    school_str = ""
    if getattr(card, "community_id", None):
        c_name = getattr(card, "community_name", "") or ""
        short_cname = f" {c_name}" if c_name else ""
        school_str = f"  ·  School: #{card.community_id}{short_cname}"
    lines.append(
        f"[{DIM}]Author: {card.author}  ·  Timestamp: {card.timestamp_display}  ·  Expr #{card.expression_id}{school_str}[/]"
    )
    lines.append("")

    # Focused Excerpt
    lines.append(f"[{BOLD}]Focused Excerpt[/]")
    group_items = ["\n".join(lines)]

    quote_clean = card.quote.strip() if getattr(card, "quote", None) else ""
    if quote_clean:
        from rich.markdown import Markdown as RichMarkdown
        from rich.padding import Padding
        group_items.append(Padding(RichMarkdown(quote_clean), (0, 2)))
    else:
        group_items.append(f"  [{DIM}](No excerpt content available)[/]")

    if getattr(card, "consensus_triangle", None):
        tri = card.consensus_triangle
        group_items.append(
            f"\n  [{ACCENT}]Triadic Consensus:[/] [{BOLD}]{tri[0]}[/], [{BOLD}]{tri[1]}[/], [{BOLD}]{tri[2]}[/] "
            f"[{DIM}](Closed cross-author dialectic)[/]"
        )

    group_items.append(Rule(style=DIM))

    use_two_col = term_width >= 100
    if use_two_col:
        col_w = max(38, (avail_w - 5) // 2)
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(width=col_w, no_wrap=False)
        tbl.add_column(width=1, style=DIM)
        tbl.add_column(width=col_w, no_wrap=False)

        left_lines: list[str] = []
        left_lines.append(f"[{BOLD} {ACCENT}]Tier 1: Local Narrative Arc[/]")
        left_lines.append(f"[{DIM}]Same-work thematic context and callbacks[/]")
        left_lines.append("")
        left_lines.append(f"[{BOLD}]◄ Context (Ignition)[/]")
        if getattr(card, "preceding_context", None):
            ctx = card.preceding_context[:220] + ("..." if len(card.preceding_context) > 220 else "")
            w_ctx = textwrap.fill(f'"{ctx}"', width=col_w - 2)
            for w in w_ctx.split("\n"):
                left_lines.append(f"  [{DIM}]{w}[/]")
        else:
            left_lines.append(f"  [{DIM}](Opening or standalone passage)[/]")
        left_lines.append("")
        left_lines.append(f"[{BOLD}]► Resolution (Follow-up)[/]")
        if getattr(card, "following_resolution", None):
            res = card.following_resolution[:220] + ("..." if len(card.following_resolution) > 220 else "")
            w_res = textwrap.fill(f'"{res}"', width=col_w - 2)
            for w in w_res.split("\n"):
                left_lines.append(f"  [{DIM}]{w}[/]")
        else:
            left_lines.append(f"  [{DIM}](Concluding or open-ended chapter segment)[/]")

        right_lines: list[str] = []
        right_lines.append(f"[{BOLD} {ACCENT}]Tier 2: Semantic Bridges[/]")
        right_lines.append(f"[{DIM}]Cross-library conceptual connections[/]")
        right_lines.append("")
        if bridges:
            for idx, b in enumerate(bridges[:3], 1):
                clique_flag = f" [{ACCENT}][Consensus Partner][/]" if b.is_clique_partner else ""
                b_attr = f"{b.author} — {b.work_title}"
                b_meta = f"Cosine: {b.cosine_similarity:.3f}"
                max_attr = max(15, col_w - len(b_meta) - 10)
                if len(b_attr) > max_attr:
                    b_attr = b_attr[:max_attr - 1] + "…"
                spaces = " " * max(1, col_w - len(b_attr) - len(b_meta) - 8)
                b_hdr = f"[{ACCENT}][B{idx}][/] [{BOLD}]{b_attr}[/]{spaces}[{DIM}]{b_meta}[/]{clique_flag}"
                right_lines.append(b_hdr)
                snip = b.snippet[:200] + ("..." if len(b.snippet) > 200 else "")
                w_snip = textwrap.fill(f'"{snip}"', width=col_w - 2)
                for w in w_snip.split("\n"):
                    right_lines.append(f"  [{DIM}]{w}[/]")
                right_lines.append("")
        else:
            right_lines.append(f"  [{DIM}](No cross-library semantic bridges detected)[/]")

        max_rows = max(len(left_lines), len(right_lines))
        for r_i in range(max_rows):
            l = left_lines[r_i] if r_i < len(left_lines) else ""
            r = right_lines[r_i] if r_i < len(right_lines) else ""
            tbl.add_row(l, "│", r)
        group_items.append(tbl)
    else:
        tbl = Table.grid(padding=(0, 0))
        tbl.add_column(width=avail_w - 2, no_wrap=False)
        tbl.add_row(f"[{BOLD} {ACCENT}]Tier 1: Local Narrative Arc[/] [{DIM}](Same-work context)[/]")
        tbl.add_row(f"[{BOLD}]◄ Context (Ignition)[/]")
        if getattr(card, "preceding_context", None):
            ctx = card.preceding_context[:220] + ("..." if len(card.preceding_context) > 220 else "")
            w_ctx = textwrap.fill(f'"{ctx}"', width=avail_w - 4)
            for w in w_ctx.split("\n"):
                tbl.add_row(f"  [{DIM}]{w}[/]")
        else:
            tbl.add_row(f"  [{DIM}](Opening or standalone passage)[/]")
        tbl.add_row("")
        tbl.add_row(f"[{BOLD}]► Resolution (Follow-up)[/]")
        if getattr(card, "following_resolution", None):
            res = card.following_resolution[:220] + ("..." if len(card.following_resolution) > 220 else "")
            w_res = textwrap.fill(f'"{res}"', width=avail_w - 4)
            for w in w_res.split("\n"):
                tbl.add_row(f"  [{DIM}]{w}[/]")
        else:
            tbl.add_row(f"  [{DIM}](Concluding or open-ended chapter segment)[/]")
        tbl.add_row("")
        tbl.add_row(Rule(style=DIM))
        tbl.add_row(f"[{BOLD} {ACCENT}]Tier 2: Semantic Bridges[/] [{DIM}](Cross-library conceptual connections)[/]")
        if bridges:
            for idx, b in enumerate(bridges[:3], 1):
                clique_flag = f" [{ACCENT}][Consensus Partner][/]" if b.is_clique_partner else ""
                b_attr = f"{b.author} — {b.work_title}"
                b_hdr = f"[{ACCENT}][B{idx}][/] [{BOLD}]{b_attr}[/] [{DIM}](Cosine: {b.cosine_similarity:.3f})[/]{clique_flag}"
                tbl.add_row(b_hdr)
                snip = b.snippet[:200] + ("..." if len(b.snippet) > 200 else "")
                w_snip = textwrap.fill(f'"{snip}"', width=avail_w - 4)
                for w in w_snip.split("\n"):
                    tbl.add_row(f"  [{DIM}]{w}[/]")
                tbl.add_row("")
        else:
            tbl.add_row(f"  [{DIM}](No cross-library semantic bridges detected)[/]")
        group_items.append(tbl)

    # Docked Quick Actions Toolbar
    actions: list[str] = []
    if cite_num is not None:
        actions.append(f"[{ACCENT}]\\play cite {cite_num}[/] [{DIM}]listen to citation[/]")
    else:
        actions.append(f"[{ACCENT}]\\play e{card.expression_id}[/] [{DIM}]listen to quote[/]")

    if bridges:
        b1 = bridges[0]
        actions.append(f"[{ACCENT}]\\play b1[/] [{DIM}]jump to {b1.author}[/]")
        actions.append(f"[{ACCENT}]\\focus b1[/] [{DIM}]focus {b1.author}[/]")

    actions.append(f"[{ACCENT}]\\synthesize \"{card.work_title}\"[/] [{DIM}]synthesize perspectives[/]")

    quick_toolbar = f"[{DIM}]Quick actions:[/]  " + "  •  ".join(actions)
    group_items.append(Rule(style=DIM))
    group_items.append(quick_toolbar)

    panel = Panel(Group(*group_items), border_style=DIM, box=BOX_STYLE, expand=False)
    console.print("")
    console.print(panel)
    console.print("")


@register_backslash("bookmark")
@register_backslash("bm")
def cmd_bookmark(session: ReplSession, args: str) -> None:
    """Inspect or create two-tiered smart contextual bookmark cards.

    Usage:
      \\bookmark <expression_id>    Show full Two-Tiered Smart Bookmark Card.
      \\bookmark cite <number>      Open card directly from a synthesis citation [1], [2]...
      \\bookmark list               List recent bookmarks.
    """
    parts = args.strip().split()
    if not parts:
        console.print("  [{DIM}]Usage: \\bookmark <expression_id> | \\bookmark cite <n> | \\bookmark list[/]")
        return

    sub = parts[0].lower()
    from audiobench.memory.bookmark_service import generate_smart_bookmark_card

    cite_num: int | None = None
    if sub == "cite":
        if len(parts) < 2 or not parts[1].isdigit():
            console.print("  [{WARNING}]Please specify citation number: \\bookmark cite <1|2|3...>[/]")
            return
        cite_num = int(parts[1])
        citations = getattr(session, "last_synthesis_citations", {})
        if cite_num not in citations:
            console.print(f"  [{WARNING}]Citation [{cite_num}] not found in last synthesis session.[/]")
            return
        expr_id = citations[cite_num]
    elif sub == "list":
        from audiobench.storage.bookmark_repository import BookmarkRepository
        repo = BookmarkRepository()
        bms = repo.list(limit=10)
        if not bms:
            console.print("  [{DIM}]No bookmarks saved yet.[/]")
            return
        table = make_table(
            "Recent Bookmarks",
            [
                ("ID", {"width": 6, "justify": "right"}),
                ("Track / File", {"style": ACCENT}),
                ("Time", {"width": 10}),
                ("Type", {"width": 10}),
                ("Notes", {}),
            ],
        )
        for b in bms:
            table.add_row(
                str(b["id"]),
                b.get("audio_file_name", "—"),
                b.get("timestamp_display", "—"),
                b.get("bookmark_type", "bookmark"),
                (b.get("notes") or "")[:40],
            )
        console.print(table)
        return
    elif sub.isdigit():
        expr_id = int(sub)
    else:
        console.print(f"  [{WARNING}]Unknown argument: '{args}'. Run \\bookmark <id> or \\bookmark cite <n>[/]")
        return

    try:
        card = generate_smart_bookmark_card(expr_id)
        _render_smart_bookmark_card(session, card, cite_num=cite_num)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not generate bookmark card for #{expr_id}: {e}[/]")



def render_synthesis_footer(resp: "SynthesisResponse") -> None:  # type: ignore[name-defined]
    """Render the synthesis diagnostics footer below the main synthesis output.

    Surfaces three signals that help the user understand retrieval behaviour:
      1. Total sources and elapsed time + model used
      2. Per-author cap signal — which authors were capped, so the user knows
         why the result has fewer than top_k fragments for a single-author query
      3. LLM fallback signal — when Ollama failed and Gemini was the actual model

    This is a named helper rather than inline code so any future display path
    can call ``render_synthesis_footer(resp)`` and get the same contract.
    """
    from audiobench.memory.synthesis_retriever import RetrievalMeta  # local import avoids circularity

    meta: RetrievalMeta | None = resp.retrieval_meta

    n_frags = len(resp.fragments)
    elapsed = resp.elapsed_seconds
    model = resp.model_used

    # --- Line 1: sources + timing + model ------------------------------------
    top_k_val = getattr(resp, "top_k", None)
    top_k_str = f" (top_k={top_k_val})" if top_k_val is not None else ""
    console.print(f"  [{DIM}]{n_frags} source{'s' if n_frags != 1 else ''}{top_k_str} · {elapsed:.2f}s · {model}[/]")

    # --- Line 2: per-author cap signal (only when cap actually fired) --------
    if meta and meta.capped_authors:
        cap_lines = []
        for a in sorted(meta.capped_authors):
            pool_n = meta.author_counts_pool.get(a, "?")
            cap_lines.append(f"{a} ({pool_n} in pool, capped at 6)")
        console.print(
            f"  [{ACCENT}]ⓘ Per-author cap fired:[/] [{DIM}]{' · '.join(cap_lines)}[/]"
        )

    # --- Line 3: LLM fallback signal (only when Ollama failed) ---------------
    if resp.fallback_from == "ollama" and resp.fallback_reason:
        reason_short = str(resp.fallback_reason)[:80]
        console.print(
            f"  [{WARNING}]ⓘ Ollama unavailable ({reason_short}) — used {model} as fallback.[/]"
        )
    elif resp.model_used == "digest_fallback":
        fb_reason = str(resp.fallback_reason or "")[:80]
        console.print(
            f"  [{WARNING}]ⓘ Both LLM providers failed ({fb_reason}) — showing grounded digest.[/]"
        )


@register_backslash("synthesize")
@register_backslash("synth")
def cmd_synthesize(session: ReplSession, args: str) -> None:
    """Synthesize perspectives across the library using Graph-Augmented MMR and LLM.

    Usage:
      \\synthesize <theme or query>
    """
    query = args.strip().strip('"').strip("'")
    if not query:
        console.print("  [{DIM}]Usage: \\synthesize <thematic question or query>[/]")
        return

    scoped_cid: int | None = None
    scoped_badge: str = ""
    if session.focus and session.focus.type == "school":
        scoped_cid = session.focus.id
        scoped_badge = f"  [{ACCENT}]‹Scoped to {session.focus.label}›[/]"

    console.print(f"  [{BOLD}]Synthesizing perspectives for:[/] \"{query}\"{scoped_badge}...")
    try:
        from audiobench.chat.synthesis_engine import generate_synthesis
        from rich.markdown import Markdown

        resp = generate_synthesis(query, community_id=scoped_cid)
        # Populate interactive flywheel citation map.
        # Unconditional overwrite — newest synthesis always wins (EC-11).
        session.last_synthesis_citations = resp.citations
        session._synthesis_command_snapshot = max(1, session._command_count)

        console.print("")
        console.print(Markdown(resp.synthesis_text))
        console.print("")

        if resp.fragments:
            playable_indices = [idx for idx, f in enumerate(resp.fragments, 1) if getattr(f, "is_playable", True)]
            text_indices = [idx for idx, f in enumerate(resp.fragments, 1) if not getattr(f, "is_playable", True)]

            for idx, f in enumerate(resp.fragments, 1):
                clique_flag = " [Consensus Partner]" if f.is_clique_partner else ""
                is_playable = getattr(f, "is_playable", True)
                author_style = BOLD if is_playable else ACCENT
                console.print(
                    f"    [{ACCENT}][{idx}][/] [{author_style}]{f.author}[/] — [{DIM}]{f.work_title}[/] "
                    f"(EffRel: {f.effective_rel:.3f}){clique_flag}"
                )

            # Footer: sources count, timing, model, cap signal, fallback signal
            render_synthesis_footer(resp)

            # Build clean, differentiated Flywheel prompt
            if playable_indices and text_indices:
                p_cites = " or ".join(str(i) for i in playable_indices[:2])
                t_cite = text_indices[0]
                t_author = resp.fragments[t_cite - 1].author
                flywheel_msg = (
                    f"Type [{ACCENT}]\\bookmark cite {t_cite}[/] for {t_author} to inspect the synthesis card, "
                    f"or [{ACCENT}]\\play cite {p_cites}[/] to listen."
                )
            elif playable_indices:
                p_first = playable_indices[0]
                p_author = resp.fragments[p_first - 1].author
                flywheel_msg = (
                    f"Type [{ACCENT}]\\bookmark cite {p_first}[/] for {p_author} to inspect the full passage card, "
                    f"or [{ACCENT}]\\play cite {p_first}[/] to listen."
                )
            elif text_indices:
                t_first = text_indices[0]
                t_author = resp.fragments[t_first - 1].author
                flywheel_msg = (
                    f"Type [{ACCENT}]\\bookmark cite {t_first}[/] for {t_author} to inspect the synthesis card."
                )
            else:
                flywheel_msg = ""

            if flywheel_msg:
                console.print(f"\n  [{BOLD}]Flywheel:[/] {flywheel_msg}")
    except Exception as e:
        console.print(f"  [{WARNING}]Synthesis failed: {e}[/]")


@register_backslash("suggest")
def cmd_suggest(session: ReplSession, args: str) -> None:
    """Suggest next listens based on knowledge graph connections, or next commands."""
    mode_cmds = args.strip().lower() in ("cmd", "cmds", "command", "commands", "history")

    # If the user specifically asks for command history or if no work is focused,
    # show command history. Otherwise, generate Next Listen recommendations.
    if not mode_cmds:
        try:
            from sqlalchemy import text as _sql
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import WorkRecord

            curr_wid: int | None = None
            with get_session() as sdb:
                if session.focus and session.focus.type == "work":
                    curr_wid = session.focus.id
                elif session.focus and session.focus.type == "file":
                    row = sdb.execute(
                        _sql("SELECT work_id FROM audio_files WHERE id = :afid"),
                        {"afid": session.focus.id},
                    ).fetchone()
                    if row:
                        curr_wid = row[0]
                elif session.last_id is not None:
                    row = sdb.execute(
                        _sql("""
                            SELECT COALESCE(
                                (SELECT af.work_id FROM transcriptions t JOIN audio_files af ON t.audio_file_id = af.id WHERE t.id = :txid),
                                (SELECT work_id FROM expressions WHERE source_type = 'transcription_chunk' AND source_id = :txid AND work_id IS NOT NULL LIMIT 1)
                            )
                        """),
                        {"txid": session.last_id},
                    ).fetchone()
                    if row:
                        curr_wid = row[0]

                if curr_wid is not None:
                    curr_work = sdb.query(WorkRecord).filter_by(id=curr_wid).first()
                    work_title = curr_work.title if curr_work else f"Work #{curr_wid}"
                    work_author = curr_work.author if curr_work and curr_work.author else ""

                    # Find candidate target works connected via semantic edges with strict work isolation
                    cand_works = sdb.execute(
                        _sql("""
                            SELECT 
                                w2.id as work_id,
                                w2.title as title,
                                w2.author as author,
                                COUNT(DISTINCT CASE 
                                    WHEN r.from_expression_id < r.to_expression_id 
                                    THEN r.from_expression_id || '_' || r.to_expression_id 
                                    ELSE r.to_expression_id || '_' || r.from_expression_id 
                                END) as edge_count,
                                MAX(r.weight) as max_weight,
                                AVG(r.weight) as avg_weight
                            FROM expression_relations r
                            JOIN expressions e1 ON (r.from_expression_id = e1.id OR r.to_expression_id = e1.id)
                            JOIN expressions e2 ON (e2.id = CASE WHEN r.from_expression_id = e1.id THEN r.to_expression_id ELSE r.from_expression_id END)
                            JOIN works w2 ON e2.work_id = w2.id
                            WHERE e1.work_id = :curr_wid
                              AND e2.work_id IS NOT NULL
                              AND e2.work_id != :curr_wid
                              AND r.relation_type = 'semantic'
                            GROUP BY w2.id
                            ORDER BY max_weight DESC, edge_count DESC
                            LIMIT 4
                        """),
                        {"curr_wid": curr_wid},
                    ).fetchall()

                    if cand_works:
                        # 3-Clique (Triadic Consensus) Detection
                        # Find all works connected to curr_wid
                        conn_to_curr_rows = sdb.execute(
                            _sql("""
                                SELECT DISTINCT e2.work_id
                                FROM expression_relations r
                                JOIN expressions e1 ON (r.from_expression_id = e1.id OR r.to_expression_id = e1.id)
                                JOIN expressions e2 ON (e2.id = CASE WHEN r.from_expression_id = e1.id THEN r.to_expression_id ELSE r.from_expression_id END)
                                WHERE e1.work_id = :curr_wid
                                  AND e2.work_id IS NOT NULL
                                  AND e2.work_id != :curr_wid
                                  AND r.relation_type = 'semantic'
                            """),
                            {"curr_wid": curr_wid},
                        ).fetchall()
                        curr_neighbors = {r[0] for r in conn_to_curr_rows}

                        author_str = f" by {work_author}" if work_author else ""
                        console.print(f"\n  [{BOLD}]Next Listen Recommendations[/] [{DIM}](based on '{work_title}'{author_str})[/]:\n")

                        for idx, cw in enumerate(cand_works, start=1):
                            target_wid = cw[0]
                            target_title = cw[1] or f"Work #{target_wid}"
                            target_author = cw[2] or "Unknown Author"
                            edge_cnt = cw[3]
                            max_w = float(cw[4])

                            # Check for 3-clique
                            conn_to_target_rows = sdb.execute(
                                _sql("""
                                    SELECT DISTINCT e2.work_id, w2.author, w2.title
                                    FROM expression_relations r
                                    JOIN expressions e1 ON (r.from_expression_id = e1.id OR r.to_expression_id = e1.id)
                                    JOIN expressions e2 ON (e2.id = CASE WHEN r.from_expression_id = e1.id THEN r.to_expression_id ELSE r.from_expression_id END)
                                    JOIN works w2 ON e2.work_id = w2.id
                                    WHERE e1.work_id = :target_wid
                                      AND e2.work_id IS NOT NULL
                                      AND e2.work_id != :target_wid
                                      AND r.relation_type = 'semantic'
                                """),
                                {"target_wid": target_wid},
                            ).fetchall()
                            target_neighbors = {r[0]: (r[1] or r[2]) for r in conn_to_target_rows}
                            mutual_wids = (curr_neighbors & set(target_neighbors.keys())) - {curr_wid, target_wid}

                            clique_badge = ""
                            if mutual_wids:
                                mutual_partner_wid = next(iter(mutual_wids))
                                partner_name = target_neighbors.get(mutual_partner_wid, f"Work #{mutual_partner_wid}")
                                clique_badge = f" [{ACCENT}]▲ Consensus with {partner_name}[/]"

                            # Best bridging excerpt
                            bridge_row = sdb.execute(
                                _sql("""
                                    SELECT e2.content, e2.source_id
                                    FROM expression_relations r
                                    JOIN expressions e1 ON (r.from_expression_id = e1.id OR r.to_expression_id = e1.id)
                                    JOIN expressions e2 ON (e2.id = CASE WHEN r.from_expression_id = e1.id THEN r.to_expression_id ELSE r.from_expression_id END)
                                    WHERE e1.work_id = :curr_wid
                                      AND e2.work_id = :target_wid
                                      AND r.relation_type = 'semantic'
                                    ORDER BY r.weight DESC
                                    LIMIT 1
                                """),
                                {"curr_wid": curr_wid, "target_wid": target_wid},
                            ).fetchone()

                            # Resolving a playable transcript ID for the target work
                            tx_row = sdb.execute(
                                _sql("""
                                    SELECT t.id 
                                    FROM transcriptions t
                                    JOIN audio_files af ON t.audio_file_id = af.id
                                    WHERE af.work_id = :twid
                                    ORDER BY t.id ASC LIMIT 1
                                """),
                                {"twid": target_wid},
                            ).fetchone()
                            play_id = tx_row[0] if tx_row else (bridge_row[1] if bridge_row and bridge_row[1] else None)

                            match_pct = int(max_w * 100)
                            console.print(f"  [{BOLD}]{idx}. {target_author} — {target_title}[/]{clique_badge}")
                            console.print(f"     [{DIM}]Strength:[/] [{ACCENT}]{match_pct}% match[/] [{DIM}]({edge_cnt} cross-work bridges)[/]")
                            if bridge_row and bridge_row[0]:
                                bridge_excerpt = bridge_row[0].strip().replace("\n", " ")
                                if len(bridge_excerpt) > 130:
                                    bridge_excerpt = bridge_excerpt[:127] + "..."
                                console.print(f"     [{DIM}]Bridge:[/] \"{bridge_excerpt}\"")
                            if play_id:
                                console.print(f"     [{DIM}]Listen:[/] [{ACCENT}]\\play {play_id}[/]  [{DIM}]or[/]  [{ACCENT}]\\related[/] [{DIM}]to explore[/]")
                            else:
                                console.print(f"     [{DIM}]Explore:[/] [{ACCENT}]\\related[/]")
                            console.print()

                        console.print(f"  [{DIM}]Tip: Type [italic]\\suggest commands[/italic] to view frequent action shortcuts.[/]\n")
                        return
        except Exception as e:
            logger.warning("Next Listen suggestion failed: %s", e)

    # ── Command History & Workflow Fallback ───────────────────────────
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
        lock_icon = "✓" if unlocked else "x"
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
                title=f"[bold][{ACCENT}]Voice Enrollment — Select Audio File[/][/]",
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
                        header = " Voice Enrollment — Pick a sample "
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
                            icon = "[DIR]" if item_path.is_dir() else "[AUD]"
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
            title=f"[bold][{ACCENT}]Voice Enrollment — Confirm[/][/]",
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


# ── Universal Playback Commands ──────────────────────────────


def _playback_backslash(slash_command: str):
    """Decorator factory: wire a backslash command to the universal playback command dispatcher."""
    def decorator(fn: Callable) -> Callable:
        name = fn.__name__.removeprefix("cmd_")

        @register_backslash(name)
        def wrapper(session: ReplSession, args: str) -> None:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.playback.controls import handle_playback_command

            handle_playback_command(slash_command, args, get_daemon_client(), console)

        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _play_school(session: ReplSession, cid: int, client: Any) -> None:
    """Queue and sequentially play representative works of a school of thought."""
    from audiobench.community.compute_medoids import select_representative_samples
    from audiobench.playback.controls import fmt_timestamp
    from audiobench.core.db_session import get_session
    from sqlalchemy import text as _sql

    # 1. Fetch community details
    with _get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT community_id, name, medoid_expr_id, node_count FROM expression_communities WHERE community_id = ? AND is_active = 1",
            (cid,),
        )
        row = cur.fetchone()
        if not row:
            console.print(f"  [{WARNING}]School #{cid} not found or inactive. Run \\schools to list available schools.[/]")
            return
        comm_id, name, medoid_id, node_count = row

        # Fetch representative sample expressions across distinct works
        samples = select_representative_samples(conn, cid, medoid_id=medoid_id, max_samples=4)

    # 2. Build candidate list: medoid first, followed by samples from other works
    candidate_eids: list[int] = []
    if medoid_id:
        candidate_eids.append(medoid_id)
    for s in samples:
        eid = s.get("id") or s.get("expression_id")
        if eid and eid not in candidate_eids:
            candidate_eids.append(eid)

    if not candidate_eids:
        console.print(f"  [{WARNING}]No representative passages available for School #{cid}.[/]")
        return

    # 3. Resolve audio files and segment timestamps
    playable_items: list[dict[str, Any]] = []
    with get_session() as sdb:
        for eid in candidate_eids:
            res = _resolve_expression_playback_info(sdb, eid)
            e_id, tx_id, w_id, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res
            if not file_path:
                continue
            start_time, end_time, speaker = _resolve_thought_segment(sdb, e_id, tx_id, content)
            if start_time == 0.0 and chap_start_time > 0.0:
                start_time = float(chap_start_time)
            playable_items.append({
                "expression_id": e_id,
                "transcription_id": tx_id,
                "audio_file_id": af_id,
                "file_name": file_name,
                "file_path": file_path,
                "work_title": work_title or file_name or f"Work #{w_id}",
                "work_author": work_author or "Unknown Author",
                "start_time": start_time,
                "end_time": end_time,
                "content": content,
            })

    if not playable_items:
        console.print(f"  [{WARNING}]No playable audio tracks found for School #{cid}.[/]")
        return

    # 4. Save to session playlist
    session.school_playlist = {
        "community_id": cid,
        "title": name,
        "items": playable_items,
        "index": 0,
    }

    # 5. Play first track
    track0 = playable_items[0]
    client.playback_play(
        track0["file_path"],
        start_pos=track0["start_time"],
        audio_file_id=track0["audio_file_id"],
        transcription_id=track0["transcription_id"],
    )

    # 6. Render queue summary (zero emojis)
    console.print(f"\n  [{SUCCESS}]> Playing School #{cid:02d}:[/] [{BOLD}]{name}[/]")
    console.print(f"  [{DIM}]{'─' * 60}[/]")
    console.print(
        f"  [{ACCENT}]Track [1/{len(playable_items)}]:[/] [{BOLD}]{track0['work_author']} — {track0['work_title']}[/] "
        f"at [{ACCENT}]{fmt_timestamp(track0['start_time'])}[/]"
    )
    if len(playable_items) > 1:
        console.print(f"\n  [{DIM}]Upcoming in queue:[/]")
        for i, trk in enumerate(playable_items[1:], 2):
            console.print(
                f"    [{DIM}][{i}][/] {trk['work_author']} — {trk['work_title']} "
                f"[{DIM}]({fmt_timestamp(trk['start_time'])})[/]"
            )
    console.print(f"  [{DIM}]{'─' * 60}[/]")
    console.print(f"  [{DIM}]Use [{ACCENT}]\\next[/] to advance or [{ACCENT}]\\prev[/] to go back.[/]\n")


def _resolve_expression_playback_info(
    session_db: Any, expr_id: int
) -> tuple[int, int | None, int | None, str, int | None, str | None, str | None, str | None, str | None, float]:
    """Retrieve playback metadata for any expression, coalescing transcriptions and chapters.

    Returns:
        (eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time)
    """
    from sqlalchemy import text as _sql

    row = session_db.execute(
        _sql("""
            SELECT e.id,
                   COALESCE(
                       t.id,
                       c.transcription_id,
                       c2.transcription_id,
                       (SELECT t_sub.id FROM transcriptions t_sub 
                        WHERE t_sub.audio_file_id = COALESCE(c.audio_file_id, c2.audio_file_id) 
                        ORDER BY (CASE WHEN t_sub.status IN ('complete', 'completed') THEN 2
                                       WHEN t_sub.segment_count > 0 THEN 1
                                       ELSE 0 END) DESC, 
                                 t_sub.segment_count DESC, 
                                 t_sub.id DESC 
                        LIMIT 1)
                   ) as transcription_id,
                   e.work_id,
                   e.content,
                   COALESCE(t.audio_file_id, c.audio_file_id, c2.audio_file_id) as audio_file_id,
                   COALESCE(af_t.file_name, af_c.file_name, af_c2.file_name) as file_name,
                   COALESCE(af_t.file_path, af_c.file_path, af_c2.file_path) as file_path,
                   w.title as work_title,
                   w.author as work_author,
                   COALESCE(c.start_time, c2.start_time, 0.0) as chapter_start_time
            FROM expressions e
            LEFT JOIN transcriptions t ON (e.source_type IN ('audio_transcript', 'transcription_chunk') AND e.source_id = t.id)
            LEFT JOIN audio_files af_t ON t.audio_file_id = af_t.id
            LEFT JOIN chapters c ON (e.source_type = 'chapter_summary' AND e.source_id = c.id)
            LEFT JOIN audio_files af_c ON c.audio_file_id = af_c.id
            LEFT JOIN chapters c2 ON (e.chapter_id IS NOT NULL AND e.chapter_id = c2.id)
            LEFT JOIN audio_files af_c2 ON c2.audio_file_id = af_c2.id
            LEFT JOIN works w ON e.work_id = w.id
            WHERE e.id = :eid
        """),
        {"eid": expr_id},
    ).fetchone()
    if not row:
        return (expr_id, None, None, "", None, None, None, None, None, 0.0)
    eid = row[0]
    tx_id = int(row[1]) if row[1] is not None else None
    wid = row[2]
    content = row[3]
    af_id = row[4]
    file_name = row[5]
    file_path = row[6]
    work_title = row[7]
    work_author = row[8]
    chap_start_time = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0
    return (eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time)


def _play_expression(
    session: ReplSession,
    client: Any,
    expr_id: int,
    label: str = "",
) -> bool:
    """Resolve and initiate playback for any semantic expression (transcript or chapter)."""
    from audiobench.core.db_session import get_session
    from audiobench.playback.controls import fmt_timestamp
    from sqlalchemy import text as _sql

    with get_session() as sdb:
        st_row = sdb.execute(
            _sql("SELECT source_type FROM expressions WHERE id = :eid"),
            {"eid": expr_id},
        ).fetchone()
        source_type = st_row[0] if st_row else ""
        if source_type in ("search_synthesis", "search_session_summary", "chat_message", "journal_entry"):
            target_desc = label.capitalize() if label else f"Expression #{expr_id}"
            type_label = (
                "prior research synthesis" if source_type == "search_synthesis"
                else "past research session summary" if source_type == "search_session_summary"
                else "chat reflection" if source_type == "chat_message"
                else "cognitive journal entry"
            )
            console.print(
                f"  [{DIM}]{target_desc} (Expression #{expr_id}) is a text-based {type_label}, "
                f"not an audio recording.[/]"
            )
            if label:
                if "citation [" in label.lower():
                    c_num = label.lower().split("[")[-1].rstrip("]")
                    bm_cmd = f"\\bookmark cite {c_num}"
                elif "bridge [" in label.lower():
                    b_num = label.lower().split("[")[-1].rstrip("]").lstrip("b")
                    bm_cmd = f"\\bookmark b{b_num}"
                else:
                    bm_cmd = f"\\bookmark {expr_id}"
                console.print(
                    f"  [{DIM}]Use [{ACCENT}]{bm_cmd}[/] to inspect the full text card and dialectic connections.[/]"
                )
            return False

        res = _resolve_expression_playback_info(sdb, expr_id)
        eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res

        if not file_path:
            tag = f" for {label}" if label else ""
            console.print(f"  [{WARNING}]Could not locate audio file{tag} (Expression #{expr_id}).[/]")
            return False

        start_time, end_time, speaker = _resolve_thought_segment(sdb, eid, tx_id, content)
        if start_time == 0.0 and chap_start_time > 0.0:
            start_time = float(chap_start_time)

        st = client.playback_status()
        if st.get("playing") and st.get("transcription_id") == tx_id and tx_id:
            client.playback_seek(start_time)
        else:
            client.playback_play(
                file_path,
                start_pos=start_time,
                audio_file_id=af_id,
                transcription_id=tx_id,
            )

        author_str = f"{work_author} — " if work_author else ""
        title_str = f"{work_title}" if work_title else file_name
        tag = f" {label}:" if label else ":"
        console.print(
            f"  [{SUCCESS}]> Playing{tag}[/] [{BOLD}]{author_str}{title_str}[/] "
            f"at [{ACCENT}]{fmt_timestamp(start_time)}[/]"
        )
        return True


@register_backslash("play")
def cmd_play(session: ReplSession, args: str) -> None:
    """Play audio for current or specified transcript, or seek to timestamp."""
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.playback.controls import handle_playback_command

    client = get_daemon_client()

    arg_clean = args.strip()

    if arg_clean.lower() in ("next", "n"):
        cmd_next(session, "")
        return

    if arg_clean.lower() in ("prev", "p", "back", "b"):
        cmd_prev(session, "")
        return

    # School playback: \play school <id>, \play school:<id>, \play s<id>, or bare \play school
    school_play_id: int | None = None
    lower_play = arg_clean.lower()
    if lower_play.startswith("school ") and lower_play[7:].strip().lstrip("#").isdigit():
        school_play_id = int(lower_play[7:].strip().lstrip("#"))
    elif lower_play.startswith("school:") and lower_play[7:].strip().lstrip("#").isdigit():
        school_play_id = int(lower_play[7:].strip().lstrip("#"))
    elif lower_play.startswith("s") and lower_play[1:].isdigit():
        school_play_id = int(lower_play[1:])
    elif lower_play == "school" and session.focus and session.focus.type == "school":
        school_play_id = session.focus.id

    if school_play_id is not None:
        _play_school(session, school_play_id, client)
        return

    # Citation playback from synthesis: \play cite <n>
    if arg_clean.startswith("cite"):
        parts = arg_clean.split()
        if len(parts) < 2 or not parts[1].isdigit():
            console.print("  [{WARNING}]Please specify citation number: \\play cite <1|2|3...>[/]")
            return
        cite_num = int(parts[1])
        citations = getattr(session, "last_synthesis_citations", {})
        if cite_num not in citations:
            console.print(f"  [{WARNING}]Citation [{cite_num}] not found. Run \\synthesize <query> first.[/]")
            return
        expr_id = citations[cite_num]
        _play_expression(session, client, expr_id, label=f"citation [{cite_num}]")
        return

    # Semantic bridge playback from bookmark: \play b1, \play b2, \play b3 (or \play bridge 1)
    b_idx: int | None = None
    if lower_play.startswith("b") and lower_play[1:].isdigit():
        b_idx = int(lower_play[1:])
    elif lower_play.startswith("bridge") and lower_play[6:].strip().isdigit():
        b_idx = int(lower_play[6:].strip())

    if b_idx is not None:
        bridges = getattr(session, "last_bookmark_bridges", {})
        if b_idx not in bridges:
            console.print(f"  [{WARNING}]Semantic bridge [B{b_idx}] not found. Run \\bookmark to view bridges.[/]")
            return
        bridge = bridges[b_idx]
        _check_bridge_staleness(session, bridge, b_idx)
        _play_expression(session, client, bridge.target_expression_id, label=f"bridge [B{b_idx}]")
        return

    # Explicit expression playback: \play expr <id> or \play e<id>
    if lower_play.startswith("expr ") or (lower_play.startswith("e") and lower_play[1:].isdigit()):
        e_str = lower_play[1:] if lower_play.startswith("e") and not lower_play.startswith("expr") else lower_play[5:].strip()
        if e_str.isdigit():
            eid = int(e_str)
            if _play_expression(session, client, eid, label=f"Expression #{eid}"):
                return

    if arg_clean.isdigit():
        idx = int(arg_clean)
        if getattr(session, "last_related", None) and 1 <= idx <= len(session.last_related):
            thought = session.last_related[idx - 1]
            if thought.file_path:
                from audiobench.playback.controls import fmt_timestamp

                st = client.playback_status()
                if st.get("playing") and st.get("transcription_id") == thought.transcript_id:
                    client.playback_seek(thought.start_time)
                else:
                    client.playback_play(
                        thought.file_path,
                        start_pos=thought.start_time,
                        audio_file_id=thought.audio_file_id,
                        transcription_id=thought.transcript_id,
                    )
                console.print(
                    f"  [{SUCCESS}]> Playing thought #{idx}[/] from [{ACCENT}]{thought.file_name}[/] "
                    f"at [{BOLD}]{fmt_timestamp(thought.start_time)}[/]"
                )
                return
        elif getattr(session, "last_bookmark_bridges", None) and idx in session.last_bookmark_bridges:
            bridge = session.last_bookmark_bridges[idx]
            _check_bridge_staleness(session, bridge, idx)
            _play_expression(session, client, bridge.target_expression_id, label=f"bridge [B{idx}]")
            return

    repo = session._get_repo() if hasattr(session, "_get_repo") else None
    if repo is None:
        try:
            from audiobench.storage.repository import TranscriptionRepository

            repo = TranscriptionRepository()
        except Exception:
            repo = None
    loaded_ids = [session.last_id] if session.last_id else []
    handle_playback_command(
        "/play",
        args,
        client,
        console,
        transcript_resolver=(lambda tid: repo.get_by_id(tid, session=session)) if repo else None,
        loaded_transcript_ids=loaded_ids,
    )


@register_backslash("next")
@register_backslash("n")
def cmd_next(session: ReplSession, args: str) -> None:
    """Advance to the next track in the school queue or citation."""
    playlist = getattr(session, "school_playlist", None)
    if playlist and playlist.get("items"):
        idx = playlist.get("index", 0) + 1
        items = playlist["items"]
        if idx >= len(items):
            console.print(f"  [{DIM}]Reached end of playlist for School #{playlist['community_id']}.[/]")
            return
        playlist["index"] = idx
        track = items[idx]
        from audiobench.daemon.factory import get_daemon_client
        from audiobench.playback.controls import fmt_timestamp

        client = get_daemon_client()
        client.playback_play(
            track["file_path"],
            start_pos=track["start_time"],
            audio_file_id=track["audio_file_id"],
            transcription_id=track["transcription_id"],
        )
        console.print(
            f"  [{SUCCESS}]> Playing track [{idx + 1}/{len(items)}]:[/] "
            f"[{BOLD}]{track['work_author']} — {track['work_title']}[/] "
            f"at [{ACCENT}]{fmt_timestamp(track['start_time'])}[/]"
        )
        return

    from audiobench.daemon.factory import get_daemon_client
    from audiobench.playback.controls import handle_playback_command

    handle_playback_command("/next", args, get_daemon_client(), console)


@register_backslash("prev")
@register_backslash("p")
def cmd_prev(session: ReplSession, args: str) -> None:
    """Return to previous track in the school queue or citation."""
    playlist = getattr(session, "school_playlist", None)
    if playlist and playlist.get("items"):
        idx = playlist.get("index", 0) - 1
        items = playlist["items"]
        if idx < 0:
            console.print(f"  [{DIM}]Already at first track of School #{playlist['community_id']}.[/]")
            return
        playlist["index"] = idx
        track = items[idx]
        from audiobench.daemon.factory import get_daemon_client
        from audiobench.playback.controls import fmt_timestamp

        client = get_daemon_client()
        client.playback_play(
            track["file_path"],
            start_pos=track["start_time"],
            audio_file_id=track["audio_file_id"],
            transcription_id=track["transcription_id"],
        )
        console.print(
            f"  [{SUCCESS}]> Playing track [{idx + 1}/{len(items)}]:[/] "
            f"[{BOLD}]{track['work_author']} — {track['work_title']}[/] "
            f"at [{ACCENT}]{fmt_timestamp(track['start_time'])}[/]"
        )
        return

    from audiobench.daemon.factory import get_daemon_client
    from audiobench.playback.controls import handle_playback_command

    handle_playback_command("/prev", args, get_daemon_client(), console)



@_playback_backslash("/seek")
def cmd_seek(session: ReplSession, args: str) -> None:
    """Seek audio to a timestamp (e.g. \\seek 6:30 or \\seek +15)."""


@_playback_backslash("/pause")
def cmd_pause(session: ReplSession, args: str) -> None:
    """Pause audio playback."""


@_playback_backslash("/resume")
def cmd_resume(session: ReplSession, args: str) -> None:
    """Resume audio playback."""


@_playback_backslash("/speed")
def cmd_speed(session: ReplSession, args: str) -> None:
    """Set playback speed (e.g. \\speed 1.5)."""


@_playback_backslash("/stop")
def cmd_stop(session: ReplSession, args: str) -> None:
    """Stop audio playback and release player."""


@_playback_backslash("/np")
def cmd_np(session: ReplSession, args: str) -> None:
    """Show current playing audio and position."""


@_playback_backslash("/explain")
def cmd_explain(session: ReplSession, args: str) -> None:
    """Display transcript context around current audio position."""


@_playback_backslash("/clip")
def cmd_clip(session: ReplSession, args: str) -> None:
    """Clip audio between two timestamps (e.g. \\clip 6:30 7:00)."""


@_playback_backslash("/lyrics")
def cmd_lyrics(session: ReplSession, args: str) -> None:
    """Launch interactive live lyrics / karaoke follow HUD for daemon playback."""


@_playback_backslash("/karaoke")
def cmd_karaoke(session: ReplSession, args: str) -> None:
    """Launch interactive live lyrics / karaoke follow HUD (alias for \\lyrics)."""


@_playback_backslash("/follow")
def cmd_follow(session: ReplSession, args: str) -> None:
    """Launch interactive live lyrics / karaoke follow HUD (alias for \\lyrics)."""


# ── Phase 5: loop / filter / chapters / read ─────────────────


@register_backslash("loop")
def cmd_loop(session: ReplSession, args: str) -> None:
    """Set or clear an A-B loop on the current audio.

    Usage:
      \\loop <start> <end>   Engage A-B loop between two timestamps
      \\loop clear           Cancel active loop
      \\loop off             Cancel active loop (alias)

    Timestamps can be seconds (90) or MM:SS (1:30).
    """
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.playback.controls import fmt_timestamp

    client = get_daemon_client()

    arg = args.strip().lower()

    if arg in ("clear", "off", ""):
        if arg == "":
            # Bare \loop → show current loop state
            st = client.playback_status()
            loop_a = st.get("loop_a")
            loop_b = st.get("loop_b")
            if loop_a is not None and loop_b is not None:
                console.print(
                    f"  [{SUCCESS}]A-B loop:[/] {fmt_timestamp(loop_a)} → {fmt_timestamp(loop_b)}"
                )
            else:
                console.print(f"  [{DIM}]No active loop. Usage: \\loop <start> <end>[/]")
            return
        # clear / off
        try:
            result = client.playback_clear_loop()
            if result.get("error"):
                console.print(f"  [{WARNING}]Could not clear loop: {result.get('error')}[/]")
            else:
                console.print(f"  [{SUCCESS}]Loop cleared.[/]")
        except Exception as exc:
            console.print(f"  [{WARNING}]Could not clear loop: {exc}[/]")
        return

    # Parse "start end" — accept seconds or MM:SS
    def _parse_ts(s: str) -> float | None:
        s = s.strip()
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    parts = arg.split()
    if len(parts) != 2:
        console.print(
            f"  [{WARNING}]Usage: \\loop <start> <end>  or  \\loop clear[/]\n"
            f"  [{DIM}]Example: \\loop 1:30 2:45[/]"
        )
        return

    start = _parse_ts(parts[0])
    end = _parse_ts(parts[1])
    if start is None or end is None:
        console.print(f"  [{WARNING}]Could not parse timestamps: {args!r}[/]")
        return
    if end <= start:
        console.print(f"  [{WARNING}]End ({fmt_timestamp(end)}) must be after start ({fmt_timestamp(start)}).[/]")
        return

    try:
        result = client.playback_loop_segment(start, end)
        if result.get("error"):
            console.print(f"  [{WARNING}]Could not set loop: {result.get('error')}[/]")
        else:
            console.print(
                f"  [{SUCCESS}]Loop requested:[/] {fmt_timestamp(start)} → {fmt_timestamp(end)}"
            )
    except Exception as exc:
        console.print(f"  [{WARNING}]Could not set loop: {exc}[/]")


@register_backslash("filter")
def cmd_filter(session: ReplSession, args: str) -> None:
    """Apply or clear a live DSP audio filter.

    Usage:
      \\filter <filter_string>    Apply ffmpeg-style audio filter
      \\filter clear              Remove active filter
      \\filter off                Remove active filter (alias)
      \\filter                    Show current filter

    Example filters:
      equalizer=f=1000:width_type=o:width=2:g=-6   Notch 1kHz
      bass=g=5                                       Boost bass +5dB
      treble=g=-3                                    Cut treble -3dB
    """
    from audiobench.daemon.factory import get_daemon_client

    client = get_daemon_client()
    arg = args.strip()

    if arg.lower() in ("clear", "off"):
        try:
            result = client.playback_set_filter("")
            if result.get("error"):
                console.print(f"  [{WARNING}]Could not clear filter: {result.get('error')}[/]")
            else:
                console.print(f"  [{SUCCESS}]Filter cleared.[/]")
        except Exception as exc:
            console.print(f"  [{WARNING}]Could not clear filter: {exc}[/]")
        return

    if not arg:
        st = client.playback_status()
        flt = st.get("filter")
        if flt:
            console.print(f"  [{SUCCESS}]Active filter:[/] [{DIM}]{flt}[/]")
        else:
            console.print(f"  [{DIM}]No active filter. Usage: \\filter <ffmpeg_filter_string>[/]")
        return

    try:
        result = client.playback_set_filter(arg)
        if result.get("error"):
            console.print(f"  [{WARNING}]Could not apply filter: {result.get('error')}[/]")
        else:
            console.print(f"  [{SUCCESS}]Filter applied:[/] [{DIM}]{arg}[/]")
    except Exception as exc:
        console.print(f"  [{WARNING}]Could not apply filter: {exc}[/]")




def _render_chapters_table(console, st: dict, chapters: list) -> None:  # noqa: ANN001
    """Render full chapter list table."""
    from audiobench.playback.controls import fmt_timestamp

    pos = float(st.get("position", 0.0))
    console.print(f"\n  [{BOLD}]Chapters ({len(chapters)})[/]")
    console.print(f"  [{DIM}]{'─' * 48}[/]")
    for ch in chapters:
        active = (
            ch.start_time <= pos
            and (
                ch == chapters[-1]
                or pos < chapters[chapters.index(ch) + 1].start_time
            )
        )
        marker = f"[{ACCENT}]▶[/] " if active else "  "
        title_str = ch.title or f"Chapter {ch.index}"
        console.print(
            f"  {marker}[{DIM}]{ch.index:>2}[/]  "
            f"[{ACCENT}]{fmt_timestamp(ch.start_time):<9}[/] {title_str}"
        )
    console.print()


def _render_chapters_hud(console, st: dict, chapters: list, client) -> None:  # noqa: ANN001
    """Render the rich chapters HUD: timeline bar, 3-node runway, 3-tone transcript.

    Args:
        console: Rich Console instance.
        st:       Current playback status dict.
        chapters: Ordered list of ChapterInfo objects for the playing file.
        client:   Daemon client (for playback_context).
    """
    from audiobench.playback.controls import fmt_timestamp

    pos = float(st.get("position", 0.0))
    total = float(st.get("duration") or 0.0)
    if total <= 0.0 and chapters:
        total = max(float(chapters[-1].end_time), pos)
    n_chapters = len(chapters)

    # ── 1. Identify active / prev / next chapters ─────────────────────────
    active_idx: int = 0
    for i, ch in enumerate(chapters):
        ch_end = chapters[i + 1].start_time if i + 1 < n_chapters else float("inf")
        if ch.start_time <= pos < ch_end:
            active_idx = i
            break
    else:
        active_idx = max(0, n_chapters - 1)

    act_ch = chapters[active_idx]
    prev_ch = chapters[active_idx - 1] if active_idx > 0 else None
    next_ch = chapters[active_idx + 1] if active_idx + 1 < n_chapters else None

    # ── 2. Book-level progress header ────────────────────────────────────
    chapter_label = f"Chapter {act_ch.index} of {n_chapters}"
    console.print(
        f"\n  [{BOLD}]Book Progress:[/] "
        f"[bold cyan]{fmt_timestamp(pos)}[/bold cyan] "
        f"[{DIM}]/ {fmt_timestamp(total)}[/]"
        f"  [{DIM}][{chapter_label}][/]"
    )

    # ── 3. Proportional timeline bar ─────────────────────────────────────
    try:
        term_w = int(console.size.width)
    except Exception:
        term_w = 80
    term_w = max(40, term_w)
    bar_w = min(term_w - 4, 80)

    # Chapter progress stats
    act_start = act_ch.start_time
    act_end = next_ch.start_time if next_ch else total
    act_dur = max(act_end - act_start, 1.0)
    act_elapsed = max(0.0, pos - act_start)
    act_pct = min(1.0, act_elapsed / act_dur)
    act_pct_int = int(act_pct * 100)
    act_remaining = max(0.0, act_dur - act_elapsed)

    if total > 0 and bar_w >= 10 and n_chapters * 3 <= bar_w:
        raw: list[float] = []
        for i, ch in enumerate(chapters):
            ch_end = chapters[i + 1].start_time if i + 1 < n_chapters else total
            dur = max(ch_end - ch.start_time, 0.0)
            raw.append(dur / total * bar_w)

        allocated = [max(2, int(r)) for r in raw]
        excess = sum(allocated) - bar_w
        if excess > 0:
            order = sorted(range(n_chapters), key=lambda x: -allocated[x])
            for idx in order:
                if excess <= 0:
                    break
                trim = min(excess, allocated[idx] - 2)
                allocated[idx] -= trim
                excess -= trim
        elif excess < 0:
            deficit = -excess
            order = sorted(range(n_chapters), key=lambda x: -allocated[x])
            for idx in order:
                if deficit <= 0:
                    break
                allocated[idx] += 1
                deficit -= 1

        top = "┌" + "┬".join("─" * s_w for s_w in allocated) + "┐"

        mid_cells: list[str] = []
        for ch, s_w in zip(chapters, allocated):
            lbl = str(ch.index).zfill(2) if ch.index < 10 else str(ch.index)
            if s_w < len(lbl):
                lbl = lbl[:s_w]
            mid_cells.append(lbl.center(s_w))
        mid = "│" + "│".join(mid_cells) + "│"

        bot_segments: list[str] = []
        cursor_abs_col = 1
        arrow_col = 1
        for i, (ch, s_w) in enumerate(zip(chapters, allocated)):
            if i == active_idx:
                off = int(act_pct * s_w)
                off = max(0, min(off, s_w - 1))
                seg_chars = list("─" * s_w)
                seg_chars[off] = "▲"
                bot_segments.append("".join(seg_chars))
                arrow_col = cursor_abs_col + off
            else:
                bot_segments.append("─" * s_w)
            cursor_abs_col += s_w + 1

        bot = "└" + "┴".join(bot_segments) + "┘"
        arrow_line = " " * (arrow_col + 2) + f"└─ ▶ Ch {act_ch.index}: {act_ch.title or 'Untitled'}"

        console.print(f"  [{DIM}]{top}[/]")
        console.print(f"  [{DIM}]{mid}[/]")
        console.print(f"  [{DIM}]{bot}[/]")
        console.print(f"  [bold cyan]{arrow_line}[/bold cyan]")
    else:
        console.print(
            f"  [{DIM}]▶ Chapter {act_ch.index}: {act_ch.title or 'Untitled'}[/]"
        )

    console.print()

    # ── 4. 3-Node Runway ─────────────────────────────────────────────────
    col1_w = max(26, int(term_w * 0.30))
    col2_w = max(36, int(term_w * 0.42))

    BAR_LEN = 14
    filled = int(act_pct * BAR_LEN)
    filled = max(0, min(filled, BAR_LEN))
    bar_filled = "━" * filled
    bar_empty = "─" * (BAR_LEN - filled)
    nib = "╸" if filled < BAR_LEN else ""
    act_bar = f"[{bar_filled}{nib}{bar_empty}]"

    prev_title = (prev_ch.title or f"Ch {prev_ch.index}") if prev_ch else "—"
    next_title = (next_ch.title or f"Ch {next_ch.index}") if next_ch else "—"
    act_title = act_ch.title or f"Ch {act_ch.index}"

    # Header row
    prev_hdr = "◀ PREVIOUS" if prev_ch else ""
    act_hdr = f"◈ ACTIVE ({act_pct_int}% • {fmt_timestamp(act_remaining)} left)"
    next_hdr = "▶ NEXT" if next_ch else ""

    console.print(
        f"  [{DIM}]{prev_hdr:<{col1_w}}[/]"
        f"  [bold cyan]{act_hdr:^{col2_w}}[/bold cyan]"
        f"  [{DIM}]{next_hdr}[/]"
    )

    # Title row
    prev_t = f"Ch {prev_ch.index}: {prev_title}" if prev_ch else ""
    act_t = f"Ch {act_ch.index}: {act_title}"
    next_t = f"Ch {next_ch.index}: {next_title}" if next_ch else ""

    def _trunc(s: str, w: int) -> str:
        return s[:w - 1] + "…" if len(s) > w else s

    console.print(
        f"  [{DIM}]{_trunc(prev_t, col1_w):<{col1_w}}[/]"
        f"  [bold cyan]{_trunc(act_t, col2_w):^{col2_w}}[/bold cyan]"
        f"  [{DIM}]{next_t}[/]"
    )

    # Timestamps + progress bar row
    prev_dur = (prev_ch.end_time - prev_ch.start_time) if prev_ch else 0.0
    prev_ts = (
        f"{fmt_timestamp(prev_ch.start_time)} ({fmt_timestamp(prev_dur)})  ✓ Finished"
        if prev_ch else ""
    )
    act_ts = (
        f"{fmt_timestamp(act_ch.start_time)} → "
        f"{fmt_timestamp(act_end)} {act_bar}"
    )
    next_dur = (next_ch.end_time - next_ch.start_time) if next_ch else 0.0
    next_ts = (
        f"{fmt_timestamp(next_ch.start_time)} ({fmt_timestamp(next_dur)})"
        if next_ch else ""
    )

    console.print(
        f"  [{DIM}]{_trunc(prev_ts, col1_w):<{col1_w}}[/]"
        f"  [bold cyan]{_trunc(act_ts, col2_w):^{col2_w}}[/bold cyan]"
        f"  [{DIM}]{next_ts}[/]"
    )
    console.print()

    # ── 5. 3-Tone Transcript Segment Flow ────────────────────────────────
    try:
        ctx = client.playback_context(position=pos, window=2)
        if isinstance(ctx, dict):
            segments = ctx.get("segments") or []
            cur_i = ctx.get("current_idx", -1)

            if isinstance(segments, list) and isinstance(cur_i, int) and cur_i >= 0:
                ch_seg_start = act_ch.start_time
                ch_seg_end = act_end

                before: list[dict] = []
                active_seg: dict | None = None
                after: list[dict] = []

                for i, seg in enumerate(segments):
                    if not isinstance(seg, dict):
                        continue
                    seg_start = float(seg.get("norm_start", seg.get("start", 0)))
                    if seg_start < ch_seg_start or seg_start >= ch_seg_end:
                        continue
                    if i < cur_i:
                        before.append(seg)
                    elif i == cur_i:
                        active_seg = seg
                    else:
                        after.append(seg)

                prev_seg = before[-1] if before else None
                next_seg = after[0] if after else None

                def _seg_line(seg: dict, indent: str = "    ") -> str:
                    ts = fmt_timestamp(float(seg.get("norm_start", seg.get("start", 0))))
                    text = seg.get("text", "").strip()
                    return f"{indent}{ts}  {text}"

                if prev_seg:
                    console.print(f"[{DIM}]{_seg_line(prev_seg)}[/]")

                if active_seg:
                    ts = fmt_timestamp(float(active_seg.get("norm_start", active_seg.get("start", 0))))
                    text = active_seg.get("text", "").strip()
                    console.print(f"  [bold cyan]▶ {ts}  {text}[/bold cyan]")

                if next_seg:
                    console.print(f"[{DIM}][italic]{_seg_line(next_seg)}[/italic][/]")
                console.print()
    except Exception:
        pass  # Transcript unavailable — render gracefully without it

    # ── 6. Footer hint ───────────────────────────────────────────────────
    console.print(
        f"  [{DIM}]Tip: \\chapters <n> to jump"
        f"  •  \\chapters all for full table"
        f"  •  \\read for interactive reader[/]\n"
    )


@register_backslash("chapters")
def cmd_chapters(session: ReplSession, args: str) -> None:
    """Show chapter list for the currently playing file.

    Usage:
      \\chapters              List chapters with HUD runway and progress
      \\chapters all          List all chapters in full table format
      \\chapters <n>          Jump to chapter n
    """
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.playback.controls import fmt_timestamp
    from audiobench.storage.chapter_repository import get_chapter_repo

    client = get_daemon_client()
    arg = args.strip()

    # Resolve current audio_file_id from playback status
    st = client.playback_status()
    if not st.get("playing"):
        console.print(f"  [{DIM}]Nothing is playing. Start playback first.[/]")
        return

    audio_file_id = st.get("audio_file_id")
    if not audio_file_id:
        console.print(f"  [{DIM}]No audio file ID available for current track.[/]")
        return

    ch_repo = get_chapter_repo()
    chapters = ch_repo.get_chapters_for_file(audio_file_id)

    if not chapters:
        console.print(f"  [{DIM}]No chapters indexed for this file. Chapters auto-seed on first play.[/]")
        return

    # Jump to chapter n
    if arg.isdigit():
        n = int(arg)
        match = next((c for c in chapters if c.index == n), None)
        if match is None:
            console.print(f"  [{WARNING}]Chapter {n} not found (file has {len(chapters)} chapters).[/]")
            return
        client.playback_seek(match.start_time)
        title_str = f" — {match.title}" if match.title else ""
        console.print(
            f"  [{SUCCESS}]Jumped to Chapter {n}{title_str}[/] "
            f"[{DIM}]({fmt_timestamp(match.start_time)})[/]"
        )
        return

    # Full table format
    if arg.lower() == "all":
        _render_chapters_table(console, st, chapters)
        return

    # ── Rich chapter display ─────────────────────────────────────────────
    _render_chapters_hud(console, st, chapters, client)



@register_backslash("read")
def cmd_read(session: ReplSession, args: str) -> None:
    """Open the chapter reader TUI for the currently playing file.

    Displays synchronized transcript with A-B loop, bookmark, and
    visual-selection controls. Local playback only for now — daemon
    mode (Phase 3b) not yet implemented.

    Usage:
      \\read                  Open reader for currently playing track
      \\read <transcript_id>  Open reader for a specific transcript
    """
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.daemon.client import DaemonClient

    client = get_daemon_client()

    # ── Acquire service adapter: daemon proxy vs local service ──
    if isinstance(client, DaemonClient):
        from audiobench.playback.daemon_proxy import DaemonPlaybackProxy
        service = DaemonPlaybackProxy(client)
    else:
        from audiobench.playback.service import get_playback_service
        service = get_playback_service()

    st = service.status()

    arg = args.strip()

    # If a transcript ID is given, load it explicitly
    if arg.isdigit():
        transcription_id = int(arg)
        audio_file_id = None
        title = None
    elif st.get("playing") or arg == "":
        # Use currently playing track
        transcription_id = st.get("transcription_id")
        audio_file_id = st.get("audio_file_id")
        title = None
        if not (transcription_id or audio_file_id):
            console.print(
                f"  [{WARNING}]Nothing is playing and no transcript ID given.[/]\n"
                f"  [{DIM}]Usage: \\read  or  \\read <transcript_id>[/]"
            )
            return
    else:
        console.print(
            f"  [{WARNING}]Nothing is playing. Start playback first, or specify a transcript ID.[/]\n"
            f"  [{DIM}]Usage: \\read  or  \\read <transcript_id>[/]"
        )
        return

    # ── Load segments from DB ─────────────────────────────────
    from audiobench.storage.repository import TranscriptionRepository

    repo = TranscriptionRepository()
    record = None

    if transcription_id:
        record = repo.get_by_id(int(transcription_id), session=None)
    elif audio_file_id:
        best_id = None
        try:
            from audiobench.cli.commands.play import _find_best_transcript_for_file
            from audiobench.storage.models import AudioFileRecord
            from audiobench.core.db_session import get_session
            from sqlalchemy import text as _sql

            with get_session() as sdb:
                row = sdb.execute(
                    _sql("SELECT file_path FROM audio_files WHERE id = :id"),
                    {"id": audio_file_id},
                ).fetchone()
                if row:
                    best_id = _find_best_transcript_for_file(repo, row[0])
        except Exception:
            pass
        if best_id:
            record = repo.get_by_id(best_id, session=None)

    segments: list[dict] = []
    if record:
        segments = record.get("segments", []) or []
        transcription_id = transcription_id or record.get("id")
        audio_file_id = audio_file_id or record.get("audio_file_id")
        title = record.get("file_name")

    if not segments:
        console.print(
            f"  [{WARNING}]No transcript segments found for this track.[/]\n"
            f"  [{DIM}]Transcribe the file first: audiobench transcribe <file>[/]"
        )
        return

    if not st.get("playing"):
        console.print(f"  [{DIM}]Note: nothing is playing — reader will open in paused state.[/]")

    # ── Launch reader ─────────────────────────────────────────
    from audiobench.cli.tui.chapter_reader import open_reader

    open_reader(
        service,
        title=title or f"Transcript #{transcription_id}",
        transcription_id=transcription_id,
        audio_file_id=audio_file_id,
        raw_segments=segments,
    )


@register_backslash("info")
def cmd_info(session: ReplSession, args: str) -> None:
    """Display 360-degree dossier for an audio file or transcript."""
    from audiobench.cli.commands.info_cmd import render_file_dossier

    target = args.strip() or (str(session.last_id) if session.last_id else None)
    if not target:
        console.print("  [dim]Usage: \\info <target> (audio file ID, transcript ID, or filename)[/]")
        return
    render_file_dossier(target, console)


@register_backslash("schools")
def cmd_schools(session: ReplSession, args: str) -> None:
    """List all emergent intellectual schools of thought."""
    from audiobench.community.community_service import list_active_schools

    try:
        with _get_db_connection() as conn:
            schools = list_active_schools(conn)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load schools of thought: {e}[/]")
        return

    if not schools:
        console.print(
            f"  [{DIM}]No active schools of thought found. "
            f"Run canonical community clustering to generate them.[/]"
        )
        return

    table = make_table(
        f"Schools of Thought ({len(schools)} Active Communities)",
        [
            ("ID", {"style": ACCENT, "no_wrap": True, "width": 5}),
            ("School of Thought", {"style": BOLD}),
            ("Chunks", {"justify": "right", "width": 8}),
            ("Core Literature / Authors", {"style": DIM}),
        ],
    )
    for s in schools:
        table.add_row(
            f"#{s['community_id']:02d}",
            s["name"],
            str(s["node_count"]),
            s["core_sources"],
        )

    console.print(table)


@register_backslash("school")
def cmd_school(session: ReplSession, args: str) -> None:
    """Inspect a specific School of Thought dossier card."""
    import textwrap
    from rich.panel import Panel
    from rich.table import Table
    from audiobench.cli.display.theme import BOX_STYLE
    from audiobench.community.community_service import get_school_details
    from audiobench.community.compute_medoids import format_sample_passage

    clean_arg = args.strip().lstrip("#")
    if not clean_arg:
        console.print(f"  [{DIM}]Usage: \\school <id> (e.g. \\school 12)[/]")
        return

    if not clean_arg.isdigit():
        console.print(f"  [{WARNING}]Invalid school ID: '{args.strip()}'. Run \\schools to see available IDs.[/]")
        return

    cid = int(clean_arg)
    try:
        with _get_db_connection() as conn:
            details = get_school_details(conn, cid)
    except Exception as e:
        console.print(f"  [{WARNING}]Error loading school #{cid}: {e}[/]")
        return

    if not details:
        console.print(f"  [{WARNING}]School #{cid} not found or inactive. Run \\schools to see available schools.[/]")
        return

    from rich.console import Group
    from rich.rule import Rule

    term_width = (console.size.width if hasattr(console, "size") and console.size else None) or console.width or 80
    avail_w = max(40, term_width - 4)

    lines = []
    lines.append(f"[{BOLD} {ACCENT}]School #{details['community_id']:02d}: {details['name']}[/]")
    lines.append(
        f"[{DIM}]Size: {details['node_count']} chunks  ·  "
        f"Resolution: {details['resolution']:.2f}  ·  "
        f"Seed: {details['seed']}  ·  "
        f"Medoid: Expr #{details['medoid']['id']}[/]"
    )
    lines.append("")

    # Medoid
    lines.append(f"[{BOLD}]Medoid Passage (Core Center)[/]")
    m = details["medoid"]
    m_author = m["work_author"]
    m_title = m["work_title"]
    m_attr = f"— {m_author}, {m_title}" if m_author else f"— {m_title}"
    if m["content"]:
        m_txt = m.get("formatted_content") or m["content"]
        lines.append(f"  [{DIM}]\"{m_txt}\"[/]")
        lines.append(f"  [{ACCENT}]{m_attr}[/]")
    else:
        lines.append(f"  [{DIM}](No medoid expression assigned)[/]")
    lines.append("")

    # Top works
    lines.append(f"[{BOLD}]Top Works Breakdown[/]")
    for tw in details["top_works"]:
        w_attr = f"{tw['author']} — {tw['title']}" if tw["author"] and tw["author"] != "Unknown Author" else tw["title"]
        lines.append(f"  [{ACCENT}]•[/] {w_attr}: [{BOLD}]{tw['count']}[/] chunks [{DIM}]({tw['pct']:.1f}%)[/]")

    # Key Sample Expressions (Two-Column Book Mode inside continuous window)
    samples = details.get("sample_expressions") or []
    use_book = term_width >= 100 and len(samples) >= 2
    mode_label = "2-column book mode" if use_book else "single column"

    lines.append("")
    lines.append(f"[{BOLD} {ACCENT}]Key Sample Expressions[/] [{DIM}]({len(samples)} core expressions · {mode_label})[/]")

    group_items = ["\n".join(lines), Rule(style=DIM)]

    if not samples:
        group_items.append(f"  [{DIM}](No additional sample expressions found)[/]")
    else:
        if use_book:
            col_w = max(38, (avail_w - 5) // 2)
            tbl = Table.grid(padding=(0, 1))
            tbl.add_column(width=col_w, no_wrap=False)
            tbl.add_column(width=1, style=DIM)
            tbl.add_column(width=col_w, no_wrap=False)

            sample_blocks = []
            for s in samples:
                s_author = s["author"]
                s_title = s["title"]
                s_attr = f"{s_author} — {s_title}" if s_author and s_author != "Unknown Author" else s_title
                s_suffix = f"Expr #{s['id']} · Str: {s['strength']:.1f}"
                max_attr = max(15, col_w - len(s_suffix) - 2)
                if len(s_attr) > max_attr:
                    s_attr = s_attr[:max_attr - 1] + "…"
                spaces = " " * max(1, col_w - len(s_attr) - len(s_suffix))
                s_hdr = f"[{BOLD}]{s_attr}[/] {spaces}[{DIM}]{s_suffix}[/]"

                formatted_txt = s.get("formatted_content") or format_sample_passage(s["content"], max_chars=280)
                wrapped = textwrap.fill(f'"{formatted_txt}"', width=col_w - 2)
                block = [s_hdr] + [f"  [{DIM}]{w}[/]" for w in wrapped.split("\n")]
                sample_blocks.append(block)

            left_lines: list[str] = []
            right_lines: list[str] = []
            for idx, b in enumerate(sample_blocks):
                target = left_lines if idx % 2 == 0 else right_lines
                target.extend(b)
                target.append("")

            max_lines = max(len(left_lines), len(right_lines))
            for r_i in range(max_lines):
                l = left_lines[r_i] if r_i < len(left_lines) else ""
                r = right_lines[r_i] if r_i < len(right_lines) else ""
                tbl.add_row(l, "│", r)
            group_items.append(tbl)
        else:
            tbl = Table.grid(padding=(0, 0))
            tbl.add_column(width=avail_w - 2, no_wrap=False)
            for s in samples:
                s_author = s["author"]
                s_title = s["title"]
                s_attr = f"{s_author} — {s_title}" if s_author and s_author != "Unknown Author" else s_title
                s_suffix = f"Expr #{s['id']} · Str: {s['strength']:.1f}"
                s_hdr = f"[{BOLD}]{s_attr}[/]  [{DIM}]{s_suffix}[/]"
                tbl.add_row(s_hdr)
                formatted_txt = s.get("formatted_content") or format_sample_passage(s["content"], max_chars=280)
                wrapped = textwrap.fill(f'"{formatted_txt}"', width=avail_w - 4)
                for w in wrapped.split("\n"):
                    tbl.add_row(f"  [{DIM}]{w}[/]")
                tbl.add_row("")
            group_items.append(tbl)

    quick_actions = (
        f"[{DIM}]Quick actions:[/]  "
        f"[{ACCENT}]\\play school {cid}[/] [{DIM}]listen to school[/]  •  "
        f"[{ACCENT}]\\related school {cid}[/] [{DIM}]school relations & bridges[/]  •  "
        f"[{ACCENT}]\\schools[/] [{DIM}]list all schools[/]"
    )
    group_items.append(Rule(style=DIM))
    group_items.append(quick_actions)

    panel = Panel(Group(*group_items), border_style=DIM, box=BOX_STYLE, expand=False)
    console.print(panel)


# ── Proposals Inspection & Governance (\proposals, \prop) ───────────────────


def _is_daemon_alive() -> bool:
    """Check if the background daemon is reachable via socket without autostarting."""
    try:
        from audiobench.daemon.client import DaemonClient

        return DaemonClient()._raw_ping()
    except Exception:
        return False


def _fetch_daemon_proposals(
    expression_id: int | None = None,
    include_all: bool = False,
    status: str | None = None,
) -> list[dict]:
    """Fetch proposals from daemon socket or fallback to audited in-process server handler."""
    if _is_daemon_alive():
        try:
            from audiobench.daemon.client import DaemonClient

            return DaemonClient().get_proposals(
                expression_id=expression_id,
                include_all=include_all,
                status=status,
            )
        except Exception:
            pass

    from audiobench.daemon.server import _handle_get_proposals

    args: dict[str, Any] = {}
    if expression_id is not None:
        args["expression_id"] = expression_id
    if include_all:
        args["include_all"] = True
    if status is not None:
        args["status"] = status
    res = _handle_get_proposals(args)
    return res.get("proposals", [])


def _authorize_daemon_proposal(expression_id: int) -> dict:
    """Authorize proposal via daemon socket or fallback to audited in-process server handler."""
    if _is_daemon_alive():
        try:
            from audiobench.daemon.client import DaemonClient

            return DaemonClient().authorize_proposal(expression_id)
        except Exception:
            pass

    from audiobench.daemon.server import _handle_authorize_proposal

    return _handle_authorize_proposal({"expression_id": expression_id})


def _defer_daemon_proposal(expression_id: int) -> dict:
    """Defer proposal via daemon socket or fallback to audited in-process server handler."""
    if _is_daemon_alive():
        try:
            from audiobench.daemon.client import DaemonClient

            return DaemonClient().defer_proposal(expression_id)
        except Exception:
            pass

    from audiobench.daemon.server import _handle_defer_proposal

    return _handle_defer_proposal({"expression_id": expression_id})


def _reject_daemon_proposal(expression_id: int) -> dict:
    """Reject proposal via daemon socket or fallback to audited in-process server handler."""
    if _is_daemon_alive():
        try:
            from audiobench.daemon.client import DaemonClient

            return DaemonClient().reject_proposal(expression_id)
        except Exception:
            pass

    from audiobench.daemon.server import _handle_reject_proposal

    return _handle_reject_proposal({"expression_id": expression_id})


def _parse_proposal_payload(content: str) -> tuple[str, str, dict]:
    """Extract proposal_type, display_title, and parsed data dictionary."""
    content_str = (content or "").strip()
    if not content_str.startswith("{"):
        preview = content_str[:70].replace("\n", " ")
        return "legacy", preview, {}

    try:
        import json

        data = json.loads(content_str)
        if not isinstance(data, dict):
            return "legacy", content_str[:70], {}

        prop_type = data.get(
            "proposal_type",
            "operator" if "operator_template" in data else "proposal",
        )
        title = data.get("title") or data.get("summary") or content_str[:70]
        return prop_type, title, data
    except Exception:
        return "legacy", content_str[:70], {}


@register_backslash("proposals")
@register_backslash("prop")
def cmd_proposals(session: ReplSession, args: str) -> None:
    """Inspect and manage daemon proposals (knowledge gaps, operators).

    Usage:
        \\proposals                      List pending proposals (proposed/deferred)
        \\proposals list [all|status]    List proposals with optional status filter
        \\proposals show <id>            Show detailed dossier for a proposal
        \\proposals <id>                 Shorthand for \\proposals show <id>
        \\proposals authorize <id>       Authorize/confirm an operator or knowledge gap
        \\proposals defer <id>           Defer proposal from active alerts
        \\proposals reject <id>          Reject and dismiss proposal
    """
    tokens = args.strip().split()
    subcmd = tokens[0].lower() if tokens else "list"

    clean_subcmd = subcmd.lstrip("#")
    if clean_subcmd.isdigit():
        _show_proposal(int(clean_subcmd))
        return

    if subcmd in ("all", "*"):
        _list_proposals(include_all=True)
        return

    if subcmd in ("list", "ls"):
        status_filter = tokens[1].lower() if len(tokens) > 1 else None
        include_all = status_filter in ("all", "*")
        eff_status = (
            None
            if include_all
            else (
                status_filter
                if status_filter in ("proposed", "deferred", "confirmed", "rejected")
                else None
            )
        )
        _list_proposals(include_all=include_all, status=eff_status)
    elif subcmd in ("show", "info", "view", "dossier"):
        raw_id = tokens[1].lstrip("#") if len(tokens) > 1 else ""
        if not raw_id.isdigit():
            console.print(f"[{ERROR}]Usage: \\proposals show <id>[/]")
            return
        _show_proposal(int(raw_id))
    elif subcmd in ("authorize", "auth", "confirm", "accept"):
        raw_id = tokens[1].lstrip("#") if len(tokens) > 1 else ""
        if not raw_id.isdigit():
            console.print(f"[{ERROR}]Usage: \\proposals authorize <id>[/]")
            return
        _authorize_proposal(int(raw_id))
    elif subcmd in ("defer", "pause"):
        raw_id = tokens[1].lstrip("#") if len(tokens) > 1 else ""
        if not raw_id.isdigit():
            console.print(f"[{ERROR}]Usage: \\proposals defer <id>[/]")
            return
        _defer_proposal(int(raw_id))
    elif subcmd in ("reject", "dismiss", "del", "rm"):
        raw_id = tokens[1].lstrip("#") if len(tokens) > 1 else ""
        if not raw_id.isdigit():
            console.print(f"[{ERROR}]Usage: \\proposals reject <id>[/]")
            return
        _reject_proposal(int(raw_id))
    else:
        console.print(
            f"[{ERROR}]Unknown subcommand: {subcmd}.[/] Type [{ACCENT}]\\help proposals[/] for usage."
        )


def _list_proposals(include_all: bool = False, status: str | None = None) -> None:
    """Render table of proposals."""
    from rich.markup import escape
    from rich.table import Table

    from audiobench.daemon.intelligence.proposal_cluster import canonical_cluster_key

    proposals = _fetch_daemon_proposals(include_all=include_all, status=status)
    if not proposals:
        target = f"'{status}'" if status else ("all" if include_all else "pending")
        console.print(f"[{DIM}]No {target} proposals found.[/]")
        return

    if not include_all:
        clusters: dict[tuple, list[dict]] = {}
        for p in proposals:
            k = canonical_cluster_key(p.get("content", ""))
            clusters.setdefault(k, []).append(p)

        display_proposals = []
        cluster_run_counts: dict[int, int] = {}
        for group in clusters.values():
            # Select representative: prefer confirmed first, then highest ID
            rep = sorted(
                group,
                key=lambda x: (x.get("inference_status") == "confirmed", x.get("id", 0)),
                reverse=True,
            )[0]
            display_proposals.append(rep)
            cluster_run_counts[rep["id"]] = len(group)

        display_proposals.sort(key=lambda x: x.get("id", 0), reverse=True)
    else:
        display_proposals = proposals
        cluster_run_counts = {}

    table = Table(
        box=BOX_STYLE,
        title=f"[{BOLD}][{ACCENT}]Daemon Proposals ({len(display_proposals)})[/][/]",
        title_justify="left",
    )
    table.add_column("ID", style=ACCENT, no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Title / Summary")
    table.add_column("Created", style=DIM, no_wrap=True)

    for p in display_proposals:
        pid = str(p["id"])
        prop_type, title, _ = _parse_proposal_payload(p.get("content", ""))
        p_status = p.get("inference_status", "proposed")

        if p_status == "proposed":
            status_str = f"[{BOLD}][{ACCENT}]proposed[/][/]"
        elif p_status == "deferred":
            status_str = f"[{DIM}]deferred[/]"
        elif p_status == "confirmed":
            status_str = f"[{SUCCESS}]confirmed[/]"
        elif p_status == "rejected":
            status_str = f"[{ERROR}]rejected[/]"
        else:
            status_str = f"[{DIM}]{p_status}[/]"

        type_color = (
            "cyan"
            if prop_type == "knowledge_gap"
            else ("magenta" if prop_type == "operator" else "white")
        )
        type_str = f"[{type_color}]{prop_type}[/]"

        escaped_title = escape(title)
        runs = cluster_run_counts.get(p["id"], 1)
        if runs > 1:
            title_display = f"{escaped_title} [{DIM}]({runs} runs)[/]"
        else:
            title_display = escaped_title

        created = str(p.get("created_at", ""))[:19]
        table.add_row(f"#{pid}", type_str, status_str, title_display, created)

    console.print(table)
    console.print(
        f"[{DIM}]Actions:[/] [{ACCENT}]\\proposals show <id>[/] [{DIM}]dossier[/] • "
        f"[{ACCENT}]\\proposals authorize <id>[/] [{DIM}]confirm[/] • "
        f"[{ACCENT}]\\proposals defer <id>[/] [{DIM}]defer[/] • "
        f"[{ACCENT}]\\proposals reject <id>[/] [{DIM}]dismiss[/]"
    )


def _normalize_recommended_ingestion(raw: Any) -> list[str]:
    """Normalize recommended_ingestion into a canonical list of strings."""
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        import re

        return [
            item.strip()
            for item in re.split(r',\s*(?=")|,\s*(?=[A-Za-z0-9])', raw)
            if item.strip()
        ]
    return []


def _show_proposal(proposal_id: int) -> None:
    """Render full dossier panel for a single proposal."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule

    from audiobench.daemon.intelligence.proposal_cluster import canonical_cluster_key

    proposals = _fetch_daemon_proposals(expression_id=proposal_id)
    if not proposals:
        console.print(f"[{ERROR}]Proposal #{proposal_id} not found.[/]")
        return

    p = proposals[0]
    content = p.get("content", "")
    prop_type, title, data = _parse_proposal_payload(content)
    status = p.get("inference_status", "proposed")
    created_at = str(p.get("created_at", ""))[:19]

    items = []
    items.append(
        f"[{BOLD}][{ACCENT}]Proposal #{proposal_id}[/][/]  [{DIM}]•[/]  [{BOLD}]{title}[/]"
    )
    items.append(
        f"[{DIM}]Type:[/] [{BOLD}]{prop_type}[/]  [{DIM}]•  Status:[/] [{BOLD}]{status}[/]  [{DIM}]•  Created:[/] {created_at}"
    )

    cluster_key = canonical_cluster_key(content)
    all_proposals = _fetch_daemon_proposals(include_all=True)
    cluster_members = [
        m for m in all_proposals
        if canonical_cluster_key(m.get("content", "")) == cluster_key
    ]
    if len(cluster_members) > 1:
        member_ids_str = ", ".join(f"#{m['id']}" for m in sorted(cluster_members, key=lambda x: x.get("id", 0)))
        items.append(
            f"[{DIM}]Cluster:[/] [{BOLD}]{len(cluster_members)} runs[/] [{DIM}]({member_ids_str})[/]"
        )
    items.append(Rule(style=DIM))

    payload = data.get("payload", data) if isinstance(data, dict) else {}
    if not isinstance(payload, dict):
        payload = data

    if prop_type == "knowledge_gap":
        diagnostic = payload.get("diagnostic") or data.get("diagnostic", "")
        if diagnostic:
            items.append(f"[{BOLD}]Missing Topic Diagnostic:[/]")
            items.append(f"  {diagnostic}")
            items.append("")

        raw_rec = payload.get("recommended_ingestion") or data.get("recommended_ingestion")
        recommended = _normalize_recommended_ingestion(raw_rec)
        if recommended:
            items.append(f"[{BOLD}]Recommended Acquisitions / Reading List:[/]")
            for item in recommended:
                items.append(f"  [{ACCENT}]•[/] {item}")
            items.append("")

        stranded = payload.get("stranded_queries") or data.get("stranded_queries", [])
        if stranded:
            items.append(f"[{BOLD}]Correlated Search Queries ({len(stranded)}):[/]")
            preview_queries = stranded[:10]
            for q in preview_queries:
                items.append(f"  [{DIM}]-[/] \"{q}\"")
            if len(stranded) > 10:
                items.append(
                    f"  [{DIM}]... and {len(stranded) - 10} more correlated queries[/]"
                )
            items.append("")

        medoid = payload.get("medoid_query") or data.get("medoid_query")
        dist = payload.get("drift_distance") if payload.get("drift_distance") is not None else payload.get("distance")
        if dist is None:
            dist = data.get("drift_distance") if data.get("drift_distance") is not None else data.get("distance")
        source = payload.get("name_source") or data.get("name_source", "unknown")
        meta_lines = []
        if medoid:
            medoid_display = medoid.strip()
            if len(medoid_display) > 160:
                medoid_display = medoid_display[:157] + "..."
            meta_lines.append(f"[{DIM}]Medoid Anchor:[/] \"{medoid_display}\"")
        if dist is not None:
            meta_lines.append(f"[{DIM}]Semantic Drift Distance:[/] {dist:.4f} (threshold: 0.20)")
        if source:
            meta_lines.append(f"[{DIM}]Naming Provider:[/] {source}")
        if meta_lines:
            items.append(Rule(title="Telemetry", style=DIM))
            for ml in meta_lines:
                items.append(f"  {ml}")

    elif prop_type == "operator":
        template = data.get("operator_template", "Unknown")
        summary = data.get("summary", "")
        rationale = data.get("rationale", "")
        items.append(f"[{BOLD}]Operator Template:[/] [{ACCENT}]{template}[/]")
        if summary:
            items.append(f"[{BOLD}]Summary:[/] {summary}")
        if rationale:
            items.append(f"[{BOLD}]Rationale:[/] {rationale}")
    else:
        items.append(f"[{DIM}]Raw Content:[/]\n{content}")

    items.append(Rule(style=DIM))
    items.append(
        f"[{DIM}]Actions:[/] [{ACCENT}]\\proposals authorize {proposal_id}[/] [{DIM}]confirm[/]  •  "
        f"[{ACCENT}]\\proposals defer {proposal_id}[/] [{DIM}]defer[/]  •  "
        f"[{ACCENT}]\\proposals reject {proposal_id}[/] [{DIM}]dismiss[/]"
    )

    console.print(
        Panel(Group(*items), border_style=ACCENT, box=BOX_STYLE, expand=False)
    )


def _authorize_proposal(proposal_id: int) -> None:
    """Execute authorize action for a proposal."""
    res = _authorize_daemon_proposal(proposal_id)
    if res.get("status") == "error":
        console.print(
            f"[{ERROR}]Failed to authorize proposal #{proposal_id}: {res.get('message', 'Unknown error')}[/]"
        )
        return
    action = res.get("action", "authorized")
    cascade_note = ""
    cascaded_count = res.get("cascaded_count", 1)
    if cascaded_count > 1:
        cluster_title = res.get("cluster_title") or f"#{proposal_id}"
        cascade_note = f" (cascaded to {cascaded_count} duplicate runs in cluster '{cluster_title}')"
    if action == "authorized":
        console.print(
            f"[{SUCCESS}]Authorized proposal #{proposal_id}{cascade_note} — operator dynamic hot-registration active.[/]"
        )
    else:
        console.print(
            f"[{SUCCESS}]Confirmed proposal #{proposal_id}{cascade_note} — knowledge gap acknowledged.[/]"
        )


def _defer_proposal(proposal_id: int) -> None:
    """Execute defer action for a proposal."""
    res = _defer_daemon_proposal(proposal_id)
    if res.get("status") == "error":
        console.print(
            f"[{ERROR}]Failed to defer proposal #{proposal_id}: {res.get('message', 'Unknown error')}[/]"
        )
        return
    cascade_note = ""
    cascaded_count = res.get("cascaded_count", 1)
    if cascaded_count > 1:
        cluster_title = res.get("cluster_title") or f"#{proposal_id}"
        cascade_note = f" (cascaded to {cascaded_count} duplicate runs in cluster '{cluster_title}')"
    console.print(
        f"[{WARNING}]Deferred proposal #{proposal_id}{cascade_note} — alerts suppressed.[/]"
    )


def _reject_proposal(proposal_id: int) -> None:
    """Execute reject action for a proposal."""
    res = _reject_daemon_proposal(proposal_id)
    if res.get("status") == "error":
        console.print(
            f"[{ERROR}]Failed to reject proposal #{proposal_id}: {res.get('message', 'Unknown error')}[/]"
        )
        return
    cascade_note = ""
    cascaded_count = res.get("cascaded_count", 1)
    if cascaded_count > 1:
        cluster_title = res.get("cluster_title") or f"#{proposal_id}"
        cascade_note = f" (cascaded to {cascaded_count} duplicate runs in cluster '{cluster_title}')"
    console.print(
        f"[{ERROR}]Rejected proposal #{proposal_id}{cascade_note} — proposal dismissed.[/]"
    )



