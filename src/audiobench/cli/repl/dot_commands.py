r"""Dot-commands for the AudioBench REPL.

Dot-commands (prefixed with `.`) are the semantic layer — they reach into the
unified expression namespace via LanceDB. They differ from backslash commands
(`\\`) in that they query meaning, not structure.

  \search "keyword"     → SQLite FTS5  — instant, exact text match
  .search "concept"     → LanceDB ANN  — semantic, cross-source, no LLM
  .ask "question"       → LanceDB+LLM  — synthesised answer (already exists via `ask` command)

Registration is side-effect-based: importing this module populates
_DOT_HANDLERS via @register_dot.

To add a new .command, decorate a function here:

    @register_dot("mycommand")
    def dot_mycommand(session: ReplSession, args: str) -> None:
        ...
"""

from __future__ import annotations

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    WARNING,
    console,
)
from audiobench.cli.repl.session import ReplSession

# ── Registry ─────────────────────────────────────────────────

_DOT_HANDLERS: dict[str, callable] = {}


def register_dot(name: str):
    """Decorator that registers a function as a dot-command handler."""
    def decorator(fn):
        _DOT_HANDLERS[name.lower()] = fn
        return fn
    return decorator


def dispatch_dot(session: ReplSession, line: str) -> bool:
    """Dispatch a dot-command (without the leading `.`).

    Returns True if handled, False if unknown.
    """
    parts = line.strip().split(None, 1)
    if not parts:
        _print_dot_help()
        return True

    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = _DOT_HANDLERS.get(cmd)
    if handler is None:
        console.print(f"  [{WARNING}]Unknown dot-command: .{cmd}[/]")
        console.print(f"  [{DIM}]Type [{ACCENT}].help[/] for available dot-commands.[/]")
        return False

    try:
        handler(session, args)
        return True
    except Exception as e:
        console.print(f"  [{WARNING}]Error in .{cmd}: {e}[/]")
        return False


def _print_dot_help() -> None:
    console.print(f"\n  [{BOLD}]Dot-commands — Semantic Layer[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"  [{ACCENT}].use[/] [{DIM}]<id>[/]         Set context to a transcript by ID")
    console.print(f"  [{ACCENT}].focus[/] [{DIM}]<id>[/]       Set focus to an audio file by ID")
    console.print(f"  [{ACCENT}].search[/] [{DIM}]\"concept\"[/]   LanceDB semantic — cross-source, no LLM")
    console.print(f"  [{ACCENT}].help[/]              This help\n")


# ── .help ─────────────────────────────────────────────────────

@register_dot("help")
def dot_help(session: ReplSession, args: str) -> None:
    """Show available dot-commands."""
    _print_dot_help()

@register_dot("use")
def dot_use(session: ReplSession, args: str) -> None:
    """Set the current context to a specific transcript ID."""
    from audiobench.cli.repl.dispatch import print_context_summary
    from audiobench.core.focused_entity import FocusedEntity
    from audiobench.storage.repository import TranscriptionRepository

    arg = args.strip()
    if not arg.isdigit():
        console.print(f"  [{WARNING}]Usage: .use <transcript_id>[/]")
        return

    tx_id = int(arg)
    repo = TranscriptionRepository()
    rec = repo.get_by_id(tx_id)
    if not rec:
        console.print(f"  [{WARNING}]Transcript #{tx_id} not found.[/]")
        return

    session.set_context(tx_id)

    # Also focus the associated audio file if possible
    audio_file_id = rec.get("audio_file_id")
    if audio_file_id:
        file_name = rec.get("file_name", f"File #{audio_file_id}")
        session.focus = FocusedEntity(type="file", id=audio_file_id, label=file_name)

    print_context_summary(session)

@register_dot("focus")
def dot_focus(session: ReplSession, args: str) -> None:
    """Alias for \\focus."""
    from audiobench.cli.repl.backslash_commands import cmd_focus
    cmd_focus(session, args)


# ── .search ──────────────────────────────────────────────────

# Source type display labels and icons
_SOURCE_ICONS: dict[str, str] = {
    "transcript_segment":  "≋",
    "audio_transcript":    "◎",
    "chat_message":        "◈",
    "ask_answer":          "◐",
    "ask_query":           "◑",
    "session_summary":     "◆",
    "drift_phase":         "◇",
    "open_thread":         "◉",
    "bookmark_note":       "◎",
    "chapter_summary":     "▸",
    "system_inference":    "⬡",
    "user_correction":     "✎",
    "journal_entry":       "📝",
}

_SOURCE_LABELS: dict[str, str] = {
    "transcript_segment":  "segment",
    "audio_transcript":    "transcript",
    "chat_message":        "chat",
    "ask_answer":          "ask answer",
    "ask_query":           "ask query",
    "session_summary":     "summary",
    "drift_phase":         "drift",
    "open_thread":         "thread",
    "bookmark_note":       "bookmark",
    "chapter_summary":     "chapter",
    "system_inference":    "inference",
    "user_correction":     "correction",
    "journal_entry":       "note",
}


def _is_daemon_available() -> bool:
    """Quick non-blocking check: is the daemon socket alive?"""
    try:
        from pathlib import Path

        from audiobench.core.settings import get_settings
        from audiobench.daemon.factory import _is_socket_alive
        socket_path = Path(get_settings().daemon_socket_path)
        return _is_socket_alive(socket_path)
    except Exception:
        return False


def _resolve_provenance(expr) -> dict:
    """Given an ExpressionRecord, resolve the human-readable provenance.

    Returns a dict with:
        file_name: str | None
        audio_file_id: int | None
        session_label: str | None
        timestamp_hint: str | None
    """
    info: dict = {
        "file_name": None,
        "audio_file_id": None,
        "session_label": None,
        "timestamp_hint": None,
    }

    try:
        from audiobench.core.db_session import get_session as db_session
        from audiobench.storage.models import (
            AskLog,
            AudioFileRecord,
            ChatSession,
            TranscriptionRecord,
        )

        source_type = expr.source_type
        source_id = expr.source_id
        session_id = expr.session_id

        with db_session() as s:
            # Transcript segment / full transcript → lookup audio file
            if source_type in ("transcript_segment", "audio_transcript", "chapter_summary"):
                if source_id:
                    tx = s.query(TranscriptionRecord).filter_by(id=source_id).first()
                    if tx:
                        af = s.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                        if af:
                            info["file_name"] = af.file_name
                            info["audio_file_id"] = af.id

            # Ask answer/query → lookup via ask log → audio file
            elif source_type in ("ask_answer", "ask_query"):
                if session_id:
                    log = s.query(AskLog).filter_by(id=session_id).first()
                    if log:
                        af = s.query(AudioFileRecord).filter_by(id=log.audio_file_id).first()
                        if af:
                            info["file_name"] = af.file_name
                            info["audio_file_id"] = af.id

            # Chat message → lookup chat session
            elif source_type == "chat_message":
                if session_id:
                    cs = s.query(ChatSession).filter_by(id=session_id).first()
                    if cs:
                        info["session_label"] = f"Chat session #{session_id}"
                        # If the chat session is linked to an audio file
                        if hasattr(cs, "audio_file_id") and cs.audio_file_id:
                            af = s.query(AudioFileRecord).filter_by(id=cs.audio_file_id).first()
                            if af:
                                info["file_name"] = af.file_name
                                info["audio_file_id"] = af.id

            # Bookmark note → lookup audio file directly
            elif source_type == "bookmark_note":
                if source_id:
                    af = s.query(AudioFileRecord).filter_by(id=source_id).first()
                    if af:
                        info["file_name"] = af.file_name
                        info["audio_file_id"] = af.id

            # Journal entry (note capture)
            elif source_type == "journal_entry":
                if source_id:
                    from audiobench.storage.models import NoteCapture, NoteCollection
                    cap = s.query(NoteCapture).filter_by(id=source_id).first()
                    if cap:
                        col = s.query(NoteCollection).filter_by(id=cap.collection_id).first()
                        if col:
                            info["note_title"] = col.title
                        info["note_id"] = cap.id

    except Exception:
        pass

    return info


def _render_results_ddm(
    display_results: list[tuple],  # (expr_or_stub, score, prov, is_redacted)
    query: str,
    mode: str = "semantic",
) -> None:
    """Render DDM-aware search results.

    Visible results render normally. Redacted results render as locked blocks
    with the exact tier required to unlock, so the user always knows what
    sensitive content exists without seeing it.
    """
    mode_label = "LanceDB Semantic" if mode == "semantic" else "SQLite FTS5"
    console.print(f"\n  [{BOLD}]Semantic Results — {mode_label}[/]")
    console.print(f"  [{DIM}]{'─' * 60}[/]")

    for i, (expr_or_stub, score, prov, is_redacted) in enumerate(display_results, 1):
        score_str = f"[{score:.2f}]" if mode == "semantic" else ""

        if is_redacted:
            stub = expr_or_stub
            tier_label = stub.get("tier_label", "Tier ?")
            tier_num = stub.get("privacy_tier", "?")
            icon = _SOURCE_ICONS.get(stub.get("source_type", ""), "●")
            label = _SOURCE_LABELS.get(stub.get("source_type", ""), "memory")
            console.print(
                f"  [{WARNING}][{i}][/] [{DIM}]{score_str}[/] "
                f"[bold]{icon}[/] [{WARNING}]{label}[/]  "
                f"[{WARNING}][REDACTED — {tier_label} required][/]"
            )
            console.print(f"       [{WARNING}]{'█' * 42}  🔒[/]")
            console.print(
                f"       [{DIM}]Run [{ACCENT}]\\unlock {tier_num}[/] to access this memory.[/]"
            )
            console.print()
        else:
            expr = expr_or_stub
            icon = _SOURCE_ICONS.get(expr.source_type, "●")
            label = _SOURCE_LABELS.get(expr.source_type, expr.source_type)
            prov_parts = []
            if prov.get("file_name"):
                prov_parts.append(prov["file_name"])
            elif prov.get("session_label"):
                prov_parts.append(prov["session_label"])
            elif prov.get("note_title"):
                prov_parts.append(prov["note_title"])
            prov_label = f'  "{prov_parts[0]}"' if prov_parts else ""
            content = expr.content.strip().replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "..."
            console.print(
                f"  [{ACCENT}][{i}][/] [{DIM}]{score_str}[/] "
                f"[bold]{icon}[/] [{ACCENT}]{label}[/]{prov_label}"
            )
            console.print(f"       [{DIM}]\"{content}\"[/]")
            console.print()

    console.print(f"  [{DIM}]{'─' * 60}[/]")


def _render_results(
    results: list[tuple],  # list of (expr, score, provenance_dict)
    query: str,
    mode: str = "semantic",
) -> None:
    """Render the search result list to the console."""
    mode_label = "LanceDB Semantic" if mode == "semantic" else "SQLite FTS5"
    console.print(f"\n  [{BOLD}]Semantic Results — {mode_label}[/]")
    console.print(f"  [{DIM}]{'─' * 60}[/]")

    for i, (expr, score, prov) in enumerate(results, 1):
        icon = _SOURCE_ICONS.get(expr.source_type, "●")
        label = _SOURCE_LABELS.get(expr.source_type, expr.source_type)
        score_str = f"[{score:.2f}]" if mode == "semantic" else ""

        # Build provenance label
        prov_parts = []
        if prov.get("file_name"):
            prov_parts.append(prov["file_name"])
        elif prov.get("session_label"):
            prov_parts.append(prov["session_label"])
        elif prov.get("note_title"):
            prov_parts.append(prov["note_title"])
        prov_label = f'  "{prov_parts[0]}"' if prov_parts else ""

        # Snippet — truncate intelligently
        content = expr.content.strip().replace("\n", " ")
        if len(content) > 120:
            content = content[:117] + "..."

        console.print(
            f"  [{ACCENT}][{i}][/] [{DIM}]{score_str}[/] "
            f"[bold]{icon}[/] [{ACCENT}]{label}[/]{prov_label}"
        )
        console.print(f"       [{DIM}]\"{content}\"[/]")
        console.print()

    console.print(f"  [{DIM}]{'─' * 60}[/]")


def _action_menu(
    session: ReplSession,
    expr,
    prov: dict,
    expr_repo,
) -> None:
    """Show the action menu for a selected expression."""
    console.print(f"\n  [{BOLD}]Expression #{expr.id}[/] [{DIM}]— {_SOURCE_LABELS.get(expr.source_type, expr.source_type)}[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    options = [
        ("v", "View full content"),
        ("g", "Show graph context (parent, relations, inferences)"),
        ("f", "Focus source file in REPL") if prov.get("audio_file_id") else None,
        ("n", "Capture to active/context note"),
        ("a", "Use as context for .ask synthesis"),
        ("s", "Show surrounding expressions"),
        ("q", "Back to results"),
    ]
    options = [o for o in options if o is not None]

    for key, desc in options:
        console.print(f"    [{ACCENT}][{key}][/] {desc}")

    while True:
        try:
            choice = input("\n  → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if choice == "q":
            return

        elif choice == "v":
            console.print(f"\n  [{BOLD}]Full Content:[/]")
            console.print(f"  [{DIM}]{'─' * 50}[/]")
            # Word-wrap the content
            for line in expr.content.strip().split("\n"):
                console.print(f"  {line}")
            console.print()

        elif choice == "g":
            _show_graph_context(expr, expr_repo)

        elif choice == "f" and prov.get("audio_file_id"):
            from audiobench.cli.repl.dispatch import print_context_summary
            from audiobench.cli.repl.session import NavigationFrame
            from audiobench.core.focused_entity import FocusedEntity
            file_id = prov["audio_file_id"]
            file_name = prov.get("file_name", f"File #{file_id}")
            session.push_frame(NavigationFrame(context=".search", state={}, intent="focused from .search"))
            session.focus = FocusedEntity(type="file", id=file_id, label=file_name)
            print_context_summary(session)
            return  # Exit the action menu and return to REPL

        elif choice == "a":
            _chain_to_ask(session, expr)
            return



        elif choice == "n":
            from audiobench.cli.repl.backslash_commands import cmd_capture
            cmd_capture(session, f"expression:{expr.id}")
            return

        elif choice == "s":
            _show_surrounding(expr, expr_repo)

        else:
            console.print(f"  [{WARNING}]Unknown action. Choose from: {', '.join(k for k, _ in options)}[/]")


def _show_graph_context(expr, expr_repo) -> None:
    """Show parent expression, inbound/outbound relations, and linked inferences."""
    console.print(f"\n  [{BOLD}]Graph Context — Expression #{expr.id}[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    # Walk to parent
    parent = expr_repo.walk_to_parent(expr.id)
    if parent:
        parent_label = _SOURCE_LABELS.get(parent.source_type, parent.source_type)
        snippet = parent.content.strip().replace("\n", " ")[:100]
        console.print(f"  [{DIM}]Parent ({parent_label}):[/] \"{snippet}...\"")
    else:
        console.print(f"  [{DIM}]No parent expression.[/]")

    # Relations
    from audiobench.memory.enums import SourceType
    relations = expr_repo.get_relations(expr.id, direction="both")
    inferences = []
    other_relations = []
    for rel in relations:
        other_id = rel.to_expression_id if rel.from_expression_id == expr.id else rel.from_expression_id
        other = expr_repo.get_by_id(other_id)
        if other:
            if other.source_type == SourceType.SYSTEM_INFERENCE.value:
                inferences.append(other)
            else:
                other_relations.append((rel.relation_type, other))

    if inferences:
        console.print(f"\n  [{BOLD}]Linked Inferences:[/]")
        for inf in inferences[:3]:
            snippet = inf.content.strip().replace("\n", " ")[:120]
            console.print(f"  [{DIM}]⬡ \"{snippet}\"[/]")

    if other_relations:
        console.print(f"\n  [{BOLD}]Relations:[/]")
        for rel_type, other in other_relations[:5]:
            other_label = _SOURCE_LABELS.get(other.source_type, other.source_type)
            snippet = other.content.strip().replace("\n", " ")[:80]
            console.print(f"  [{DIM}]{rel_type} → ({other_label}) \"{snippet}...\"[/]")

    console.print()


def _show_surrounding(expr, expr_repo) -> None:
    """Show sibling expressions by walking to the parent and listing its children."""
    console.print(f"\n  [{BOLD}]Surrounding Context[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    parent = expr_repo.walk_to_parent(expr.id)
    if not parent:
        console.print(f"  [{DIM}]No parent — cannot show surrounding context.[/]")
        return

    # Find sibling expressions that share the same parent via SOURCE relations
    try:
        from audiobench.core.db_session import get_session as db_session
        from audiobench.storage.models import ExpressionRelation

        with db_session() as s:
            sibling_ids = (
                s.query(ExpressionRelation.from_expression_id)
                .filter_by(to_expression_id=parent.id, relation_type="source")
                .order_by(ExpressionRelation.created_at)
                .all()
            )
            sibling_ids = [row[0] for row in sibling_ids]

        current_pos = sibling_ids.index(expr.id) if expr.id in sibling_ids else -1

        for pos, sid in enumerate(sibling_ids):
            sib = expr_repo.get_by_id(sid)
            if not sib:
                continue
            snippet = sib.content.strip().replace("\n", " ")[:100]
            marker = f"[{ACCENT}]→[/] " if sid == expr.id else "  "
            console.print(f"  {marker}[{DIM}][{pos+1}] \"{snippet}\"[/]")

    except Exception as e:
        console.print(f"  [{WARNING}]Could not load siblings: {e}[/]")

    console.print()


def _chain_to_ask(session: ReplSession, expr) -> None:
    """Prompt for a question, then invoke .ask with this expression pre-loaded as context."""
    console.print(f"\n  [{BOLD}]Ask about this expression:[/]")
    console.print(f"  [{DIM}]This expression will be included as context in the synthesis.[/]\n")
    try:
        question = input("  ? ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if not question:
        return

    from audiobench.cli.repl.dispatch import dispatch_command
    if session.last_id:
        dispatch_command(session, ["ask", str(session.last_id), question])
    else:
        console.print(f"  [{DIM}]No focused transcript. Set focus with [{ACCENT}]\\focus <id>[/] first.[/]")


# ── Dynamic Data Masking (DDM) intercept ────────────────────

def _apply_ddm(
    enriched: list[tuple],
    session: ReplSession,
) -> tuple[list[tuple], int]:
    """Apply Dynamic Data Masking to search results.

    Searches the ENTIRE corpus (no WHERE filter — this is DDM, not RLS).
    After retrieval, results whose privacy_tier exceeds the session's
    effective_tier() have their content replaced with a redaction stub.

    Returns:
        (display_results, redacted_count) where display_results is a list
        of (expr_or_stub, score, prov, is_redacted) tuples.

    Design:
        - Public results (Tier 0) always visible.
        - Redacted items still appear in the list with their position and
          score so the user knows relevant-but-locked memories exist.
        - The stub tells the user exactly which \\unlock tier they need.
    """
    current_tier = session.effective_tier()
    display_results = []
    redacted_count = 0

    for expr, score, prov in enriched:
        expr_tier = getattr(expr, "privacy_tier", 0)
        if expr_tier <= current_tier:
            display_results.append((expr, score, prov, False))
        else:
            tier_labels = {1: "Relational", 2: "Intimate", 3: "Override"}
            tier_label = tier_labels.get(expr_tier, f"Tier {expr_tier}")
            # Build a minimal stub dict to carry the redaction metadata
            stub = {
                "_redacted": True,
                "privacy_tier": expr_tier,
                "tier_label": tier_label,
                "score": score,
                # Preserve provenance so the user can see WHAT source it's from
                "source_type": expr.source_type,
            }
            display_results.append((stub, score, prov, True))
            redacted_count += 1

    return display_results, redacted_count


@register_dot("search")
def dot_search(session: ReplSession, args: str) -> None:
    """Semantic search across the unified expression namespace.

    Usage:  .search "persistence through difficulty"
            .search "concept" --top-k 15
            .search "concept" --preset deep
            .search "concept" --source-type chat_message
    """
    import shlex

    if not args.strip():
        console.print(f"  [{DIM}]Usage: [{ACCENT}].search \"concept or idea\"[/][/]")
        return

    # Parse args
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()

    query = parts[0]
    top_k = 8
    preset = "balanced"
    source_type_filter: str | None = None
    no_fallback = False

    i = 1
    while i < len(parts):
        p = parts[i]
        if p == "--top-k" and i + 1 < len(parts):
            try:
                top_k = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        elif p == "--preset" and i + 1 < len(parts):
            preset = parts[i + 1]
            i += 2
        elif p == "--source-type" and i + 1 < len(parts):
            source_type_filter = parts[i + 1]
            i += 2
        elif p == "--no-fallback":
            no_fallback = True
            i += 1
        else:
            i += 1

    # Check daemon availability
    daemon_up = _is_daemon_available()

    from audiobench.storage.expression_repository import ExpressionRepository
    expr_repo = ExpressionRepository()

    if daemon_up:
        # ── Semantic path: LanceDB ────────────────────────────
        console.print(f"  [{DIM}]Searching the expression namespace...[/]")
        try:
            from audiobench.daemon.factory import get_daemon_client

            use_bm25 = preset in ("balanced", "deep")
            use_dense = True
            use_colbert = preset in ("balanced", "deep")

            client = get_daemon_client()
            raw_results = client.search(
                query=query,
                top_k=top_k,
                use_bm25=use_bm25,
                use_dense=use_dense,
                use_colbert=use_colbert,
            )
        except Exception as e:
            console.print(f"  [{WARNING}]Semantic search failed: {e}[/]")
            raw_results = []

        if not raw_results:
            console.print(f"  [{DIM}]No results for \"{query}\".[/]")
            return

        # Fetch ExpressionRecords and provenance
        enriched = []
        for r in raw_results:
            expr_id = r.get("expression_id")
            if not expr_id:
                continue
            expr = expr_repo.get_by_id(expr_id)
            if not expr:
                continue
            if source_type_filter and expr.source_type != source_type_filter:
                continue
            score = float(r.get("score", 0.0))
            prov = _resolve_provenance(expr)
            enriched.append((expr, score, prov))

        if not enriched:
            console.print(f"  [{DIM}]No results matched after filtering.[/]")
            return

        # ── DDM: mask results above session tier ────────────────────────────
        display_results, redacted_count = _apply_ddm(enriched, session)

        if redacted_count > 0:
            noun = "memory" if redacted_count == 1 else "memories"
            console.print(
                f"  [yellow]⚠[/]  [{WARNING}]{redacted_count} {noun} relevant but locked.[/] "
                f"[{DIM}]Run [{ACCENT}]\\unlock[/] for full access.[/]"
            )

        # Pass only non-redacted items to the interactive loop
        visible = [(e, s, p) for e, s, p, r in display_results if not r]
        _render_results_ddm(display_results, query, mode="semantic")
        _interactive_result_loop(session, visible, expr_repo)

    else:
        # ── Fallback: SQLite FTS ──────────────────────────────
        if no_fallback or preset == "deep":
            console.print(
                f"  [{WARNING}]Semantic engine unavailable and --no-fallback specified.[/]\n"
                f"  [{DIM}]Start the daemon: [{ACCENT}]audiobench daemon start[/][/]"
            )
            return

        console.print(
            f"  [yellow]⚡[/] Semantic engine unavailable — showing text search results instead.\n"
            f"  [{DIM}]audiobench daemon start[/] for full semantic search.\n"
        )

        from audiobench.storage.repository import TranscriptionRepository
        repo = TranscriptionRepository()
        fts_results = repo.search(query, limit=top_k)

        if not fts_results:
            console.print(f"  [{DIM}]No results for \"{query}\".[/]")
            return

        # Convert FTS results to the display format by building lightweight expr stubs
        _render_fts_fallback(fts_results, query)
        _interactive_fts_loop(session, fts_results)


def _render_fts_fallback(results: list[dict], query: str) -> None:
    """Render SQLite FTS results in a readable numbered list."""
    console.print(f"\n  [{BOLD}]Text Search Results — SQLite FTS5[/]")
    console.print(f"  [{DIM}]{'─' * 60}[/]")
    for i, r in enumerate(results, 1):
        file_name = r.get("file_name", f"Transcript #{r.get('id', '?')}")
        snippet = (r.get("text_preview") or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        console.print(f"  [{ACCENT}][{i}][/] [{ACCENT}]◎[/] transcript  \"{file_name}\"")
        if snippet:
            console.print(f"       [{DIM}]\"{snippet}\"[/]")
        console.print()
    console.print(f"  [{DIM}]{'─' * 60}[/]")


def _interactive_result_loop(session: ReplSession, enriched: list[tuple], expr_repo) -> None:
    """Let the user select a semantic result to inspect or act on."""
    while True:
        console.print(f"  [{DIM}]Select result [1-{len(enriched)}], [{ACCENT}]q[/] to quit[/]")
        try:
            choice = input("  → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("q", "quit", ""):
            return

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(enriched):
                expr, score, prov = enriched[idx]
                _action_menu(session, expr, prov, expr_repo)
                # If user focused, exit loop (REPL prompt has changed)
                if session.focus:
                    return
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(enriched)}.[/]")
        else:
            console.print(f"  [{WARNING}]Enter a number or q.[/]")


def _interactive_fts_loop(session: ReplSession, results: list[dict]) -> None:
    """Let the user select an FTS result to focus on."""
    while True:
        console.print(f"  [{DIM}]Select result [1-{len(results)}] to focus, [{ACCENT}]q[/] to quit[/]")
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
                audio_file_id = r.get("audio_file_id") or r.get("id")
                file_name = r.get("file_name", f"File #{audio_file_id}")
                if audio_file_id:
                    from audiobench.cli.repl.dispatch import print_context_summary
                    from audiobench.cli.repl.session import NavigationFrame
                    from audiobench.core.focused_entity import FocusedEntity
                    session.push_frame(NavigationFrame(context=".search", state={}, intent="focused from .search (FTS)"))
                    session.focus = FocusedEntity(type="file", id=audio_file_id, label=file_name)
                    print_context_summary(session)
                    return
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(results)}.[/]")
        else:
            console.print(f"  [{WARNING}]Enter a number or q.[/]")
