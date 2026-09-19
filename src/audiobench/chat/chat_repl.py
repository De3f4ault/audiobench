"""Interactive Chat REPL Loop."""

from __future__ import annotations

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table

from audiobench.chat.chat_session import ChatSession
from audiobench.chat.chat_store import ChatRepository
from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    CHAT_CODE_THEME,
    DIM,
    PROMPT,
    SUCCESS,
    WARNING,
    chat_console,
    console,
    error_panel,
)
import os
import re
import textwrap
from collections import defaultdict

from audiobench.core.db_engine import init_db
from audiobench.core.error_types import AudioBenchError
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.playback.controls import (
    PLAYBACK_COMMANDS,
    extract_timestamps,
    handle_playback_command,
    render_citations_banner,
    render_jump_bar,
    render_playback_status_line,
)
from audiobench.storage.repository import TranscriptionRepository

logger = get_logger("chat.repl")


def _short_source(file_path: str, max_len: int = 55) -> str:
    """Return a readable short name from a full file path.

    Strips audio extensions and leading 8-char hash prefixes (e.g. '760d70bc_Why Eye...' → 'Why Eye...').
    Uses middle truncation if the name exceeds max_len.
    """
    if not file_path:
        return "Unknown"
    name = os.path.basename(file_path)
    for ext in (".mp4", ".mp3", ".m4a", ".wav", ".webm", ".m4b", ".ogg", ".flac"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    if "_" in name and len(name.split("_")[0]) == 8:
        name = name.split("_", 1)[1]
    name = name.strip()
    if len(name) > max_len:
        half = (max_len - 1) // 2
        name = name[:half] + "…" + name[-half:]
    return name


def _fmt_timestamp(s: float) -> str:
    """Format seconds into m:ss or h:mm:ss."""
    s = max(0.0, s)
    total_sec = int(s)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    sec = total_sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

from audiobench.chat.command_meta import render_chat_help

# Deprecated alias for backwards compatibility
CHAT_HELP_TEXT = "Use render_chat_help() to display slash commands dynamically."



class ChatREPL:
    """Manages the interactive chat loop, slash commands, and rendering."""

    def __init__(
        self,
        session=None,
        tx_repo=None,
        chat_repo=None,
        settings=None,
        session_type: str = "chat",
        preloaded_fragments=None,
        preloaded_title: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        think: bool = True,
        resume_id: int | None = None,
        project_id: int | None = None,
        current_session_number: int | None = None,
    ):
        init_db()
        self.settings = settings or get_settings()
        self.tx_repo = tx_repo or TranscriptionRepository()
        self.chat_repo = chat_repo or ChatRepository()

        if session is None:
            model_name = model or self.settings.ollama_model
            if model_name.lower().startswith("gemini"):
                from audiobench.chat.providers.gemini_provider import GeminiClient
                client = GeminiClient(model=model_name)
            else:
                client = OllamaClient(
                    base_url=self.settings.ollama_base_url,
                    model=model_name,
                )
            self.session = ChatSession(
                client=client,
                chat_repo=self.chat_repo,
                model=model_name,
                temperature=temperature,
                conversation_id=resume_id,
                show_thinking=think,
            )
            if resume_id is not None:
                self.session.restore_from_db(tx_repo=self.tx_repo)
        else:
            self.session = session

        self.client = self.session._client
        self.temperature = self.session._temperature
        self.session_type = session_type
        self.preloaded_fragments = preloaded_fragments
        self.preloaded_title = preloaded_title

        # Ensure active model and provider are available, auto-falling back if missing
        if session is None and resume_id is not None and not model:
            self._ensure_model_available()
        self.project_id = project_id
        self.current_session_number = current_session_number
        self._last_hint_at: float = 0.0  # epoch timestamp of last BM25 hint

        # Create the initial conversation entry if preloaded_title or non-default session_type is specified
        if not self.session.conversation_id and (self.preloaded_title or self.session_type != "chat"):
            title = self.preloaded_title or "New Chat"
            self.session._conversation_id = self.chat_repo.create_conversation(
                model=self.session.model,
                title=title,
                session_type=self.session_type,
            )

        import time as _time
        from pathlib import Path as _Path
        self._time = _time
        self._Path = _Path

        from audiobench.cli.shared.repl_shell import make_prompt_session
        from audiobench.chat.chat_completer import ChatAutoSuggest, ChatSlashCompleter

        _history_file = self.settings.data_dir / "chat_history.txt"
        _history_file.parent.mkdir(parents=True, exist_ok=True)
        self._completer = ChatSlashCompleter(self)
        self._auto_suggest = ChatAutoSuggest()
        self._active_citations: list[tuple[str, float]] = []
        if resume_id is not None and self.session.messages:
            from audiobench.playback.controls import extract_timestamps

            for msg in reversed(self.session.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    cited = extract_timestamps(msg["content"])
                    if cited:
                        self._active_citations = cited
                        break

        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        def _buffer_is_empty(event) -> bool:
            return event.app.current_buffer.text == ""

        for _d in "123456789":
            @kb.add(_d)
            def _key_digit(event, digit=_d):
                if _buffer_is_empty(event) and self._active_citations and int(digit) <= len(self._active_citations):
                    event.app.exit(result=f"\x00{digit}")
                else:
                    event.current_buffer.insert_text(digit)

        @kb.add("right")
        def _key_right(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00NEXT")
            else:
                event.current_buffer.cursor_right()

        @kb.add("left")
        def _key_left(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00PREV")
            else:
                event.current_buffer.cursor_left()

        @kb.add("n")
        def _key_n(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00NEXT")
            else:
                event.current_buffer.insert_text("n")

        @kb.add("b")
        def _key_b(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00PREV")
            else:
                event.current_buffer.insert_text("b")

        @kb.add("p")
        def _key_p(event):
            if _buffer_is_empty(event) and self._daemon:
                event.app.exit(result="\x00SPACE")
            else:
                event.current_buffer.insert_text("p")

        @kb.add("]")
        def _key_bracket_right(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00NEXT")
            else:
                event.current_buffer.insert_text("]")

        @kb.add("[")
        def _key_bracket_left(event):
            if _buffer_is_empty(event) and self._active_citations:
                event.app.exit(result="\x00PREV")
            else:
                event.current_buffer.insert_text("[")

        @kb.add(" ")
        def _key_space(event):
            if _buffer_is_empty(event):
                event.app.exit(result="\x00SPACE")
            else:
                event.current_buffer.insert_text(" ")

        self._pt_session: PromptSession = make_prompt_session(
            history_path=_history_file,
            slash_completer=self._completer,
            auto_suggest=self._auto_suggest,
            key_bindings=kb,
        )

        try:
            from audiobench.daemon.factory import get_daemon_client
            self._daemon = get_daemon_client()
        except Exception as exc:
            logger.debug("Daemon not available for playback: %s", exc)
            self._daemon = None

    def _ensure_model_available(self) -> None:
        """Verify the active model and client are accessible, falling back gracefully if missing."""
        cur_m = (self.session.model or "").lower()
        client = getattr(self.session, "_client", None)
        if client is None:
            return

        if cur_m.startswith("gemini"):
            if not client.is_available():
                # Gemini client not available (e.g. missing GEMINI_API_KEY)
                from audiobench.chat.providers.ollama_provider import OllamaClient

                o_client = OllamaClient(base_url=self.settings.ollama_base_url)
                if o_client.is_available():
                    avail = o_client.list_models()
                    fallback = (
                        self.settings.ollama_model
                        if self.settings.ollama_model in avail
                        else (avail[0] if avail else self.settings.ollama_model)
                    )
                    console.print(
                        f"  [yellow]Notice: Gemini not available (missing GEMINI_API_KEY). Switched to '{fallback}'.[/]"
                    )
                    self.session.switch_model(fallback)
                    self.client = self.session._client
            return

        # Ollama client check
        if not client.is_available():
            from audiobench.chat.providers.gemini_provider import GeminiClient

            g_client = GeminiClient(model=self.settings.gemini_model)
            if g_client.is_available():
                console.print(
                    f"  [yellow]Notice: Ollama not reachable. Switched to Gemini ({self.settings.gemini_model}).[/]"
                )
                self.session.switch_model("gemini")
                self.client = self.session._client
            return

        # Ollama is reachable: check if model exists
        try:
            avail = client.list_models()
            if avail and self.session.model not in avail:
                fallback = (
                    self.settings.ollama_model
                    if self.settings.ollama_model in avail
                    else avail[0]
                )
                logger.info(
                    "Restored model '%s' not found on Ollama server. Falling back to '%s'",
                    self.session.model,
                    fallback,
                )
                self.session.switch_model(fallback)
                self.client = self.session._client
        except Exception:
            pass

    def _build_study_context(self) -> str:
        """Build a study context block from prior session memoirs.

        Compression degrades by age:
          - N-1 session: FULL (narrative + insights + threads)
          - N-2 to N-3:  DIGEST (truncated narrative + insights + threads)
          - N-4+:        KEY_ONLY (insights + threads only)

        Open threads are ALWAYS included at every compression level.
        Estimated token budget: < 8000 words.
        """
        if self.project_id is None:
            return ""


        from audiobench.core.db_session import get_session as _db
        from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir
        from audiobench.storage.models import ConversationSummary, StudySession

        current_n = self.current_session_number or 1

        with _db() as db:
            # Fetch all closed sessions for this project, ordered by session number
            prior_sessions = (
                db.query(StudySession)
                .filter(
                    StudySession.project_id == self.project_id,
                    StudySession.closed_at.isnot(None),
                )
                .order_by(StudySession.id)
                .all()
            )
            # For each session, load its memoir via ConversationSummary
            session_memoirs: list[tuple[int, Memoir]] = []
            for idx, s in enumerate(prior_sessions, 1):
                if s.memoir_id is None:
                    continue
                # Find ConversationSummary that linked to this expression
                cs = db.query(ConversationSummary).filter_by(
                    expression_id=s.memoir_id
                ).first()
                if cs is None:
                    continue
                memoir = Memoir(
                    narrative=cs.narrative,
                    key_insights=cs.key_insights,
                    open_threads=cs.open_threads,
                    refined_title=cs.refined_title,
                )
                session_memoirs.append((idx, memoir))

        if not session_memoirs:
            return ""

        parts: list[str] = []
        parts.append("# Prior Study Sessions\n")

        for session_num, memoir in session_memoirs:
            age = current_n - session_num  # 1 = N-1, 2 = N-2, ...
            if age == 1:
                level = CompressionLevel.FULL
            elif age <= 3:
                level = CompressionLevel.DIGEST
            else:
                level = CompressionLevel.KEY_ONLY

            title = memoir.refined_title or f"Session {session_num}"
            compressed = compress_memoir(memoir, level)
            parts.append(f"## Session {session_num}: {title}\n\n{compressed}\n")

        return "\n".join(parts)

    def _fetch_memory_hints(
        self,
        user_text: str,
        debounce_seconds: float = 30.0,
        top_k: int = 3,
    ) -> list[str]:
        """Silently run a BM25 search and return matching segment texts as hints.

        Only fires when:
          1. user_text ends with '?' (looks like a question)
          2. Enough time has passed since the last hint (debounce)

        Returns a list of short text hints (or empty list if suppressed).
        Never prints to stdout. Never raises.
        """
        import time

        # Only fire on questions
        stripped = user_text.strip()
        if not stripped.endswith("?"):
            return []

        # Debounce: don't fetch if a hint was already fetched recently
        now = time.time()
        if now - self._last_hint_at < debounce_seconds:
            return []

        try:
            from audiobench.memory.query_reformulator import ReformulatedQuery
            from audiobench.memory.retrieval_streams import FTS5Stream

            # Minimal reformulation: use the raw question as BM25 keywords
            # (strip the '?' and punctuation)
            keywords = stripped.rstrip("?").strip()
            rq = ReformulatedQuery(
                original=stripped,
                bm25_keywords=keywords,
                semantic_query=stripped,
                hyde_anchor=stripped,
            )
            hits = FTS5Stream().retrieve(rq, top_k=top_k)
            self._last_hint_at = now
            return [h.text for h in hits]
        except Exception:
            # Must never propagate
            return []

    # ── Transcript Picker Helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_load_ids(arg: str) -> list[int]:
        """Parse explicit numeric transcript IDs from /load arg string.

        Accepts:
          - Space or comma separated digits: "1182 1181" "1182,1181"
          - Hash-prefixed IDs: "#1182 #1181"
          - Ranges: "1180-1183"  (treated as IDs, not list positions)

        Returns empty list when *arg* contains any non-numeric word (i.e. a
        keyword query), so the caller knows to fall through to the picker.
        """
        if not arg or not arg.strip():
            return []

        import re  # local import — this is a @staticmethod with no module-level re
        tokens = [t.strip("#,").strip() for t in arg.replace(",", " ").split()]
        ids: list[int] = []
        for tok in tokens:
            if tok.isdigit():
                ids.append(int(tok))
            elif re.fullmatch(r"\d+-\d+", tok):
                lo, hi = tok.split("-")
                ids.extend(range(int(lo), int(hi) + 1))
            else:
                # Any non-numeric token → treat whole arg as a keyword query
                return []
        return ids

    def _show_transcript_picker(self, initial_query: str = "") -> list[int]:
        """Interactive transcript browser / picker for /load.

        Displays a paginated catalog of transcripts, supports live keyword
        filtering (searches SQLite directly — no memory load), multi-select,
        and erases the entire picker UI from the terminal upon exit so the
        chat scrollback stays clean.

        Returns:
            List of selected transcript IDs (may be empty if cancelled).
        """
        import math, sys as _sys

        def _fmt_dur(secs: float | None) -> str:
            if not secs:
                return "–"
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h {m:02d}m"
            return f"{m}m {s:02d}s"

        def _fmt_date(iso: str) -> str:
            if not iso:
                return ""
            return iso[:10]  # YYYY-MM-DD

        def _render_catalog(rows: list[dict], query: str, term_width: int) -> int:
            """Print the catalog table and prompt. Returns number of lines printed."""
            label = f"filter: '{query}'" if query else "recent"
            header = f"  ── Transcripts ({label} · {len(rows)} found) "
            bar_fill = "─" * max(0, term_width - len(header) - 2)
            console.print(f"[dim]{header}{bar_fill}[/dim]")

            if not rows:
                console.print(f"  [{DIM}]No transcripts found.[/]")
            else:
                # Calculate column widths
                name_max = max(20, term_width - 52)
                for i, row in enumerate(rows, 1):
                    name = row["file_name"]
                    if len(name) > name_max:
                        name = name[:name_max - 1] + "…"
                    words = f"{row['word_count']:,}w" if row.get("word_count") else ""
                    dur = _fmt_dur(row.get("duration"))
                    dt = _fmt_date(row.get("created_at", ""))
                    console.print(
                        f"  [{ACCENT}][{i:>2}][/] [dim]#{row['id']:<5}[/] {name:<{name_max}} "
                        f"[dim]{words:>7}  {dur:>8}  {dt}[/]"
                    )

            console.print()
            console.print(
                f"  [{DIM}]Pick [bold]1-{len(rows)}[/bold], "
                f"[bold]IDs[/bold] (e.g. #1182), "
                f"[bold]1-3[/bold] range, "
                f"type a [bold]keyword[/bold] to filter, "
                f"or [bold]0[/bold]/[bold]q[/bold] to cancel:[/]"
            )

            # Count lines rendered (used for ANSI erase)
            # header + rows + blank + prompt line = len(rows) + 3
            return len(rows) + 3

        query = initial_query.strip()
        chosen_ids: list[int] = []

        while True:
            rows = self.tx_repo.search_transcripts(query=query, limit=15)
            tw = console.width or 100

            # Measure rendered height precisely via capture
            with console.capture() as _cap:
                _render_catalog(rows, query, tw)
                console.print("  › ")
            _render_lines = _cap.get().count("\n") + 1  # +1 for the actual input line

            # Actually render
            _render_catalog(rows, query, tw)
            console.print("  › ", end="", highlight=False)

            try:
                raw = input("").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()
                raw = ""

            # Erase picker from terminal (cursor up N lines then clear to bottom)
            console.file.write(f"\033[{_render_lines}A\033[J")
            console.file.flush()

            if not raw or raw in ("0", "q", "Q", "cancel"):
                break  # cancelled

            if raw.lower() in ("all", "clear", "reset"):
                query = ""
                continue

            # Try to parse as numeric selection (list positions, IDs, ranges)
            parsed_ids = self._parse_load_ids(raw)

            # Distinguish list-position picks (small numbers ≤ len(rows)) from raw IDs
            if parsed_ids:
                for n in parsed_ids:
                    if 1 <= n <= len(rows):
                        # List position → resolve to real DB id
                        chosen_ids.append(rows[n - 1]["id"])
                    elif any(r["id"] == n for r in rows):
                        # Matches a #ID in current visible list
                        chosen_ids.append(n)
                    else:
                        # Treat as a raw DB id (may not be in current page)
                        chosen_ids.append(n)
                break  # done selecting

            # Non-numeric → treat as a new keyword filter
            query = raw

        return chosen_ids

    def _render_search_picker(
        self,
        frags: list,
        scope_label: str,
        query: str,
    ) -> list:
        """Render search results in Book Mode (2 columns when wide, single when narrow).

        Matches the exact Book Mode layout from audiobench search (memory_cmd.py).
        Prompts user to select fragments to inject into chat context.
        Erases the search results display upon selection/exit via ANSI escape codes
        so the REPL history remains clean.
        """
        term_width = console.size.width or 80

        # Group fragments by source
        groups: dict[str, list[tuple[int, object]]] = defaultdict(list)
        for idx, fr in enumerate(frags, 1):
            sf = fr.source_file or f"Transcript #{fr.transcription_id}"
            groups[sf].append((idx, fr))

        # Outer ranking: sum RRF scores per source
        ordered_sources = sorted(
            groups.keys(),
            key=lambda sf: sum(getattr(fr, "rrf_score", 0.0) for _, fr in groups[sf]),
            reverse=True,
        )

        use_book = term_width >= 100 and len(ordered_sources) >= 2

        def _render_group(source_file: str, wrap_w: int) -> tuple[str, list[str]]:
            group = groups[source_file]
            group_sorted = sorted(group, key=lambda x: getattr(x[1], "start_time", 0.0))
            source_name = _short_source(source_file)
            n_group = len(group_sorted)
            header = f"[dim]──[/dim] [bold]{source_name}[/bold]  [dim]{n_group} fragment{'s' if n_group != 1 else ''}[/dim]"
            body: list[str] = []
            for frag_pos, (global_idx, fr) in enumerate(group_sorted):
                ts = f"[cyan]{_fmt_timestamp(fr.start_time)} → {_fmt_timestamp(fr.end_time)}[/cyan]"
                score = f"[dim]score {getattr(fr, 'rrf_score', 0.0):.3f}[/dim]"
                body.append(f"  [bold cyan][{global_idx}][/bold cyan]  {ts}  {score}")
                raw_text = fr.text.replace("\n", " ").strip()
                wrapped = textwrap.fill(raw_text, width=wrap_w)
                for t_line in wrapped.split("\n"):
                    body.append(f"     {t_line}")
                if frag_pos < len(group_sorted) - 1:
                    body.append("")
            return header, body

        content_lines: list[str] = []
        left_groups: list[str] = []
        right_groups: list[str] = []

        if use_book:
            # Balance sources between left/right columns by estimated height
            col_w_raw = max(40, (term_width - 3) // 2)
            left_h = 0
            right_h = 0
            for sf in ordered_sources:
                _, body = _render_group(sf, col_w_raw - 4)
                h = len(body) + 2
                if left_h <= right_h:
                    left_groups.append(sf)
                    left_h += h
                else:
                    right_groups.append(sf)
                    right_h += h
        else:
            wrap_width = min(term_width - 6, 90)
            for sf in ordered_sources:
                hdr, body = _render_group(sf, wrap_width)
                content_lines.append(hdr)
                content_lines.extend(body)
                content_lines.append("")

        border_w = min(term_width - 4, 110)
        mode_label = "2-column book mode" if use_book else "single column"

        with console.capture() as capture:
            console.print()
            console.print(
                f"  [{ACCENT}]Search results[/] [{DIM}]({scope_label} · {len(frags)} found · {mode_label})[/]"
            )
            console.print(f"  [{DIM}]{'─' * border_w}[/]")

            if use_book:
                col_w = max(40, (term_width - 6) // 2)

                def _col_text(source_files: list[str]) -> str:
                    parts: list[str] = []
                    for sf in source_files:
                        hdr, body = _render_group(sf, col_w - 4)
                        parts.append(hdr)
                        parts.extend(body)
                        parts.append("")
                    return "\n".join(parts)

                left_text = _col_text(left_groups)
                right_text = _col_text(right_groups)

                tbl = Table.grid(padding=(0, 1))
                tbl.add_column(width=col_w, no_wrap=False)
                tbl.add_column(width=1, style=DIM)   # separator column
                tbl.add_column(width=col_w, no_wrap=False)

                # Zip rows so the separator │ appears on every line
                left_rows = left_text.split("\n")
                right_rows = right_text.split("\n")
                max_rows = max(len(left_rows), len(right_rows))
                left_rows += [""] * (max_rows - len(left_rows))
                right_rows += [""] * (max_rows - len(right_rows))

                for l_row, r_row in zip(left_rows, right_rows):
                    tbl.add_row(l_row, "│", r_row)

                console.print(tbl)
            else:
                for line in content_lines:
                    console.print(f"  {line}")

            console.print(f"  [{DIM}]{'─' * border_w}[/]")
            console.print(
                f"  [{DIM}]Inject into context? "
                f"[[bold]1-{len(frags)}[/bold], [bold]all[/bold], "
                f"[bold]none[/bold], [bold]p<N>[/bold] preview / Enter to skip]: [/]",
                end="",
            )

        rendered = capture.get()
        _render_lines = rendered.count("\n") + 1

        console.file.write(rendered)
        console.file.flush()
        try:
            pick_raw = input("").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            pick_raw = ""

        # Erase search results from terminal (same as /load picker)
        console.file.write(f"\033[{_render_lines}A\033[J")
        console.file.flush()

        if pick_raw.lower().startswith("p") and pick_raw[1:].strip().isdigit():
            p_idx = int(pick_raw[1:].strip())
            if 1 <= p_idx <= len(frags):
                target_fr = frags[p_idx - 1]
                if self._daemon:
                    start_s = getattr(target_fr, "start_time", 0.0)
                    src_f = getattr(target_fr, "source_file", None)
                    tx_id = getattr(target_fr, "transcription_id", None)
                    if src_f and os.path.exists(src_f):
                        self._daemon.playback_play(
                            src_f,
                            start_pos=start_s,
                            transcription_id=tx_id,
                        )
                    elif tx_id:
                        rec = self.tx_repo.get_by_id(tx_id)
                        if rec and rec.get("file_path"):
                            self._daemon.playback_play(
                                rec["file_path"],
                                start_pos=start_s,
                                audio_file_id=rec.get("audio_file_id"),
                                transcription_id=tx_id,
                            )
                console.print(f"  [{SUCCESS}]✓ Previewing [{p_idx}] from {_fmt_timestamp(getattr(target_fr, 'start_time', 0.0))}[/]")
                return self._render_search_picker(frags, scope_label, query)

        if not pick_raw or pick_raw.lower() in ("none", "n", "0", "skip", "q", "cancel"):
            return []

        if pick_raw.lower() == "all":
            return frags

        parsed_nums = self._parse_load_ids(pick_raw)
        to_inject = []
        for n in parsed_nums:
            if 1 <= n <= len(frags):
                to_inject.append(frags[n - 1])

        return to_inject

    def _render_context_view(self) -> None:
        """Display full breakdown of loaded transcripts and injected search fragments."""
        details = self.session.get_context_details()
        txs = details["transcripts"]
        frags = details["fragments"]

        console.print()
        # 1. Loaded Transcripts section
        if txs:
            tx_header = (
                f"  [{ACCENT}]Loaded Transcripts[/] "
                f"[{DIM}]({len(txs)} file{'s' if len(txs) != 1 else ''} · "
                f"{details['total_words']:,} words · limit {details['max_words']:,})[/]"
            )
            console.print(tx_header)
            console.print(f"  [{DIM}]{'─' * 60}[/]")
            for t in txs:
                skip_tag = "  [yellow][!] [context limit reached - skipped from prompt][/yellow]" if t["skipped"] else ""
                ts_badge = f"  [{SUCCESS}]⏱ timestamps active[/]" if t.get("has_timestamps") else f"  [{DIM}](no timestamps)[/]"
                console.print(
                    f"    [{DIM}]#{t['id']}[/] {t['file_name']} "
                    f"[{DIM}]({t['word_count']:,} words)[/]{ts_badge}{skip_tag}"
                )
        else:
            console.print(f"  [{DIM}]No transcripts loaded (use /load <ID>)[/]")

        # 2. Injected Search Fragments section
        console.print()
        if frags:
            console.print(
                f"  [{ACCENT}]Injected Search Fragments[/] "
                f"[{DIM}]({len(frags)} fragment{'s' if len(frags) != 1 else ''} · in prompt context)[/]"
            )
            console.print(f"  [{DIM}]{'─' * 60}[/]")
            for i, f in enumerate(frags, 1):
                src = _short_source(f["source_file"])
                query_str = f" [dim](from query: [italic]{f.get('query', '')}[/italic])[/dim]" if f.get("query") else ""
                console.print(
                    f"    [{ACCENT}][{i}][/] [{BOLD}]{src}[/] · [{DIM}]{f['timestamp']}[/]{query_str}"
                )
                text = f["text"].replace("\n", " ").strip()
                wrapped = textwrap.fill(text, width=min(console.size.width - 10, 85))
                for line in wrapped.split("\n"):
                    console.print(f"        [{DIM}]{line}[/]")
                console.print()
            console.print(f"  [{DIM}]Tip: Use /unsearch or /context clear fragments to remove injected snippets.[/]")
        else:
            console.print(f"  [{DIM}]No search fragments injected (use /search <query> to find & inject snippets)[/]")
        console.print()

    def _show_unfocus_picker(self) -> list[int]:
        """Interactive picker to select transcripts to remove from context.

        Lists all currently loaded transcripts with their index [1..N], word counts,
        and flags which ones hit the context limit. Erases itself upon selection.
        """
        details = self.session.get_context_details()
        txs = details["transcripts"]
        if not txs:
            console.print(f"  [{DIM}]No transcripts currently loaded in context.[/]")
            return []

        border_w = min(console.size.width - 4, 80)
        catalog_lines = [
            "",
            f"  [{ACCENT}]Loaded Transcripts in Context[/] "
            f"[{DIM}]({len(txs)} files · {details['total_words']:,} words · limit {details['max_words']:,})[/]",
            f"  [{DIM}]{'─' * border_w}[/]",
        ]

        for idx, t in enumerate(txs, 1):
            tag = "  [yellow][!] [skipped: limit reached][/yellow]" if t["skipped"] else "  [green][+] in prompt[/green]"
            name = _short_source(t["file_name"], max_len=45)
            catalog_lines.append(
                f"  [{ACCENT}][{idx:2d}][/] [{DIM}]#{t['id']:<5}[/] {name:<45} [{DIM}]({t['word_count']:,} words)[/]{tag}"
            )

        catalog_lines.append(f"  [{DIM}]{'─' * border_w}[/]")

        with console.capture() as capture:
            for line in catalog_lines:
                console.print(line)
            console.print(
                f"  [{DIM}]Unfocus [[bold]1-{len(txs)}[/bold], [bold]all[/bold], "
                f"[bold]skipped[/bold], [bold]0[/bold] cancel]: [/]",
                end="",
            )

        rendered = capture.get()
        _render_lines = rendered.count("\n") + 1

        console.file.write(rendered)
        console.file.flush()
        try:
            raw = input("").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            raw = ""

        # Erase picker from terminal (cursor up N lines then clear to bottom)
        console.file.write(f"\033[{_render_lines}A\033[J")
        console.file.flush()

        if not raw or raw in ("0", "q", "cancel"):
            return []

        if raw.lower() == "all":
            return [t["id"] for t in txs]

        if raw.lower() in ("skipped", "skip"):
            return [t["id"] for t in txs if t["skipped"]]

        parsed_ids = self._parse_load_ids(raw)
        chosen_ids = []
        for n in parsed_ids:
            if 1 <= n <= len(txs):
                chosen_ids.append(txs[n - 1]["id"])
            elif any(t["id"] == n for t in txs):
                chosen_ids.append(n)

        return chosen_ids

    def _handle_slash_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        parts = cmd.strip().split(None, 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/exit", "/quit", "/q"):
            return True

        elif command == "/help":
            render_chat_help()

        elif command == "/context":
            if arg and arg.strip().isdigit():
                tid = int(arg.strip())
                record = self.tx_repo.get_by_id(tid)
                if not record:
                    console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                    return False
                self.session.load_transcripts([record])
                console.print(
                    f"  [{SUCCESS}]✓ Loaded #{tid} "
                    f"{record['file_name']} "
                    f"({record['word_count']:,} words)[/]"
                )
            elif arg.strip().lower() in ("clear fragments", "clear search", "unsearch", "clear-fragments"):
                count = self.session.clear_injected_fragments()
                console.print(f"  [{SUCCESS}]✓ Cleared {count} injected search fragment(s)[/]")
            else:
                self._render_context_view()

        elif command in ("/unsearch", "/clear-search", "/clear-fragments"):
            count = self.session.clear_injected_fragments()
            console.print(f"  [{SUCCESS}]✓ Cleared {count} injected search fragment(s)[/]")

        elif command in ("/load", "/focus", "/browse"):
            # Collect explicit numeric IDs from arg, e.g. "/load 1182 1181 #1180"
            explicit_ids = self._parse_load_ids(arg)
            if explicit_ids:
                # Direct ID load — no picker needed
                for tid in explicit_ids:
                    record = self.tx_repo.get_by_id(tid)
                    if not record:
                        console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                    else:
                        self.session.load_transcripts([record])
                        console.print(
                            f"  [{SUCCESS}]✓ Loaded #{tid} "
                            f"{record['file_name']} "
                            f"({record['word_count']:,} words)[/]"
                        )
            else:
                # No explicit IDs — use the interactive picker (with optional keyword pre-filter)
                initial_query = arg.strip() if arg else ""
                chosen_ids = self._show_transcript_picker(initial_query)
                for tid in chosen_ids:
                    record = self.tx_repo.get_by_id(tid)
                    if not record:
                        console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                    else:
                        self.session.load_transcripts([record])
                        console.print(
                            f"  [{SUCCESS}]✓ Loaded #{tid} "
                            f"{record['file_name']} "
                            f"({record['word_count']:,} words)[/]"
                        )

        elif command == "/clear":
            self.session.clear_history()
            console.print(
                f"  [{SUCCESS}]✓ Conversation cleared (new session #{self.session.conversation_id})[/]"
            )
            console.print(f"  [{DIM}]Context reset — use /load <ID> to add transcripts[/]")

        elif command in ("/remove", "/unfocus"):
            if not self.session._transcripts:
                console.print(f"  [{DIM}]No transcripts currently loaded in context.[/]")
                return False

            raw_arg = arg.strip().lower()
            if raw_arg == "all":
                ids_to_remove = list(self.session._transcript_ids)
            elif raw_arg in ("skipped", "skip"):
                details = self.session.get_context_details()
                ids_to_remove = [t["id"] for t in details["transcripts"] if t["skipped"]]
            elif raw_arg:
                # Direct numeric IDs/ranges passed, e.g. "/unfocus 1360 1359" or "#1360"
                parsed = self._parse_load_ids(raw_arg)
                ids_to_remove = [n for n in parsed if n in self.session._transcript_ids]
                if not ids_to_remove:
                    console.print(f"  [{DIM}]None of the specified IDs are in context.[/]")
                    return False
            else:
                # Interactive picker
                ids_to_remove = self._show_unfocus_picker()

            if not ids_to_remove:
                return False

            removed = self.session.remove_transcripts(ids_to_remove)
            if removed:
                console.print(
                    f"  [{SUCCESS}]✓ Unfocused {len(removed)} transcript(s) from context[/] "
                    f"[{DIM}](#{', #'.join(str(i) for i in removed)})[/]"
                )
            else:
                console.print(f"  [{DIM}]No matching transcripts found in context.[/]")

        elif command == "/model":
            target = arg.strip() if arg else ""

            if not target or target.lower() == "list":
                from audiobench.chat.providers.ollama_provider import AIError, OllamaClient

                current = self.session.model
                try:
                    client = OllamaClient(base_url=self.settings.ollama_base_url)
                    models = client.list_models()
                except AIError as e:
                    console.print(f"  [yellow]Ollama not reachable ({e}) — keeping current: {current}[/]")
                    return False
                except Exception as e:
                    console.print(f"  [red]Error listing models: {e}[/]")
                    return False

                gemini_avail = bool(self.settings.gemini_api_key)
                gemini_model = self.settings.gemini_model

                console.print()
                console.print("  [bold]Available Models:[/bold]")
                for i, m in enumerate(models, 1):
                    marker = "  [green]← current[/green]" if m == current else ""
                    console.print(f"    [[bold cyan]{i}[/bold cyan]] {m}{marker}")

                if gemini_avail:
                    g_marker = "  [green]← current[/green]" if current.lower().startswith("gemini") else ""
                    console.print(f"    [[bold cyan]G[/bold cyan]] Gemini ([dim]{gemini_model}[/dim]){g_marker}")

                console.print(f"    [[bold cyan]0[/bold cyan]] Keep current ([dim]{current}[/dim])")
                console.print()

                try:
                    raw = input("    › Pick model [0]: ").strip()
                except (KeyboardInterrupt, EOFError):
                    console.print()
                    return False

                if not raw or raw == "0":
                    console.print(f"  [{DIM}]Kept current model: {current}[/]")
                    return False

                if raw.lower() in ("g", "gemini"):
                    target = "gemini"
                elif raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(models):
                        target = models[idx]
                    else:
                        console.print("  [red]Invalid choice — must be a listed number or 'G'[/red]")
                        return False
                elif raw in models:
                    target = raw
                else:
                    prefix_matches = [m for m in models if m.startswith(raw)]
                    if len(prefix_matches) == 1:
                        target = prefix_matches[0]
                    else:
                        console.print(f"  [red]Unrecognized model: '{raw}'[/red]")
                        return False

            elif target.lower() in ("ollama", "default"):
                target = self.settings.ollama_model
            elif target.lower() == "gemini":
                target = "gemini"

            try:
                self.session.switch_model(target)
                self.client = self.session._client
                console.print(f"  [{SUCCESS}]✓ Switched to {self.session.model}[/]")
            except RuntimeError as exc:
                console.print(f"  [red]✗ Could not switch model:[/] {exc}")


        elif command == "/think":
            self.session.show_thinking = not self.session.show_thinking
            state = "on" if self.session.show_thinking else "off"
            console.print(f"  [{SUCCESS}]✓ Thinking display: {state}[/]")

        elif command == "/history":
            convs = self.chat_repo.list_conversations(limit=10)
            if not convs:
                console.print(f"  [{DIM}]No past conversations[/]")
                return False
            console.print()
            for c in convs:
                tid_list = c.get("transcript_ids", [])
                ctx = f" (transcripts: {tid_list})" if tid_list else ""
                console.print(
                    f"    [{ACCENT}]#{c['id']}[/] "
                    f"{c['title']} "
                    f"[{DIM}]({c['message_count']} msgs, "
                    f"{c['model']}){ctx}[/]"
                )
            console.print()

        elif command in ("/sessions", "/switch"):
            recent_chats = self.chat_repo.list_conversations(limit=20, session_type="chat")
            if not recent_chats:
                console.print(f"  [{DIM}]No recent sessions[/]")
                return False

            from audiobench.cli.shared.session_picker import show_session_picker
            picker_sessions = [
                {
                    "id": c["id"],
                    "title": c["title"],
                    "created_at": c["created_at"],
                    "detail": f"{c['message_count']} msgs, {c['model']}",
                }
                for c in recent_chats
            ]
            try:
                chosen_id = show_session_picker(console, picker_sessions, noun="conversation")
            except KeyboardInterrupt:
                return False

            if chosen_id is not None and chosen_id != self.session.conversation_id:
                self.session = ChatSession(
                    client=self.session._client,
                    chat_repo=self.chat_repo,
                    model=self.session.model,
                    temperature=self.session._temperature,
                    conversation_id=chosen_id,
                    show_thinking=self.session.show_thinking,
                )
                self.session.restore_from_db(tx_repo=self.tx_repo)
                console.print(f"  [{SUCCESS}]✓ Switched to conversation #{chosen_id}[/]")
            return False

        elif command == "/rename":
            if self.session.conversation_id is None:
                console.print(f"  [{DIM}]Cannot rename: start a conversation first[/]")
                return False
            
            new_title = arg.strip()
            if not new_title:
                # Ask LLM to auto-rename based on recent messages
                if not self.session._messages:
                    console.print(f"  [{DIM}]Cannot auto-rename: conversation is empty. Usage: /rename <title>[/]")
                    return False
                
                last_user = next((m["content"] for m in reversed(self.session._messages) if m["role"] == "user"), "")
                last_assistant = next((m["content"] for m in reversed(self.session._messages) if m["role"] == "assistant"), "")
                
                with console.status(f"[{DIM}]Generating title...[/]"):
                    from audiobench.chat.context_builder import TITLE_PROMPT
                    prompt = TITLE_PROMPT.format(
                        first_message=last_user[:500],
                        first_response=last_assistant[:500]
                    )
                    try:
                        if hasattr(self.session._client, "generate"):
                            new_title = self.session._client.generate(
                                prompt=prompt,
                                temperature=0.3,
                            ).strip()
                        else:
                            result = self.session._client.chat(
                                [{"role": "user", "content": prompt}], temperature=0.3
                            )
                            new_title = result.get("content", "").strip()
                        if not new_title:
                            raise ValueError("Empty response")
                    except Exception as e:
                        console.print(f"  [{DIM}]Auto-rename failed: {e}. Usage: /rename <title>[/]")
                        return False

            self.chat_repo.update_title(self.session.conversation_id, new_title)
            console.print(f"  [{SUCCESS}]✓ Renamed to: {new_title}[/]")
            return False

        elif command == "/export":
            if not self.session.messages:
                console.print(f"  [{DIM}]Nothing to export yet[/]")
                return False
            fname = arg.strip() if arg.strip() else None
            if not fname:
                slug = f"chat_{self.session.conversation_id or 'new'}_{int(self._time.time())}"
                fname = f"{slug}.md"
            path = self._Path(fname).expanduser()
            lines = [f"# Chat #{self.session.conversation_id or 'new'}\n"]
            lines.append(f"Model: {self.session.model}  \n")
            lines.append("---\n")
            for msg in self.session.messages:
                if msg["role"] == "user":
                    lines.append(f"**You:** {msg['content']}\n")
                elif msg["role"] == "assistant":
                    lines.append(f"**AI:**\n\n{msg['content']}\n")
                lines.append("---\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            console.print(f"  [{SUCCESS}]✓ Exported to {path}[/]")

        elif command == "/retry":
            self.session._retry_requested = True
            return False

        elif command == "/compare":
            if not arg:
                cmp_model = getattr(self.session, "_compare_model", None)
                if cmp_model:
                    console.print(
                        f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                        f"  [{DIM}]Primary:   {self.session.model}[/]\n"
                        f"  [{DIM}]Secondary: {cmp_model}[/]\n"
                        f"  [{DIM}]Use /compare off to disable[/]"
                    )
                else:
                    console.print(
                        f"  [{DIM}]Comparison mode is OFF[/]\n"
                        f"  [{DIM}]Usage: /compare <model> to enable[/]\n"
                        f"  [{DIM}]Example: /compare qwen4-next:110b-cloud[/]"
                    )
                return False
            if arg.strip().lower() == "off":
                old = getattr(self.session, "_compare_model", None)
                self.session._compare_model = None
                if old:
                    console.print(
                        f"  [{SUCCESS}]✓ Comparison mode OFF[/] [{DIM}](was comparing with {old})[/]"
                    )
                else:
                    console.print(f"  [{DIM}]Comparison mode was already off[/]")
                return False
            new_model = arg.strip()
            old = getattr(self.session, "_compare_model", None)
            self.session._compare_model = new_model
            if old and old != new_model:
                console.print(
                    f"  [{ACCENT}]⚡ Switched comparison:[/] [{DIM}]{old}[/] → [{BOLD}]{new_model}[/]"
                )
            else:
                console.print(
                    f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                    f"  [{DIM}]Every prompt will compare {self.session.model} vs {new_model}[/]\n"
                    f"  [{DIM}]/compare off to disable[/]"
                )
            return False

        elif command == "/bookmarks":
            from audiobench.core.db_engine import init_db
            from audiobench.storage.bookmark_repository import (
                BOOKMARK_TYPES,
                BookmarkRepository,
            )
            from audiobench.storage.bookmark_repository import (
                _format_timestamp as _bfmt,
            )

            init_db()
            bm_repo = BookmarkRepository()

            if arg and arg.strip().isdigit():
                tid = int(arg.strip())
                record = self.tx_repo.get_by_id(tid)
                if not record:
                    console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                    return False
                audio_id = record.get("audio_file_id")
                if not audio_id:
                    console.print(f"  [{DIM}]No audio file linked to #{tid}[/]")
                    return False
                bookmarks = bm_repo.list_for_file(audio_id)
                label = f"#{tid} {record.get('file_name', '')}"
            else:
                bookmarks = bm_repo.list_all(limit=15)
                label = "All files"

            if not bookmarks:
                console.print(f"  [{DIM}]No bookmarks found[/]")
                return False

            console.print()
            console.print(f"  [{ACCENT}]Bookmarks — {label}[/]")
            for b in bookmarks:
                emoji = BOOKMARK_TYPES.get(b["bookmark_type"], "🔖")
                time_str = _bfmt(b["timestamp"])
                if b.get("is_region") and b.get("end_timestamp"):
                    time_str += f"→{_bfmt(b['end_timestamp'])}"
                console.print(f"    [{DIM}]#{b['id']}[/] {emoji} {time_str}  {b['name'][:40]}")
            console.print()

        elif command == "/autocomplete":
            raw = arg.strip().lower()
            current_mode = getattr(self.settings, "chat_autocomplete_verbosity", "brief")
            valid_modes = ("minimal", "brief", "detailed")

            if not raw:
                console.print(
                    f"\n  [{ACCENT}]Autocomplete Mode:[/] [bold]{current_mode}[/]\n"
                    f"  [{DIM}]Available verbosity levels:[/]\n"
                    f"    [{BOLD}]minimal[/]   - command names only (no descriptions)\n"
                    f"    [{BOLD}]brief[/]     - command names + one-line tldr summary (default)\n"
                    f"    [{BOLD}]detailed[/]  - command names + parameter synopsis + summary\n\n"
                    f"  [{DIM}]Switch mode: /autocomplete [minimal|brief|detailed][/]\n"
                )
                return False

            if raw not in valid_modes:
                console.print(
                    f"  [{DIM}]Invalid mode: '{raw}'. Choose from: {', '.join(valid_modes)}[/]"
                )
                return False

            self.settings.chat_autocomplete_verbosity = raw
            try:
                self.settings.save()
                console.print(
                    f"  [{SUCCESS}][+] Autocomplete mode set to [bold]{raw}[/bold] (saved to settings.json)[/]"
                )
            except Exception as exc:
                console.print(
                    f"  [{SUCCESS}][+] Autocomplete mode set to [bold]{raw}[/bold] (session only: {exc})[/]"
                )
            return False

        elif command in ("/search", "/find"):
            # Universal inline search.
            #   /search <query>              → scoped to loaded transcripts
            #   /search --all <query>        → global library search
            #   /search ! <query>            → global (shorthand)
            #   /search -n 8 <query>         → limit results to 8
            #   /search --top 8 <query>      → same as -n
            #   Flags may be combined: /search --all -n 3 <query>
            raw_arg = arg.strip()
            global_scope = False
            top_k = getattr(self.settings, "chat_search_default_k", 8)

            # Parse flags from front
            tokens = raw_arg.split()
            query_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok in ("--all", "!"):
                    global_scope = True
                elif tok in ("-n", "--top") and i + 1 < len(tokens):
                    try:
                        top_k = max(1, min(int(tokens[i + 1]), 20))
                        i += 1
                    except ValueError:
                        query_tokens.append(tok)
                else:
                    query_tokens.append(tok)
                i += 1

            raw_arg = " ".join(query_tokens).strip()

            if not raw_arg:
                console.print(
                    f"  [{DIM}]Usage: /search [-n <k>] [--all] <query>[/]\n"
                    f"  [{DIM}]  --all / !   search entire library (not just loaded transcripts)[/]\n"
                    f"  [{DIM}]  -n <k>      show up to k results (default {top_k}, max 20)[/]"
                )
                return False

            scoped_ids = self.session._transcript_ids if not global_scope else []
            if global_scope or not scoped_ids:
                scope_label = "global library"
            elif len(scoped_ids) <= 5:
                scope_label = f"scoped to #{', #'.join(str(i) for i in scoped_ids)}"
            else:
                scope_label = f"scoped to {len(scoped_ids)} loaded transcripts"

            with console.status(f"  [{DIM}]Searching ({scope_label})...[/]"):
                try:
                    from audiobench.memory.query_engine import ResearchEngine
                    engine = ResearchEngine()
                    result = engine.search(raw_arg, top_k=top_k * 3, preset="fast", skip_synthesis=True)
                    frags = result.sources or []
                    # Post-filter to loaded transcripts when in targeted scope
                    if scoped_ids and not global_scope:
                        frags = [
                            f for f in frags
                            if f.transcription_id in scoped_ids
                        ]
                    frags = frags[:top_k]
                except Exception as exc:
                    console.print(f"  [{DIM}]Search failed: {exc}[/]")
                    return False

            if not frags:
                if scoped_ids and not global_scope:
                    console.print(
                        f"  [{DIM}]No matches found in active transcripts "
                        f"(#{', #'.join(str(i) for i in scoped_ids)}).\n"
                        f"  Tip: use /search --all {raw_arg!r} to search your entire library.[/]"
                    )
                else:
                    console.print(f"  [{DIM}]No results found for: {raw_arg!r}[/]")
                return False

            to_inject = self._render_search_picker(frags, scope_label, raw_arg)
            if to_inject:
                self.session.inject_search_fragments(to_inject, query=raw_arg)
                sources_str = ", ".join(dict.fromkeys(_short_source(f.source_file) for f in to_inject))
                console.print(
                    f"  [{SUCCESS}]✓ Injected {len(to_inject)} fragment(s) into context[/] "
                    f"[{DIM}]({sources_str} · query: '{raw_arg}')[/]"
                )

        elif command == "/info":
            from audiobench.cli.commands.info_cmd import render_file_dossier

            target = arg.strip() or (
                str(next(iter(self.session._transcript_ids))) if self.session._transcript_ids else None
            )
            if not target:
                console.print("  [dim]Usage: /info <target> (audio file ID, transcript ID, or filename)[/]")
                return False
            render_file_dossier(target, console)
            return False

        elif command in PLAYBACK_COMMANDS or (command.startswith("/") and command[1:].isdigit()):
            if not self._daemon:
                console.print(f"  [{DIM}]Daemon not available for playback.[/]")
                return False

            on_explain = None
            if command == "/explain":
                # Note: self._stream_and_render sends the prompt to session.send(),
                # which executes an AI turn and persists the prompt and response into chat history.
                def on_explain(prompt_text: str) -> None:
                    self._stream_and_render(prompt_text)

            def_record = None
            if self.session._transcript_ids:
                def_record = self.tx_repo.get_by_id(next(iter(self.session._transcript_ids)))

            try:
                handle_playback_command(
                    command,
                    arg,
                    self._daemon,
                    console,
                    transcript_resolver=self.tx_repo.get_by_id,
                    loaded_transcript_ids=list(self.session._transcript_ids) if self.session._transcript_ids else [],
                    on_explain_context=on_explain,
                    active_citations=self._active_citations,
                    default_record=def_record,
                )
            except Exception as exc:
                console.print(f"  [{WARNING}]Playback error: {exc}[/]")

        else:
            console.print(f"  [{DIM}]Unknown command: {command} (type /help for commands)[/]")

        return False

    def _render_comparison_pair(self, msg_a: dict, msg_b: dict) -> None:
        """Render a comparison pair as side-by-side panels."""
        layout = Layout()
        layout.split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        from typing import Any
        for side, msg in [("left", msg_a), ("right", msg_b)]:
            parts: list[Any] = []
            if msg.get("thinking") and self.session.show_thinking:
                think_preview = msg["thinking"][:300]
                if len(msg["thinking"]) > 300:
                    think_preview += "…"
                parts.append(Text(f"💭 {think_preview}", style="dim italic"))
            parts.append(RichMarkdown(msg["content"], code_theme=CHAT_CODE_THEME))
            model_label = msg.get("model_name") or "Model"
            border = "cyan" if side == "left" else "magenta"
            layout[side].update(Panel(Group(*parts), title=model_label, border_style=border))
        console.print(layout)

    def _stream_and_render(self, user_text: str) -> None:
        """Send user input and render the streamed response."""
        console.print()
        try:
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            token_count = 0
            t_start = self._time.monotonic()

            with Live(
                console=chat_console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                for chunk in self.session.send(user_text):
                    thinking = chunk.get("thinking", "")
                    content = chunk.get("content", "")

                    if thinking:
                        thinking_parts.append(thinking)

                    if content:
                        content_parts.append(content)
                        token_count += 1

                    display_parts = []

                    if thinking_parts and self.session.show_thinking:
                        think_text = "".join(thinking_parts)
                        think_lines = think_text.splitlines()
                        if len(think_lines) > 5:
                            think_text = "…\n" + "\n".join(think_lines[-5:])
                        display_parts.append(
                            Text(f"💭 {think_text}", style="dim italic"),
                        )

                    if content_parts:
                        full_text = "".join(content_parts)
                        preview_lines = full_text.splitlines()
                        if len(preview_lines) > 8:
                            preview = "\n".join(preview_lines[-8:])
                            display_parts.append(
                                Text("  ⋮\n", style="dim"),
                            )
                        else:
                            preview = full_text
                        display_parts.append(Text(preview))
                        elapsed_so_far = self._time.monotonic() - t_start
                        tps_so_far = token_count / elapsed_so_far if elapsed_so_far > 0 else 0
                        display_parts.append(
                            Text(
                                f"\n  ▍ {token_count} tokens · {tps_so_far:.0f} tok/s",
                                style="dim",
                            ),
                        )

                    if display_parts:
                        live.update(Group(*display_parts))

            if content_parts:
                final_md = "".join(content_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )
            elif thinking_parts:
                final_md = "".join(thinking_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )

            self.session.finalize_response()

            elapsed = self._time.monotonic() - t_start
            if token_count > 0 and elapsed > 0:
                tps = token_count / elapsed
                console.print(
                    f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
                )
            console.print()

            # Display permanent citation banner if timestamps were cited
            if content_parts:
                full_response = "".join(content_parts)
                cited_ts = extract_timestamps(full_response)
                if cited_ts:
                    self._active_citations = cited_ts
                    render_citations_banner(console, cited_ts)

        except KeyboardInterrupt:
            if content_parts:
                self.session.finalize_response()
            console.print()
            console.print(f"  [{DIM}]Generation interrupted[/]")
            console.print()

        except (AIError, AudioBenchError) as e:
            err_msg = str(e)
            cur_m = (self.session.model or "").lower()

            # Auto-recovery 1: Gemini model wrongly routed to Ollama
            if "gemini" in cur_m and not getattr(self, "_retrying_stream", False):
                try:
                    from audiobench.chat.providers.gemini_provider import GeminiClient

                    g_client = GeminiClient(model=self.session.model)
                    if g_client.is_available():
                        console.print(
                            f"  [yellow]Notice: Re-routing '{self.session.model}' to Gemini provider...[/]"
                        )
                        self.session.switch_model(self.session.model)
                        self.client = self.session._client
                        self._retrying_stream = True
                        try:
                            self._stream_and_render(user_text)
                            return
                        finally:
                            self._retrying_stream = False
                except Exception:
                    pass

            # Auto-recovery 2: Model not found on Ollama server -> fallback to an available model
            if (
                "not found" in err_msg.lower()
                or "not available on ollama" in err_msg.lower()
                or "try pulling it" in err_msg.lower()
            ) and not getattr(self, "_retrying_stream", False):
                try:
                    client = getattr(self.session, "_client", None)
                    if client and hasattr(client, "list_models"):
                        avail = client.list_models()
                        if avail:
                            fallback = (
                                self.settings.ollama_model
                                if self.settings.ollama_model in avail
                                and self.settings.ollama_model != self.session.model
                                else [m for m in avail if m != self.session.model][0]
                            )
                            console.print(
                                f"  [yellow]Notice: Model '{self.session.model}' not found on server. Auto-switching to '{fallback}'...[/]"
                            )
                            self.session.switch_model(fallback)
                            self.client = self.session._client
                            self._retrying_stream = True
                            try:
                                self._stream_and_render(user_text)
                                return
                            finally:
                                self._retrying_stream = False
                except Exception:
                    pass

            console.print(error_panel("AI Error", err_msg))
            console.print()

        except Exception as e:
            console.print(error_panel("Unexpected Error", str(e)))
            console.print()


    def _compare_and_render(self, user_text: str, compare_model: str) -> None:
        """Run comparison between primary and secondary model."""
        console.print()
        try:
            from audiobench.chat.compare import ModelComparison

            cmp_messages = self.session._build_api_messages()
            cmp_messages.append({"role": "user", "content": user_text})

            comparison = ModelComparison(
                client=self.client,
                messages=cmp_messages,
                model_a=self.session.model,
                model_b=compare_model,
                temperature=self.temperature,
                show_thinking=self.session.show_thinking,
            )
            result = comparison.run()

            conv_id = self.session.ensure_conversation()
            self.chat_repo.add_message(conv_id, "user", user_text)
            self.session._messages.append({"role": "user", "content": user_text})

            for side in ("model_a", "model_b"):
                res = result[side]
                self.chat_repo.add_message(
                    self.session.conversation_id,
                    "assistant",
                    res["content"],
                    thinking=res["thinking"],
                    model_name=res["model_name"],
                )
                self.session._messages.append(
                    {
                        "role": "assistant",
                        "content": res["content"],
                        "thinking": res["thinking"],
                        "model_name": res["model_name"],
                    }
                )

            elapsed = result["elapsed"]
            total_tokens = result["model_a"]["tokens"] + result["model_b"]["tokens"]
            tps = total_tokens / elapsed if elapsed > 0 else 0
            console.print(f"  [{DIM}]{total_tokens} tok · {tps:.0f} tok/s · {elapsed:.1f}s[/]")
            console.print()

            if self.session.turn_count <= 1:
                self.session.generate_title_async(user_text, result["model_a"]["content"])

        except KeyboardInterrupt:
            console.print()
            console.print(f"  [{DIM}]Comparison interrupted[/]")
            console.print()

        except Exception as e:
            console.print(error_panel("Comparison Error", str(e)))
            console.print()

    def _read_multiline(self) -> str:
        """Read multi-line input via prompt_toolkit (Alt+Enter or \"\"\" to end)."""
        console.print(
            f'  [{DIM}]Multi-line mode — type """ on its own line or press Alt+Enter to submit:[/]'
        )
        try:
            text = self._pt_session.prompt(
                ANSI("\033[38;5;240m... \033[0m"),
                multiline=True,
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        text = text.strip()
        if text.startswith('"""'):
            text = text[3:]
        if text.endswith('"""'):
            text = text[:-3]
        return str(text.strip())

    def _trigger_summary(self) -> None:
        """Trigger summary generation in a background thread."""
        import threading

        def run_summary():
            import json

            from audiobench.chat.summary_generator import SummaryGenerator
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.enums import SourceType
            from audiobench.storage.expression_repository import ExpressionRepository

            gen = SummaryGenerator()
            result = gen.generate(self.session.messages)
            if not result:
                return

            if result.refined_title:
                self.chat_repo.update_title(self.session.conversation_id, result.refined_title)

            expr_repo = ExpressionRepository()
            expr = expr_repo.register(
                content=result.narrative,
                source_type=SourceType.SESSION_SUMMARY.value,
                source_id=self.session.conversation_id,
                session_type="chat",
                session_id=self.session.conversation_id,
            )

            self.chat_repo.save_summary(
                conversation_id=self.session.conversation_id,
                narrative=result.narrative,
                drift_phases=json.dumps(result.drift_phases),
                key_insights=json.dumps(result.key_insights),
                open_threads=json.dumps(result.open_threads),
                refined_title=result.refined_title,
                generated_by=gen.model_name,
                expression_id=expr.id,
            )

            try:
                daemon = get_daemon_client()
                daemon.embed(
                    expression_id=expr.id,
                    content=result.narrative,
                    source_type=SourceType.SESSION_SUMMARY,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_summary, daemon=True)
        thread.start()

    def _push_exit_frame(self) -> None:
        if not self.session.conversation_id:
            return
        try:
            from audiobench.cli.repl.session import NavigationFrame, ReplSession
            ctx = click.get_current_context(silent=True)
            repl_session: ReplSession | None = None
            while ctx is not None:
                obj = getattr(ctx, "obj", None)
                if isinstance(obj, ReplSession):
                    repl_session = obj
                    break
                ctx = getattr(ctx, "parent", None)
            if repl_session is not None:
                repl_session.push_frame(NavigationFrame(
                    context="chat",
                    state={"conversation_id": self.session.conversation_id},
                    intent="mid-conversation",
                ))
        except Exception:
            pass

    def _render_history(self) -> None:
        if self.session.messages:
            console.print(f"  [{DIM}]─── Previous Messages ───[/]")
            console.print()
            msgs = self.session.messages
            i = 0
            while i < len(msgs):
                msg = msgs[i]
                if msg["role"] == "user":
                    console.print(f"  [{PROMPT}]>>> {msg['content']}[/]")
                    console.print()
                    i += 1
                elif msg["role"] == "assistant":
                    if (
                        i + 1 < len(msgs)
                        and msgs[i + 1]["role"] == "assistant"
                        and msg.get("model_name") != msgs[i + 1].get("model_name")
                    ):
                        self._render_comparison_pair(msg, msgs[i + 1])
                        console.print()
                        i += 2
                    elif msg["content"].strip():
                        if msg.get("thinking") and self.session.show_thinking:
                            think_preview = msg["thinking"][:200]
                            if len(msg["thinking"]) > 200:
                                think_preview += "…"
                            console.print(
                                Padding(
                                    Text(f"💭 {think_preview}", style="dim italic"),
                                    (0, 2, 0, 4),
                                )
                            )
                        md = RichMarkdown(
                            msg["content"],
                            code_theme=CHAT_CODE_THEME,
                        )
                        chat_console.print(Padding(md, (0, 2, 1, 2)))
                        console.print()
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
            console.print(f"  [{DIM}]─── End of History ───[/]")
            console.print()
            if self._active_citations:
                from audiobench.playback.controls import render_citations_banner

                render_citations_banner(console, self._active_citations)
                console.print()

    def run(self, resume_id: int | None = None) -> None:
        console.print()
        conv_label = f" [#{resume_id}]" if resume_id else ""
        if self.preloaded_title:
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — {self.preloaded_title}{conv_label}")
        else:
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Chat{conv_label}")

        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Model:    {self.session.model}")
        ctx_lines = self.session.get_context_summary()
        console.print(f"    Context:  {ctx_lines[0]}")
        for line in ctx_lines[1:]:
            console.print(f"              {line}")
        think_label = "on" if self.session.show_thinking else "off"
        console.print(f"    Thinking: {think_label}")
        if resume_id and self.session.turn_count > 0:
            console.print(f"    Resumed:  {self.session.turn_count} previous turn(s)")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print()

        if resume_id:
            self._render_history()

        last_user_input: str | None = None
        self.session._retry_requested = False
        if not hasattr(self.session, "_compare_model"):
            self.session._compare_model = None

        while True:
            try:
                cmp_active = getattr(self.session, "_compare_model", None)
                status_str = (
                    render_playback_status_line(self._daemon, self._active_citations)
                    if self._daemon
                    else None
                )
                status_prefix = f"\033[38;5;240m  {status_str}\033[0m\n" if status_str else ""
                if cmp_active:
                    prompt_str = ANSI(f"{status_prefix}\033[38;5;214m⚡ >>> \033[0m")
                else:
                    prompt_str = ANSI(f"{status_prefix}\033[38;5;48m>>> \033[0m")
                user_input = self._pt_session.prompt(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                if self.session.conversation_id:
                    console.print(
                        f"  [{SUCCESS}]✓ Conversation "
                        f"#{self.session.conversation_id} saved "
                        f"({self.session.turn_count * 2} messages)[/]"
                    )
                    if self.session.turn_count >= 3:
                        self._trigger_summary()
                self._push_exit_frame()
                console.print(f"  [{DIM}]Goodbye![/]")
                console.print()
                break

            if not user_input:
                continue

            # ── Sentinel from empty-buffer keybindings ───────────────────────
            if user_input.startswith("\x00"):
                self._handle_sentinel_action(user_input)
                continue

            if user_input == '"""':
                user_input = self._read_multiline()
                if not user_input.strip():
                    continue

            if user_input.startswith("\\"):
                user_input = "/" + user_input[1:]
            # Intercept bare quit words so they don't get sent to the LLM
            if user_input.strip().lower() in ("q", "quit", "exit", ":q", ":wq"):
                user_input = "/exit"
            if user_input.startswith("/"):
                should_exit = self._handle_slash_command(user_input)

                if getattr(self.session, "_retry_requested", False):
                    self.session._retry_requested = False
                    if last_user_input and self.session.messages:
                        self.session._messages = [m for m in self.session._messages if m != self.session._messages[-1]]
                        if self.session._messages and self.session._messages[-1]["role"] == "user":
                            self.session._messages.pop()
                        console.print(f"  [{DIM}]Regenerating...[/]")
                        self._stream_and_render(last_user_input)
                    else:
                        console.print(f"  [{DIM}]Nothing to retry[/]")
                    continue

                if should_exit:
                    if self.session.conversation_id:
                        console.print(
                            f"  [{SUCCESS}]✓ Conversation "
                            f"#{self.session.conversation_id} saved "
                            f"({self.session.turn_count * 2} messages)"
                            f"[/]"
                        )
                        if self.session.turn_count >= 3:
                            self._trigger_summary()
                    self._push_exit_frame()
                    console.print(f"  [{DIM}]Goodbye![/]")
                    console.print()
                    break
                continue

            last_user_input = user_input
            self._active_citations = []  # Retire prior turn's active citation hotkeys for new turn

            compare_model = getattr(self.session, "_compare_model", None)
            if compare_model:
                self._compare_and_render(user_input, compare_model)
            else:
                self._stream_and_render(user_input)

    def _handle_sentinel_action(self, sentinel: str) -> None:
        """Dispatch empty-buffer hotkeys without sending input to LLM."""
        if not self._daemon:
            return

        action = sentinel[1:]
        def_record = None
        if self.session._transcript_ids:
            def_record = self.tx_repo.get_by_id(next(iter(self.session._transcript_ids)))

        if action == "SPACE":
            st = self._daemon.playback_toggle()
            state = "Paused" if st.get("paused") else "Playing"
            console.print(f"  [{SUCCESS}][DONE] {state}[/]")
            return

        if action == "NEXT":
            handle_playback_command(
                "/next", "", self._daemon, console,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return

        if action == "PREV":
            handle_playback_command(
                "/prev", "", self._daemon, console,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return

        if action.isdigit():
            handle_playback_command(
                f"/{action}", "", self._daemon, console,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return
