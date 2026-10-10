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
from rich.markup import escape

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
    format_duration,
)
import os
import re
import textwrap
import time
from collections import defaultdict
from pathlib import Path

from audiobench.core.db_engine import init_db
from audiobench.core.error_types import AudioBenchError
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.playback.controls import (
    PLAYBACK_COMMANDS,
    extract_timestamps,
    handle_playback_command,
    is_bg_exit_requested,
    pause_playback_if_owned,
    render_citations_banner,
    render_jump_bar,
    render_playback_status_line,
)
from audiobench.storage.bookmark_repository import BookmarkRepository
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
        preloaded_query: str | None = None,
        preloaded_answer: str | None = None,
        search_session_id: int | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        think: bool = True,
        resume_id: int | None = None,
        project_id: int | None = None,
        current_session_number: int | None = None,
        start_voice: bool = False,
    ):
        init_db()
        self.settings = settings or get_settings()
        self.tx_repo = tx_repo or TranscriptionRepository()
        self.chat_repo = chat_repo or ChatRepository()
        self.start_voice = start_voice

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
        self.preloaded_query = preloaded_query
        self.preloaded_answer = preloaded_answer
        self.search_session_id = search_session_id

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
                search_session_id=self.search_session_id,
            )
        elif self.session.conversation_id and self.search_session_id:
            self.chat_repo.update_search_session_id(self.session.conversation_id, self.search_session_id)

        # Inject preloaded search fragments into chat context
        if self.preloaded_fragments and self.session:
            q_text = self.preloaded_query or self.preloaded_title or ""
            self.session.inject_search_fragments(self.preloaded_fragments, query=q_text)

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
        active_resume_id = resume_id if resume_id is not None else (self.session.conversation_id if self.session else None)
        if active_resume_id is not None and self.session and self.session.messages:
            from audiobench.playback.controls import extract_timestamps

            for msg in reversed(self.session.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    cited = extract_timestamps(msg["content"])
                    if cited:
                        self._active_citations = cited
                        break

        # Rehydrate citations from preloaded answer or fragments if not yet set
        if not self._active_citations and self.preloaded_answer:
            from audiobench.playback.controls import extract_timestamps

            cited = extract_timestamps(self.preloaded_answer)
            if cited:
                self._active_citations = cited

        if not self._active_citations and self.preloaded_fragments:
            for frag in self.preloaded_fragments:
                sf = getattr(frag, "source_file", None) or ""
                st = getattr(frag, "start_time", 0.0)
                if (sf, st) not in self._active_citations:
                    self._active_citations.append((sf, st))

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

        self._stt_session_id: str | None = None
        try:
            from audiobench.daemon.factory import get_daemon_client
            self._daemon = get_daemon_client(autostart=False)
            if self._daemon and hasattr(self._daemon, "whisper_acquire"):
                self._stt_session_id = f"chat_repl_{os.getpid()}"
                self._daemon.whisper_acquire(
                    self._stt_session_id,
                    model=self.settings.stt_model or "base",
                )
        except Exception as exc:
            logger.debug("Daemon not available for playback or STT: %s", exc)
            self._daemon = None
            self._stt_session_id = None

        self._exit_bg: bool = False

    def _cleanup_stt_session(self) -> None:
        """Release warm Whisper session in daemon upon chat exit."""
        if self._daemon and getattr(self, "_stt_session_id", None):
            try:
                self._daemon.whisper_release(self._stt_session_id)
            except Exception:
                pass
            self._stt_session_id = None

    def _pause_session_playback(self, reason: str = "exit", force_bg: bool = False) -> None:
        """Pause playback and save resume position if daemon is playing this session's audio."""
        pause_playback_if_owned(
            self._daemon,
            self.session._transcript_ids if self.session else None,
            console,
            reason=reason,
            force_bg=force_bg,
        )

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
                        rec = self.tx_repo.get_by_id(tx_id, session=None)
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
                ts_badge = f"  [{SUCCESS}][TS] timestamps active[/]" if t.get("has_timestamps") else f"  [{DIM}](no timestamps)[/]"
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
            self._exit_bg = is_bg_exit_requested(arg)
            return True

        elif command == "/help":
            render_chat_help()

        elif command == "/context":
            if arg and arg.strip().isdigit():
                tid = int(arg.strip())
                record = self.tx_repo.get_by_id(tid, session=None)
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
                    record = self.tx_repo.get_by_id(tid, session=None)
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
                    record = self.tx_repo.get_by_id(tid, session=None)
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
            self._pause_session_playback(reason="switch")
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
                self._pause_session_playback(reason="switch")
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
                        f"  [{ACCENT}][COMPARE] Comparison mode ON[/]\n"
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
                    f"  [{ACCENT}][COMPARE] Switched comparison:[/] [{DIM}]{old}[/] → [{BOLD}]{new_model}[/]"
                )
            else:
                console.print(
                    f"  [{ACCENT}][COMPARE] Comparison mode ON[/]\n"
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
                record = self.tx_repo.get_by_id(tid, session=None)
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
                emoji = BOOKMARK_TYPES.get(b["bookmark_type"], "[BM]")
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

        elif command in ("/voice", "/v"):
            record_audio = "--no-record" not in arg and "--no-audio" not in arg
            clean_arg = arg.replace("--no-record", "").replace("--no-audio", "").strip().lower()

            if clean_arg in ("status", "info"):
                from audiobench.chat.voice_session import _LOADED_WHISPER_MODELS, is_internet_connected
                from audiobench.tts.engine import PiperTTSEngine
                engine = PiperTTSEngine(voices_dir=self.settings.voices_dir)
                is_voice_local = (engine._voices_dir / f"{self.settings.tts_voice}.onnx").exists()
                whisper_cached = self.settings.stt_model in _LOADED_WHISPER_MODELS
                has_gemini = bool(self.settings.gemini_api_key)
                online = is_internet_connected()

                console.print()
                console.print(f"  [{BOLD} {ACCENT}]AudioBench[/] — Voice & Audio Configuration")
                console.print(f"  [{DIM}]{'─' * 54}[/]")
                console.print(f"    STT Provider:       [bold cyan]{self.settings.stt_provider}[/bold cyan] [dim](auto / local / cloud)[/dim]")
                console.print(f"    Whisper STT Model:  [bold cyan]{self.settings.stt_model}[/bold cyan] [dim]({'cached in RAM' if whisper_cached else 'loads on-demand'})[/dim]")
                console.print(f"    Gemini Cloud STT:   {'[green]Available[/green]' if has_gemini else '[yellow]No API key[/yellow]'} [dim](gemini-2.5-flash)[/dim]")
                console.print(f"    TTS Voice Model:    [bold cyan]{self.settings.tts_voice}[/bold cyan] [dim]({'downloaded' if is_voice_local else 'needs download'})[/dim]")
                console.print(f"    Live Duplex Model:  gemini-2.5-flash-native-audio-preview")
                console.print(f"    Internet Link:      {'[green]Online[/green]' if online else '[red]Offline[/red]'}")
                console.print(f"    Session Recording:  {'enabled (session WAV saved)' if record_audio else 'disabled'}")
                console.print(f"  [{DIM}]{'─' * 54}[/]")
                console.print(f"  [{DIM}]Commands: /voice provider · /voice model · /voice picker · /voice live[/dim]")
                console.print()
                return False

            elif clean_arg.startswith("provider") or clean_arg.startswith("engine"):
                tokens = clean_arg.split(None, 1)
                provider_arg = tokens[1].strip() if len(tokens) > 1 else ""
                valid_providers = ("auto", "local", "cloud")
                if not provider_arg:
                    current_prov = getattr(self.settings, "stt_provider", "auto")
                    console.print()
                    console.print(f"  [{BOLD} {ACCENT}]Speech-to-Text (STT) Provider Selection[/]")
                    console.print(f"  [{DIM}]{'─' * 52}[/]")
                    console.print(f"    [[bold cyan]1[/bold cyan]] auto   — Cloud (Gemini) when online + API key, offline Whisper fallback {'[green]← active[/green]' if current_prov == 'auto' else ''}")
                    console.print(f"    [[bold cyan]2[/bold cyan]] local  — Local Whisper only (private, zero network dependency) {'[green]← active[/green]' if current_prov == 'local' else ''}")
                    console.print(f"    [[bold cyan]3[/bold cyan]] cloud  — Gemini 2.5 Flash STT only (fast, high accuracy on jargon) {'[green]← active[/green]' if current_prov == 'cloud' else ''}")
                    console.print(f"    [[bold cyan]0[/bold cyan]] Keep current ({current_prov})")
                    console.print()
                    try:
                        raw = input("    › Pick provider [0]: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        console.print()
                        return False
                    if not raw or raw == "0":
                        console.print(f"  [{DIM}]Kept current provider: {current_prov}[/]")
                        return False
                    mapping = {"1": "auto", "2": "local", "3": "cloud", "auto": "auto", "local": "local", "cloud": "cloud"}
                    provider_arg = mapping.get(raw.lower(), "")
                    if not provider_arg:
                        console.print(f"  [red]Invalid provider option: '{raw}'[/red]")
                        return False

                if provider_arg in valid_providers:
                    self.settings.stt_provider = provider_arg
                    self.settings.save()
                    console.print(f"  [{SUCCESS}]✓ STT Provider set to:[/] [bold cyan]{provider_arg}[/bold cyan]")
                else:
                    console.print(f"  [red]Invalid provider: '{provider_arg}'. Choose from: auto, local, cloud[/red]")
                return False

            elif clean_arg.startswith("model") or clean_arg.startswith("stt-model"):
                tokens = clean_arg.split(None, 1)
                model_arg = tokens[1].strip() if len(tokens) > 1 else ""
                model_specs = [
                    ("tiny", "~75MB", "Fastest transcription, lowest memory footprint"),
                    ("base", "~145MB", "Fast, balanced accuracy for general speech (default)"),
                    ("small", "~480MB", "High accuracy, moderate compute requirements"),
                    ("medium", "~1.5GB", "Studio-level accuracy, higher RAM"),
                    ("turbo", "~1.6GB", "State-of-the-art accuracy at 8× real-time speed"),
                ]
                valid_models = {"tiny", "base", "small", "medium", "turbo", "large-v3-turbo"}
                current_model = getattr(self.settings, "stt_model", "base")

                if not model_arg:
                    console.print()
                    console.print(f"  [{BOLD} {ACCENT}]Whisper Speech-to-Text Model Selection[/]")
                    console.print(f"  [{DIM}]{'─' * 62}[/]")
                    for idx, (m_name, m_size, m_desc) in enumerate(model_specs, 1):
                        is_curr = (current_model == m_name or (current_model == "large-v3-turbo" and m_name == "turbo"))
                        marker = " [green]← active[/green]" if is_curr else ""
                        console.print(f"    [[bold cyan]{idx}[/bold cyan]] [bold]{m_name:<7}[/bold] ({m_size:<7}) — {m_desc}{marker}")
                    console.print(f"    [[bold cyan]0[/bold cyan]] Keep current ({current_model})")
                    console.print()
                    try:
                        raw = input("    › Pick Whisper model [0]: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        console.print()
                        return False
                    if not raw or raw == "0":
                        console.print(f"  [{DIM}]Kept current model: {current_model}[/]")
                        return False
                    mapping = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "turbo"}
                    model_arg = mapping.get(raw.lower(), raw.lower())

                if model_arg == "large-v3-turbo":
                    model_arg = "turbo"

                if model_arg in valid_models:
                    self.settings.stt_model = model_arg
                    self.settings.save()
                    console.print(f"  [{SUCCESS}]✓ Whisper STT model set to:[/] [bold cyan]{model_arg}[/bold cyan]")
                else:
                    console.print(f"  [red]Invalid model: '{model_arg}'. Choose from: tiny, base, small, medium, turbo[/red]")
                return False

            elif clean_arg in ("test", "audition"):
                from audiobench.chat.voice_picker import audition_voice
                audition_voice(self.settings.tts_voice, console=console)
                return False

            elif clean_arg in ("picker", "set", "choose", "select"):
                from audiobench.chat.voice_picker import show_voice_picker
                chosen = show_voice_picker(console=console, current_voice=self.settings.tts_voice)
                if chosen:
                    self.settings.tts_voice = chosen
                    self.settings.save()
                    console.print(f"  [{SUCCESS}]✓ TTS voice set to: {chosen}[/]")
                return False

            elif clean_arg in ("live", "call", "stream"):
                from audiobench.chat.voice_session import VoiceChatSession
                conv_id = self.session.ensure_conversation() if hasattr(self.session, "ensure_conversation") and getattr(self.session, "_chat_repo", None) else None
                v_session = VoiceChatSession(
                    conversation_id=conv_id,
                    console=console,
                    settings=self.settings,
                    record_audio=record_audio,
                )
                turns, wav_path, total_duration = v_session.run_continuous_live()
                if turns and conv_id is not None:
                    audio_fid, tx_id, seg_ids = self.tx_repo.save_voice_chat_session(
                        conversation_id=conv_id,
                        audio_file_path=wav_path,
                        turns=turns,
                        duration_seconds=total_duration,
                        model_name=self.session.model,
                    )
                    self.chat_repo.link_voice_session(
                        conversation_id=conv_id,
                        audio_file_id=audio_fid,
                        transcription_id=tx_id,
                    )

                    for idx, turn in enumerate(turns):
                        s_id = seg_ids[idx] if idx < len(seg_ids) else None
                        self.chat_repo.add_message(
                            conversation_id=conv_id,
                            role=turn["role"],
                            content=turn["text"],
                            modality="voice_live",
                            duration_seconds=max(0.0, turn.get("end_time", 0.0) - turn.get("start_time", 0.0)),
                            audio_path=wav_path,
                            voice_name=turn.get("voice_name"),
                            segment_id=s_id,
                        )
                return False

            else:
                from audiobench.chat.voice_session import VoiceChatSession
                conv_id = self.session.ensure_conversation() if hasattr(self.session, "ensure_conversation") and getattr(self.session, "_chat_repo", None) else None
                v_session = VoiceChatSession(
                    conversation_id=conv_id,
                    console=console,
                    settings=self.settings,
                    record_audio=record_audio,
                )
                try:
                    spoken_text, wav_path, duration = v_session.record_single_turn(silence_timeout=5.0)
                except KeyboardInterrupt:
                    console.print()
                    console.print(f"  [{DIM}]Cancelled.[/]")
                    return False
                if spoken_text:
                    console.print(f"  [{DIM}]Prompt: \"{spoken_text}\"[/]")
                    seg_id = None
                    if wav_path and conv_id is not None:
                        single_turn = [{"role": "user", "text": spoken_text, "start_time": 0.0, "end_time": duration}]
                        _, _, seg_ids = self.tx_repo.save_voice_chat_session(
                            conversation_id=conv_id,
                            audio_file_path=wav_path,
                            turns=single_turn,
                            duration_seconds=duration,
                            model_name=self.session.model,
                        )
                        if seg_ids:
                            seg_id = seg_ids[0]

                    self._stream_and_render(
                        spoken_text,
                        modality="voice",
                        duration_seconds=duration,
                        audio_path=wav_path,
                        segment_id=seg_id,
                    )
                else:
                    console.print(f"  [{DIM}]No speech detected.[/]")
                return False

        elif command in ("/voices", "/voice-picker"):
            from audiobench.chat.voice_picker import show_voice_picker
            chosen = show_voice_picker(console=console, current_voice=self.settings.tts_voice)
            if chosen:
                self.settings.tts_voice = chosen
            return False

        elif command == "/speak":
            raw_arg = arg.strip() if arg else ""
            from audiobench.chat.voice_picker import ensure_voice_downloaded, strip_markdown_for_speech
            from audiobench.tts.engine import PiperTTSEngine, TTSError

            # Parse flags
            tokens = raw_arg.split()
            show_lyrics = False
            foreground_mode = False
            auto_pause = True
            text_tokens = []

            for t in tokens:
                if t in ("-l", "--lyrics"):
                    show_lyrics = True
                elif t in ("-f", "--foreground"):
                    foreground_mode = True
                elif t == "--no-pause":
                    auto_pause = False
                else:
                    text_tokens.append(t)

            target = " ".join(text_tokens).strip()

            text_to_speak = ""
            if target.startswith("msg ") or target.startswith("message "):
                parts = target.split()
                if len(parts) > 1 and parts[1].isdigit():
                    m_id = int(parts[1])
                    from audiobench.core.db_session import get_session
                    from audiobench.storage.models import ChatMessage
                    with get_session() as db:
                        msg = db.query(ChatMessage).filter_by(id=m_id).first()
                        if msg and msg.content:
                            text_to_speak = msg.content
                        else:
                            console.print(f"  [{DIM}]Message #{m_id} not found or empty.[/]")
                            return False
            elif target:
                text_to_speak = target
            else:
                for msg in reversed(self.session.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        text_to_speak = msg["content"]
                        break

            if not text_to_speak:
                console.print(f"  [{DIM}]No assistant message or text to speak.[/]")
                return False

            if not ensure_voice_downloaded(self.settings.tts_voice, console=console):
                return False

            cleaned = strip_markdown_for_speech(text_to_speak)
            voice = self.settings.tts_voice

            if foreground_mode or not self._daemon:
                console.print(f"  [cyan][Voice][/cyan] [dim]Speaking with {voice}... [bold](Space: pause · q: stop)[/bold][/dim]")
                try:
                    engine = PiperTTSEngine(voices_dir=self.settings.voices_dir)
                    engine.play(cleaned, voice=voice)
                except KeyboardInterrupt:
                    PiperTTSEngine.stop()
                    console.print("\n  [dim][Voice] Playback stopped.[/dim]")
                except TTSError as e:
                    console.print(f"  [yellow]TTS Error: {e}[/yellow]")
                except Exception as e:
                    console.print(f"  [yellow]TTS Playback failed: {e}[/yellow]")
                return False

            # Daemon background playback with sentence timestamps
            try:
                cache_dir = Path(self.settings.data_dir) / "cache" / "tts"
                cache_dir.mkdir(parents=True, exist_ok=True)
                speech_wav = cache_dir / f"speak_{os.getpid()}_{int(time.time())}.wav"

                engine = PiperTTSEngine(voices_dir=self.settings.voices_dir)
                wav_path, segments = engine.synthesize_with_timestamps(
                    cleaned, voice=voice, output_path=speech_wav
                )

                self._daemon.playback_play(
                    str(wav_path),
                    segments=segments,
                    auto_resume=auto_pause,
                )

                if show_lyrics:
                    from audiobench.playback.lyrics_hud import show_daemon_lyrics
                    show_daemon_lyrics(self._daemon, console, tx_repo=self.tx_repo)
                else:
                    pause_info = " · auto-resumes previous track" if auto_pause else ""
                    console.print(
                        f"  [cyan][Voice][/cyan] [dim]Speaking with {voice} in background "
                        f"[bold](Space: pause · /stop · /lyrics to follow{pause_info})[/bold][/dim]"
                    )
            except TTSError as e:
                console.print(f"  [yellow]TTS Error: {e}[/yellow]")
            except Exception as e:
                logger.error("Daemon /speak failed: %s", e)
                console.print(f"  [yellow]TTS Playback failed: {e}[/yellow]")
            return False

        elif command == "/play" and (arg.strip().startswith("msg") or arg.strip().startswith("message")):
            parts = arg.strip().split()
            if len(parts) > 1 and parts[1].isdigit():
                msg_id = int(parts[1])
                from audiobench.core.db_session import get_session
                from audiobench.storage.models import (
                    AudioFileRecord,
                    ChatMessage,
                    SegmentRecord,
                    TranscriptionRecord,
                )
                audio_path = None
                msg_content = None
                start_pos = 0.0
                with get_session() as db:
                    msg = db.query(ChatMessage).filter_by(id=msg_id).first()
                    if msg:
                        audio_path = msg.audio_path
                        msg_content = msg.content
                        seg = None
                        if msg.segment_id:
                            seg = db.query(SegmentRecord).filter_by(id=msg.segment_id).first()
                            if seg:
                                start_pos = seg.start_time

                        # If segment was purged, missing, or detached, probe active transcription
                        if seg is None and (audio_path or (msg.conversation and msg.conversation.audio_file_id)):
                            audio_rec = None
                            if audio_path:
                                audio_rec = db.query(AudioFileRecord).filter_by(file_path=audio_path).first()
                                if not audio_rec:
                                    audio_rec = db.query(AudioFileRecord).filter(
                                        AudioFileRecord.file_name == Path(audio_path).name
                                    ).first()
                            if not audio_rec and msg.conversation and msg.conversation.audio_file_id:
                                audio_rec = db.query(AudioFileRecord).filter_by(id=msg.conversation.audio_file_id).first()

                            if audio_rec:
                                if not audio_path:
                                    audio_path = audio_rec.file_path
                                active_tx = (
                                    db.query(TranscriptionRecord)
                                    .filter(
                                        TranscriptionRecord.audio_file_id == audio_rec.id,
                                        TranscriptionRecord.pipeline_phase.in_(["complete", "completed"])
                                    )
                                    .order_by(TranscriptionRecord.id.desc())
                                    .first()
                                )
                                if active_tx and msg_content:
                                    matched_seg = None
                                    # 1. Probe explicit quotes first
                                    quotes = re.findall(r'["\']([^"\']{10,})["\']', msg_content)
                                    for q in quotes:
                                        q_words = [w for w in re.sub(r"[^\w\s]", " ", q).split() if len(w) > 3][:4]
                                        if len(q_words) >= 2:
                                            pat = "%" + "%".join(q_words) + "%"
                                            matched_seg = (
                                                db.query(SegmentRecord)
                                                .filter(
                                                    SegmentRecord.transcription_id == active_tx.id,
                                                    SegmentRecord.text.like(pat)
                                                )
                                                .order_by(SegmentRecord.start_time)
                                                .first()
                                            )
                                            if matched_seg:
                                                break

                                    # 2. Sliding word windows from message content
                                    if not matched_seg:
                                        clean_probe = re.sub(r"[^\w\s]", " ", msg_content.strip()[:120]).strip()
                                        words = [w for w in clean_probe.split() if len(w) > 3]
                                        if len(words) >= 3:
                                            for i in range(min(5, len(words) - 2)):
                                                pat = "%" + "%".join(words[i : i + 3]) + "%"
                                                matched_seg = (
                                                    db.query(SegmentRecord)
                                                    .filter(
                                                        SegmentRecord.transcription_id == active_tx.id,
                                                        SegmentRecord.text.like(pat)
                                                    )
                                                    .order_by(SegmentRecord.start_time)
                                                    .first()
                                                )
                                                if matched_seg:
                                                    break
                                    if matched_seg:
                                        start_pos = matched_seg.start_time
                                        msg.segment_id = matched_seg.id
                                        db.commit()

                if audio_path and Path(audio_path).exists() and self._daemon:
                    self._daemon.playback_play(audio_path, start_pos=start_pos)
                    console.print(f"  [{SUCCESS}][DONE] Playing voice message #{msg_id}[/] [{DIM}]({audio_path} from {start_pos:.1f}s)[/]")
                elif msg_content:
                    from audiobench.chat.voice_picker import ensure_voice_downloaded, strip_markdown_for_speech
                    from audiobench.tts.engine import PiperTTSEngine
                    if not ensure_voice_downloaded(self.settings.tts_voice, console=console):
                        return False
                    cleaned = strip_markdown_for_speech(msg_content)
                    console.print(f"  [cyan][Voice][/cyan] [dim]Synthesizing message #{msg_id} with {self.settings.tts_voice}... [bold](Space: pause · q: stop)[/bold][/dim]")
                    try:
                        engine = PiperTTSEngine(voices_dir=self.settings.voices_dir)
                        engine.play(cleaned, voice=self.settings.tts_voice)
                    except KeyboardInterrupt:
                        PiperTTSEngine.stop()
                        console.print("\n  [dim][Voice] Playback stopped.[/dim]")
                    except Exception as e:
                        console.print(f"  [yellow]TTS Playback failed: {e}[/yellow]")
                else:
                    console.print(f"  [{DIM}]Message #{msg_id} has no audio or content.[/]")
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
                def_record = self.tx_repo.get_by_id(next(iter(self.session._transcript_ids)), session=None)

            try:
                handle_playback_command(
                    command,
                    arg,
                    self._daemon,
                    console,
                    # Wrap with session=None so the playback resolver never returns
                    # owner-voice content regardless of ChatSession state.
                    transcript_resolver=lambda tid: self.tx_repo.get_by_id(tid, session=None),
                    loaded_transcript_ids=list(self.session._transcript_ids) if self.session._transcript_ids else [],
                    on_explain_context=on_explain,
                    active_citations=self._active_citations,
                    default_record=def_record,
                )
            except Exception as exc:
                console.print(f"  [{WARNING}]Playback error: {escape(str(exc))}[/]")

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
                parts.append(Text(f"[thinking] {think_preview}", style="dim italic"))
            parts.append(RichMarkdown(msg["content"], code_theme=CHAT_CODE_THEME))
            model_label = msg.get("model_name") or "Model"
            border = "cyan" if side == "left" else "magenta"
            layout[side].update(Panel(Group(*parts), title=model_label, border_style=border))
        console.print(layout)

    def _stream_and_render(
        self,
        user_text: str,
        modality: str = "text",
        duration_seconds: float | None = None,
        audio_path: str | None = None,
        segment_id: int | None = None,
    ) -> None:
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
                for chunk in self.session.send(
                    user_text,
                    modality=modality,
                    duration_seconds=duration_seconds,
                    audio_path=audio_path,
                    segment_id=segment_id,
                ):
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
                            Text(f"[thinking] {think_text}", style="dim italic"),
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
                                    Text(f"[thinking] {think_preview}", style="dim italic"),
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
            if not self._active_citations and self.session and self.session.messages:
                from audiobench.playback.controls import extract_timestamps

                for msg in reversed(self.session.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        cited = extract_timestamps(msg["content"])
                        if cited:
                            self._active_citations = cited
                            break

            if self._active_citations:
                from audiobench.playback.controls import render_citations_banner

                render_citations_banner(console, self._active_citations)
                console.print()

    def run(self, resume_id: int | None = None) -> None:
        """Run the chat REPL session, ensuring STT daemon session cleanup upon exit."""
        try:
            self._run_loop(resume_id=resume_id)
        finally:
            self._cleanup_stt_session()

    def _run_loop(self, resume_id: int | None = None) -> None:
        console.print()

        # Seed initial turn if preloaded_answer is provided and this is a brand new session
        if self.preloaded_answer and self.session and self.session.turn_count == 0:
            query_text = self.preloaded_query or self.preloaded_title or "Search Query"
            self.session.seed_initial_turn(query_text, self.preloaded_answer)

        active_resume_id = resume_id if resume_id is not None else (self.session.conversation_id if (self.session and self.session.turn_count > 0) else None)
        conv_label = f" [#{active_resume_id}]" if active_resume_id else ""
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
        if active_resume_id and self.session.turn_count > 0:
            console.print(f"    Resumed:  {self.session.turn_count} previous turn(s)")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print()

        if active_resume_id and self.session.turn_count > 0:
            self._render_history()
        elif self._active_citations:
            from audiobench.playback.controls import render_citations_banner

            render_citations_banner(console, self._active_citations)
            console.print()

        last_user_input: str | None = None
        self.session._retry_requested = False
        if not hasattr(self.session, "_compare_model"):
            self.session._compare_model = None

        if self.start_voice:
            self.start_voice = False
            self._handle_slash_command("/voice live")

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
                    prompt_str = ANSI(f"{status_prefix}\033[38;5;214m[cmp] >>> \033[0m")
                else:
                    prompt_str = ANSI(f"{status_prefix}\033[38;5;48m>>> \033[0m")
                user_input = self._pt_session.prompt(prompt_str).strip()
            except KeyboardInterrupt:
                now = self._time.monotonic()
                last_sig = getattr(self, "_last_interrupt_time", 0.0)
                if now - last_sig < 1.5:
                    console.print()
                    self._pause_session_playback(reason="exit", force_bg=False)
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
                self._last_interrupt_time = now
                console.print(f"\n  [{DIM}](Press Ctrl+C again or type /exit to quit)[/]")
                continue
            except EOFError:
                console.print()
                self._pause_session_playback(reason="exit", force_bg=False)
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
            tokens = user_input.strip().lower().split()
            if tokens and tokens[0] in ("q", "quit", "exit", ":q", ":wq"):
                self._exit_bg = is_bg_exit_requested(tokens[1:])
                user_input = "/exit"
            try:
                if user_input.startswith("/"):
                    try:
                        should_exit = self._handle_slash_command(user_input)
                    except Exception as cmd_err:
                        logger.exception("Error executing command '%s'", user_input)
                        console.print(f"  [red]Command error: {cmd_err}[/red]")
                        continue

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
                        self._pause_session_playback(reason="exit", force_bg=getattr(self, "_exit_bg", False))
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
            except KeyboardInterrupt:
                try:
                    from audiobench.tts.engine import PiperTTSEngine
                    PiperTTSEngine.stop()
                except Exception:
                    pass
                console.print("\n  [dim]Cancelled.[/dim]")
                continue

    def _handle_sentinel_action(self, sentinel: str) -> None:
        """Dispatch empty-buffer hotkeys without sending input to LLM."""
        if not self._daemon:
            return

        action = sentinel[1:]
        def_record = None
        if self.session._transcript_ids:
            def_record = self.tx_repo.get_by_id(next(iter(self.session._transcript_ids)), session=None)

        session_tids = list(self.session._transcript_ids) if self.session._transcript_ids else []
        # Locked resolver: bake session=None into all playback lookups from chat.
        _locked_resolver = lambda tid: self.tx_repo.get_by_id(tid, session=None)  # noqa: E731

        if action == "SPACE":
            handle_playback_command(
                "/toggle", "", self._daemon, console,
                transcript_resolver=_locked_resolver,
                loaded_transcript_ids=session_tids,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return

        if action == "NEXT":
            handle_playback_command(
                "/next", "", self._daemon, console,
                transcript_resolver=_locked_resolver,
                loaded_transcript_ids=session_tids,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return

        if action == "PREV":
            handle_playback_command(
                "/prev", "", self._daemon, console,
                transcript_resolver=_locked_resolver,
                loaded_transcript_ids=session_tids,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return

        if action.isdigit():
            handle_playback_command(
                f"/{action}", "", self._daemon, console,
                transcript_resolver=_locked_resolver,
                loaded_transcript_ids=session_tids,
                active_citations=self._active_citations,
                default_record=def_record,
            )
            return
