"""Context-aware autocomplete and suggestions for Memory Search REPL.

Provides:
  - SearchSlashCompleter: 2-column popup menu with command descriptions and sub-options.
  - SearchSemanticCompleter: Daemon semantic autocomplete for query text.
  - SearchAutoSuggest: Ghost text suggestion combining history and command synopses.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion

from audiobench.cli.shared.repl_shell import _pad_meta
from audiobench.memory.search_meta import (
    SEARCH_COMMAND_REGISTRY,
    CommandMeta,
    get_all_search_command_names,
    get_search_command_meta,
)

if TYPE_CHECKING:
    from audiobench.cli.commands.memory_cmd import SearchSessionState


class SearchSlashCompleter(Completer):
    """Context-aware slash command completer for the Search REPL."""

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        if not text_before.startswith("/"):
            return

        parts = text_before.split(None, 1)

        # Position 0: typing the slash command
        if len(parts) == 1 and not text_before.endswith(" "):
            prefix = parts[0].lower()
            for meta in SEARCH_COMMAND_REGISTRY:
                for name in meta.all_names:
                    if name.startswith(prefix):
                        yield Completion(
                            name,
                            start_position=-len(prefix),
                            display=name,
                            display_meta=_pad_meta(meta.summary),
                        )
            return

        # Position 1+: sub-options for /set, /switch, etc.
        cmd_name = parts[0].lower()
        arg_part = parts[1] if len(parts) > 1 else ""

        if cmd_name in ("/set", "/settings", "/config"):
            meta = get_search_command_meta("/set")
            if meta and meta.choices:
                for choice in meta.choices:
                    if choice.startswith(arg_part):
                        yield Completion(
                            choice,
                            start_position=-len(arg_part),
                            display=choice,
                            display_meta=_pad_meta(f"Configure {choice}"),
                        )


def _clean_passage_snippet(content: str, max_chars: int = 50) -> str:
    """Truncate passage cleanly at a word boundary with quotation marks."""
    cleaned = " ".join(content.strip().split())
    if len(cleaned) <= max_chars:
        return f'"{cleaned}"'
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 20:
        truncated = truncated[:last_space]
    truncated = truncated.rstrip(",;:-.?! ")
    return f'"{truncated}…"'


class SearchSemanticCompleter(Completer):
    """Multi-bucket knowledge & semantic completer for plain query text (triggered on Tab).

    Surfaces 4 distinct cognitive tiers:
      1. [History] Recent past search queries
      2. [Author] / [Book] Canonical entities from indexed works
      3. [School] Discovered philosophical communities
      4. [Passage] Attributed concept passages from the daemon vector index
    """

    _debounce_ms: int = 150

    def __init__(self, state: SearchSessionState) -> None:
        self._state = state
        self._latest: str = ""
        try:
            from audiobench.daemon.factory import get_daemon_client
            self._client = get_daemon_client()
        except Exception:
            self._client = None

    def get_completions(self, document, complete_event):
        if not getattr(self._state, "autocomplete", True):
            return

        raw_text = document.text_before_cursor
        trimmed = raw_text.strip()
        if raw_text.startswith("/") or len(trimmed) < 2:
            return

        # Only debounce during continuous typing; Tab requests proceed immediately
        if not complete_event.completion_requested:
            self._latest = raw_text
            time.sleep(self._debounce_ms / 1000.0)
            if self._latest != raw_text:
                return

        word = document.get_word_before_cursor(WORD=True)
        seen_displays: set[str] = set()

        # ── 1. Query History ([History]) ─────────────────────────────────────
        try:
            from audiobench.core.db_session import get_session
            from sqlalchemy import text as _sql

            with get_session() as sdb:
                hist_rows = sdb.execute(
                    _sql("""
                        SELECT DISTINCT query_text
                        FROM search_queries
                        WHERE query_text LIKE :pattern
                          AND length(query_text) > 3
                          AND query_text NOT LIKE '/%'
                          AND query_text NOT LIKE '\\%'
                        ORDER BY id DESC
                        LIMIT 3
                    """),
                    {"pattern": f"%{trimmed}%"},
                ).fetchall()

                for h in hist_rows:
                    q_text = " ".join(h[0].split())
                    disp_q = q_text if len(q_text) <= 55 else q_text[:52] + "…"
                    if disp_q not in seen_displays:
                        seen_displays.add(disp_q)
                        yield Completion(
                            q_text,
                            start_position=-len(raw_text),
                            display=disp_q,
                            display_meta=_pad_meta("[History]"),
                        )

                # ── 2. Entities: Authors & Works ([Author], [Book]) ──────────
                target_word = word if len(word) >= 2 else trimmed
                author_rows = sdb.execute(
                    _sql("""
                        SELECT author, count(id) as work_count
                        FROM works
                        WHERE author LIKE :pattern
                          AND author IS NOT NULL
                        GROUP BY author
                        ORDER BY work_count DESC
                        LIMIT 2
                    """),
                    {"pattern": f"%{target_word}%"},
                ).fetchall()

                for a_row in author_rows:
                    author = a_row[0]
                    w_cnt = a_row[1]
                    cnt_label = f" ({w_cnt} works)" if w_cnt > 1 else ""
                    if author not in seen_displays:
                        seen_displays.add(author)
                        start_pos = -len(word) if word else -len(raw_text)
                        yield Completion(
                            f'author:"{author}"',
                            start_position=start_pos,
                            display=author,
                            display_meta=_pad_meta(f"[Author]{cnt_label}"),
                        )

                work_rows = sdb.execute(
                    _sql("""
                        SELECT title, author
                        FROM works
                        WHERE title LIKE :pattern
                          AND title IS NOT NULL
                        ORDER BY id DESC
                        LIMIT 2
                    """),
                    {"pattern": f"%{target_word}%"},
                ).fetchall()

                for w_row in work_rows:
                    w_title = w_row[0]
                    w_author = w_row[1] or "Unknown Author"
                    disp_title = w_title if len(w_title) <= 45 else w_title[:42] + "…"
                    author_short = w_author.split()[-1] if w_author else ""
                    meta_suffix = f" {author_short}" if author_short else ""
                    if disp_title not in seen_displays:
                        seen_displays.add(disp_title)
                        start_pos = -len(word) if word else -len(raw_text)
                        yield Completion(
                            f'work:"{w_title}"',
                            start_position=start_pos,
                            display=disp_title,
                            display_meta=_pad_meta(f"[Book]{meta_suffix}"),
                        )

                # ── 3. Schools of Thought ([School]) ─────────────────────────
                comm_rows = sdb.execute(
                    _sql("""
                        SELECT community_id, name
                        FROM expression_communities
                        WHERE name LIKE :pattern
                        ORDER BY community_id ASC
                        LIMIT 2
                    """),
                    {"pattern": f"%{target_word}%"},
                ).fetchall()

                for c_row in comm_rows:
                    cid = c_row[0]
                    c_name = c_row[1]
                    disp_name = f"#{cid} {c_name}" if len(c_name) <= 42 else f"#{cid} {c_name[:37]}…"
                    if disp_name not in seen_displays:
                        seen_displays.add(disp_name)
                        start_pos = -len(word) if word else -len(raw_text)
                        yield Completion(
                            f"school:{cid}",
                            start_position=start_pos,
                            display=disp_name,
                            display_meta=_pad_meta("[School]"),
                        )
        except Exception:
            pass

        # ── 4. Attributed Concept Passages ([Passage]) ───────────────────────
        if self._client is not None:
            try:
                results = self._client.autocomplete(trimmed, top_k=4)
                for r in results:
                    content = r.get("text", "")
                    if not content:
                        continue
                    clean_content = " ".join(content.split())
                    disp_snippet = _clean_passage_snippet(clean_content, max_chars=48)
                    if disp_snippet in seen_displays:
                        continue
                    seen_displays.add(disp_snippet)

                    work_title = r.get("work_title")
                    work_author = r.get("work_author")
                    speaker = r.get("speaker")
                    source_type = r.get("source_type")

                    if work_author and work_title:
                        author_last = work_author.split()[-1] if not work_author.startswith("[") else work_author
                        short_title = work_title if len(work_title) <= 20 else work_title[:18] + "…"
                        meta = f"[Passage] {author_last} — {short_title}"
                    elif work_author:
                        meta = f"[Passage] {work_author}"
                    elif speaker:
                        meta = f"[Passage] {speaker}"
                    elif source_type and source_type != "audio_transcript":
                        meta = f"[{source_type}]"
                    else:
                        meta = "[Passage]"

                    clean_snippet = disp_snippet.strip('"').rstrip("…")
                    yield Completion(
                        f'"{clean_snippet}"',
                        start_position=-len(raw_text),
                        display=disp_snippet,
                        display_meta=_pad_meta(meta),
                    )
            except Exception:
                pass


class SearchAutoSuggest(AutoSuggest):
    """Two-tier auto-suggestion:

    1. Past query history match (AutoSuggestFromHistory) - instant
    2. Command synopsis template - instant
    3. Non-blocking semantic continuation from background cache
    """

    def __init__(self, state: SearchSessionState) -> None:
        import threading

        self._state = state
        self._history = AutoSuggestFromHistory()
        self._threading = threading
        self._cache: dict[str, str] = {}
        self._last_fetch_query: str = ""
        self._last_fetch_time: float = 0.0
        try:
            from audiobench.daemon.factory import get_daemon_client
            self._client = get_daemon_client()
        except Exception:
            self._client = None

    def _fetch_background(self, text: str) -> None:
        if not self._client:
            return
        try:
            results = self._client.autocomplete(text, top_k=1)
            if results:
                matched_text = " ".join(results[0].get("text", "").split()).strip()
                if matched_text:
                    self._cache[text] = matched_text
        except Exception:
            pass

    def get_suggestion(self, buffer, document) -> Suggestion | None:
        hist_sugg = self._history.get_suggestion(buffer, document)
        if hist_sugg:
            return hist_sugg

        text = document.text

        # If typing a slash command, suggest synopsis
        if text.startswith("/") and text.endswith(" "):
            tokens = text.split()
            if len(tokens) == 1:
                cmd_name = tokens[0].lower()
                meta = get_search_command_meta(cmd_name)
                if meta and meta.synopsis:
                    return Suggestion(meta.synopsis)
            return None

        # Semantic ghost text from memory (strictly non-blocking)
        if not getattr(self._state, "autocomplete", True) or self._client is None:
            return None

        if text.startswith("/") or len(text.strip()) < 4 or document.cursor_position < len(text):
            return None

        # If cached, return suggestion immediately without blocking UI
        if text in self._cache:
            matched_text = self._cache[text]
            text_lower = text.lower()
            matched_lower = matched_text.lower()

            if matched_lower.startswith(text_lower):
                rest = matched_text[len(text):]
                if rest:
                    return Suggestion(rest)

            pos = matched_lower.find(text_lower)
            if pos != -1:
                rest = matched_text[pos + len(text):]
                if rest:
                    return Suggestion(rest)
            return None

        # Debounced trigger in background thread so prompt_toolkit main thread never stalls
        now = time.time()
        if text != self._last_fetch_query and (now - self._last_fetch_time > 0.15):
            self._last_fetch_query = text
            self._last_fetch_time = now
            self._threading.Thread(
                target=self._fetch_background, args=(text,), daemon=True
            ).start()

        return None
