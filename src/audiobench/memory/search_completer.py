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


class SearchSemanticCompleter(Completer):
    """Semantic fragment completions for plain query text (triggered on Tab)."""

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
        if self._client is None:
            return
        text = document.text_before_cursor
        if text.startswith("/") or len(text.strip()) < 3:
            return
        self._latest = text
        time.sleep(self._debounce_ms / 1000.0)
        if self._latest != text:
            return
        try:
            results = self._client.autocomplete(text, top_k=6)
            for r in results:
                content = r.get("text", "")
                if not content:
                    continue
                clean_content = " ".join(content.split())
                speaker = r.get("speaker")
                source_type = r.get("source_type")
                display = clean_content if len(clean_content) <= 65 else clean_content[:62] + "…"
                if speaker:
                    meta = f"[{speaker}]"
                elif source_type and source_type != "audio_transcript":
                    meta = f"[{source_type}]"
                else:
                    meta = ""
                yield Completion(
                    clean_content,
                    start_position=-len(text),
                    display=display,
                    display_meta=f"   {meta}" if meta else "",
                )
        except Exception:
            pass


class SearchAutoSuggest(AutoSuggest):
    """Two-tier auto-suggestion:

    1. Past query history match (AutoSuggestFromHistory)
    2. Semantic transcript continuation from daemon
    3. Command synopsis template
    """

    def __init__(self, state: SearchSessionState) -> None:
        self._state = state
        self._history = AutoSuggestFromHistory()
        try:
            from audiobench.daemon.factory import get_daemon_client
            self._client = get_daemon_client()
        except Exception:
            self._client = None

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

        # Semantic ghost text from memory
        if not getattr(self._state, "autocomplete", True) or self._client is None:
            return None

        if text.startswith("/") or len(text.strip()) < 4 or document.cursor_position < len(text):
            return None

        try:
            results = self._client.autocomplete(text, top_k=1)
            if not results:
                return None
            matched_text = " ".join(results[0].get("text", "").split()).strip()
            if not matched_text:
                return None

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
        except Exception:
            pass

        return None
