"""Context-aware autocomplete and ghost suggestions for Chat REPL slash commands.

Provides:
  - ChatSlashCompleter: 2-column popup with command descriptions, flags, choices, and IDs.
  - ChatAutoSuggest: History-first ghost text falling back to command synopsis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion

from audiobench.chat.command_meta import (
    COMMAND_REGISTRY,
    CommandMeta,
    get_all_command_names,
    get_command_meta,
)

if TYPE_CHECKING:
    from audiobench.chat.chat_repl import ChatREPL


def _pad_meta(desc: str) -> str:
    """Pad description metadata with leading spaces for visual separation."""
    if not desc:
        return ""
    return f"   {desc}"


class ChatSlashCompleter(Completer):
    """Context-aware slash command completer with configurable verbosity."""

    def __init__(self, repl: ChatREPL):
        self.repl = repl

    def _get_verbosity(self) -> str:
        """Fetch current verbosity level from settings."""
        try:
            return getattr(self.repl.settings, "chat_autocomplete_verbosity", "brief")
        except Exception:
            return "brief"

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        if not text_before.startswith("/"):
            return

        tokens = text_before.split()
        word = document.get_word_before_cursor(WORD=True)
        verbosity = self._get_verbosity()

        # ── Position 0: typing the command name itself ───────────────────────
        if len(tokens) == 0 or (len(tokens) == 1 and not text_before.endswith(" ")):
            yield from self._complete_command_name(word, verbosity)
            return

        # ── Position 1+: command already entered ─────────────────────────────
        cmd_name = tokens[0]
        meta = get_command_meta(cmd_name)
        if not meta:
            return

        remaining_tokens = tokens[1:]
        is_trailing_space = text_before.endswith(" ")

        # 1. Complete partial flag or trigger flag completion
        if word.startswith("-") or word.startswith("!"):
            yield from self._complete_flags(meta, word)
            return

        # 2. Check if previous token was a flag taking values (e.g., -n or --top)
        if len(tokens) >= 2 and is_trailing_space:
            prev_tok = tokens[-1]
            for flag in meta.flags:
                if prev_tok in flag.opts and flag.takes_value and flag.suggested_values:
                    for val in flag.suggested_values:
                        yield Completion(val, start_position=0, display_meta=_pad_meta(f"{val} results"))
                    return

        # 3. If trailing space and flags are available, offer unselected flags
        if is_trailing_space and meta.flags:
            used_opts = set(remaining_tokens)
            for flag in meta.flags:
                if not any(opt in used_opts for opt in flag.opts):
                    opt_display = flag.opts[0]
                    desc = flag.summary
                    if flag.takes_value and flag.value_hint:
                        desc = f"{flag.value_hint} · {desc}"
                    yield Completion(opt_display, start_position=0, display_meta=_pad_meta(desc))

        # 4. Contextual sub-choice completions (e.g. /voice provider, /voice model)
        if cmd_name in ("/voice", "/v") and len(remaining_tokens) >= 1:
            sub = remaining_tokens[0].lower()
            if sub in ("provider", "engine") and (is_trailing_space or len(remaining_tokens) == 2):
                partial = word if not is_trailing_space else ""
                start_pos = -len(partial) if partial else 0
                provider_choices = [
                    ("auto", "Cloud Gemini when online with API key, offline fallback"),
                    ("local", "Local Whisper only (private, offline, zero cloud)"),
                    ("cloud", "Gemini 2.5 Flash STT (fast, high technical accuracy)"),
                ]
                for p_name, p_desc in provider_choices:
                    if not partial or p_name.startswith(partial):
                        yield Completion(p_name, start_position=start_pos, display_meta=_pad_meta(p_desc))
                return
            elif sub in ("model", "stt-model") and (is_trailing_space or len(remaining_tokens) == 2):
                partial = word if not is_trailing_space else ""
                start_pos = -len(partial) if partial else 0
                model_choices = [
                    ("tiny", "~75MB, fastest, lowest memory"),
                    ("base", "~145MB, standard fast, balanced"),
                    ("small", "~480MB, balanced quality & speed"),
                    ("medium", "~1.5GB, studio quality transcription"),
                    ("turbo", "~1.6GB, large-v3 accuracy at 8× speed"),
                ]
                for m_name, m_desc in model_choices:
                    if not partial or m_name.startswith(partial):
                        yield Completion(m_name, start_position=start_pos, display_meta=_pad_meta(m_desc))
                return

        # 5. Complete command choices (e.g., /compare, /export, /autocomplete, /voice)
        if meta.choices and len(remaining_tokens) <= 1:
            partial = word if not is_trailing_space else ""
            start_pos = -len(partial) if partial else 0
            for choice in meta.choices:
                if not partial or choice.startswith(partial):
                    yield Completion(choice, start_position=start_pos, display_meta=_pad_meta(f"{meta.name} option"))

        # 5. Complete transcript IDs (e.g., for /unfocus, /context)
        if meta.takes_transcript_ids:
            partial_id = word.lstrip("#") if not is_trailing_space else ""
            start_pos = -len(word) if not is_trailing_space else 0
            active_ids = getattr(self.repl.session, "_transcript_ids", [])
            for tid in active_ids:
                s_tid = str(tid)
                if not partial_id or s_tid.startswith(partial_id):
                    yield Completion(s_tid, start_position=start_pos, display_meta=_pad_meta(f"Active transcript #{tid}"))

    def _complete_command_name(self, word: str, verbosity: str):
        """Yield all command names matching the current word prefix."""
        seen: set[str] = set()
        for name in get_all_command_names():
            if name.startswith(word) and name not in seen:
                seen.add(name)
                meta = get_command_meta(name)
                desc = meta.get_display_meta(verbosity) if meta else ""
                yield Completion(name, start_position=-len(word), display_meta=_pad_meta(desc))

    def _complete_flags(self, meta: CommandMeta, word: str):
        """Complete flag options matching current word prefix."""
        for flag in meta.flags:
            for opt in flag.opts:
                if opt.startswith(word):
                    desc = flag.summary
                    if flag.takes_value and flag.value_hint:
                        desc = f"{flag.value_hint} · {desc}"
                    yield Completion(opt, start_position=-len(word), display_meta=_pad_meta(desc))


class ChatAutoSuggest(AutoSuggest):
    """Two-tier ghost suggestion for Chat REPL:

    1. History-first (previous user commands or messages)
    2. Command synopsis template when user types a command + space
    """

    def __init__(self):
        self._history = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document):
        # 1. Check history first
        suggestion = self._history.get_suggestion(buffer, document)
        if suggestion:
            return suggestion

        text = document.text
        if not text.startswith("/") or not text.endswith(" "):
            return None

        tokens = text.split()
        if len(tokens) == 1:
            cmd_name = tokens[0]
            meta = get_command_meta(cmd_name)
            if meta and meta.synopsis:
                return Suggestion(meta.synopsis)

        return None
