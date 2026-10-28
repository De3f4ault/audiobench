from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, AutoSuggest
from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PtStyle
from prompt_toolkit.lexers import Lexer

def _pad_meta(desc: str) -> str:
    """Pad metadata string with leading spaces for visual separation."""
    if not desc:
        return ""
    return f"   {desc}"


class SlashCommandCompleter(Completer):
    """Complete slash commands when the buffer starts with '/'."""

    def __init__(self, commands: list[str] | dict[str, str] | list[tuple[str, str]] | None = None):
        if isinstance(commands, dict):
            self.command_dict = dict(commands)
        elif commands and isinstance(commands[0], tuple):
            self.command_dict = dict(commands)
        else:
            self.command_dict = {c: "" for c in (commands or [])}

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in self.command_dict.items():
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=_pad_meta(desc),
                )

class UnifiedCompleter(Completer):
    """Route to slash or semantic completer based on buffer content."""
    def __init__(self, slash_completer: Completer, semantic_completer: Completer | None = None):
        self._slash = slash_completer
        # Note: if the provided semantic_completer is not thread-safe, the caller should wrap it in ThreadedCompleter
        self._semantic = semantic_completer

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            yield from self._slash.get_completions(document, complete_event)
        elif self._semantic:
            yield from self._semantic.get_completions(document, complete_event)

def make_prompt_session(
    slash_commands: list[str] | None = None,
    history_path: Path = Path("history.txt"),
    style_overrides: dict | None = None,
    semantic_completer: Completer | None = None,
    key_bindings = None,
    lexer: Lexer | None = None,
    auto_suggest: AutoSuggest | None = None,
    complete_while_typing: bool | Any | None = None,
    slash_completer: Completer | None = None,
) -> PromptSession:
    """Build a prompt_toolkit PromptSession configured for a consistent REPL experience.

    Features:
      - Adaptive autocomplete (instant on slash commands, Tab-triggered on queries)
      - History tracking
      - Ghost-text auto-suggest
      - Multi-bucket semantic completions
      - Refined dark-mode palette tokens
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default styling for autocomplete popup and ghost text
    base_style = {
        "completion-menu":                       "bg:#141414 #d8d8d8",
        "completion-menu.completion":            "bg:#141414 #d8d8d8",
        "completion-menu.completion.current":    "bg:#004f6e #ffffff bold",
        "completion-menu.meta.completion":       "bg:#141414 #7d8590",
        "completion-menu.meta.completion.current": "bg:#004f6e #a5f3fc",
        "auto-suggestion":                       "#4a5568",
    }
    if style_overrides:
        base_style.update(style_overrides)
        
    prompt_style = PtStyle.from_dict(base_style)
    
    slash = slash_completer or SlashCommandCompleter(slash_commands or [])
    completer = UnifiedCompleter(slash, semantic_completer)

    if complete_while_typing is None:
        from prompt_toolkit.filters import Condition

        def _should_complete_while_typing() -> bool:
            try:
                from prompt_toolkit.application.current import get_app
                app = get_app()
                text = app.current_buffer.text_before_cursor
                return text.lstrip().startswith("/")
            except Exception:
                return False

        effective_complete_while_typing = Condition(_should_complete_while_typing)
    else:
        effective_complete_while_typing = complete_while_typing
    
    return PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=auto_suggest or AutoSuggestFromHistory(),
        completer=completer,
        style=prompt_style,
        key_bindings=key_bindings,
        lexer=lexer,
        include_default_pygments_style=False if lexer else True,
        complete_while_typing=effective_complete_while_typing,
    )
