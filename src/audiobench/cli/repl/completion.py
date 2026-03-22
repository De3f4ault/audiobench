"""REPL tab completion — readline-based completion for commands and dot-commands."""

from __future__ import annotations

import readline

from audiobench.cli.repl.dot_commands import DOT_COMMANDS
from audiobench.cli.repl.session import ReplSession


def setup_completion(session: ReplSession) -> None:
    """Set up readline tab-completion."""
    commands = sorted(session.cli_group.commands.keys())
    dot_cmds = sorted(DOT_COMMANDS.keys())
    meta_words = ["help", "exit", "quit", "clear", "commands"]
    slash_cmds = [
        "/help",
        "/commands",
        "/clear",
        "/exit",
        "/context",
    ]
    all_completions = commands + dot_cmds + slash_cmds + meta_words

    def completer(text: str, state: int) -> str | None:
        matches = [c for c in all_completions if c.startswith(text)]
        if not matches:
            import glob

            matches = glob.glob(text + "*")
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")
