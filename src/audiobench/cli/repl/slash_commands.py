"""REPL slash/meta commands — /help, /exit, /commands, /clear, /context.

Provides:
    - ALIASES: Bare-word aliases (help → /help, exit → /exit, etc.)
    - handle_slash_command(): Router for /commands. Returns True to exit.
"""

from __future__ import annotations

import os

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console
from audiobench.cli.repl.dispatch import print_context_summary
from audiobench.cli.repl.session import ReplSession

ALIASES = {
    "help": "/help",
    "exit": "/exit",
    "quit": "/exit",
    "q": "/exit",
    "clear": "/clear",
    "commands": "/commands",
}


def handle_slash_command(cmd: str, session: ReplSession) -> bool:
    """Handle a meta command. Returns True to exit REPL."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()

    if command in ("/exit", "/quit", "/q"):
        return True

    elif command in ("/help", "/h", "/?"):
        from audiobench.cli.repl.banner import print_full_help

        print_full_help(session)

    elif command == "/clear":
        os.system("clear" if os.name != "nt" else "cls")

    elif command == "/commands":
        cmds = sorted(session.cli_group.commands.keys())
        console.print(f"\n  [{BOLD}][{ACCENT}]Commands ({len(cmds)})[/][/]\n")
        categories = {
            "Core": [
                "transcribe",
                "subtitle",
                "listen",
                "speak",
                "download-voice",
            ],
            "AI": ["summarize", "ask", "chat"],
            "Data": [
                "history",
                "search",
                "show",
                "export",
                "delete",
            ],
            "Config": ["preset", "install-completion"],
            "Analytics": ["vocab"],
            "System": [
                "download",
                "info",
                "doctor",
                "status",
                "cleanup",
            ],
            "Interactive": ["repl"],
        }
        for cat, cat_cmds in categories.items():
            available = [c for c in cat_cmds if c in cmds]
            if available:
                cmd_str = "  ".join(f"[{ACCENT}]{c}[/]" for c in available)
                console.print(f"    [{BOLD}]{cat:<12}[/] {cmd_str}")
        console.print(f"\n  [{DIM}]Type <command> --help for details[/]\n")

    elif command == "/context":
        if session.context_record:
            print_context_summary(session)
        else:
            console.print(f"  [{DIM}]No context set. Use .use <ID> or transcribe a file.[/]")

    else:
        console.print(f"  [{DIM}]Unknown: {command} — type /help or help[/]")

    return False
