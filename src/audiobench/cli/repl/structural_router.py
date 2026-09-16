"""Smarter `?` Routing — fast-path structural questions to SQLite instead of LLM.

The Command Graph gives us perfect knowledge of the user's history. When they ask
"what do I usually do after transcribing?" we don't need to embed the question and
search the vector index. We can parse the intent and hit SQLite directly.
"""

from __future__ import annotations

import re

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console, make_table
from audiobench.cli.repl.session import ReplSession


def handle_structural_question(session: ReplSession, question: str) -> bool:
    """Check if a ? question matches a structural intent. 
    
    Returns True if the question was fully handled, False to fall back to the LLM `ask`.
    """
    q_lower = question.lower().strip()

    # Intent 1: "what do I usually do after X"
    match = re.match(r"(?:what|which) (?:do i|commands) (?:usually )?(?:run|do) after ([\w\\]+)", q_lower)
    if match:
        cmd = match.group(1).lstrip("\\")
        _suggest_next_command(cmd)
        return True

    # Intent 2: "how many files did I transcribe [recently]"
    if "how many" in q_lower and "transcribe" in q_lower:
        days = 7
        if "today" in q_lower:
            days = 1
        elif "month" in q_lower:
            days = 30

        _count_transcriptions(days)
        return True

    # Intent 3: "what is the most common command"
    if "most common command" in q_lower or "command stats" in q_lower:
        _show_graph_stats()
        return True

    return False


def _suggest_next_command(cmd: str) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        suggestions = repo.get_next_command_suggestions(after_command=cmd, limit=3)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not query command graph: {e}[/]")
        return

    if not suggestions:
        console.print(f"  [{DIM}]No data on what follows [{ACCENT}]\\{cmd}[/][/]")
        return

    console.print(f"\n  [{BOLD}]Command Graph:[/] After [{ACCENT}]\\{cmd}[/], you usually run:")
    for s in suggestions:
        console.print(f"    [{ACCENT}]\\{s['command']}[/] [{DIM}]({s['count']} times)[/]")
    console.print()


def _count_transcriptions(days: int) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        count = repo.count_transcriptions_since(days=days)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not query command graph: {e}[/]")
        return

    time_str = f"in the last {days} days" if days > 1 else "today"
    console.print(f"\n  [{BOLD}]Command Graph:[/] You transcribed [{ACCENT}]{count}[/] files {time_str}.\n")


def _show_graph_stats() -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        stats = repo.get_command_stats(days=30)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not query command graph: {e}[/]")
        return

    if not stats:
        console.print(f"  [{DIM}]No command history yet.[/]")
        return

    table = make_table(
        "Top Commands (Last 30 Days)",
        [("Command", {"style": ACCENT}), ("Count", {"justify": "right"})]
    )
    for row in stats[:5]:
        table.add_row(f"\\{row['command']}", str(row['count']))

    console.print()
    console.print(table)
