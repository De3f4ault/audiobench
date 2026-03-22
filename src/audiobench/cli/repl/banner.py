"""REPL banner, onboarding, help text, and goodbye messages."""

from __future__ import annotations

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    console,
    format_duration,
)
from audiobench.cli.repl.dot_commands import DOT_COMMANDS
from audiobench.cli.repl.session import ReplSession


def print_banner(session: ReplSession) -> None:
    cmd_count = len(session.cli_group.commands)
    dot_count = len(DOT_COMMANDS)

    console.print(f"""
  [{BOLD}][{ACCENT}]╭────────────────────────────────────────────╮[/][/]
  [{BOLD}][{ACCENT}]│  {APP_NAME} REPL{" " * 31}│[/][/]
  [{BOLD}][{ACCENT}]│  v{APP_VERSION}  •  {cmd_count} commands  \
•  {dot_count} dot-commands{" " * 6}│[/][/]
  [{BOLD}][{ACCENT}]╰────────────────────────────────────────────╯[/][/]
""")


def print_full_help(session: ReplSession) -> None:
    """Print comprehensive REPL help."""
    ctx_label = f"#{session.last_id}" if session.last_id else "none"
    ctx_name = ""
    if session.context_record:
        fname = session.context_record.get("file_name", "")
        ctx_name = f" ({fname})" if fname else ""

    console.print(f"""
  [{BOLD}][{ACCENT}]{APP_NAME} REPL — Interactive Shell[/][/]
  [{DIM}]{"─" * 50}[/]

  [{BOLD}]1. Run any command[/] [{DIM}](without 'audiobench' prefix)[/]
     [{ACCENT}]transcribe interview.mp3[/]     Transcribe a file
     [{ACCENT}]history --tail 5[/]            Recent transcriptions
     [{ACCENT}]search "meeting"[/]            Search transcripts
     [{ACCENT}]doctor[/]                       Check system health

  [{BOLD}]2. Context-aware[/] [{DIM}](current: {ctx_label}{ctx_name})[/]
     When context is set, these auto-fill the transcript ID:
     [{ACCENT}]show[/]   [{ACCENT}]ask "..."[/]   [{ACCENT}]summarize[/]   \
[{ACCENT}]vocab[/]   [{ACCENT}]export -f srt[/]

  [{BOLD}]3. Dot commands[/] [{DIM}](quick actions on context)[/]
     [{ACCENT}].stats[/]  [{ACCENT}].show[/]  [{ACCENT}].segments[/]  \
[{ACCENT}].vocab[/]  [{ACCENT}].info[/]  [{ACCENT}].find "..."[/]
     [{ACCENT}].play[/]  [{ACCENT}].play 01:25[/]  [{ACCENT}].play segment 3[/]  \
[{ACCENT}].open[/]  [{ACCENT}].path[/]
     [{ACCENT}].ask "..."[/]  [{ACCENT}].chat[/]  [{ACCENT}].summarize[/]  \
[{ACCENT}].export srt[/]  [{ACCENT}].edit[/]
     [{ACCENT}].use <ID>[/]  [{ACCENT}].clear[/]  [{ACCENT}].next[/]  \
[{ACCENT}].prev[/]  [{ACCENT}].recent[/]  [{ACCENT}].search "..."[/]

  [{BOLD}]4. Shortcuts[/]
     [{ACCENT}]$last[/]  Expands to context ID ({ctx_label})
     [{ACCENT}]? ...[/]  AI question shorthand: \
[{ACCENT}]? what are the key points?[/]

  [{BOLD}]5. Meta[/]
     [{ACCENT}]help[/] [{ACCENT}]/help[/]  This help       \
[{ACCENT}]/commands[/]  All commands
     [{ACCENT}]/clear[/]  Clear screen     \
[{ACCENT}]/context[/]  Show context
     [{ACCENT}]/exit[/]  Quit             \
[{ACCENT}]!<cmd>[/]    Shell escape
""")


def print_onboarding(session: ReplSession) -> None:
    """Show a helpful onboarding for new or returning users."""
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=5)
        if records:
            console.print(f"  [{BOLD}]Recent transcriptions:[/]")
            for r in records:
                duration = r.get("duration", 0) or 0
                dur_str = format_duration(duration) if duration else "?"
                console.print(
                    f"    [{ACCENT}]#{r['id']:<4}[/] "
                    f"{r.get('file_name', '?'):<30} "
                    f"[{DIM}]{r.get('word_count', 0):>5,} words "
                    f" {dur_str:>8}[/]"
                )
            console.print(
                f"\n  [{DIM}]Tip:[/] "
                f"[{ACCENT}].use {records[0]['id']}[/] "
                f"[{DIM}]to set context, or[/] "
                f"[{ACCENT}]transcribe <file>[/] "
                f"[{DIM}]for a new one. Type[/] "
                f"[{ACCENT}]help[/] [{DIM}]for guide.[/]"
            )
        else:
            console.print(
                f"  [{DIM}]No transcriptions yet. "
                f"Start with:[/] "
                f"[{ACCENT}]transcribe <audio_file>[/]"
            )
    except Exception:
        console.print(
            f"  [{DIM}]Type[/] [{ACCENT}]help[/] "
            f"[{DIM}]for commands, or[/] "
            f"[{ACCENT}]transcribe <file>[/] "
            f"[{DIM}]to begin.[/]"
        )
    console.print()


def print_goodbye(session: ReplSession) -> None:
    if session._command_count > 0:
        console.print(f"  [{DIM}]Session: {session._command_count} command(s) run[/]")
    console.print(f"  [{DIM}]Goodbye![/]\n")
