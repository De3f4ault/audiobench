"""REPL banner, onboarding, help text, and goodbye messages."""

from __future__ import annotations

import datetime

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    console,
    format_duration,
)
from audiobench.cli.repl.session import ReplSession


def print_banner(session: ReplSession) -> None:
    cmd_count = len(session.cli_group.commands)

    from rich.panel import Panel

    from audiobench.cli.display.theme import BOX_STYLE

    content = f"[{BOLD}][{ACCENT}]v{APP_VERSION}  •  {cmd_count} commands[/][/]"
    console.print(
        Panel(
            content,
            title=f"[{BOLD}][{ACCENT}]{APP_NAME} REPL[/][/]",
            title_align="left",
            border_style=ACCENT,
            box=BOX_STYLE,
            expand=False,
        )
    )


def print_full_help(session: ReplSession) -> None:
    """Print comprehensive REPL help."""
    ctx_label = f"#{session.last_id}" if session.last_id else "none"
    ctx_name = ""
    if session.focus:
        ctx_name = f" ({session.focus.label})"

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

  [{BOLD}]3. Context commands[/] [{DIM}](quick actions on context)[/]
     [{ACCENT}]\\stats[/]  [{ACCENT}]\\show[/]  [{ACCENT}]\\focus[/]  \
[{ACCENT}]\\ls[/]
     [{ACCENT}]\\ask "..."[/]  [{ACCENT}]\\chat[/]  [{ACCENT}]\\summarize[/]  \
[{ACCENT}]\\search[/]  [{ACCENT}]\\help[/]

  [{BOLD}]4. Shell Passthrough[/]
     [{ACCENT}]!ls -la[/]  [{ACCENT}]!pwd[/]  [{ACCENT}]!ffmpeg -i ...[/]

  [{BOLD}]5. Dot-commands — Semantic Layer[/]
     [{ACCENT}].search[/] [{DIM}]"concept"[/]   LanceDB ANN — cross-source, no LLM
     [{ACCENT}].help[/]               Dot-command reference

  [{BOLD}]6. Search Modes[/]
     [{ACCENT}]\\search[/] [{DIM}]"keyword"[/]  SQLite FTS — instant, exact text
     [{ACCENT}].search[/] [{DIM}]"concept"[/]  LanceDB    — semantic, all sources
     [{ACCENT}]?[/] [{DIM}]"question"[/]       LLM synthesis — generates answer

  [{BOLD}]7. Shortcuts[/]
     [{ACCENT}]$last[/]  Expands to context ID ({ctx_label})
     [{ACCENT}]? ...[/]  AI question shorthand: \
[{ACCENT}]? what are the key points?[/]

  [{BOLD}]8. Meta[/]
     [{ACCENT}]help[/] [{ACCENT}]/help[/]  This help       \
[{ACCENT}]/commands[/]  All commands
     [{ACCENT}]/clear[/]  Clear screen     \
[{ACCENT}]/context[/]  Show context
     [{ACCENT}]/exit[/]  Quit             \
[{ACCENT}]!<cmd>[/]    Shell escape
""")


# Rotating tips — one shown per session (changes daily)
_TIPS = [
    f"Run commands in the background with [{ACCENT}]transcribe file.mp4 &[/] — stay in the REPL while it processes.",
    f"Use [{ACCENT}]work <file>[/] to set context on an audio file before transcribing — commands like .play and .info work immediately.",
    f"[{ACCENT}]? what were the action items?[/] — the [b]?[/] shorthand routes your question to the current transcript's AI.",
    f"[{ACCENT}].find 'keyword'[/] jumps to every segment where that word appears, with timestamps.",
    f"[{ACCENT}].export srt[/] / [b].export json[/] — re-export the current transcript without re-transcribing.",
    f"[{ACCENT}]/context[/] shows a full summary of the current file and transcript focus.",
    f"[{ACCENT}].next[/] / [b].prev[/] navigate through your transcription history without leaving the REPL.",
    f"[{ACCENT}]transcribe --diarize meeting.m4a[/] identifies speakers. Run [b].diarize[/] afterwards to re-run it interactively.",
    f"[{ACCENT}].edit[/] opens the transcript in $EDITOR — changes are saved back to the database.",
    f"[{ACCENT}]preset create meeting --model large-v3 --accurate[/] saves your preferred settings as a named preset.",
    f"[{ACCENT}].chat[/] opens a full conversational AI session over the current transcript.",
    f"[{ACCENT}].jobs[/] shows all background transcription jobs and their status at a glance.",
]


def _pick_tip() -> str:
    """Pick a daily tip (stable within the same day)."""
    day_of_year = datetime.date.today().timetuple().tm_yday
    return _TIPS[day_of_year % len(_TIPS)]


def print_onboarding(session: ReplSession) -> None:
    """Show a helpful onboarding for new or returning users."""
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=5)
        if records:
            # ── Returning user: show recent transcriptions ──
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
            # Quick-start tip: load the most recent file
            most_recent_file = records[0].get("file_name", "<file>")
            console.print(
                f"\n  [{DIM}]Tip:[/] "
                f"[{ACCENT}]work {most_recent_file}[/] "
                f"[{DIM}]to resume where you left off. Type[/] "
                f"[{ACCENT}]help[/] [{DIM}]for guide.[/]"
            )
            # Rotating daily tip
            console.print(f"  [{DIM}]━━ Today's tip:[/] {_pick_tip()}")
        else:
            # ── First-run: focused getting-started guide ──
            console.print(f"  [{BOLD}]Welcome to AudioBench![/]")
            console.print()
            console.print(f"  [{DIM}]Get started in 3 steps:[/]")
            console.print(
                f"    [{ACCENT}]1.[/] [{DIM}]Drop a file in:[/]  [{ACCENT}]work interview.mp3[/]"
            )
            console.print(
                f"    [{ACCENT}]2.[/] [{DIM}]Transcribe it:[/]  [{ACCENT}]transcribe[/]  [{DIM}](or[/] [{ACCENT}]transcribe --interactive[/] [{DIM}]for guided setup)[/]"
            )
            console.print(
                f"    [{ACCENT}]3.[/] [{DIM}]Ask questions:[/] [{ACCENT}]? what were the main points?[/]"
            )
            console.print()
            console.print(
                f"  [{DIM}]Type[/] [{ACCENT}]help[/] [{DIM}]for a full guide, or[/] [{ACCENT}]/commands[/] [{DIM}]to see everything.[/]"
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
