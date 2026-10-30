"""REPL banner, onboarding, help text, and goodbye messages."""

from __future__ import annotations

import datetime

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    ERROR,
    WARNING,
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
     [{ACCENT}]show[/]   [{ACCENT}]ask "..."[/]   [{ACCENT}]summarize[/]   [{ACCENT}]vocab[/]   [{ACCENT}]export -f srt[/]

  [{BOLD}]3. Knowledge Graph & Cross-Work Intelligence[/]
     [{ACCENT}]\\synthesize "topic"[/]       Dialectic synthesis across library via Graph MMR
     [{ACCENT}]\\suggest[/]                  Next Listen recommendations & bridge discovery
     [{ACCENT}]\\related[/]                  Explore semantic relatives for focused file
     [{ACCENT}]\\play cite <n>[/]            Listen to cited segment directly in daemon
     [{ACCENT}]\\bookmark cite <n>[/]        Save smart bookmark card from synthesis citation
     [{ACCENT}]\\graph seed[/]               Seed cross-work semantic edges in background
     [{ACCENT}]\\proposals[/]                Review & authorize daemon proposals (knowledge gaps & operators)

  [{BOLD}]4. Context & Action Commands[/] [{DIM}](quick actions on context)[/]
     [{ACCENT}]\\show[/]   [{ACCENT}]\\stats[/]   [{ACCENT}]\\focus[/]   [{ACCENT}]\\ls[/]   [{ACCENT}]\\info[/]
     [{ACCENT}]\\chat[/]   [{ACCENT}]\\ask "..."[/]   [{ACCENT}]\\summarize[/]   [{ACCENT}]\\search[/]   [{ACCENT}]\\bookmark[/]

  [{BOLD}]5. Dot-commands — Semantic Layer[/]
     [{ACCENT}].search[/] [{DIM}]"concept"[/]   LanceDB ANN — cross-source, no LLM
     [{ACCENT}].help[/]               Dot-command reference

  [{BOLD}]6. Search & Synthesis Modes[/]
     [{ACCENT}]\\search[/] [{DIM}]"keyword"[/]  SQLite FTS — instant, exact text
     [{ACCENT}].search[/] [{DIM}]"concept"[/]  LanceDB    — semantic, all sources
     [{ACCENT}]?[/] [{DIM}]"question"[/]       LLM synthesis — focused transcript
     [{ACCENT}]\\synthesize[/] [{DIM}]"topic"[/] Knowledge graph — library-wide synthesis

  [{BOLD}]7. Shortcuts[/]
     [{ACCENT}]$last[/]  Expands to context ID ({ctx_label})
     [{ACCENT}]? ...[/]  AI question shorthand: [{ACCENT}]? what are the key points?[/]
     [{ACCENT}]!ls -la[/] Shell escape passthrough

  [{BOLD}]8. Help & Meta[/]
     [{ACCENT}]help[/] [{ACCENT}]/help[/]      REPL overview
     [{ACCENT}]\\help <cmd>[/]      Detailed command docs & tips (e.g. \\help synthesize)
     [{ACCENT}]/commands[/]        List all CLI commands
     [{ACCENT}]/clear[/]           Clear screen
     [{ACCENT}]/context[/]         Show focused entity
     [{ACCENT}]/exit[/]            Quit REPL
""")


# Rotating tips — one shown per session (changes daily)
_TIPS = [
    f"Run commands in the background with [{ACCENT}]transcribe file.mp4 &[/] — stay in the REPL while it processes.",
    f"Use [{ACCENT}]work <file>[/] to set context on an audio file before transcribing — commands like \\play and \\info work immediately.",
    f"[{ACCENT}]? what were the action items?[/] — the [b]?[/] shorthand routes your question to the current transcript's AI.",
    f"[{ACCENT}]\\search 'keyword'[/] or [{ACCENT}].search 'keyword'[/] searches semantic memory and transcripts.",
    f"[{ACCENT}]\\show[/] prints the current transcript text directly in the terminal.",
    f"[{ACCENT}]\\context[/] shows a full summary of the current file and transcript focus.",
    f"[{ACCENT}]\\focus <id>[/] or [{ACCENT}].use <id>[/] loads any transcript or audio file into the active session.",
    f"[{ACCENT}]transcribe --diarize meeting.m4a[/] identifies speakers and marks voiceprints.",
    f"[{ACCENT}]\\related[/] shows semantic connections, works, and expressions linked to the focused file.",
    f"[{ACCENT}]\\synthesize 'concept'[/] (or [{ACCENT}]\\synth[/]) synthesizes cross-work ideas across your entire library.",
    f"After [{ACCENT}]\\synthesize[/], use [{ACCENT}]\\play cite 1[/] to listen to the source or [{ACCENT}]\\bookmark cite 1[/] to save it.",
    f"[{ACCENT}]\\suggest[/] surfaces high-affinity Next Listen paths and triadic consensus connections.",
    f"Type [{ACCENT}]\\help <command>[/] (e.g. [{ACCENT}]\\help synthesize[/]) for syntax, options, and flywheel tips.",
    f"preset create meeting --model large-v3 --accurate saves your preferred settings as a named preset.",
    f"[{ACCENT}]\\chat[/] opens a conversational AI session over the current transcript.",
    f"[{ACCENT}]\\jobs[/] or [{ACCENT}]jobs[/] shows all background transcription jobs and their status at a glance.",
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
                status = r.get("status") or r.get("pipeline_phase") or "complete"
                duration = r.get("duration", 0) or 0
                dur_str = format_duration(duration) if duration else "?"
                if status == "failed":
                    status_str = f"[{ERROR}]failed (0 words)[/]"
                elif status in ("transcribing", "queued", "running"):
                    status_str = f"[{WARNING}]in progress[/]"
                else:
                    status_str = f"[{DIM}]{r.get('word_count', 0):>5,} words  {dur_str:>8}[/]"
                console.print(
                    f"    [{ACCENT}]#{r['id']:<4}[/] "
                    f"{r.get('file_name', '?'):<35} "
                    f"{status_str}"
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

        # Check for pending proposals (shown for both first-run and returning sessions)
        try:
            from audiobench.cli.repl.backslash_commands import _fetch_daemon_proposals
            from audiobench.daemon.intelligence.proposal_cluster import canonical_cluster_key

            pending_props = _fetch_daemon_proposals()
            if pending_props:
                clusters = {}
                for p in pending_props:
                    ck = canonical_cluster_key(p.get("content", ""))
                    clusters.setdefault(ck, []).append(p)
                cluster_count = len(clusters)
                total_runs = len(pending_props)
                plural = "proposal" if cluster_count == 1 else "proposals"
                runs_note = f" [{DIM}]({total_runs} runs)[/]" if total_runs > cluster_count else ""
                console.print(
                    f"\n  [{BOLD}][{WARNING}]Pending Proposals:[/] [{BOLD}]{cluster_count} {plural}[/]{runs_note} "
                    f"[{DIM}](type[/] [{ACCENT}]\\proposals[/] [{DIM}]to inspect and authorize)[/]"
                )
        except Exception:
            pass
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
