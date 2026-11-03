"""Command metadata and schema registry for Chat REPL slash commands.

Provides descriptions, parameter specs, flag definitions, and synopsis strings
to power tldr-style autocompletion, ghost suggestions, and help documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlagMeta:
    """Specification of an inline command flag."""

    opts: list[str]
    summary: str
    takes_value: bool = False
    value_hint: str = ""
    default: str = ""
    suggested_values: list[str] = field(default_factory=list)


@dataclass
class CommandMeta:
    """Specification of a Chat REPL slash command."""

    name: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    synopsis: str = ""
    flags: list[FlagMeta] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    example: str = ""
    takes_transcript_ids: bool = False

    @property
    def all_names(self) -> list[str]:
        return [self.name] + self.aliases

    def get_display_meta(self, verbosity: str = "brief") -> str:
        """Return the display description according to verbosity mode."""
        v = (verbosity or "brief").lower()
        if v == "minimal":
            return ""
        if v == "detailed":
            if self.synopsis:
                return f"{self.synopsis} · {self.summary}"
            return self.summary
        # Default: brief
        return self.summary


# ── Command Registry ──────────────────────────────────────────────────────────

COMMAND_REGISTRY: list[CommandMeta] = [
    CommandMeta(
        name="/search",
        aliases=["/find"],
        summary="Search transcripts & inject snippets into context",
        synopsis="[-n k] [--all] <query>",
        example="/search --all -n 3 eye contact",
        flags=[
            FlagMeta(
                opts=["--all", "!"],
                summary="Search entire library (global scope)",
            ),
            FlagMeta(
                opts=["-n", "--top"],
                summary="Limit result count (1–20, default 8)",
                takes_value=True,
                value_hint="<k>",
                default="8",
                suggested_values=["3", "5", "8", "12", "20"],
            ),
        ],
    ),
    CommandMeta(
        name="/unsearch",
        aliases=["/clear-search", "/clear-fragments"],
        summary="Clear injected search snippets from context",
        synopsis="",
        example="/unsearch",
    ),
    CommandMeta(
        name="/load",
        aliases=["/focus", "/browse"],
        summary="Browse catalog or load transcripts by query or ID",
        synopsis="[id|query]",
        example="/load 1182",
        takes_transcript_ids=True,
    ),
    CommandMeta(
        name="/unfocus",
        aliases=["/remove"],
        summary="Remove transcripts from context (interactive picker)",
        synopsis="[id|all|skipped]",
        example="/unfocus skipped",
        choices=["all", "skipped"],
        takes_transcript_ids=True,
    ),
    CommandMeta(
        name="/context",
        summary="Inspect active transcripts & injected search snippets",
        synopsis="[id|clear|clear fragments]",
        example="/context clear fragments",
        choices=["clear", "clear fragments"],
        takes_transcript_ids=True,
    ),
    CommandMeta(
        name="/model",
        summary="Switch active LLM model or view current",
        synopsis="[name]",
        example="/model gemini",
        choices=["ollama", "gemini", "default"],
    ),
    CommandMeta(
        name="/compare",
        summary="Toggle side-by-side model comparison or turn off",
        synopsis="<model|off>",
        example="/compare gemini",
        choices=["off", "ollama", "gemini"],
    ),
    CommandMeta(
        name="/think",
        summary="Toggle thinking/reasoning display",
        synopsis="",
        example="/think",
    ),
    CommandMeta(
        name="/clear",
        summary="Clear chat history and active context",
        synopsis="",
        example="/clear",
    ),
    CommandMeta(
        name="/history",
        summary="Browse and resume past conversations",
        synopsis="",
        example="/history",
    ),
    CommandMeta(
        name="/sessions",
        aliases=["/switch"],
        summary="List study sessions or switch to session ID",
        synopsis="[id]",
        example="/sessions",
    ),
    CommandMeta(
        name="/rename",
        summary="Rename current conversation title",
        synopsis="<title>",
        example="/rename Deep Research Study",
    ),
    CommandMeta(
        name="/export",
        summary="Export conversation to markdown, json, or txt",
        synopsis="[format] [path]",
        example="/export md",
        choices=["md", "json", "txt"],
    ),
    CommandMeta(
        name="/retry",
        summary="Regenerate last assistant response",
        synopsis="",
        example="/retry",
    ),
    CommandMeta(
        name="/bookmarks",
        summary="View bookmarks for active transcripts",
        synopsis="[type]",
        example="/bookmarks",
    ),
    CommandMeta(
        name="/autocomplete",
        summary="Configure autocomplete verbosity (minimal/brief/detailed)",
        synopsis="[minimal|brief|detailed]",
        example="/autocomplete detailed",
        choices=["minimal", "brief", "detailed"],
    ),
    CommandMeta(
        name="/play",
        summary="Play audio for loaded transcript or seek to timestamp",
        synopsis="[timestamp|id]",
        example="/play 6:30",
        takes_transcript_ids=True,
    ),
    CommandMeta(
        name="/seek",
        summary="Seek audio to absolute (6:30) or relative (+15) timestamp",
        synopsis="<timestamp>",
        example="/seek 10:30",
    ),
    CommandMeta(
        name="/pause",
        aliases=["/resume", "/toggle"],
        summary="Pause, resume, or toggle audio playback",
        synopsis="",
        example="/pause",
    ),
    CommandMeta(
        name="/speed",
        summary="Set playback speed multiplier (e.g., 1.5)",
        synopsis="<multiplier>",
        example="/speed 1.5",
        choices=["0.75", "1.0", "1.25", "1.5", "2.0"],
    ),
    CommandMeta(
        name="/stop",
        summary="Stop audio playback and release player",
        synopsis="",
        example="/stop",
    ),
    CommandMeta(
        name="/np",
        aliases=["/nowplaying", "/status"],
        summary="Display current playback position and status",
        synopsis="",
        example="/np",
    ),
    CommandMeta(
        name="/explain",
        summary="Explain speaker's context around current audio position",
        synopsis="[timestamp]",
        example="/explain",
    ),
    CommandMeta(
        name="/clip",
        summary="Extract an audio clip between two timestamps with ffmpeg",
        synopsis="<start> <end> [filename]",
        example="/clip 6:30 7:00 highlight.mp3",
    ),
    CommandMeta(
        name="/next",
        aliases=["/n"],
        summary="Jump to next cited timestamp in audio",
        synopsis="",
        example="/next",
    ),
    CommandMeta(
        name="/prev",
        aliases=["/p", "/back", "/b"],
        summary="Jump to previous cited timestamp in audio",
        synopsis="",
        example="/prev",
    ),
    CommandMeta(
        name="/lyrics",
        aliases=["/karaoke", "/follow"],
        summary="Open synchronized live lyrics / follow HUD for playing audio",
        synopsis="",
        example="/lyrics",
    ),
    CommandMeta(
        name="/info",
        aliases=["/dossier"],
        summary="View 360-degree dossier for current or specified audio file",
        synopsis="[id|#tx|name]",
        example="/info",
    ),
    CommandMeta(
        name="/voice",
        aliases=["/v"],
        summary="Voice mode (single-turn push-to-talk, continuous live dialogue, voice picker, STT settings)",
        synopsis="[live|status|picker|provider|model|test] [--no-record]",
        example="/voice provider auto",
        choices=["live", "status", "picker", "provider", "model", "test"],
        flags=[
            FlagMeta(
                opts=["--no-record", "--no-audio"],
                summary="Do not save session audio to disk",
            ),
        ],
    ),
    CommandMeta(
        name="/speak",
        aliases=["/read"],
        summary="Speak text or latest assistant message aloud using Piper TTS",
        synopsis="[msg <id>|text]",
        example="/speak",
    ),
    CommandMeta(
        name="/voices",
        aliases=["/voice-picker"],
        summary="Open interactive Piper TTS voice audition and picker",
        synopsis="",
        example="/voices",
    ),
    CommandMeta(
        name="/help",
        summary="Show help and command list",
        synopsis="",
        example="/help",
    ),
    CommandMeta(
        name="/exit",
        aliases=["/quit", "/q"],
        summary="Exit the chat REPL",
        synopsis="",
        example="/exit",
    ),
]


# ── Lookup Indexes ────────────────────────────────────────────────────────────

_NAME_INDEX: dict[str, CommandMeta] = {}
for _meta in COMMAND_REGISTRY:
    for _n in _meta.all_names:
        _NAME_INDEX[_n] = _meta


def get_command_meta(name: str) -> CommandMeta | None:
    """Find command metadata by command name or alias."""
    return _NAME_INDEX.get(name)


def get_all_command_names() -> list[str]:
    """Return all command names and aliases."""
    return list(_NAME_INDEX.keys())


def render_chat_help() -> None:
    """Render comprehensive interactive help guide generated directly from COMMAND_REGISTRY."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console

    console.print()
    console.print(f"  [{BOLD}]AudioBench Chat — Slash Commands & Help[/]")
    console.print(f"  [{DIM}]{'─' * 56}[/]")
    console.print()

    # Core table of slash commands
    for meta in COMMAND_REGISTRY:
        syn = f" {meta.synopsis}" if meta.synopsis else ""
        alias_str = f" [dim](alias: {', '.join(meta.aliases)})[/dim]" if meta.aliases else ""
        cmd_head = f"    [{BOLD}]{meta.name}[/][{DIM}]{syn}[/]{alias_str}"
        console.print(f"{cmd_head:<48}  {meta.summary}")

        # Show flags or choices if defined
        if meta.flags:
            for flag in meta.flags:
                opts = ", ".join(flag.opts)
                v_hint = f" {flag.value_hint}" if flag.value_hint else ""
                console.print(f"      [{ACCENT}]{opts}{v_hint}[/]  [{DIM}]— {flag.summary}[/]")
        if meta.example:
            console.print(f"      [{DIM}]example: [italic]{meta.example}[/italic][/]")

    console.print()
    console.print(f"  [{BOLD}]Multi-line Input[/]")
    console.print(f"  [{DIM}]{'─' * 56}[/]")
    console.print("  Type [bold]triple-quotes (\"\"\")[/] on its own line or press [bold]Alt+Enter[/bold] to submit a block.\n")

