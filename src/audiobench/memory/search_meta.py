"""Command metadata and schema registry for Memory Search REPL slash commands.

Provides descriptions, parameter specs, flag definitions, and synopsis strings
to power tldr-style autocompletion, ghost suggestions, and help documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlagMeta:
    """Specification of an inline command flag or option."""

    opts: list[str]
    summary: str
    takes_value: bool = False
    value_hint: str = ""
    default: str = ""
    suggested_values: list[str] = field(default_factory=list)


@dataclass
class CommandMeta:
    """Specification of a Search REPL slash command."""

    name: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    synopsis: str = ""
    flags: list[FlagMeta] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    example: str = ""

    @property
    def all_names(self) -> list[str]:
        return [self.name] + self.aliases


SEARCH_COMMAND_REGISTRY: list[CommandMeta] = [
    CommandMeta(
        name="/set",
        aliases=["/settings", "/config"],
        summary="Configure search presets, layout, and parameters",
        synopsis="<option> [value]",
        example="/set deep",
        choices=[
            "fast",
            "balanced",
            "deep",
            "synthesis",
            "layout book",
            "layout list",
            "autocomplete on",
            "autocomplete off",
            "width",
            "model",
            "diversity-weight",
            "reset",
        ],
    ),
    CommandMeta(
        name="/focus",
        summary="Scope search to a specific transcript ID",
        synopsis="<id>",
        example="/focus 1182",
    ),
    CommandMeta(
        name="/unfocus",
        summary="Clear active transcript scoping",
        synopsis="",
        example="/unfocus",
    ),
    CommandMeta(
        name="/pin",
        summary="Pin search snippet to persistent scratchpad",
        synopsis="<id>",
        example="/pin 1",
    ),
    CommandMeta(
        name="/unpin",
        summary="Unpin snippet from scratchpad",
        synopsis="<id>",
        example="/unpin 1",
    ),
    CommandMeta(
        name="/pins",
        summary="Display all currently pinned snippets",
        synopsis="",
        example="/pins",
    ),
    CommandMeta(
        name="/history",
        summary="View search query history and prior results",
        synopsis="",
        example="/history",
    ),
    CommandMeta(
        name="/summary",
        summary="Generate AI summary across current results",
        synopsis="",
        example="/summary",
    ),
    CommandMeta(
        name="/sessions",
        summary="List previous search sessions",
        synopsis="",
        example="/sessions",
    ),
    CommandMeta(
        name="/switch",
        summary="Switch to a past search session by ID",
        synopsis="<id>",
        example="/switch 3",
    ),
    CommandMeta(
        name="/show",
        summary="Inspect full transcript text by ID",
        synopsis="<id>",
        example="/show 1182",
    ),
    CommandMeta(
        name="/rename",
        summary="Rename current search session title",
        synopsis="<title>",
        example="/rename Machine Learning Review",
    ),
    CommandMeta(
        name="/export",
        summary="Export current search results to markdown file",
        synopsis="[path]",
        example="/export results.md",
    ),
    CommandMeta(
        name="/forget",
        summary="Clear active session memory and scratchpad",
        synopsis="",
        example="/forget",
    ),
    CommandMeta(
        name="/help",
        aliases=["/?"],
        summary="Show interactive commands and keyboard shortcuts",
        synopsis="",
        example="/help",
    ),
    CommandMeta(
        name="/exit",
        aliases=["/quit", "/q"],
        summary="Exit the search session",
        synopsis="",
        example="/exit",
    ),
]


_NAME_INDEX: dict[str, CommandMeta] = {}
for _meta in SEARCH_COMMAND_REGISTRY:
    for _n in _meta.all_names:
        _NAME_INDEX[_n] = _meta


def get_search_command_meta(name: str) -> CommandMeta | None:
    """Find command metadata by command name or alias."""
    return _NAME_INDEX.get(name)


def get_all_search_command_names() -> list[str]:
    """Return all command names and aliases."""
    return list(_NAME_INDEX.keys())


def render_search_help() -> None:
    """Render comprehensive interactive help guide dynamically from SEARCH_COMMAND_REGISTRY."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console

    console.print()
    console.print(f"  [{BOLD}]AudioBench Semantic Search — Interactive Console Help[/]")
    console.print(f"  [{DIM}]{'─' * 56}[/]")
    console.print()
    console.print(f"  [{ACCENT}]Commands & Options:[/]  (Press [bold]Tab[/bold] on [dim]/[/dim] to autocomplete)")

    for meta in SEARCH_COMMAND_REGISTRY:
        syn = f" {meta.synopsis}" if meta.synopsis else ""
        alias_str = f" [dim](alias: {', '.join(meta.aliases)})[/dim]" if meta.aliases else ""
        cmd_head = f"    [{BOLD}]{meta.name}[/][{DIM}]{syn}[/]{alias_str}"
        console.print(f"{cmd_head:<40}  {meta.summary}")
        if meta.choices:
            console.print(f"      [{DIM}]choices: {', '.join(meta.choices[:6])}...[/]")
        if meta.example:
            console.print(f"      [{DIM}]example: [italic]{meta.example}[/italic][/]")

    console.print()
    console.print(f"  [{ACCENT}]Keyboard Shortcuts:[/]  [dim](When input line is empty)[/dim]")
    console.print(
        f"    [{BOLD}]1–9[/]   — Open Fragment Reader for result #1–9\n"
        f"    [{BOLD}]E[/]     — Expand/collapse view between Book Mode and List\n"
        f"    [{BOLD}]C[/]     — Copy active synthesis to clipboard\n"
        f"    [{BOLD}]S[/]     — Start fresh search query\n"
        f"    [{BOLD}]Q[/]     — Quit search session (or Ctrl+D)\n"
    )
    console.print(
        f"  [{DIM}]Tip: Type words or \"quoted phrases\" to search. Use AND, OR, NOT for Boolean filtering.[/]\n"
    )
