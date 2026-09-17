"""Command metadata and schema registry for Jobs REPL slash commands.

Provides command definitions, aliases, synopses, and help documentation
to power context-aware autocompletion and dynamic help.
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
    """Specification of a Jobs REPL slash command."""

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


JOBS_COMMAND_REGISTRY: list[CommandMeta] = [
    CommandMeta(
        name="/list",
        aliases=[],
        summary="Full view: active queue + recent jobs + summary",
        synopsis="",
        example="/list",
    ),
    CommandMeta(
        name="/status",
        aliases=[],
        summary="Compact: running jobs only with elapsed time",
        synopsis="",
        example="/status",
    ),
    CommandMeta(
        name="/refresh",
        aliases=[],
        summary="Full repaint with a timestamp header",
        synopsis="",
        example="/refresh",
    ),
    CommandMeta(
        name="/ls",
        aliases=[],
        summary="One-liner: summary counts + currently running filenames",
        synopsis="",
        example="/ls",
    ),
    CommandMeta(
        name="/tail",
        aliases=["/follow"],
        summary="Stream real-time progress events or execution logs from a job",
        synopsis="<id>",
        example="/tail 6",
    ),
    CommandMeta(
        name="/log",
        aliases=["/fg", "/logs"],
        summary="Inspect stdout/stderr execution log of a job",
        synopsis="<id>",
        example="/log 6",
    ),
    CommandMeta(
        name="/retry",
        aliases=["/rerun"],
        summary="Re-enqueue failed or cancelled jobs",
        synopsis="<id> | --all",
        example="/retry --all",
        flags=[
            FlagMeta(
                opts=["--all"],
                summary="Retry all failed and cancelled jobs",
            )
        ],
    ),
    CommandMeta(
        name="/cancel",
        aliases=["/stop", "/kill"],
        summary="Cancel running or pending jobs",
        synopsis="<id> | --all | --batch <id>",
        example="/cancel 6",
        flags=[
            FlagMeta(
                opts=["--all"],
                summary="Cancel all active and pending jobs",
            ),
            FlagMeta(
                opts=["--batch"],
                summary="Cancel all jobs in a batch",
                takes_value=True,
                value_hint="<batch_id>",
            ),
        ],
    ),
    CommandMeta(
        name="/clean",
        aliases=["/prune"],
        summary="Delete completed, failed, and cancelled jobs from DB",
        synopsis="[--all]",
        example="/clean",
        flags=[
            FlagMeta(
                opts=["--all"],
                summary="Also purge running jobs (dangerous)",
            )
        ],
    ),
    CommandMeta(
        name="/clear",
        aliases=["/cls"],
        summary="Clear terminal screen and show fresh queue",
        synopsis="",
        example="/clear",
    ),
    CommandMeta(
        name="/help",
        aliases=["/?"],
        summary="Show interactive commands and usage guide",
        synopsis="",
        example="/help",
    ),
    CommandMeta(
        name="/exit",
        aliases=["/quit", "/q"],
        summary="Exit the interactive jobs console",
        synopsis="",
        example="/exit",
    ),
]


_NAME_INDEX: dict[str, CommandMeta] = {}
for _meta in JOBS_COMMAND_REGISTRY:
    for _n in _meta.all_names:
        _NAME_INDEX[_n] = _meta


def get_job_command_meta(name: str) -> CommandMeta | None:
    """Find command metadata by command name or alias."""
    return _NAME_INDEX.get(name)


def get_all_job_command_names() -> list[str]:
    """Return all command names and aliases."""
    return list(_NAME_INDEX.keys())


def render_jobs_help() -> None:
    """Render comprehensive interactive help guide dynamically from JOBS_COMMAND_REGISTRY."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console

    console.print()
    console.print(f"  [{BOLD}]AudioBench Jobs — Interactive Console Help[/]")
    console.print(f"  [{DIM}]{'─' * 56}[/]")
    console.print()
    console.print(f"  [{ACCENT}]Commands & Syntax:[/]  (Press [bold]Tab[/bold] on [dim]/[/dim] to autocomplete)")

    for meta in JOBS_COMMAND_REGISTRY:
        syn = f" {meta.synopsis}" if meta.synopsis else ""
        alias_str = f" [dim](alias: {', '.join(meta.aliases)})[/dim]" if meta.aliases else ""
        cmd_head = f"    [{BOLD}]{meta.name}[/][{DIM}]{syn}[/]{alias_str}"
        console.print(f"{cmd_head:<42}  {meta.summary}")

        if meta.flags:
            for flag in meta.flags:
                opts = ", ".join(flag.opts)
                v_hint = f" {flag.value_hint}" if flag.value_hint else ""
                console.print(f"      [{ACCENT}]{opts}{v_hint}[/]  [{DIM}]— {flag.summary}[/]")

    console.print()
    console.print(f"  [{ACCENT}]Examples:[/]  [dim](IDs can be typed as 6 or #6)[/dim]")
    console.print(
        f"    [{DIM}]• Follow live transcription:[/]   [bold]/tail 6[/]\n"
        f"    [{DIM}]• Inspect worker logs:[/]         [bold]/log 6[/]\n"
        f"    [{DIM}]• Cancel running task:[/]         [bold]/cancel 6[/]\n"
        f"    [{DIM}]• Re-run failed task:[/]          [bold]/retry 6[/]\n"
        f"    [{DIM}]• Re-run all failed jobs:[/]      [bold]/retry --all[/]\n"
        f"    [{DIM}]• Stop everything active:[/]      [bold]/cancel --all[/]\n"
    )
    console.print(
        f"  [{DIM}]Tip: Detach from /tail or /log at any time with [bold]Ctrl+C[/bold] without stopping the job.[/]\n"
    )
