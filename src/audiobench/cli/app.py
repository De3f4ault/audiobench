"""AudioBench CLI — entry point.

Defines the top-level Click group with global options (verbose, debug, version).
Commands are registered from the cli.commands package.
"""

from __future__ import annotations

import click

from audiobench.cli.display.theme import APP_VERSION
from audiobench.cli.plugins.custom_group import DefaultGroup
from audiobench.core.logger_factory import setup_logging


@click.group(cls=DefaultGroup, default_command="transcribe", invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed log output")
@click.option("--debug", is_flag=True, help="Debug logging")
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Machine-readable JSON output (where supported)",
)
@click.version_option(version=APP_VERSION, prog_name="audiobench")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool, json_mode: bool) -> None:
    """AudioBench — offline audio workbench.

    \b
    Transcribe files:
      audiobench transcribe meeting.m4a                 Print to stdout
      audiobench transcribe meeting.m4a -f srt          Save as meeting.srt
      audiobench transcribe meeting.m4a -o notes.srt    Auto-detect SRT format
      audiobench transcribe *.m4a -o ./out/             Batch to directory
      audiobench transcribe meeting.m4a --fast          Fast preset
      audiobench transcribe meeting.m4a -q | grep word  Pipe-friendly

    \b
    Manage:
      audiobench history                                Past transcriptions
      audiobench search "keyword"                       Search text
      audiobench export 3 -f vtt                        Re-export as VTT
      audiobench download large-v3-turbo                Pre-download model
      audiobench delete 3                               Remove from history
      audiobench info                                   System info
    """
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["json_mode"] = json_mode


# ── Register all commands ───────────────────────────────────
# Import command modules — each module attaches its commands
# to the `cli` group via add_command() in __init__.py

from audiobench.cli.commands import register_all  # noqa: E402

register_all(cli)
