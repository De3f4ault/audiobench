"""AudioBench CLI — entry point.

Defines the top-level Click group with global options (verbose, debug, version).
Commands are registered from the cli.commands package.
"""

from __future__ import annotations

import click

from audiobench.cli.display.theme import APP_VERSION
from audiobench.cli.plugins.custom_group import DefaultGroup
from audiobench.core.logger_factory import setup_logging


@click.group(cls=DefaultGroup, default_command="repl", invoke_without_command=True)
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

    Launch the interactive REPL (default):
      audiobench

    \b
    Transcribe & Process:
      audiobench transcribe meeting.m4a -f srt          Save as meeting.srt
      audiobench transcribe *.m4a -o ./out/             Batch to directory
      audiobench summarize 3 --interactive              AI summary of transcript #3
      audiobench clean --interactive                    Clean up transcript text
      audiobench bookmark --interactive                 Manage timestamps & regions
      audiobench convert meeting.m4a -o meeting.mp3     Format conversion
      audiobench merge pt1.wav pt2.wav -o full.wav      Merge audio files

    \b
    Manage:
      audiobench config --interactive                   Interactive configuration
      audiobench history                                Past transcriptions
      audiobench search "keyword"                       Search text
      audiobench export 3 --interactive                 Re-export transcript
    """
    import os

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    setup_logging(log_level)
    from audiobench.events import get_bus
    from audiobench.observatory.db import init_journal_db
    from audiobench.observatory.subscriber import get_subscriber

    init_journal_db()
    get_bus().on("*", get_subscriber().record)

    ctx.ensure_object(dict)
    ctx.obj["json_mode"] = json_mode

    # Crash recovery: reset any stale "running" jobs whose PIDs are dead.
    # Runs on every invocation — cheap (one DB query), no-op when clean.
    from audiobench.jobs.scheduler import startup_recovery as _job_recovery
    _job_recovery()


# ── Register all commands ───────────────────────────────────
# Import command modules — each module attaches its commands
# to the `cli` group via add_command() in __init__.py

from audiobench.cli.commands import register_all  # noqa: E402

register_all(cli)
