"""Command registration — auto-discovers and registers all CLI commands.

Scans the `audiobench.cli.commands` package for Click commands,
imports each module, and registers every `click.BaseCommand` found
at module level. Also registers the REPL and user plugins.

No manual wiring needed — drop a new `*.py` file into commands/
with a `@click.command()` function, and it's available immediately.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import click


def register_all(cli: click.Group) -> None:
    """Auto-discover and register all CLI commands.

    1. Scans audiobench.cli.commands for .py modules (excluding __init__)
    2. Imports each module and finds click.BaseCommand objects
    3. Registers the REPL from audiobench.cli.repl
    4. Loads user plugins from ~/.audiobench/plugins/
    """
    import click as _click

    import audiobench.cli.commands as commands_pkg

    # ── Auto-discover command modules ──
    for module_info in pkgutil.iter_modules(commands_pkg.__path__):
        if module_info.name.startswith("_"):
            continue  # Skip __init__, __pycache__, etc.

        module = importlib.import_module(f"audiobench.cli.commands.{module_info.name}")

        # Find all Click commands/groups at module level
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if isinstance(attr, _click.Command):
                cli.add_command(attr)

    # ── REPL (lives in cli/repl/, not cli/commands/) ──
    from audiobench.cli.repl import repl

    cli.add_command(repl)

    # ── User plugins ──
    from audiobench.cli.plugins.loader import register_plugins

    register_plugins(cli)
