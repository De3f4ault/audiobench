"""Plugin loader — discover and load user plugins from data/plugins/.

Plugins are Python files that define Click commands. Each plugin file should
define a function called `register(cli)` that adds commands to the CLI group.

Example plugin (data/plugins/my_tool.py):

    import click

    @click.command()
    @click.argument("text")
    def shout(text):
        \"\"\"Shout some text.\"\"\"
        click.echo(text.upper())

    def register(cli):
        cli.add_command(shout)

Plugins are loaded after all built-in commands are registered, so they can
override or extend any existing functionality.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

if TYPE_CHECKING:
    import click

logger = get_logger("plugins")


def _plugins_dir() -> Path:
    """Get the plugins directory from settings (project-local data/plugins/)."""
    return get_settings().data_dir / "plugins"


def discover_plugins() -> list[Path]:
    """Find all .py plugin files in the plugins directory."""
    plugins_dir = _plugins_dir()
    if not plugins_dir.is_dir():
        return []

    return sorted(p for p in plugins_dir.glob("*.py") if not p.name.startswith("_"))


def load_plugin(path: Path) -> object | None:
    """Load a single plugin module from a file path."""
    module_name = f"audiobench_plugin_{path.stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot load plugin: %s", path)
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        logger.info("Loaded plugin: %s", path.name)
        return module

    except Exception as e:
        logger.warning("Failed to load plugin %s: %s", path.name, e)
        return None


def register_plugins(cli: click.Group) -> int:
    """Discover, load, and register all user plugins.

    Returns the number of successfully registered plugins.
    """
    plugin_files = discover_plugins()
    if not plugin_files:
        return 0

    count = 0
    for path in plugin_files:
        module = load_plugin(path)
        if module is None:
            continue

        # Call register(cli) if defined
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            try:
                register_fn(cli)
                count += 1
                logger.info("Registered plugin: %s", path.name)
            except Exception as e:
                logger.warning("Plugin %s register() failed: %s", path.name, e)
        else:
            # Auto-register: look for Click commands at module level
            import click as _click

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, _click.BaseCommand):
                    cli.add_command(attr)
                    count += 1
                    logger.info(
                        "Auto-registered command '%s' from %s",
                        attr.name,
                        path.name,
                    )

    return count


def ensure_plugins_dir() -> Path:
    """Create the plugins directory if it doesn't exist."""
    plugins_dir = _plugins_dir()
    plugins_dir.mkdir(parents=True, exist_ok=True)
    return plugins_dir
