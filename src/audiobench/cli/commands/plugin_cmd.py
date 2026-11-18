"""Plugin management commands: list, create, install, remove."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    error_panel,
)
from audiobench.cli.plugins.loader import discover_plugins, ensure_plugins_dir, load_plugin
from audiobench.core.settings import get_settings

# ── Plugin command group ───────────────────────────────────────────────────────


@click.group(name="plugin")
def plugin_cmd() -> None:
    """Manage AudioBench plugins.

    \b
    Plugins extend AudioBench with new commands and event hooks.
    They live in the plugins directory (run 'plugin list' to see the path).

    \b
    Examples:
      audiobench plugin list                     Show installed plugins
      audiobench plugin create my_tool           Scaffold a new plugin
      audiobench plugin install /path/to/foo.py  Copy a plugin into the directory
      audiobench plugin remove my_tool           Delete an installed plugin
    """


# ── list ──────────────────────────────────────────────────────────────────────


@plugin_cmd.command(name="list")
def list_plugins() -> None:
    """List all installed plugins with their status."""
    plugins_dir = ensure_plugins_dir()
    plugin_files = discover_plugins()

    console.print(f"\n  [{BOLD}]Plugins directory:[/] [{ACCENT}]{plugins_dir}[/]\n")

    if not plugin_files:
        console.print(f"  [{DIM}]No plugins installed. Use 'audiobench plugin create <name>' to scaffold one.[/]\n")
        return

    from rich.table import Table
    table = Table(title="Installed Plugins", border_style="dim", title_style="bold")
    for col in ["Plugin", "File", "Commands", "Events", "Status"]:
        table.add_column(col)
    for path in plugin_files:
        module = load_plugin(path)
        if module is None:
            table.add_row(path.stem, path.name, "—", "—", f"[{WARNING}]load error[/]")
            continue

        # Count Click commands
        import click as _click
        cmds = [
            a for a in dir(module)
            if not a.startswith("_") and isinstance(getattr(module, a), _click.BaseCommand)
        ]
        has_register = callable(getattr(module, "register", None))
        cmd_count = f"[{ACCENT}]{len(cmds)}[/]" if cmds or has_register else f"[{DIM}]0[/]"

        has_events = "✓" if callable(getattr(module, "setup_events", None)) else f"[{DIM}]—[/]"
        table.add_row(path.stem, path.name, cmd_count, has_events, f"[{SUCCESS}]ok[/]")

    console.print(table)
    console.print()


# ── create ────────────────────────────────────────────────────────────────────


_PLUGIN_TEMPLATE = '''\
"""AudioBench plugin: {name}

Drop this file into your AudioBench plugins directory to activate it.
Run: audiobench plugin list   to confirm it is loaded.
"""

from __future__ import annotations

import click


# ── Commands ──────────────────────────────────────────────────────────────────


@click.command(name="{name}")
@click.argument("text", required=False)
def {fn_name}(text: str | None) -> None:
    """{name} — describe what this command does."""
    click.echo(f"[{name}] received: {{text!r}}")


def register(cli: click.Group) -> None:
    """Called by AudioBench to add this plugin's commands to the CLI."""
    cli.add_command({fn_name})


# ── Event hooks ───────────────────────────────────────────────────────────────
# Uncomment and adapt the examples below to react to AudioBench events.


def setup_events(bus) -> None:
    """Called by AudioBench to register event subscribers."""

    # Fired every time a transcription is saved to the database.
    # @bus.on("transcription.complete")
    # def on_transcription(tx_id: int, file_path: str, **kw) -> None:
    #     print(f"[{name}] transcription #{tx_id} done: {{file_path}}")

    # Fired every time an AI summary is generated.
    # @bus.on("summary.complete")
    # def on_summary(tx_id: int, summary: str, **kw) -> None:
    #     print(f"[{name}] summary for #{tx_id} ready.")

    # Fired every time a new file is imported into the library.
    # @bus.on("import.complete")
    # def on_import(audio_file_id: int, file_path: str, **kw) -> None:
    #     print(f"[{name}] imported {{file_path}}")

    pass
'''


@plugin_cmd.command(name="create")
@click.argument("name")
@click.option("--force", is_flag=True, help="Overwrite if plugin already exists")
def create_plugin(name: str, force: bool) -> None:
    """Scaffold a new plugin with the given NAME.

    Creates a ready-to-edit plugin file in the plugins directory.
    """
    # Sanitise name
    slug = name.lower().replace(" ", "_").replace("-", "_")
    plugins_dir = ensure_plugins_dir()
    dest = plugins_dir / f"{slug}.py"

    if dest.exists() and not force:
        console.print(error_panel(
            "Already exists",
            f"{dest.name} already exists. Use --force to overwrite.",
        ))
        sys.exit(1)

    fn_name = slug
    content = _PLUGIN_TEMPLATE.format(name=slug, fn_name=fn_name)
    dest.write_text(content, encoding="utf-8")

    console.print(f"\n  [{SUCCESS}]✓ Created plugin: [{ACCENT}]{dest}[/][/]")
    console.print(f"  [{DIM}]Edit the file above, then restart the REPL or run 'audiobench plugin list' to verify.[/]\n")


# ── install ───────────────────────────────────────────────────────────────────


@plugin_cmd.command(name="install")
@click.argument("source", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite if a plugin with this name exists")
def install_plugin(source: str, force: bool) -> None:
    """Install a plugin from a local file path SOURCE.

    Copies the file into the AudioBench plugins directory.
    """
    src = Path(source)
    if src.suffix != ".py":
        console.print(error_panel("Invalid file", "Plugins must be .py files."))
        sys.exit(1)

    plugins_dir = ensure_plugins_dir()
    dest = plugins_dir / src.name

    if dest.exists() and not force:
        console.print(error_panel(
            "Already installed",
            f"A plugin named '{src.stem}' already exists. Use --force to overwrite.",
        ))
        sys.exit(1)

    shutil.copy2(src, dest)
    console.print(f"\n  [{SUCCESS}]✓ Installed: [{ACCENT}]{dest.name}[/][/]")
    console.print(f"  [{DIM}]Restart the REPL or run 'audiobench plugin list' to activate.[/]\n")


# ── remove ────────────────────────────────────────────────────────────────────


@plugin_cmd.command(name="remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def remove_plugin(name: str, yes: bool) -> None:
    """Remove an installed plugin by NAME (without the .py extension)."""
    plugins_dir = get_settings().data_dir / "plugins"
    target = plugins_dir / f"{name}.py"

    if not target.exists():
        console.print(error_panel("Not found", f"No plugin named '{name}' is installed."))
        sys.exit(1)

    if not yes:
        confirm = click.confirm(f"  Remove plugin '{name}'?", default=False)
        if not confirm:
            console.print(f"  [{DIM}]Aborted.[/]")
            return

    target.unlink()
    console.print(f"\n  [{SUCCESS}]✓ Removed plugin: [{ACCENT}]{name}[/][/]\n")


# ── events (introspection) ────────────────────────────────────────────────────


@plugin_cmd.command(name="events")
def list_events() -> None:
    """Show all currently registered event handlers on the EventBus."""
    from audiobench.events import get_bus

    listeners = get_bus().listeners()
    if not listeners:
        console.print(f"\n  [{DIM}]No event handlers registered yet.[/]\n")
        return

    console.print()
    from rich.table import Table
    table = Table(title="EventBus Listeners", border_style="dim", title_style="bold")
    for col in ["Event", "Handler"]:
        table.add_column(col)
    for event, handlers in sorted(listeners.items()):
        for handler in handlers:
            table.add_row(f"[{ACCENT}]{event}[/]", handler)
    console.print(table)
    console.print()
