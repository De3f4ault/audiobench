"""Preset + Config commands — save/load/list named configurations."""

from __future__ import annotations

import click

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    make_table,
)

# ── Presets directory ───────────────────────────────────────


def _presets_dir():
    """Get the presets directory, creating it if needed."""

    from audiobench.core.settings import get_settings

    d = get_settings().data_dir / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_preset(name: str) -> dict | None:
    """Load a preset by name, returns dict or None."""
    import tomllib

    path = _presets_dir() / f"{name}.toml"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def _save_preset(name: str, data: dict) -> None:
    """Save a preset to TOML file."""
    import tomli_w

    path = _presets_dir() / f"{name}.toml"
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _save_preset_raw(name: str, data: dict) -> None:
    """Save a preset to TOML file (manual serialization, no tomli_w dependency)."""
    path = _presets_dir() / f"{name}.toml"
    lines = [f"# AudioBench preset: {name}", ""]
    for key, value in sorted(data.items()):
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, list):
            items = ", ".join(f'"{v}"' for v in value)
            lines.append(f"{key} = [{items}]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Preset Group ────────────────────────────────────────────


@click.group()
def preset() -> None:
    """Manage named configuration presets.

    \b
    Examples:
      audiobench preset create meeting --model large-v3 --speed accurate --enhance
      audiobench preset list
      audiobench preset show meeting
      audiobench preset delete meeting
      audiobench transcribe file.m4a --preset meeting
    """


@preset.command("create")
@click.argument("name")
@click.option("-m", "--model", default=None, help="Whisper model")
@click.option(
    "--speed",
    type=click.Choice(["fast", "balanced", "accurate"]),
    default=None,
    help="Speed preset",
)
@click.option("-l", "--language", default=None, help="Language code")
@click.option("-f", "--format", "output_format", default=None, help="Output format")
@click.option("--enhance", is_flag=True, default=False, help="Enable audio enhancement")
@click.option("--translate", is_flag=True, default=False, help="Translate to English")
@click.option("--diarize", is_flag=True, default=False, help="Enable speaker diarization")
@click.option("--filter", "audio_filter", default=None, help="Custom ffmpeg filter")
@click.option("--prompt", "initial_prompt", default=None, help="Initial context prompt")
def preset_create(
    name: str,
    model: str | None,
    speed: str | None,
    language: str | None,
    output_format: str | None,
    enhance: bool,
    translate: bool,
    diarize: bool,
    audio_filter: str | None,
    initial_prompt: str | None,
) -> None:
    """Create a named preset with specific options.

    \b
    Examples:
      audiobench preset create meeting --model large-v3 --speed accurate
      audiobench preset create podcast --language en --format srt --enhance
    """
    data = {}
    if model:
        data["model"] = model
    if speed:
        data["speed"] = speed
    if language:
        data["language"] = language
    if output_format:
        data["format"] = output_format
    if enhance:
        data["enhance"] = True
    if translate:
        data["translate"] = True
    if diarize:
        data["diarize"] = True
    if audio_filter:
        data["filter"] = audio_filter
    if initial_prompt:
        data["prompt"] = initial_prompt

    if not data:
        console.print(error_panel("Empty preset", "Specify at least one option to save."))
        return

    _save_preset_raw(name, data)
    console.print(f"  [{SUCCESS}]✓[/] Preset [{ACCENT}]{name}[/] saved")
    for k, v in data.items():
        console.print(f"    [{DIM}]{k}: {v}[/]")


@preset.command("list")
def preset_list() -> None:
    """List all saved presets."""
    presets_dir = _presets_dir()
    files = sorted(presets_dir.glob("*.toml"))

    if not files:
        console.print(
            f"  [{DIM}]No presets yet. Create one with: audiobench preset create <name>[/]"
        )
        return

    table = make_table(
        "Saved Presets",
        [
            ("Name", {"style": ACCENT}),
            ("Settings", {}),
        ],
    )

    for f in files:
        data = _load_preset(f.stem)
        if data:
            summary = ", ".join(f"{k}={v}" for k, v in data.items())
            table.add_row(f.stem, summary)

    console.print(table)


@preset.command("show")
@click.argument("name")
def preset_show(name: str) -> None:
    """Show details of a preset."""
    data = _load_preset(name)
    if not data:
        console.print(error_panel(f"Preset '{name}' not found"))
        return

    console.print(f"\n  [{BOLD} {ACCENT}]Preset: {name}[/]")
    for k, v in data.items():
        console.print(f"    {k}: [{ACCENT}]{v}[/]")


@preset.command("delete")
@click.argument("name")
def preset_delete(name: str) -> None:
    """Delete a preset."""
    path = _presets_dir() / f"{name}.toml"
    if not path.exists():
        console.print(error_panel(f"Preset '{name}' not found"))
        return
    path.unlink()
    console.print(f"  [{SUCCESS}]✓[/] Deleted preset [{ACCENT}]{name}[/]")


# ── Shell Completions ───────────────────────────────────────


@click.command("install-completion")
@click.argument(
    "shell",
    type=click.Choice(["bash", "zsh", "fish"]),
)
def install_completion(shell: str) -> None:
    """Install shell tab completions for audiobench.

    \b
    Examples:
      audiobench install-completion bash
      audiobench install-completion zsh
      audiobench install-completion fish
    """
    from pathlib import Path

    instructions = {
        "bash": (
            'eval "$(_AUDIOBENCH_COMPLETE=bash_source audiobench)"',
            Path.home() / ".bashrc",
        ),
        "zsh": (
            'eval "$(_AUDIOBENCH_COMPLETE=zsh_source audiobench)"',
            Path.home() / ".zshrc",
        ),
        "fish": (
            "_AUDIOBENCH_COMPLETE=fish_source audiobench | source",
            Path.home() / ".config" / "fish" / "config.fish",
        ),
    }

    line, rc_file = instructions[shell]

    # Check if already installed
    if rc_file.exists():
        content = rc_file.read_text(encoding="utf-8")
        if "_AUDIOBENCH_COMPLETE" in content:
            console.print(f"  [{DIM}]Completions already installed in {rc_file}[/]")
            return

    # Append to rc file
    with open(rc_file, "a", encoding="utf-8") as f:
        f.write(f"\n# AudioBench shell completions\n{line}\n")

    console.print(f"  [{SUCCESS}]✓[/] Installed {shell} completions")
    console.print(f"    [{DIM}]Added to: {rc_file}[/]")
    console.print(f"    [{DIM}]Restart your shell or run: source {rc_file}[/]")
