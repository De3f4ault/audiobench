"""Checkout menu for the Staging Cart."""

from __future__ import annotations

from rich.table import Table

from audiobench.cli.display.theme import DIM, console
from audiobench.cli.wizard import prompt_menu
from audiobench.core.db_session import get_session
from audiobench.storage.models import StagingCartItem


def edit_cart_items() -> None:
    """Prompt the user to edit engine/model/preset for items in the staging cart."""
    try:
        import questionary
    except ImportError:
        console.print(f"  [{DIM}]Missing 'questionary' package. Cannot edit cart.[/]")
        return

    with get_session() as session:
        items = session.query(StagingCartItem).all()
        if not items:
            console.print(f"  [{DIM}]Cart is empty.[/]")
            return

        choices = []
        for i, item in enumerate(items, 1):
            file_name = item.audio_file.file_name if item.audio_file else "Unknown"
            choices.append(
                questionary.Choice(
                    f"[{i}] {file_name} ({item.engine} - {item.model_name})",
                    item.id
                )
            )

        selected_ids = questionary.checkbox("Select files to edit:", choices=choices).ask()
        if not selected_ids:
            return

        engine = questionary.select(
            "New Engine:",
            choices=["skip (keep current)", "gemini", "whisper", "deepgram"]
        ).ask()

        model = questionary.select(
            "New Model:",
            choices=["skip (keep current)", "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
            default="large-v3-turbo"
        ).ask()

        preset = questionary.select(
            "New Preset:",
            choices=["skip (keep current)", "fast", "balanced", "accurate"]
        ).ask()

        strategy = questionary.select(
            "New Strategy:",
            choices=["skip (keep current)", "batch", "chunk", "concurrent"]
        ).ask()

        for item in items:
            if item.id in selected_ids:
                if engine and engine != "skip (keep current)":
                    item.engine = engine
                if model and model != "skip (keep current)":
                    item.model_name = model
                if preset and preset != "skip (keep current)":
                    item.speed_preset = preset
                if strategy and strategy != "skip (keep current)":
                    item.strategy = strategy

        session.commit()
        console.print(f"  [{DIM}]Updated {len(selected_ids)} items.[/]")


def display_cart() -> list[StagingCartItem]:
    """Fetch and display the current staging cart."""
    with get_session() as session:
        items = session.query(StagingCartItem).all()
        if not items:
            return []

        table = Table(show_header=True, header_style="bold cyan", title="\nTranscription Queue")
        table.add_column("#")
        table.add_column("File")
        table.add_column("Engine")
        table.add_column("Model")
        table.add_column("Preset")
        table.add_column("Strategy")

        for i, item in enumerate(items, 1):
            file_name = item.audio_file.file_name if item.audio_file else "Unknown"
            table.add_row(
                str(i),
                file_name,
                item.engine,
                item.model_name,
                item.speed_preset,
                item.strategy,
            )

        console.print(table)
        return items


def prompt_checkout_cart() -> str | None:
    """Displays the staging cart and prompts for the next action.

    Returns:
        "now", "later", "edit", "clear", or None if cancelled.
    """
    items = display_cart()
    if not items:
        return None

    options = [
        ("Now", "Run immediately in the foreground", "now"),
        ("Later", "Send to background Job Queue", "later"),
        ("Edit", "Change engines or presets", "edit"),
        ("Clear", "Empty the staging cart", "clear"),
        ("Cancel", "Return to main menu", "cancel"),
    ]

    return prompt_menu("Checkout Actions", options, default_idx=0)
