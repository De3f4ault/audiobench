"""CLI commands for reviewing quarantined privacy classifications."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from audiobench.cli.display.theme import ACCENT, APP_NAME, BOLD, DIM
from audiobench.storage.repository import TranscriptionRepository

console = Console()


@click.group()
def review() -> None:
    """Review quarantined or pending privacy classifications."""
    pass


@review.command("audiobooks")
@click.option("--approve-all", is_flag=True, help="Batch-approve all pending files as audiobooks (Tier 0).")
@click.option("--scan-all", is_flag=True, help="Batch-reject and send all pending files to biometric scan queue.")
def review_audiobooks(approve_all: bool, scan_all: bool) -> None:
    """Inspect and resolve audio files in the pending privacy review queue."""
    repo = TranscriptionRepository()
    pending = repo.get_pending_privacy_reviews()

    console.print(f"\n  [{BOLD} {ACCENT}]{APP_NAME} — Pending Privacy Reviews[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    if not pending:
        console.print("  [green]✓ No audio files pending privacy review.[/]\n")
        return

    table = Table(title=f"{len(pending)} File(s) Awaiting Operator Confirmation", show_header=True)
    table.add_column("#", style=DIM, width=4)
    table.add_column("ID", style=BOLD, width=6)
    table.add_column("File Name", style="cyan", max_width=35)
    table.add_column("Detected Work / Author", style="white", max_width=30)
    table.add_column("Duration", style=DIM, width=10)
    table.add_column("Review Reason", style="yellow")

    for idx, item in enumerate(pending, 1):
        dur_mins = f"{item['duration_seconds'] / 60:.1f}m" if item.get("duration_seconds") else "unknown"
        work_desc = item.get("detected_title") or "-"
        if item.get("detected_author"):
            work_desc = f"{work_desc} ({item['detected_author']})"

        table.add_row(
            str(idx),
            str(item["id"]),
            item["file_name"],
            work_desc,
            dur_mins,
            item["reason"],
        )

    console.print(table)
    console.print()

    if approve_all:
        resolved = repo.resolve_all_privacy_reviews(as_audiobook=True)
        console.print(f"  [green]✓ Approved {resolved} file(s) as Audiobooks (Tier 0).[/]\n")
        return

    if scan_all:
        resolved = repo.resolve_all_privacy_reviews(as_audiobook=False)
        console.print(f"  [yellow]✓ Sent {resolved} file(s) to biometric scan queue.[/]\n")
        return

    console.print("  Actions:")
    console.print("    [[bold green]a[/]] Approve all as Audiobooks (Tier 0, bypass biometric pass)")
    console.print("    [[bold yellow]s[/]] Scan all with Biometrics (enqueue ECAPA-TDNN evaluation)")
    console.print("    [[bold red]q[/]] Quit / keep pending in safe quarantine\n")

    choice = click.prompt("  Choose action", type=click.Choice(["a", "s", "q"], case_sensitive=False), default="q")
    choice = choice.lower()

    if choice == "a":
        resolved = repo.resolve_all_privacy_reviews(as_audiobook=True)
        console.print(f"\n  [green]✓ Approved {resolved} file(s) as Audiobooks (Tier 0).[/]\n")
    elif choice == "s":
        resolved = repo.resolve_all_privacy_reviews(as_audiobook=False)
        console.print(f"\n  [yellow]✓ Sent {resolved} file(s) to biometric scan queue.[/]\n")
    else:
        console.print("\n  [dim]No changes made. Files remain in pending review quarantine.[/]\n")
