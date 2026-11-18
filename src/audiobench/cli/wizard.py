"""Interactive wizard helpers for guided CLI flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

try:
    import readline  # enables arrow keys and editing for standard input()
except ImportError:
    pass

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console

T = TypeVar("T")

# Tune this after use — 150ms is a baseline, not a measured value.
# Check daemon logs post-ship: if autocomplete spam returns, lower it.
# If suggestions feel laggy, raise it.
_AUTOCOMPLETE_DEBOUNCE_MS: int = 150

def prompt_menu(
    title: str,
    options: list[tuple[str, str, T]],
    default_idx: int | None = None,
) -> T:
    """Show a numbered menu and prompt the user for a choice.

    Args:
        title: The menu title.
        options: List of tuples (label, description, value).
        default_idx: 0-indexed default option if the user just presses Enter.

    Returns:
        The selected value.
    """
    console.print(f"\n  [{BOLD}]{title}[/]")
    for i, (label, desc, _) in enumerate(options):
        idx = i + 1
        default_marker = "  ← default" if default_idx == i else ""
        desc_padded = f"{desc:<25}"
        console.print(
            f"    [{ACCENT}][{idx}][/] {label:<15} [{DIM}]{desc_padded}[/]{default_marker}"
        )

    while True:
        try:
            choice = input("  → ").strip()
            if not choice and default_idx is not None:
                return options[default_idx][2]

            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    return options[idx - 1][2]

            console.print(
                f"  [{WARNING}]Invalid choice. Please enter a number from 1 to {len(options)}.[/]"
            )
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_bool(
    prompt: str,
    default: bool = False,
) -> bool:
    """Prompt for a yes/no answer.

    Args:
        prompt: The question to ask.
        default: Default answer if empty.

    Returns:
        True for yes, False for no.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    console.print(f"\n  {prompt}  [{DIM}]{suffix}[/]")

    while True:
        try:
            choice = input("  → ").strip().lower()
            if not choice:
                return default
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False
            console.print(f"  [{WARNING}]Please answer 'y' or 'n'.[/]")
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_string(
    prompt: str,
    default: str = "",
    validator: Callable[[str], bool] | None = None,
    validation_msg: str = "Invalid input.",
    enable_autocomplete: bool = False,
) -> str:
    """Prompt for a string input.

    Args:
        prompt: The prompt text.
        default: Default if empty.
        validator: Optional function to validate the input.
        validation_msg: Error message if validation fails.
        enable_autocomplete: If True, uses prompt_toolkit with daemon autocomplete.

    Returns:
        The user's string.
    """
    console.print(f"\n  {prompt}")

    if enable_autocomplete:
        try:
            import time

            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter

            class DaemonSemanticCompleter(Completer):
                def __init__(self) -> None:
                    from audiobench.daemon.factory import get_daemon_client
                    self.client = get_daemon_client()
                    self._latest_text = ""

                def get_completions(self, document, complete_event):
                    text = document.text
                    self._latest_text = text

                    if len(text) < 3:
                        return

                    # Debounce fast typing
                    time.sleep(_AUTOCOMPLETE_DEBOUNCE_MS / 1000.0)
                    if self._latest_text != text:
                        return  # Superseded by newer keystrokes
                    try:
                        # fetch fast-path results
                        results = self.client.autocomplete(text, top_k=7)
                        for r in results:
                            # r is {"expression_id": ..., "text": ..., "speaker": ..., "source_type": ...}
                            content = r.get("text", "")
                            if content:
                                speaker = r.get("speaker")
                                source_type = r.get("source_type")
                                display_text = content if len(content) <= 60 else content[:57] + "..."
                                display_meta = f"[{speaker}]" if speaker else f"[{source_type}]" if source_type else ""
                                yield Completion(
                                    content,
                                    start_position=-len(text),
                                    display=display_text,
                                    display_meta=display_meta
                                )
                    except Exception:
                        pass

            session = PromptSession(completer=ThreadedCompleter(DaemonSemanticCompleter()))
        except ImportError:
            # Fallback to standard input if prompt_toolkit is somehow missing
            enable_autocomplete = False

    while True:
        try:
            if enable_autocomplete:
                choice = session.prompt("  → ").strip()
            else:
                choice = input("  → ").strip()

            if not choice:
                choice = default

            if validator and not validator(choice):
                console.print(f"  [{WARNING}]{validation_msg}[/]")
                continue

            return choice
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_transcription(title: str = "Select a transcription", limit: int = 15) -> int:
    """Prompt the user to select a recent transcription.

    Args:
        title: Title of the menu.
        limit: Number of recent transcriptions to show.

    Returns:
        The selected transcription ID.
    """
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    recent = repo.get_history(limit=limit)

    if not recent:
        console.print(f"  [{WARNING}]No transcriptions found in history.[/]")
        raise KeyboardInterrupt

    options = []
    for rec in recent:
        file_name = rec.get("file_name", "Unknown")
        # Truncate long file names
        if len(file_name) > 30:
            file_name = file_name[:27] + "..."

        tx_id = rec.get("id")
        created = rec.get("created_at", "")[:16]
        desc = f"ID: {tx_id}  |  {created}"
        options.append((file_name, desc, tx_id))

    return prompt_menu(title, options)


def prompt_file(title: str = "Enter file path") -> str:
    """Prompt the user for a valid file path."""
    import os

    def validate_file(p: str) -> bool:
        p = os.path.expanduser(p)
        return os.path.isfile(p)

    path = prompt_string(
        title,
        validator=validate_file,
        validation_msg="File does not exist or is not a regular file.",
    )
    return os.path.expanduser(path)


# ── Chapter selection helpers ──────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_chapter_table(chapters: list, max_head: int = 15, max_tail: int = 5) -> None:
    """Print a formatted table of detected chapters.

    For long books (> max_head + max_tail chapters), the middle is folded into
    a single '── N chapters hidden ──' row so the table stays readable.

    Args:
        chapters:  List of ChapterInfo objects.
        max_head:  Max chapters to show at the top of the table.
        max_tail:  Max chapters to show at the bottom of the table.
    """
    from rich.box import SIMPLE_HEAVY
    from rich.table import Table

    total = len(chapters)
    fold_threshold = max_head + max_tail

    table = Table(
        box=SIMPLE_HEAVY,
        border_style=ACCENT,
        header_style=f"bold {ACCENT}",
        show_edge=True,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("#", style="bold", justify="right", no_wrap=True)
    table.add_column("Title", style="default", min_width=28, max_width=52)
    table.add_column("Start", style=DIM, justify="right", no_wrap=True)
    table.add_column("Duration", style=DIM, justify="right", no_wrap=True)

    def _add_row(chap) -> None:
        ghost_mark = f" [{DIM}](ghost)[/]" if chap.is_ghost else ""
        table.add_row(
            str(chap.index + 1),  # display 1-indexed
            f"{chap.title}{ghost_mark}",
            _fmt_time(chap.start_time),
            _fmt_time(chap.duration_seconds),
        )

    if total <= fold_threshold:
        for chap in chapters:
            _add_row(chap)
    else:
        for chap in chapters[:max_head]:
            _add_row(chap)
        hidden = total - max_head - max_tail
        table.add_row(
            "…",
            f"[{DIM}]── {hidden} chapters hidden ──[/]",
            "",
            "",
        )
        for chap in chapters[-max_tail:]:
            _add_row(chap)

    console.print()
    console.print(f"  [{BOLD}]{total} chapter{'s' if total != 1 else ''} detected[/]")
    console.print(table)


def prompt_chapters(chapters: list) -> str | None:
    """Show a chapter table and interactively ask which chapters to transcribe.

    Handles the following user inputs:
    - Empty / Enter      → all chapters (returns None so the caller passes no filter)
    - "all"              → explicitly all chapters
    - "1,3,5"            → specific indices
    - "1-5"              → a range
    - "1-3, 7, 10-12"   → mixed ranges and singles
    - Any ghost-only
      selection          → warns the user and re-prompts

    Args:
        chapters: List of ChapterInfo objects from ChapterDetector.

    Returns:
        A raw selection string like "1-5,10" to pass to --chapters, or None for all.
    """
    if not chapters:
        return None

    # Build an index-lookup (1-based display → ChapterInfo) for ghost validation
    index_map = {chap.index + 1: chap for chap in chapters}  # display index → chap
    total = len(chapters)

    print_chapter_table(chapters)

    console.print(
        f"  [{DIM}]Enter chapter numbers to transcribe.[/]\n"
        f"  [{DIM}]Examples:  1-5   ·   2,4,7   ·   1-3,10-12   ·   all[/]"
    )

    while True:
        try:
            raw = input(f"\n  Chapters [1-{total}] (Enter = all) → ").strip()
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt

        # Empty → all
        if not raw or raw.lower() == "all":
            return None  # caller will treat None as "all chapters"

        # Validate the input parses cleanly into integers that exist
        try:
            selected: list[int] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo_str, hi_str = part.split("-", 1)
                    lo, hi = int(lo_str.strip()), int(hi_str.strip())
                    if lo < 1 or hi > total or lo > hi:
                        raise ValueError(f"Range {lo}-{hi} is out of bounds (1-{total})")
                    selected.extend(range(lo, hi + 1))
                else:
                    n = int(part)
                    if n < 1 or n > total:
                        raise ValueError(f"Chapter {n} is out of bounds (1-{total})")
                    selected.append(n)

            selected = sorted(set(selected))
        except ValueError as exc:
            console.print(f"  [{WARNING}]Invalid selection: {exc}. Please try again.[/]")
            continue

        # Warn if every selected chapter is a ghost (would produce no output)
        all_ghost = all(index_map[n].is_ghost for n in selected if n in index_map)
        if all_ghost:
            console.print(
                f"  [{WARNING}]All selected chapters are ghost (zero-duration) chapters "
                f"and would be skipped. Please choose different chapters.[/]"
            )
            continue

        # Re-normalise back to a compact range string for the CLI
        return _compress_selection(selected)


def _compress_selection(indices: list[int]) -> str:
    """Compress a sorted list of ints into a compact range string.

    e.g. [1, 2, 3, 5, 7, 8] → "1-3,5,7-8"
    """
    if not indices:
        return ""

    parts: list[str] = []
    start = prev = indices[0]

    for n in indices[1:]:
        if n == prev + 1:
            prev = n
        else:
            parts.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n

    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)

