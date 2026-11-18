"""``audiobench clean`` — retroactive LLM transcript cleaning.

Applies the segment-level transcript refiner to previously saved
transcriptions in the database.  Works at the segment level: only
the text of each segment is updated; timestamps are never touched.
All downstream features (play, lyric mode, export, chat) automatically
receive the cleaned text after running this command.
"""

from __future__ import annotations

import shutil
import sys

import click

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
)


@click.command()
@click.argument("ids", nargs=-1, type=int, required=False)
@click.option("--all", "clean_all", is_flag=True, help="Clean all unrefined transcriptions")
@click.option(
    "--force",
    is_flag=True,
    help="Re-clean already-refined transcriptions",
)
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override the cleaning model (default: clean_model setting)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without writing to the database",
)
@click.option("--gemini", is_flag=True, help="Use Gemini (gemini-2.5-flash) instead of Ollama")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode (no progress output)")
@click.option("-i", "--interactive", "interactive_mode", is_flag=True, help="Interactive wizard")
def clean(
    ids: tuple[int, ...],
    clean_all: bool,
    force: bool,
    model_override: str | None,
    dry_run: bool,
    gemini: bool,
    quiet: bool,
    interactive_mode: bool = False,
) -> None:
    """Clean transcript text using an LLM (fixes spelling, spacing, punctuation).

    Cleans at the segment level — timestamps are never touched.  All features
    (play, lyric mode, export, chat) automatically use the cleaned text.
    The original Whisper output is preserved in the database and accessible
    via ``audiobench show <ID> --raw``.

    \b
    Examples:
      audiobench clean 3                      Clean transcription #3
      audiobench clean 1 2 5                  Clean multiple IDs
      audiobench clean --all                  Clean all unrefined transcriptions
      audiobench clean --all --force          Re-clean everything
      audiobench clean 3 --dry-run            Preview changes only
      audiobench clean 3 --model qwen4:14b    Use a different model
      audiobench clean 3 --gemini             Use Gemini instead of Ollama
    """
    from audiobench.chat.providers.gemini_provider import GeminiClient
    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.core.settings import get_settings
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.refiner import TranscriptRefiner

    if interactive_mode:
        from audiobench.cli.wizard import prompt_bool, prompt_transcription

        try:
            target_id = prompt_transcription("Select a transcription to clean")
            ids = (target_id,)
            clean_all = False
            force = prompt_bool("Re-clean even if already refined?", default=False)
            gemini = prompt_bool("Use Gemini instead of local Ollama?", default=False)
            dry_run = prompt_bool("Preview changes only (dry-run)?", default=False)
        except KeyboardInterrupt:
            sys.exit(0)

    if not ids and not clean_all:
        console.print(
            error_panel(
                "No input", "Provide transcription ID(s) or use --all\nOr use --interactive"
            )
        )
        sys.exit(1)

    init_db()
    repo = TranscriptionRepository()
    settings = get_settings()

    if gemini:
        model = model_override or "gemini-2.5-flash"
    else:
        model = model_override or settings.clean_model

    # ── Resolve which IDs to clean ──
    if clean_all:
        history = repo.get_history(limit=9999)
        if force:
            target_ids = [r["id"] for r in history]
        else:
            # Use refined_at from history rows — no extra DB calls needed
            target_ids = [r["id"] for r in history if not r.get("refined_at")]
    else:
        target_ids = list(ids)
        if not force:
            # Skip already-refined unless --force
            filtered = []
            for tid in target_ids:
                status = repo.get_refinement_status(tid)
                if status is None:
                    if not quiet:
                        console.print(f"  [{DIM}]#{tid} not found — skipping[/]")
                    continue
                if status["is_refined"]:
                    if not quiet:
                        console.print(
                            f"  [{DIM}]#{tid} already refined "
                            f"({status['refined_at'][:10]}) — use --force to re-clean[/]"
                        )
                    continue
                filtered.append(tid)
            target_ids = filtered

    if not target_ids:
        if not quiet:
            console.print(f"  [{DIM}]Nothing to clean.[/]")
        return

    # ── Set up client ──
    if gemini:
        client = GeminiClient(model=model)
        if not client.is_available():
            console.print(
                error_panel(
                    "Gemini not available",
                    "Please set GEMINI_API_KEY in your environment or config.",
                )
            )
            sys.exit(1)
    else:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model,
        )
        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not available",
                    f"Cannot reach {settings.ollama_base_url}. "
                    "Is Ollama running with cloud routing enabled?",
                )
            )
            sys.exit(1)

    refiner = TranscriptRefiner(client, model=model)

    if not quiet:
        mode_label = "[dim](dry run)[/dim]" if dry_run else ""
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]AudioBench[/] — Transcript Cleaning {mode_label}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Model:   [{ACCENT}]{model}[/]")
        console.print(f"    Targets: {len(target_ids)} transcription(s)")
        console.print(f"  [{DIM}]{'─' * 44}[/]")

    cleaned_count = 0
    skipped_count = 0

    for tid in target_ids:
        data = repo.get_by_id(tid, session=None)
        if not data:
            if not quiet:
                console.print(f"  [{DIM}]#{tid} not found — skipping[/]")
            skipped_count += 1
            continue

        segments = data.get("segments", [])
        if not segments:
            if not quiet:
                console.print(f"  [{DIM}]#{tid} has no segments — skipping[/]")
            skipped_count += 1
            continue

        seg_texts = [s["text"] for s in segments]

        # Capture loop vars by value in closures
        _tid = tid
        _fname = data["file_name"]

        _state = {"batch": 0, "total_batches": 0, "tokens": 0}
        _is_tty = not quiet and sys.stdout.isatty()

        def _render(extra: str = "") -> None:
            """Overwrite the current terminal line with progress info."""
            if not _is_tty:
                return
            batch_label = (
                f"batch {_state['batch']}/{_state['total_batches']}"
                if _state["total_batches"]
                else "starting"
            )
            tok_label = f"{_state['tokens']} tok"
            line = f"\r  Cleaning #{_tid} {_fname}  [{batch_label}  {tok_label}]{extra}"
            width = shutil.get_terminal_size(fallback=(100, 24)).columns
            sys.stdout.write(line.ljust(width)[:width])
            sys.stdout.flush()

        def on_progress(done: int, total: int) -> None:
            _state["batch"] = done
            _state["total_batches"] = total
            _state["tokens"] = 0  # reset token counter per batch
            _render()

        def on_token(token: str) -> None:
            _state["tokens"] += len(token)
            _render()

        if not quiet and not _is_tty:
            console.print(f"  Cleaning [bold]#{_tid}[/] {_fname}", end="", flush=True)

        cleaned_texts, has_failures = refiner.refine_segments(
            seg_texts,
            on_progress=on_progress,
            on_token=on_token,
        )

        if not quiet:
            if _is_tty:
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                console.print()  # newline after non-tty inline status

        changed = sum(1 for a, b in zip(seg_texts, cleaned_texts, strict=True) if a != b)

        if changed == 0:
            if not quiet:
                console.print(f"[{DIM}]— no changes[/]")
            skipped_count += 1
            continue

        if dry_run:
            # Show a compact diff
            if not quiet:
                console.print(f"[{ACCENT}]~ {changed} segment(s) would change[/]")
                _print_diff(seg_texts, cleaned_texts, limit=5)
        else:
            # Apply to DB
            ok_segs = repo.update_segments(tid, cleaned_texts)
            if ok_segs:
                refined_full = " ".join(t.strip() for t in cleaned_texts if t.strip())
                if not has_failures:
                    # Preserve existing raw_text if already set (--force re-clean path)
                    # raw_text from get_by_id() is "" on first clean, original Whisper on subsequent
                    existing_raw = data.get("raw_text", "")
                    raw_text = existing_raw if existing_raw else data.get("full_text", "")
                    repo.update_full_text(tid, refined_full, raw_text)
                    if not quiet:
                        console.print(f"[{SUCCESS}]✓ {changed} segment(s) cleaned[/]")
                else:
                    # Don't set refined_at, so it will be retried next time without --force
                    repo.update_text(tid, refined_full)
                    if not quiet:
                        console.print(f"[{DIM}]~ {changed} segment(s) cleaned (incomplete, will retry)[/]")
                cleaned_count += 1
            else:
                if not quiet:
                    console.print(f"[{DIM}]! update failed[/]")
                skipped_count += 1

    # ── Summary ──
    if not quiet:
        console.print()
        if dry_run:
            console.print(f"  [{DIM}]Dry run complete. Run without --dry-run to apply changes.[/]")
        else:
            console.print(
                f"  [{SUCCESS}]✓[/] Cleaned: {cleaned_count}  |  [{DIM}]Skipped: {skipped_count}[/]"
            )


def _print_diff(
    original: list[str],
    cleaned: list[str],
    limit: int = 5,
) -> None:
    """Print a compact per-segment diff for --dry-run mode."""
    shown = 0
    for i, (orig, new) in enumerate(zip(original, cleaned, strict=True)):
        if orig == new:
            continue
        if shown >= limit:
            remaining = sum(1 for a, b in zip(original[i:], cleaned[i:], strict=True) if a != b)
            if remaining:
                console.print(f"  [{DIM}]  … and {remaining} more change(s)[/]")
            break
        console.print(f"  [{DIM}]  [{i}] -{orig.strip()}[/]")
        console.print(f"        [{SUCCESS}]  +{new.strip()}[/]")
        shown += 1
