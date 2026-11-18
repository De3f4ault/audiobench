"""Summarize (AI summary) command."""

from __future__ import annotations

import click

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
)
from audiobench.core.settings import get_settings


def _generate_summary(
    client, record: dict, custom_prompt: str | None, model_name: str, title_suffix: str = ""
) -> str | None:
    """Generate and save a summary for a transcript record."""
    from audiobench.chat.context_builder import (
        TRANSCRIPT_SYSTEM,
        action_items,
    )
    from audiobench.chat.context_builder import (
        summarize as summarize_prompt,
    )
    from audiobench.chat.providers.ollama_provider import AIError

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Summary{title_suffix}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:  [{ACCENT}]#{record['id']} {record['file_name']}[/]")
    console.print(f"    Model:   {model_name}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # Build prompt
    if custom_prompt and "action" in custom_prompt.lower():
        prompt = action_items(record["full_text"])
    elif custom_prompt:
        prompt = f"{custom_prompt}\n\nTRANSCRIPT:\n{record['full_text']}"
    else:
        prompt = summarize_prompt(record["full_text"])

    try:
        console.print(f"  [{DIM}]Generating...[/]")
        console.print()

        summary_text = ""
        for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
            console.print(token, end="")
            summary_text += token

        console.print()
        console.print()
        return summary_text
    except AIError as e:
        console.print(error_panel("AI Error", str(e)))
        return None


@click.command()
@click.argument("transcript_id", type=int, required=False)
@click.option("--model", default=None, help="Ollama model (default: from settings)")
@click.option(
    "--prompt",
    "custom_prompt",
    default=None,
    help="Custom instruction (e.g., 'Focus on action items')",
)
@click.option("-i", "--interactive", "interactive_mode", is_flag=True, help="Interactive wizard")
@click.option("--chapter", type=int, default=None, help="Summarize a specific chapter")
@click.option("--all-chapters", is_flag=True, help="Summarize all chapters individually")
@click.option(
    "--rollup", is_flag=True, help="Summarize all chapters and roll up into a master summary"
)
def summarize(
    transcript_id: int | None,
    model: str | None,
    custom_prompt: str | None,
    interactive_mode: bool = False,
    chapter: int | None = None,
    all_chapters: bool = False,
    rollup: bool = False,
) -> None:
    """Summarize a transcript using local AI (Ollama).

    \b
    Examples:
      audiobench summarize 3                         Summarize transcript #3
      audiobench summarize 3 --chapter 2             Summarize chapter 2
      audiobench summarize 3 --rollup                Map-Reduce summary of all chapters
    """
    import sys

    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    if interactive_mode:
        from audiobench.cli.wizard import prompt_menu, prompt_transcription

        try:
            if not transcript_id:
                transcript_id = prompt_transcription("Select a transcript to summarize")
            if not custom_prompt:
                custom_prompt = prompt_menu(
                    "What kind of summary?",
                    [
                        ("General", "A concise general summary", None),
                        (
                            "Action Items",
                            "Extract action items and next steps",
                            "Focus on action items",
                        ),
                        (
                            "Bullet Points",
                            "Key takeaways in bullet points",
                            "Extract key takeaways as bullet points",
                        ),
                    ],
                )
        except KeyboardInterrupt:
            sys.exit(0)

    if not transcript_id:
        console.print(error_panel("Usage: audiobench summarize [ID]\nOr use --interactive"))
        sys.exit(1)

    init_db()
    repo = TranscriptionRepository()
    from audiobench.storage.chapter_repository import get_chapter_repo

    chap_repo = get_chapter_repo()
    record = repo.get_by_id(transcript_id, session=None)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        sys.exit(1)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=model_name,
    )
    if not client.is_available():
        console.print(
            error_panel(
                "Ollama not running",
                f"Start with: ollama serve\nThen pull the model: ollama pull {model_name}",
            )
        )
        sys.exit(1)

    audio_file_id = record.get("audio_file_id")

    # 1) Specific Chapter
    if chapter is not None:
        if not audio_file_id:
            console.print(
                error_panel(
                    "No audio file linked", "Cannot find chapters without an audio_file_id."
                )
            )
            sys.exit(1)
        chap = chap_repo.get_chapter_by_index(audio_file_id, chapter)
        if not chap or not chap["transcription_id"]:
            console.print(
                error_panel(
                    "Chapter Not Ready",
                    f"Chapter {chapter} is either missing or not transcribed yet.",
                )
            )
            sys.exit(1)

        chap_record = repo.get_by_id(chap["transcription_id"], session=None)
        if not chap_record:
            console.print(error_panel("Chapter Not Found", "Chapter transcript is missing."))
            sys.exit(1)

        summary_text = _generate_summary(
            client, chap_record, custom_prompt, model_name, title_suffix=f" (Chapter {chapter})"
        )
        if summary_text:
            _save_report(settings, chap_record, summary_text, suffix=f"_ch{chapter}")
            try:
                from audiobench.events import get_bus
                get_bus().emit(
                    "summary.complete",
                    tx_id=chap["transcription_id"],
                    summary=summary_text,
                )
            except Exception:
                pass
        return

    # 2) All Chapters or Rollup
    if all_chapters or rollup:
        if not audio_file_id:
            console.print(
                error_panel(
                    "No audio file linked", "Cannot find chapters without an audio_file_id."
                )
            )
            sys.exit(1)

        from audiobench.storage.chapter_repository import get_chapter_repo

        chapters = get_chapter_repo().list_for_file(audio_file_id)
        if not chapters:
            console.print(error_panel("No Chapters", "This file has no chapters defined."))
            sys.exit(1)

        # Collect chapter summaries
        chapter_summaries = []
        for i, chap in enumerate(chapters, start=1):
            # Look up whether this chapter has a transcription via the DB
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import ChapterRecord

            with get_session() as db:
                db_chap = db.query(ChapterRecord).filter_by(id=chap.id).first()
                tx_id = db_chap.transcription_id if db_chap else None

            if not tx_id:
                console.print(f"  [{DIM}]Skipping Chapter {i} ({chap.title}) — Not transcribed.[/]")
                continue

            chap_record = repo.get_by_id(tx_id, session=None)
            if chap_record:
                console.print(f"  [{ACCENT}]Summarizing Chapter {i}: {chap.title}[/]")
                summary_text = _generate_summary(
                    client, chap_record, custom_prompt, model_name, title_suffix=f" (Chapter {i})"
                )
                if summary_text:
                    chapter_summaries.append((i, chap.title, summary_text))
                    _save_report(settings, chap_record, summary_text, suffix=f"_ch{i}")

        # 3) Rollup Phase
        if rollup and chapter_summaries:
            console.print(
                f"  [{BOLD} {ACCENT}]Rolling up {len(chapter_summaries)} chapter summaries into a Master Summary...[/]"
            )

            # Build the master prompt
            master_prompt = "Please provide a comprehensive, high-level summary of the entire document based on the following chapter summaries:\n\n"
            for idx, title, c_sum in chapter_summaries:
                master_prompt += f"## Chapter {idx}: {title}\n{c_sum}\n\n"

            from audiobench.chat.context_builder import TRANSCRIPT_SYSTEM

            console.print()
            summary_text = ""
            for token in client.stream(master_prompt, system_prompt=TRANSCRIPT_SYSTEM):
                console.print(token, end="")
                summary_text += token

            console.print()
            console.print()

            # Save master report
            _save_report(
                settings, record, summary_text, suffix="_master", extra_content=master_prompt
            )

        return

    # 4) Default: Summarize single master transcript
    summary_text = _generate_summary(client, record, custom_prompt, model_name)
    if summary_text:
        _save_report(settings, record, summary_text)
        try:
            from audiobench.events import get_bus
            get_bus().emit(
                "summary.complete",
                tx_id=transcript_id,
                summary=summary_text,
            )
        except Exception:
            pass


def _save_report(
    settings, record, summary_text: str, suffix: str = "", extra_content: str = ""
) -> None:
    from pathlib import Path

    reports_dir = settings.data_dir / "exports" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(record["file_name"]).stem
    report_path = reports_dir / f"{base_name}_summary{suffix}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        if suffix == "_master":
            f.write(
                f"# Master Summary: {record['file_name']}\n\n{summary_text}\n\n---\n\n{extra_content}"
            )
        else:
            f.write(f"# Summary: {record['file_name']}\n\n{summary_text}")

    console.print(f"  [{SUCCESS}]✓ Summary saved to: [{ACCENT}]{report_path}[/]")
