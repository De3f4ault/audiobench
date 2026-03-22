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


@click.command()
@click.argument("transcript_id", type=int)
@click.option("--model", default=None, help="Ollama model (default: from settings)")
@click.option(
    "--prompt",
    "custom_prompt",
    default=None,
    help="Custom instruction (e.g., 'Focus on action items')",
)
def summarize(transcript_id: int, model: str | None, custom_prompt: str | None) -> None:
    """Summarize a transcript using local AI (Ollama).

    \b
    Examples:
      audiobench summarize 3                         Summarize transcript #3
      audiobench summarize 3 --model deepseek-v3.2   Use a specific model
      audiobench summarize 3 --prompt "Focus on action items"
    """
    from audiobench.chat.context_builder import (
        TRANSCRIPT_SYSTEM,
        action_items,
    )
    from audiobench.chat.context_builder import (
        summarize as summarize_prompt,
    )
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Summary")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:  [{ACCENT}]#{transcript_id} {record['file_name']}[/]")
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

    # Stream response
    try:
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
            return

        console.print(f"  [{DIM}]Generating...[/]")
        console.print()

        for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
            console.print(token, end="")

        console.print()
        console.print()
        console.print(f"  [{SUCCESS}]✓ Summary complete[/]")

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))
