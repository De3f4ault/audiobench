"""Speak + Download-Voice (TTS) commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    DIM,
    SUCCESS,
    console,
    error_panel,
)
from audiobench.core.settings import get_settings

# ── Speak Command ───────────────────────────────────────────


@click.command()
@click.argument("text_or_file", required=False, default=None)
@click.option(
    "--id", "transcript_id", type=int, default=None, help="Speak transcript # from history"
)
@click.option(
    "--voice",
    default=None,
    help="Piper voice name (default: from settings)",
)
@click.option(
    "-o", "--output", "output_path", default=None, help="Save to WAV file instead of playing"
)
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def speak(
    text_or_file: str | None,
    transcript_id: int | None,
    voice: str | None,
    output_path: str | None,
    quiet: bool,
) -> None:
    """Speak text aloud using Piper TTS.

    \b
    Examples:
      audiobench speak "Hello world"                 Speak text directly
      audiobench speak notes.txt                     Speak a text file
      audiobench speak --id 3                        Speak transcript from history
      audiobench speak "Hello" -o greeting.wav       Save to file
      audiobench speak --voice en_US-lessac-medium "Test"
    """
    from audiobench.tts.engine import PiperTTSEngine, TTSError

    settings = get_settings()
    voice_name = voice or settings.tts_voice

    # Determine text to speak
    if transcript_id is not None:
        # Speak from history
        from audiobench.core.db_engine import init_db
        from audiobench.storage.repository import TranscriptionRepository

        init_db()
        repo = TranscriptionRepository()
        record = repo.get_by_id(transcript_id)
        if not record:
            console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
            return
        text = record["full_text"]
        if not quiet:
            fname = record.get("file_name", "unknown")
            console.print(f"  [{DIM}]Speaking transcript #{transcript_id}: {fname}[/]")

    elif text_or_file is not None:
        # Check if it's a file path (short strings only — long text can't be paths)
        maybe_file = Path(text_or_file) if len(text_or_file) < 256 else None
        try:
            is_file = maybe_file and maybe_file.exists() and maybe_file.is_file()
        except OSError:
            is_file = False
        if is_file:
            text = maybe_file.read_text(encoding="utf-8")
            if not quiet:
                console.print(f"  [{DIM}]Speaking file: {maybe_file.name}[/]")
        else:
            text = text_or_file
    else:
        # Read from stdin
        text = sys.stdin.read()
        if not text.strip():
            console.print(error_panel("No input", "Provide text, a file, --id, or pipe to stdin"))
            return

    if not quiet:
        console.print(f"  [{ACCENT}]Voice: {voice_name}[/]")
        preview = text[:80] + "..." if len(text) > 80 else text
        console.print(f"  [{DIM}]{preview}[/]")

    try:
        engine = PiperTTSEngine(voices_dir=settings.voices_dir)

        if output_path:
            result = engine.synthesize(text, voice=voice_name, output_path=output_path)
            if not quiet:
                console.print(f"  [{SUCCESS}]✓ Saved to: {result}[/]")
        else:
            engine.play(text, voice=voice_name)
            if not quiet:
                console.print(f"  [{SUCCESS}]✓ Playback complete[/]")

    except TTSError as e:
        console.print(error_panel("TTS Error", str(e)))


# ── Download Voice Command ──────────────────────────────────


@click.command("download-voice")
@click.argument("voice_name")
def download_voice(voice_name: str) -> None:
    """Download a Piper TTS voice model.

    \b
    Examples:
      audiobench download-voice en_US-amy-medium
      audiobench download-voice en_US-lessac-high
      audiobench download-voice de_DE-thorsten-medium
    """
    from audiobench.tts.engine import PiperTTSEngine, TTSError

    settings = get_settings()

    console.print(f"  [{ACCENT}]Downloading voice: {voice_name}[/]")

    try:
        engine = PiperTTSEngine(voices_dir=settings.voices_dir)
        model_path = engine.download_voice(voice_name)
        console.print(f"  [{SUCCESS}]✓ Voice downloaded to: {model_path}[/]")
    except TTSError as e:
        console.print(error_panel("Download failed", str(e)))
