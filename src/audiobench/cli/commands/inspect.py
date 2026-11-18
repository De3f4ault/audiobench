"""Visual inspection commands — waveform and spectrogram generation."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    format_size,
)


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--waveform/--no-waveform",
    default=True,
    show_default=True,
    help="Generate waveform image",
)
@click.option(
    "--spectrum/--no-spectrum",
    default=True,
    show_default=True,
    help="Generate spectrogram image",
)
@click.option("-o", "--output-dir", default=None, help="Output directory (default: same as input)")
@click.option("--width", default=1920, show_default=True, type=int, help="Image width in pixels")
@click.option("--color", default="#4488ff", show_default=True, help="Waveform color (hex)")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def inspect(
    file: str,
    waveform: bool,
    spectrum: bool,
    output_dir: str | None,
    width: int,
    color: str,
    quiet: bool,
) -> None:
    """Generate visual representations of audio files.

    Creates waveform and/or spectrogram PNG images for
    visual inspection of audio quality.

    \b
    Examples:
      audiobench inspect meeting.m4a                    Both waveform + spectrogram
      audiobench inspect meeting.m4a --no-spectrum      Waveform only
      audiobench inspect meeting.m4a -o ./images/       Custom output directory
      audiobench inspect lecture.wav --width 3840        4K width
    """
    import time

    from audiobench.transcribe.audio_converter import generate_spectrogram, generate_waveform

    file_path = Path(file).resolve()
    out_dir = Path(output_dir) if output_dir else file_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = file_path.stem

    if not waveform and not spectrum:
        console.print(error_panel("Nothing to generate", "Use --waveform and/or --spectrum"))
        sys.exit(1)

    if not quiet:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Inspect")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    File:   [{ACCENT}]{file_path.name}[/]")
        tasks = []
        if waveform:
            tasks.append("waveform")
        if spectrum:
            tasks.append("spectrogram")
        console.print(f"    Tasks:  {', '.join(tasks)}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")

    start = time.perf_counter()
    generated = []

    try:
        if waveform:
            wf_path = out_dir / f"{stem}_waveform.png"
            if not quiet:
                console.print(f"  [{DIM}]Generating waveform...[/]")
            generate_waveform(file_path, wf_path, width=width, color=color)
            generated.append(wf_path)

        if spectrum:
            sp_path = out_dir / f"{stem}_spectrogram.png"
            if not quiet:
                console.print(f"  [{DIM}]Generating spectrogram...[/]")
            generate_spectrogram(file_path, sp_path, width=width)
            generated.append(sp_path)

        elapsed = time.perf_counter() - start

        if not quiet:
            console.print()
            console.print(
                f"  [{SUCCESS}]✓ Generated {len(generated)} image(s) in {elapsed:.1f}s[/]"
            )
            for p in generated:
                console.print(f"    [{ACCENT}]{p.name}[/] ({format_size(p.stat().st_size)})")
            console.print()

            # Print chapters if any
            from audiobench.cli.display.theme import format_duration, make_table
            from audiobench.core.db_engine import get_session, init_db
            from audiobench.storage.chapter_repository import get_chapter_repo
            from audiobench.storage.repository import AudioFileRecord
            from audiobench.storage.utils import get_file_hash

            init_db()
            fhash = get_file_hash(file_path)
            with get_session() as session:
                audio = session.query(AudioFileRecord).filter_by(file_hash=fhash).first()
                if audio:
                    chapters = get_chapter_repo().get_chapters(audio.id)
                    if chapters:
                        # Also query raw DB records for transcription_id (not on ChapterInfo)
                        from audiobench.storage.models import ChapterRecord

                        tx_ids = {
                            r.chapter_index: r.transcription_id
                            for r in session.query(ChapterRecord)
                            .filter_by(audio_file_id=audio.id)
                            .all()
                        }
                        table = make_table(
                            f"Chapters for {file_path.name}",
                            [
                                ("#", {"style": DIM, "width": 4}),
                                ("Title", {"style": ACCENT}),
                                ("Duration", {"width": 10, "justify": "right"}),
                                ("Status", {"width": 12}),
                            ],
                        )
                        for ch in chapters:
                            dur_str = format_duration(ch.duration_seconds)
                            if ch.is_ghost:
                                status = "[yellow]Skipped (0s)[/]"
                            elif tx_ids.get(ch.index):
                                status = f"[{SUCCESS}]Transcribed[/]"
                            else:
                                status = f"[{DIM}]Pending[/]"
                            table.add_row(str(ch.index), ch.title, dur_str, status)
                        console.print(table)

    except Exception as e:
        console.print(error_panel("Inspection failed", str(e)))
        sys.exit(1)
