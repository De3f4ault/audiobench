"""Transcribe + Subtitle commands."""

from __future__ import annotations

import signal
import sys
import time
import uuid
from pathlib import Path

import click

from audiobench.cli.display import job_stream
from audiobench.cli.display.phase_tracker import PhaseTracker
from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    format_duration,
    format_size,
    make_table,
    stdout,
    summary_panel,
)
from audiobench.cli.io.file_collector import collect_files
from audiobench.cli.io.output_resolver import parse_formats, resolve_output
from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS
from audiobench.core.settings import get_settings
from audiobench.jobs import scheduler


def _detect_explicit_source(raw_files: tuple, is_recursive: bool, from_f: str | None) -> bool:
    """Check if inputs are explicitly specified filenames (not directories or recursive/file lists)."""
    if is_recursive or from_f:
        return False
    if not raw_files:
        return False
    for f in raw_files:
        try:
            if Path(str(f)).is_dir():
                return False
        except Exception:
            pass
    return True


def _build_extra_flags(
    *,
    output_format: str | None = None,
    output_path: str | None = None,
    language: str | None = None,
    model: str | None = None,
    speed_preset: str | None = None,
    no_cache: bool = False,
    no_timestamps: bool = False,
    quiet: bool = False,
    enhance: bool = False,
    trim: bool = False,
    denoise: bool = False,
    audio_filter: str | None = None,
    initial_prompt: str | None = None,
    translate: bool = False,
    diarize: bool = False,
    diarize_mode: str | None = None,
    diarize_threshold: float = 0.65,
    collision: str | None = None,
    mirror: bool = False,
    preset_name: str | None = None,
    id_only: bool = False,
    engine_name: str | None = None,
    align: bool | None = None,
    map_speakers: str | None = None,
    auto_name: bool = False,
    target_chapters: str | None = None,
    resume: bool = False,
    sensitive: bool = False,
) -> list[str]:
    """Serialize transcribe options to forwarded CLI flags."""
    extra_flags: list[str] = []
    if output_format:
        extra_flags += ["-f", output_format]
    if output_path:
        extra_flags += ["-o", output_path]
    if language:
        extra_flags += ["--language", language]
    if model:
        extra_flags += ["--model", model]
    if speed_preset and speed_preset != "balanced":
        extra_flags.append(f"--{speed_preset}")
    if no_cache:
        extra_flags.append("--no-cache")
    if no_timestamps:
        extra_flags.append("--no-timestamps")
    if quiet:
        extra_flags.append("--quiet")
    if enhance:
        extra_flags.append("--enhance")
    if trim:
        extra_flags.append("--trim")
    if denoise:
        extra_flags.append("--denoise")
    if audio_filter:
        extra_flags += ["--audio-filter", audio_filter]
    if initial_prompt:
        extra_flags += ["--initial-prompt", initial_prompt]
    if translate:
        extra_flags.append("--translate")
    if diarize:
        extra_flags.append("--diarize")
    if diarize_mode and diarize_mode != "fast":
        extra_flags += ["--diarize-mode", diarize_mode]
    if diarize_threshold != 0.65:
        extra_flags += ["--diarize-threshold", str(diarize_threshold)]
    if collision and collision != "skip":
        extra_flags += ["--collision", collision]
    if mirror:
        extra_flags.append("--mirror")
    if preset_name:
        extra_flags += ["--preset", preset_name]
    if id_only:
        extra_flags.append("--id-only")
    if engine_name:
        extra_flags += ["--engine", engine_name]
    if align is not None:
        extra_flags.append("--align" if align else "--no-align")
    if map_speakers:
        extra_flags += ["--map-speakers", map_speakers]
    if auto_name:
        extra_flags.append("--auto-name")
    if target_chapters:
        extra_flags += ["--target-chapters", target_chapters]
    if resume:
        extra_flags.append("--resume")
    if sensitive:
        extra_flags.append("--sensitive")
    return extra_flags


# ── Transcribe Command ──────────────────────────────────────


@click.command()
@click.argument("files", nargs=-1, type=click.Path(), required=False)
@click.option(
    "-f",
    "--format",
    "output_format",
    default=None,
    help="Output format: txt, srt, vtt, json (or comma-separated: srt,json  or 'all')",
)
@click.option("-o", "--output", "output_path", default=None, help="Output path (file or directory)")
@click.option(
    "-l", "--language", default=None, help="Language code (e.g., en, sw). Default: auto-detect"
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]),
    help="Whisper model",
)
@click.option("--fast", "speed_preset", flag_value="fast", help=" Fast: beam=1, batch=4")
@click.option(
    "--balanced",
    "speed_preset",
    flag_value="balanced",
    default=True,
    help=" Balanced: beam=3, batch=4 (default)",
)
@click.option(
    "--accurate", "speed_preset", flag_value="accurate", help="Accurate: beam=5, sequential"
)
@click.option("--no-cache", is_flag=True, help="Re-transcribe even if cached")
@click.option("--no-timestamps", is_flag=True, help="Disable word timestamps")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode (raw output only, for piping)")
@click.option("--check", is_flag=True, help="Show file metadata only (no transcription)")
@click.option("--enhance", is_flag=True, help="Apply noise reduction + normalization filters")
@click.option("--trim", is_flag=True, help="Remove leading/trailing silence before transcription")
@click.option(
    "--denoise",
    is_flag=True,
    help="Apply AI noise reduction (RNNoise neural network, auto-downloads model)",
)
@click.option("--filter", "audio_filter", default=None, help="Custom ffmpeg audio filter graph")
@click.option(
    "--prompt",
    "initial_prompt",
    default=None,
    help="Guide model with context (e.g., 'Conversation in Swahili and English')",
)
@click.option(
    "--translate",
    is_flag=True,
    help="Translate speech to English (any language → English)",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Identify speakers (Gemini: built-in, Whisper: requires pyannote.audio + HF token)",
)
@click.option(
    "--diarize-mode",
    type=click.Choice(["fast", "accurate"]),
    default="fast",
    show_default=True,
    help="fast: uses Whisper VAD + AHC (default). accurate: uses Pyannote.",
)
@click.option(
    "--diarize-threshold",
    type=float,
    default=0.65,
    show_default=True,
    help="Distance threshold for AHC clustering in fast diarization mode.",
)
@click.option(
    "-R",
    "--recursive",
    is_flag=True,
    help="Recurse into subdirectories when input is a directory",
)
@click.option(
    "--ext",
    "extensions",
    default=None,
    help="Filter by extension (e.g., --ext mp3,m4a). Default: all supported",
)
@click.option(
    "--from-file",
    "from_file",
    default=None,
    type=click.Path(exists=True),
    help="Read input paths from a manifest file (one per line)",
)
@click.option(
    "--exclude",
    default=None,
    help='Exclude files matching glob patterns (e.g., --exclude "*_draft*,temp_*")',
)
@click.option(
    "--collision",
    type=click.Choice(["overwrite", "skip", "rename"]),
    default="overwrite",
    show_default=True,
    help="Strategy when output file already exists",
)
@click.option(
    "--mirror",
    is_flag=True,
    help="Preserve directory structure in output (dir→dir mirror mode)",
)
@click.option(
    "--preset",
    "preset_name",
    default=None,
    help="Load a saved preset (e.g., --preset meeting)",
)
@click.option(
    "--id-only",
    is_flag=True,
    help="Output only transcription IDs (for piping to other commands)",
)
@click.option(
    "--map-speakers",
    "map_speakers",
    default=None,
    help="Manually map generic speakers to real names (e.g. 'Speaker 1=Lex, Speaker 2=Bob')",
)
@click.option(
    "--auto-name",
    is_flag=True,
    help="Use Gemini to automatically detect and map real names for speakers (requires --diarize)",
)
@click.option(
    "--notify",
    is_flag=True,
    help="Send desktop notification when transcription completes",
)
@click.option(
    "--engine",
    "engine_name",
    type=click.Choice(["whisper", "gemini"]),
    default=None,
    help="Transcription engine: whisper (local, default) or gemini (cloud)",
)
@click.option(
    "--align/--no-align",
    default=None,
    help="Force hybrid alignment (or disable it) for long audio on Gemini engine",
)
@click.option(
    "-b",
    "--background",
    is_flag=True,
    help="Force queuing to background without attaching/streaming live output",
)
@click.option(
    "--chapters",
    "target_chapters",
    default=None,
    help="Transcribe specific chapters (e.g. '1,3,5' or '1-5')",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Skip chapters that are already marked as completed",
)
@click.option(
    "-p",
    "--parallel",
    type=int,
    default=1,
    help="Number of chapters to transcribe in parallel",
)
@click.option(
    "--skip-ghost/--no-skip-ghost",
    default=True,
    help="Skip ghost chapters (chapters with start_time == end_time)",
)
@click.option(
    "--job-id",
    type=int,
    default=None,
    hidden=True,
    help="Internal: ID of the background job for status reporting",
)
@click.option(
    "--events-file",
    type=click.Path(),
    default=None,
    hidden=True,
    help="Internal: path to events file for live NDJSON streaming",
)
@click.option(
    "--strategy",
    type=click.Choice(["chunk", "batch", "concurrent"]),
    default="batch",
    show_default=True,
    help="Pipeline execution strategy. 'batch' transcribes all chapters first then diarizes all. 'chunk' interleaves. 'concurrent' overlaps both.",
)
@click.option(
    "--pipeline-workers",
    type=int,
    default=2,
    help="Worker count for --strategy concurrent.",
)
@click.option(
    "--parallel-gpus",
    is_flag=True,
    help="Run transcription on multiple GPUs in parallel",
)
@click.option(
    "--workers",
    type=int,
    default=0,
    help="Number of parallel GPU workers (0 = auto-detect based on GPUs)",
)
@click.option(
    "--interactive",
    "interactive_mode",
    is_flag=True,
    help="Guided wizard: prompts for engine, model, diarization, and other options",
)
@click.option(
    "--sensitive",
    is_flag=True,
    help="Mark transcription as Tier 3 (Override) immediately, bypassing biometric engine",
)
def transcribe(
    files: tuple[str, ...],
    output_format: str | None,
    output_path: str | None,
    language: str | None,
    model: str | None,
    speed_preset: str,
    no_cache: bool,
    no_timestamps: bool,
    quiet: bool,
    check: bool,
    enhance: bool,
    trim: bool,
    denoise: bool,
    audio_filter: str | None,
    initial_prompt: str | None,
    translate: bool,
    diarize: bool,
    diarize_mode: str,
    diarize_threshold: float,
    recursive: bool,
    extensions: str | None,
    from_file: str | None,
    exclude: str | None,
    collision: str,
    mirror: bool,
    preset_name: str | None,
    id_only: bool,
    notify: bool,
    engine_name: str | None,
    map_speakers: str | None,
    auto_name: bool,
    background: bool,
    job_id: int | None,
    events_file: str | None,
    interactive_mode: bool,
    target_chapters: str | None,
    resume: bool,
    strategy: str,
    pipeline_workers: int,
    parallel: int,
    skip_ghost: bool,
    parallel_gpus: bool,
    workers: int,
    sensitive: bool,
    align: bool | None,
) -> None:
    """Transcribe audio or video files to text.

    \b
    Examples:
      audiobench transcribe meeting.m4a                  Print to stdout
      audiobench transcribe meeting.m4a -f srt           Auto-save meeting.srt
      audiobench transcribe book.m4b --strategy batch    Batch mode (default)
      audiobench transcribe book.m4b --strategy chunk    Sequential per-chapter
      audiobench transcribe book.m4b --strategy concurrent Overlapped pipeline
      audiobench transcribe meeting.m4a -o notes.srt     Format from extension
      audiobench transcribe *.m4a -o ./out/              Batch to directory
      audiobench transcribe --fast lecture.mp3            Fast preset
      audiobench transcribe --translate audio_sw.m4a      Translate to English
      audiobench transcribe --diarize meeting.m4a         Identify speakers
      audiobench transcribe -q meeting.m4a | grep word   Pipe-friendly
      audiobench transcribe --interactive meeting.m4a     Guided setup wizard

    \b
    Directory & batch input:
      audiobench transcribe ./audiobooks/                Walk a directory
      audiobench transcribe ./audiobooks/ -R             Recurse into subdirs
      audiobench transcribe ./recordings/ --ext mp3,m4a  Filter by extension
      audiobench transcribe --from-file list.txt         Read paths from file
      find . -name '*.m4a' | audiobench transcribe -     Read from stdin
      audiobench transcribe . --exclude '*_draft*'       Exclude patterns

    \b
    Output control:
      audiobench transcribe dir/ -o out/ --mirror        Preserve dir structure
      audiobench transcribe dir/ --collision skip        Skip existing outputs
      audiobench transcribe dir/ --collision rename      Auto-rename conflicts
      audiobench transcribe file.m4a -f srt,json         Export to both formats
      audiobench transcribe file.m4a -f all              Export to all 4 formats

    \b
    Presets & automation:
      audiobench preset create meeting --model large-v3 --speed accurate
      audiobench transcribe file.m4a --preset meeting    Use saved preset
      audiobench transcribe dir/ --id-only               Output IDs only (piping)
    """
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    # ── Interactive wizard ──────────────────────────────────
    if interactive_mode:
        from audiobench.cli.wizard import prompt_bool, prompt_menu, prompt_string

        if not files:
            source_action = prompt_menu(
                "Select Transcription Source",
                [
                    ("Library", "Select files from your untranscribed library", "library"),
                    ("Import", "Browse OS to import new files", "import"),
                    ("Path", "Paste the file's path manually", "path"),
                    ("Exit", "Cancel and exit", "exit"),
                ],
                default_idx=0,
            )

            if source_action == "exit":
                return
            elif source_action == "path":
                from audiobench.cli.wizard import prompt_file
                p = prompt_file("Enter file path")
                files = (p,)
            elif source_action == "import":
                from audiobench.cli.commands.import_cmd import run_import_flow
                from audiobench.core.db_session import get_session
                from audiobench.storage.models import AudioFileRecord
                imported_ids = run_import_flow()
                if not imported_ids:
                    return
                with get_session() as session:
                    rec = session.query(AudioFileRecord).get(imported_ids[0])
                    if rec: files = (rec.file_path,)
            elif source_action == "library":
                from audiobench.cli.tui.library_tui import launch_library_tui
                from audiobench.core.db_session import get_session
                from audiobench.storage.models import AudioFileRecord
                result = launch_library_tui()
                selected_ids = result.get("selected_ids", [])
                if not selected_ids:
                    return
                with get_session() as session:
                    rec = session.query(AudioFileRecord).get(selected_ids[0])
                    if rec: files = (rec.file_path,)

            if not files:
                return

        # Resolve file for display
        display_file = str(files[0])
        file_size_str = ""
        try:
            p = Path(display_file)
            if p.exists():
                sz = p.stat().st_size / (1024 * 1024)
                from audiobench.transcribe.audio_converter import probe

                info = probe(display_file)
                file_size_str = f" · {format_duration(info.duration)} · {sz:.0f} MB"
        except Exception:
            pass

        from rich.panel import Panel

        from audiobench.cli.display.theme import BOX_STYLE

        console.print(
            Panel(
                f"[{BOLD}][{ACCENT}]{Path(display_file).name:<32} {file_size_str:<12}[/][/]",
                title=f"[{BOLD}][{ACCENT}]AudioBench Transcription Wizard[/][/]",
                title_align="left",
                border_style=ACCENT,
                box=BOX_STYLE,
                expand=False,
            )
        )

        try:
            from audiobench.chapters.detector import ChapterDetector

            detector = ChapterDetector()
            detected_chapters = detector.detect(Path(display_file))
        except Exception:
            detected_chapters = []

        try:
            if detected_chapters:
                from audiobench.cli.wizard import prompt_chapters

                chapter_selection = prompt_chapters(detected_chapters)
                if chapter_selection is not None:
                    # User picked specific chapters — pass the compact range string
                    target_chapters = chapter_selection
                # else: None → transcribe all; leave target_chapters as-is (None)

            # Engine
            chosen_engine = prompt_menu(
                "Engine",
                [
                    ("Whisper", "local, private, offline", "whisper"),
                    ("Gemini", "cloud, faster, smarter", "gemini"),
                ],
                default_idx=0,
            )
            engine_name = chosen_engine

            # Align (Gemini only)
            if chosen_engine == "gemini":
                chosen_align = prompt_menu(
                    "Hybrid Alignment (Gemini)",
                    [
                        ("auto", "Enable automatically for long files", "auto"),
                        ("yes", "Force enable alignment", "yes"),
                        ("no", "Force disable alignment", "no"),
                    ],
                    default_idx=0,
                )
                if chosen_align == "yes":
                    align = True
                elif chosen_align == "no":
                    align = False
                else:
                    align = None

            # Model (only shown for Whisper)
            if chosen_engine == "whisper" and not model:
                model = prompt_menu(
                    "Model",
                    [
                        ("tiny", "fastest, lowest quality", "tiny"),
                        ("base", "fast, decent", "base"),
                        ("medium", "balanced", "medium"),
                        ("large-v3", "slow, best quality", "large-v3"),
                        ("large-v3-turbo", "fast, near-large quality", "large-v3-turbo"),
                    ],
                    default_idx=2,
                )

            # Speed preset (Whisper only)
            if chosen_engine == "whisper":
                speed_preset = prompt_menu(
                    "Speed preset",
                    [
                        ("fast", "beam=1, batch=4", "fast"),
                        ("balanced", "beam=3, batch=4", "balanced"),
                        ("accurate", "beam=5, sequential", "accurate"),
                    ],
                    default_idx=1,
                )

            # Language
            lang_raw = prompt_string(
                "Language  [auto-detect — press Enter to skip, or type e.g. en, sw, fr]",
                default="",
            )
            if lang_raw.strip():
                language = lang_raw.strip()

            # Diarization
            diarize = prompt_bool("Speaker diarization?", default=False)
            if diarize:
                prompt_menu(
                    "Diarization model",
                    [
                        ("3.1 (legacy)", "slower, broader support", "speaker-diarization-3.1"),
                        ("3.0 (stable)", "stable general-purpose", "speaker-diarization-3.0"),
                    ],
                    default_idx=0,
                )  # model arg not yet wired into PyannoteDiarizer — stored for future use

            # Output format
            chosen_fmt = prompt_menu(
                "Output format",
                [
                    ("none", "print to terminal only", ""),
                    ("txt", "plain text file", "txt"),
                    ("srt", "SRT subtitles", "srt"),
                    ("vtt", "WebVTT subtitles", "vtt"),
                    ("json", "structured JSON", "json"),
                    ("pdf", "professional PDF document", "pdf"),
                    ("all", "all formats", "all"),
                ],
                default_idx=0,
            )
            if chosen_fmt:
                output_format = chosen_fmt

            # Output directory
            if output_format:
                out_raw = prompt_string(
                    "Output directory  [Enter to save next to source]",
                    default="",
                )
                if out_raw.strip():
                    output_path = out_raw.strip()

            # Pre-run summary
            console.print(f"\n  [{BOLD}]Ready to transcribe:[/]")
            console.print(f"    [{DIM}]File:[/]     {display_file}")
            if detected_chapters:
                console.print(f"    [{DIM}]Chapters:[/] {target_chapters if target_chapters else 'all'}")
            console.print(f"    [{DIM}]Engine:[/]   {engine_name}")
            if engine_name == "gemini":
                align_str = "auto" if align is None else ("yes" if align else "no")
                console.print(f"    [{DIM}]Align:[/]    {align_str}")
            if model:
                console.print(f"    [{DIM}]Model:[/]    {model}")
            console.print(f"    [{DIM}]Speed:[/]    {speed_preset}")
            if language:
                console.print(f"    [{DIM}]Lang:[/]     {language}")
            console.print(f"    [{DIM}]Diarize:[/]  {'yes' if diarize else 'no'}")
            if output_format:
                console.print(f"    [{DIM}]Format:[/]   {output_format}")
            if output_path:
                console.print(f"    [{DIM}]Output:[/]   {output_path}")
            console.print()

            go = prompt_bool("Start transcription?", default=True)
            if not go:
                console.print(f"  [{DIM}]Cancelled.[/]")
                return

        except KeyboardInterrupt:
            console.print()
            return

    if background and not SUPPORTS_BACKGROUND_JOBS:
        console.print(
            f"  [{WARNING}]Background jobs are only supported on Linux/macOS. Running in foreground...[/]"
        )
        background = False

    # E1: Load preset defaults (CLI flags override)
    if preset_name:
        from audiobench.cli.commands.config_cmd import _load_preset

        preset_data = _load_preset(preset_name)
        if not preset_data:
            console.print(error_panel(f"Preset '{preset_name}' not found"))
            return

        # Apply preset values only where CLI didn't specify
        if not model and "model" in preset_data:
            model = preset_data["model"]
        if speed_preset == "balanced" and "speed" in preset_data:
            speed_preset = preset_data["speed"]
        if not language and "language" in preset_data:
            language = preset_data["language"]
        if not output_format and "format" in preset_data:
            output_format = preset_data["format"]
        if not enhance and preset_data.get("enhance"):
            enhance = True
        if not translate and preset_data.get("translate"):
            translate = True
        if not diarize and preset_data.get("diarize"):
            diarize = True
        if not audio_filter and "filter" in preset_data:
            audio_filter = preset_data["filter"]
        if not initial_prompt and "prompt" in preset_data:
            initial_prompt = preset_data["prompt"]

        if not quiet:
            console.print(f"  [{DIM}]Using preset: {preset_name}[/]")

    # E2: --id-only implies quiet
    if id_only:
        quiet = True

    # ── Unified Staging Loop (Router) ──
    if not files and not from_file:
        from audiobench.cli.commands.import_cmd import run_import_flow
        from audiobench.cli.tui.library_tui import launch_library_tui
        from audiobench.cli.wizard import prompt_menu
        from audiobench.cli.wizard_checkout import prompt_checkout_cart
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import StagingCartItem

        while True:
            # Check cart status
            with get_session() as session:
                cart_count = session.query(StagingCartItem).count()

            options = []
            if cart_count > 0:
                options.append(
                    ("Checkout", f"Review and transcribe {cart_count} staged file(s)", "checkout")
                )

            options.extend(
                [
                    ("Library", "Select files from your untranscribed library", "library"),
                    ("Import", "Browse OS to import new files", "import"),
                    ("Path", "Paste a file path manually", "path"),
                    ("Exit", "Cancel and exit", "exit"),
                ]
            )

            action = prompt_menu("Select Transcription Source", options, default_idx=0)

            if action == "exit":
                return

            if action == "path":
                import os

                from audiobench.cli.wizard import prompt_file
                p = prompt_file("Enter file path")
                if not os.path.exists(p):
                    console.print(f"  [{WARNING}]File does not exist: {p}[/]")
                    continue

                # Import it invisibly so it has an audio_file_id
                from audiobench.storage.repository import AudioFileRepository
                from audiobench.transcribe.audio_converter import probe
                try:
                    info = probe(p)
                    repo = AudioFileRepository()
                    rec_id = repo.add_file(p, info.duration, info.format_name, info.size_bytes)
                    with get_session() as session:
                        if not session.query(StagingCartItem).filter_by(audio_file_id=rec_id).first():
                            session.add(StagingCartItem(audio_file_id=rec_id))
                        session.commit()
                except Exception as e:
                    console.print(f"  [{WARNING}]Failed to add path: {e}[/]")
                continue

            if action == "library":
                result = launch_library_tui()
                tui_action = result.get("action")
                selected_ids = result.get("selected_ids", [])

                if selected_ids:
                    with get_session() as session:
                        for sid in selected_ids:
                            if (
                                not session.query(StagingCartItem)
                                .filter_by(audio_file_id=sid)
                                .first()
                            ):
                                session.add(StagingCartItem(audio_file_id=sid))
                        session.commit()

                if tui_action == "switch_to_import":
                    action = "import"
                elif tui_action == "transcribe":
                    action = "checkout"

            if action == "import":
                imported_ids = run_import_flow()
                if imported_ids:
                    with get_session() as session:
                        for sid in imported_ids:
                            if (
                                not session.query(StagingCartItem)
                                .filter_by(audio_file_id=sid)
                                .first()
                            ):
                                session.add(StagingCartItem(audio_file_id=sid))
                        session.commit()
                continue

            if action == "checkout":
                checkout_action = prompt_checkout_cart()
                if checkout_action == "cancel" or not checkout_action:
                    continue
                elif checkout_action == "clear":
                    with get_session() as session:
                        session.query(StagingCartItem).delete()
                        session.commit()
                    console.print(f"  [{DIM}]Cart cleared.[/]")
                    continue
                elif checkout_action == "edit":
                    from audiobench.cli.wizard_checkout import edit_cart_items
                    edit_cart_items()
                    continue
                elif checkout_action in ("now", "later"):
                    with get_session() as session:
                        items = session.query(StagingCartItem).all()
                        count = len(items)
                        batch_id = f"cart_{uuid.uuid4().hex[:8]}" if count > 1 else None
                        for idx, item in enumerate(items):
                            file_path = item.audio_file.file_path if item.audio_file else None
                            if file_path:
                                item_args = ["transcribe", file_path]
                                if item.engine:
                                    item_args.extend(["--engine", item.engine])
                                if item.model_name:
                                    item_args.extend(["--model", item.model_name])
                                if item.speed_preset:
                                    item_args.append(f"--{item.speed_preset}")

                                scheduler.enqueue(
                                    job_type="transcribe",
                                    args=item_args,
                                    slot="transcription",
                                    file_label=Path(file_path).name,
                                    batch_id=batch_id,
                                    batch_label="Staging Cart",
                                    batch_index=idx + 1,
                                    batch_total=count,
                                )
                        session.query(StagingCartItem).delete()
                        session.commit()

                    if count == 0:
                        console.print(f"  [{DIM}]Cart was empty.[/]")
                        return

                    if checkout_action == "now":
                        console.print(
                            f"\n  [{ACCENT}]Processing {count} files sequentially in foreground...[/]"
                        )
                        scheduler.drain_slot("transcription")
                        return
                    elif checkout_action == "later":
                        scheduler.ensure_worker()
                        console.print(
                            f"\n  [{ACCENT}]✓ Added {count} files to background queue.[/]"
                        )
                        return

    parsed_chapters = None
    if target_chapters:
        if target_chapters.lower() == "all":
            parsed_chapters = "all"
        else:
            parsed_chapters = []
            for part in target_chapters.split(","):
                part = part.strip()
                if "-" in part:
                    start_str, end_str = part.split("-", 1)
                    parsed_chapters.extend(range(int(start_str), int(end_str) + 1))
                else:
                    parsed_chapters.append(int(part))
            parsed_chapters = sorted(list(set(parsed_chapters)))

    resolved_files = collect_files(
        files,
        recursive=recursive,
        extensions=extensions,
        from_file=from_file,
        exclude=exclude,
    )

    if not resolved_files:
        console.print(
            error_panel("No files found", "No supported audio/video files matched the input.")
        )
        raise click.exceptions.Exit(1)

    if not quiet and len(resolved_files) != len(files or ()):
        # Show discovery summary when directory/glob expanded the input
        console.print(f"  [{DIM}]Found {len(resolved_files)} file(s) to process[/]")

    settings = get_settings()
    if model:
        settings.model_name = model

    # Build filter list (smart ordering: highpass → denoise → trim → loudnorm)
    from audiobench.transcribe.audio_converter import build_filter_chain

    filters = build_filter_chain(
        enhance=enhance,
        denoise=denoise,
        trim=trim,
        custom=audio_filter,
    )

    # --check: show metadata only, no transcription
    if check:
        from audiobench.transcribe.audio_converter import probe

        for file_path in resolved_files:
            input_p = Path(str(file_path))
            info = probe(str(file_path))
            table = make_table(
                f"File: {input_p.name}",
                [
                    ("Property", {"style": BOLD}),
                    ("Value", {}),
                ],
            )
            table.add_row("Codec", info.codec)
            table.add_row("Duration", format_duration(info.duration))
            table.add_row("Sample Rate", f"{info.sample_rate} Hz")
            table.add_row("Channels", str(info.channels))
            if info.bitrate:
                table.add_row("Bitrate", f"{info.bitrate // 1000} kbps")
            table.add_row("Container", info.format_name)
            table.add_row("Size", format_size(input_p.stat().st_size))
            if filters:
                table.add_row("Filters", ", ".join(filters))
            console.print(table)
            console.print(f"  [{SUCCESS}]Ready to transcribe.[/]")
        return

    preset_icons = {"fast": "fast", "balanced": "balanced", "accurate": "accurate"}
    preset_label = preset_icons.get(speed_preset, speed_preset)

    # C1: Determine base directory for mirror mode
    input_base_dir: str | None = None
    if mirror and files:
        # Use the first directory argument as the base
        for p in files:
            if Path(p).is_dir():
                input_base_dir = p
                break

    # C3: Parse multi-format string
    multi_formats = parse_formats(output_format)
    # If parse_formats returns formats, use the first as primary
    primary_format = multi_formats[0] if multi_formats else None
    extra_formats = multi_formats[1:] if len(multi_formats) > 1 else []

    if parallel_gpus:
        from audiobench.transcription.parallel import ParallelTranscriber

        opts = {
            "output_path": output_path,
            "primary_format": primary_format,
            "extra_formats": extra_formats,
            "settings_output_format": settings.output_format,
            "input_base_dir": input_base_dir,
            "collision": collision,
            "language": language,
            "word_timestamps": not no_timestamps,
            "skip_cache": no_cache,
            "speed_preset": speed_preset,
            "initial_prompt": initial_prompt,
            "translate": translate,
            "enable_diarization": diarize,
            "diarize_mode": diarize_mode,
            "diarize_threshold": diarize_threshold,
            "map_speakers": map_speakers,
            "auto_name": auto_name,
            "filters": filters,
            "engine_name": engine_name,
            "target_chapters": parsed_chapters,
            "resume": resume,
            "parallel_chapters": parallel,
            "skip_ghost": skip_ghost,
            "workers": workers,
            "model": settings.model_name,
            "sensitive": sensitive,
            "align": align,
        }

        pt = ParallelTranscriber()
        pt.run(resolved_files, opts)
        return

    # ── Unified Intelligent Dispatcher & Deduplication ─────────
    if not parallel_gpus and not job_id and not check:
        skip_done = not (no_cache or (collision and collision == "overwrite"))
        files_to_process = scheduler.deduplicate_files(resolved_files, skip_done=skip_done)

        if not files_to_process:
            console.print(
                f"  [{SUCCESS}]All {len(resolved_files)} file(s) are already completed or queued.[/]"
            )
            return

        if len(files_to_process) < len(resolved_files):
            console.print(
                f"  [{DIM}]Skipping {len(resolved_files) - len(files_to_process)} already completed or queued file(s).[/]"
            )

        if SUPPORTS_BACKGROUND_JOBS:
            is_explicit = _detect_explicit_source(files, recursive, from_file)
            extra_flags = _build_extra_flags(
                output_format=output_format,
                output_path=output_path,
                language=language,
                model=model,
                speed_preset=speed_preset,
                no_cache=no_cache,
                no_timestamps=no_timestamps,
                quiet=quiet,
                enhance=enhance,
                trim=trim,
                denoise=denoise,
                audio_filter=audio_filter,
                initial_prompt=initial_prompt,
                translate=translate,
                diarize=diarize,
                diarize_mode=diarize_mode,
                diarize_threshold=diarize_threshold,
                collision=collision,
                mirror=mirror,
                preset_name=preset_name,
                id_only=id_only,
                engine_name=engine_name,
                align=align,
                map_speakers=map_speakers,
                auto_name=auto_name,
                target_chapters=target_chapters,
                resume=resume,
                sensitive=sensitive,
            )

            resolved_engine = engine_name or getattr(settings, "engine", "faster-whisper")
            is_cloud = (resolved_engine == "gemini")
            target_slot = "transcription:cloud" if is_cloud else "transcription:local"

            PRIORITY_THRESHOLD = 5
            is_priority = is_explicit and len(files_to_process) <= PRIORITY_THRESHOLD and not background

            if is_priority:
                batch_id = f"batch_{uuid.uuid4().hex[:8]}" if len(files_to_process) > 1 else None
                batch_label = Path(str(files_to_process[0])).name if len(files_to_process) > 1 else None
                total_count = len(files_to_process)
                queued_ids: list[int] = []

                for idx, f in enumerate(files_to_process):
                    file_args = ["transcribe", str(f.resolve()), *extra_flags]
                    jid = scheduler.enqueue_priority(
                        job_type="transcribe",
                        args=file_args,
                        slot=target_slot,
                        file_label=f.name,
                        batch_id=batch_id,
                        batch_label=batch_label,
                        batch_index=idx + 1 if batch_id else None,
                        batch_total=total_count if batch_id else None,
                    )
                    queued_ids.append(jid)

                scheduler.ensure_worker()

                lane_label = "cloud lane" if is_cloud else "local lane"
                if len(queued_ids) == 1:
                    console.print(f"  [{ACCENT}][P][/] Queued Job #{queued_ids[0]} at priority ({lane_label}, next in queue)")
                else:
                    console.print(
                        f"  [{ACCENT}][P][/] Queued {len(queued_ids)} jobs at priority ({lane_label}: #{queued_ids[0]}..#{queued_ids[-1]})"
                    )

                job_stream.stream_jobs_sequence(queued_ids)
                return

            else:
                # Normal queue path: bulk, folder, recursive, >5 files, or --background
                batch_id = f"batch_{uuid.uuid4().hex[:8]}"
                batch_label = str(files[0]) if files else "Batch"
                total_count = len(files_to_process)

                for idx, f in enumerate(files_to_process):
                    file_args = ["transcribe", str(f.resolve()), *extra_flags]
                    scheduler.enqueue(
                        job_type="transcribe",
                        args=file_args,
                        slot=target_slot,
                        file_label=f.name,
                        batch_id=batch_id,
                        batch_label=batch_label,
                        batch_index=idx + 1,
                        batch_total=total_count,
                        priority=100,
                    )

                scheduler.ensure_worker()
                lane_label = "cloud queue" if is_cloud else "background queue"
                console.print(
                    f"  [{SUCCESS}]Queued {total_count} file(s) into {lane_label}.[/]\n"
                    f"  [{DIM}]The background worker will process them sequentially.[/]\n"
                    f"  [{DIM}]Use [bold]audiobench jobs[/bold] to monitor progress.[/]"
                )
                return
        else:
            # Fallback for platforms without background worker support (e.g. Windows)
            resolved_files = files_to_process

    pipeline = TranscriptionPipeline()
    results: list[dict] = []

    for file_path in resolved_files:
        input_p = Path(str(file_path))
        file_size = input_p.stat().st_size

        # Resolve output path and format
        resolved_output, resolved_format = resolve_output(
            str(file_path),
            output_path,
            primary_format,
            settings.output_format,
            input_base_dir=input_base_dir,
            collision=collision,
        )

        # C2: collision=skip → resolve_output returns None path
        if resolved_output is None and (output_path or primary_format) and collision == "skip":
            if not quiet:
                console.print(f"  [{DIM}]Skipped (exists): {input_p.name}[/]")
            continue

        # ── Header ──
        if not quiet:
            console.print()
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/]")
            console.print(f"  [{DIM}]{'─' * 44}[/]")
            console.print(f"    File:    [{ACCENT}]{input_p.name}[/] ({format_size(file_size)})")
            console.print(f"    Model:   {settings.model_name} | Preset: {preset_label}")
            if engine_name == "gemini":
                console.print(f"    Engine:  [bold]Gemini[/] ({settings.gemini_model})")
            if translate:
                console.print("    Task:    [bold]Translate → English[/]")
            if diarize:
                console.print("    Diarize: [bold]Speaker identification[/]")
            if filters:
                console.print(f"    Filters: [{DIM}]{', '.join(filters)}[/]")
            if resolved_output:
                console.print(f"    Output:  [{DIM}]{resolved_output}[/]")
            if extra_formats:
                console.print(f"    Formats: [{DIM}]{', '.join(multi_formats)}[/]")
            console.print(f"  [{DIM}]{'─' * 44}[/]")

        start_time = time.perf_counter()
        tracker = PhaseTracker(quiet=quiet, events_file=events_file)
        tracker.start()

        # SIGINT handler for graceful partial save
        original_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(
            signum: int,
            frame: object,
            _tracker: PhaseTracker = tracker,
            _file_path: str = str(file_path),
            _original: object = original_handler,
        ) -> None:
            partial_path = _tracker.save_partial(_file_path)
            if partial_path:
                console.print(
                    f"\n  [{ACCENT}]Interrupted. Partial transcript saved to: {partial_path}[/]"
                )
            else:
                console.print(f"\n  [{DIM}]Interrupted. No segments transcribed yet.[/]")
            signal.signal(signal.SIGINT, _original)  # restore
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_interrupt)

        try:
            transcript = pipeline.transcribe_file(
                file_path=str(file_path),
                language=language,
                output_format=resolved_format,
                output_path=resolved_output,
                word_timestamps=not no_timestamps,
                skip_cache=no_cache,
                speed_preset=speed_preset,
                initial_prompt=initial_prompt,
                translate=translate,
                enable_diarization=diarize,
                diarize_mode=diarize_mode,
                diarize_threshold=diarize_threshold,
                sensitive=sensitive,
                map_speakers=map_speakers,
                auto_name=auto_name,
                on_phase=tracker.update,
                on_segment=tracker.on_segment,
                filters=filters,
                engine_name=engine_name,
                job_id=job_id,
                target_chapters=parsed_chapters,
                resume=resume,
                parallel=parallel,
                skip_ghost=skip_ghost,
                align=align,
            )

            # NOTE: job_id here is a UnifiedJob.id. Completion is recorded by
            # worker_daemon polling the subprocess exit code — no write back needed.
            # JobRepository().mark_job_done(job_id) was a no-op here since
            # 2026-09-17: it targeted the legacy 'jobs' table, not unified_jobs.

            # For non-streaming engines (Gemini), segments aren't always emitted
            # during generation, so force a final update to show them if missing.
            if engine_name == "gemini" and not tracker.segments:
                for seg in transcript.segments:
                    tracker.on_segment(seg)

            if not tracker.segments and transcript.segments:
                tracker.segments = list(transcript.segments)

            tracker.finalize()

            elapsed = time.perf_counter() - start_time
            speed_ratio = transcript.duration_seconds / elapsed if elapsed > 0 else 0

            # ── Output ──
            if quiet:
                if resolved_format == "pdf":
                    from audiobench.export.pdf import PDFExporter

                    # Ensure transcript acts like a dict for PDFExporter
                    data = transcript.dict()
                    data["file_name"] = Path(str(file_path)).stem
                    PDFExporter().export_transcript(
                        data, resolved_output or f"{data['file_name']}.pdf"
                    )
                else:
                    from audiobench.output.base import get_formatter

                    formatter = get_formatter(resolved_format)
                    stdout.print(formatter.format(transcript), highlight=False)
            else:
                # Transcript text was already displayed progressively
                # during transcription via on_segment callbacks.
                # Just show the summary panel below.

                # ── Summary ──
                console.print()
                if getattr(transcript, "is_cached", False):
                    status_line = f"  [{SUCCESS}]✓ Retrieved from cache[/]  [{DIM}]•  existing transcript[/]"
                else:
                    status_line = (
                        f"  [{SUCCESS}]✓ Done in {format_duration(elapsed)}[/]"
                        f"  [{DIM}]•  {speed_ratio:.1f}x real-time[/]"
                    )

                console.print(
                    summary_panel(
                        [
                            status_line,
                            "",
                            f"  Language   [{BOLD}]{transcript.language}[/] "
                            f"({transcript.language_probability * 100:.0f}%)"
                            f"     Segments  {transcript.segment_count}",
                            f"  Words      {transcript.word_count}"
                            f"              Audio     "
                            f"{format_duration(transcript.duration_seconds)}",
                        ]
                    )
                )

                if resolved_output:
                    console.print(f"  [{DIM}]Saved → {resolved_output}[/]")

            # C3: Multi-format — save additional formats
            if extra_formats and transcript:
                from audiobench.output.base import get_formatter as get_fmt

                for extra_fmt in extra_formats:
                    extra_out, _ = resolve_output(
                        str(file_path),
                        output_path,
                        extra_fmt,
                        extra_fmt,
                        input_base_dir=input_base_dir,
                        collision=collision,
                    )
                    if extra_out is None:
                        if not quiet:
                            console.print(f"  [{DIM}]Skipped {extra_fmt} (exists)[/]")
                        continue
                    if extra_fmt == "pdf":
                        from audiobench.export.pdf import PDFExporter

                        data = transcript.dict()
                        data["file_name"] = Path(str(file_path)).stem
                        PDFExporter().export_transcript(data, extra_out)
                    else:
                        fmt_obj = get_fmt(extra_fmt)
                        content = fmt_obj.format(transcript)
                        Path(extra_out).parent.mkdir(parents=True, exist_ok=True)
                        Path(extra_out).write_text(content, encoding="utf-8")
                    if not quiet:
                        console.print(f"  [{DIM}]Saved → {extra_out}[/]")

            results.append(
                {
                    "file": input_p.name,
                    "words": transcript.word_count,
                    "duration": transcript.duration_seconds,
                    "elapsed": elapsed,
                    "speed": speed_ratio,
                    "language": transcript.language,
                }
            )

        except Exception as e:
            tracker.abort(str(e))
            if not quiet:
                console.print(error_panel(f"Failed: {input_p.name}", str(e)))
            else:
                print(f"Error: {input_p.name}: {e}", file=sys.stderr)
            if job_id:
                sys.exit(1)

    if not results and resolved_files:
        raise click.exceptions.Exit(1)

    # ── Batch summary ──
    if len(results) > 1 and not quiet:
        console.print()
        table = make_table(
            "Batch Summary",
            [
                ("File", {"style": ACCENT}),
                ("Words", {"justify": "right", "width": 6}),
                ("Duration", {"justify": "right", "width": 10}),
                ("Processed", {"justify": "right", "width": 10}),
                ("Speed", {"justify": "right", "width": 8}),
            ],
        )
        for r in results:
            table.add_row(
                r["file"],
                str(r["words"]),
                format_duration(r["duration"]),
                format_duration(r["elapsed"]),
                f"{r['speed']:.1f}x",
            )
        console.print(table)

    # ── Desktop notification ──
    if notify and results:
        _send_notification(results)


def _send_notification(results: list[dict]) -> None:
    """Send a desktop notification on transcription completion."""
    import subprocess

    total_words = sum(r["words"] for r in results)
    total_dur = sum(r["duration"] for r in results)
    n_files = len(results)

    title = "AudioBench — Transcription Complete"
    if n_files == 1:
        body = f"{results[0]['file']}: {total_words:,} words, {format_duration(total_dur)}"
    else:
        body = (
            f"{n_files} files: "
            f"{total_words:,} total words, "
            f"{format_duration(total_dur)} total audio"
        )

    try:
        import sys as _sys

        if _sys.platform == "darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{body}" with title "{title}"',
                ],
                check=False,
            )
        elif _sys.platform == "win32":
            pass  # Windows toast not implemented
        else:
            subprocess.run(
                ["notify-send", title, body, "-i", "audio-x-generic"],
                check=False,
            )
    except FileNotFoundError:
        pass  # notify-send not installed — silently skip


# ── Subtitle Command ────────────────────────────────────────


@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None, help="Output video path")
@click.option(
    "--hard",
    "hard_burn",
    is_flag=True,
    help="Burn subtitles into video pixels (permanent)",
)
@click.option("-l", "--language", default=None, help="Language code (e.g., en, sw)")
@click.option("--translate", is_flag=True, help="Translate subtitles to English")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def subtitle(
    video: str,
    output_path: str | None,
    hard_burn: bool,
    language: str | None,
    translate: bool,
    quiet: bool,
) -> None:
    """Transcribe a video and embed subtitles into it.

    \b
    Examples:
      audiobench subtitle lecture.mp4                    Soft-embed subtitles
      audiobench subtitle lecture.mp4 --hard             Burn into video pixels
      audiobench subtitle lecture.mp4 -o subtitled.mp4   Custom output path
      audiobench subtitle lecture.mp4 --translate        Subtitles in English
    """
    import tempfile

    from audiobench.output.srt import SrtFormatter
    from audiobench.transcribe.audio_converter import SUPPORTED_VIDEO_FORMATS, embed_subtitles
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    video_path = Path(video)
    ext = video_path.suffix.lstrip(".").lower()

    if ext not in SUPPORTED_VIDEO_FORMATS:
        console.print(
            error_panel(
                "Unsupported format",
                f".{ext} is not a supported video format. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}",
            )
        )
        return

    # Resolve output path
    out = Path(output_path) if output_path else video_path.with_stem(f"{video_path.stem}_subtitled")

    if not quiet:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Subtitle Embedding")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Video:   [{ACCENT}]{video_path.name}[/]")
        console.print(f"    Output:  [{DIM}]{out.name}[/]")
        mode_desc = "Hard burn (permanent)" if hard_burn else "Soft embed (selectable track)"
        console.print(f"    Mode:    {mode_desc}")
        if translate:
            console.print("    Task:    [bold]Translate → English[/]")
        console.print(f"  [{DIM}]{'─' * 44}[/]")

    start_time = time.perf_counter()

    try:
        # Step 1: Transcribe the video's audio track
        if not quiet:
            console.print(f"  [{DIM}]Transcribing audio track...[/]")

        pipeline = TranscriptionPipeline()
        transcript = pipeline.transcribe_file(
            file_path=video,
            language=language,
            output_format="srt",
            word_timestamps=True,
            translate=translate,
        )

        # Step 2: Generate temporary SRT file
        formatter = SrtFormatter()
        srt_content = formatter.format(transcript)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, prefix="audiobench_sub_"
        ) as tmp:
            tmp.write(srt_content)
            srt_path = tmp.name

        if not quiet:
            console.print(f"  [{DIM}]Generated {transcript.segment_count} subtitle segments[/]")
            console.print(f"  [{DIM}]Embedding subtitles...[/]")

        # Step 3: Embed subtitles into video
        embed_subtitles(video_path, srt_path, out, hard_burn=hard_burn)

        # Cleanup temp SRT
        import contextlib

        with contextlib.suppress(OSError):
            Path(srt_path).unlink()

        elapsed = time.perf_counter() - start_time

        if not quiet:
            out_size = out.stat().st_size
            console.print()
            console.print(f"  [{SUCCESS}]✓ Subtitles embedded successfully[/]")
            console.print(f"    Output:   [{ACCENT}]{out}[/] ({format_size(out_size)})")
            console.print(f"    Segments: {transcript.segment_count}")
            console.print(f"    Elapsed:  {format_duration(elapsed)}")
            console.print()

    except Exception as e:
        console.print(error_panel("Subtitle embedding failed", str(e)))
