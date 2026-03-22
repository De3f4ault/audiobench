"""System commands — info, download, doctor."""

from __future__ import annotations

import sys

import click

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    make_table,
)
from audiobench.core.settings import get_settings

# ── Info Command ────────────────────────────────────────────


@click.command()
def info() -> None:
    """Show system info and current settings."""
    settings = get_settings()

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    device_str = settings.resolve_device()
    device_label = f"{device_str} ({'CUDA' if cuda_available else 'CPU only'})"

    table = make_table(
        f"{APP_NAME} v{APP_VERSION}",
        [
            ("Setting", {"style": BOLD}),
            ("Value", {}),
        ],
    )

    table.add_row("Model", settings.model_name)
    table.add_row("Device", device_label)
    table.add_row("Compute Type", settings.resolve_compute_type())
    table.add_row("CPU Threads", str(settings.resolve_cpu_threads()))
    table.add_row("Speed Preset", settings.speed_preset)
    table.add_row("Beam Size", str(settings.resolve_beam_size()))
    table.add_row("Batch Size", str(settings.resolve_batch_size()))
    table.add_row("Language", settings.language or "auto-detect")
    table.add_row("Output Format", settings.output_format)
    table.add_row("Word Timestamps", str(settings.word_timestamps))
    table.add_row("Diarization", str(settings.enable_diarization))
    table.add_row("Database", settings.database_url)
    table.add_row("Models Dir", str(settings.models_dir))
    table.add_row("HF Token", "✓ set" if settings.hf_token else "– not set")
    table.add_row("Log Level", settings.log_level)

    console.print(table)

    # Engines
    from audiobench.transcribe.engines.engine_registry import list_engines

    console.print(f"  [{DIM}]Engines: {', '.join(list_engines())}[/]")

    # Formats
    from audiobench.transcribe.audio_converter import AudioLoader

    formats = AudioLoader.get_supported_formats()
    console.print(f"  [{DIM}]Audio: {', '.join(sorted(formats['audio']))}[/]")
    console.print(f"  [{DIM}]Video: {', '.join(sorted(formats['video']))}[/]")

    # Presets
    console.print()
    pt = make_table(
        "Speed Presets",
        [
            ("Preset", {}),
            ("Beam", {"justify": "center", "width": 6}),
            ("Batch", {"justify": "center", "width": 6}),
            ("Description", {}),
        ],
    )
    pt.add_row("fast", "1", "4", "Maximum speed, good quality")
    pt.add_row("balanced", "3", "4", "Good balance (default)")
    pt.add_row("accurate", "5", "1", "Best quality, slower")
    console.print(pt)


# ── Download Command ────────────────────────────────────────


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(
        [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "large-v3-turbo",
        ]
    ),
)
def download(model_name: str) -> None:
    """Pre-download a Whisper model for offline use."""
    console.print(f"  [{ACCENT}]Downloading model:[/] {model_name} ...")

    try:
        from faster_whisper import WhisperModel

        WhisperModel(model_name, device="cpu", compute_type="int8")
        console.print(f"  [{SUCCESS}]✓[/] Model '{model_name}' downloaded and cached.")
    except Exception as e:
        console.print(error_panel(f"Download failed: {model_name}", str(e)))
        sys.exit(1)


# ── Doctor Command ──────────────────────────────────────────


@click.command()
def doctor() -> None:
    """Check system dependencies and health.

    \b
    Checks:
      ffmpeg, ffprobe, faster-whisper, piper-tts, ollama,
      CUDA availability, disk space, database connectivity.
    """
    import shutil

    console.print(f"\n  [{BOLD} {ACCENT}]{APP_NAME} Doctor[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")

    checks = []

    # ffmpeg
    ff = shutil.which("ffmpeg")
    checks.append(("ffmpeg", "✓" if ff else "✗", ff or "NOT FOUND"))

    # ffprobe
    fp = shutil.which("ffprobe")
    checks.append(("ffprobe", "✓" if fp else "✗", fp or "NOT FOUND"))

    # faster-whisper
    try:
        import faster_whisper

        v = getattr(faster_whisper, "__version__", "installed")
        checks.append(("faster-whisper", "✓", v))
    except ImportError:
        checks.append(("faster-whisper", "✗", "NOT INSTALLED"))

    # piper-tts
    piper = shutil.which("piper")
    checks.append(("piper-tts", "✓" if piper else "–", piper or "not found (optional)"))

    # ollama
    ollama = shutil.which("ollama")
    checks.append(("ollama", "✓" if ollama else "–", ollama or "not found (optional)"))

    # CUDA
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            checks.append(("CUDA", "✓", name))
        else:
            checks.append(("CUDA", "–", "not available (CPU mode)"))
    except ImportError:
        checks.append(("CUDA", "–", "torch not installed"))

    # Database
    try:
        from audiobench.core.db_engine import init_db

        init_db()
        checks.append(("Database", "✓", get_settings().database_url))
    except Exception as e:
        checks.append(("Database", "✗", str(e)))

    # Display results
    table = make_table(
        "Health Check",
        [
            ("Component", {"style": BOLD}),
            ("Status", {"width": 6, "justify": "center"}),
            ("Details", {}),
        ],
    )

    for name, status, detail in checks:
        style = SUCCESS if status == "✓" else (DIM if status == "–" else "red")
        table.add_row(name, f"[{style}]{status}[/]", detail)

    console.print(table)

    failed = sum(1 for _, s, _ in checks if s == "✗")
    if failed:
        console.print(f"\n  [red]{failed} required dependency missing![/]")
    else:
        console.print(f"\n  [{SUCCESS}]All required dependencies OK[/]")


# ── Status Command ──────────────────────────────────────────


@click.command()
def status() -> None:
    """Show usage statistics and disk usage.

    \b
    Displays:
      Database size, transcription count, total hours,
      model cache size, voice cache size, preset count.
    """
    from pathlib import Path

    settings = get_settings()

    console.print(f"\n  [{BOLD} {ACCENT}]{APP_NAME} Status[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")

    # Database stats
    try:
        from audiobench.core.db_engine import init_db
        from audiobench.storage.repository import TranscriptionRepository

        init_db()
        repo = TranscriptionRepository()
        records = repo.get_history(limit=10000)
        total_count = len(records)
        total_duration = sum(r.get("duration", 0) or 0 for r in records)
        total_words = sum(r.get("word_count", 0) or 0 for r in records)
    except Exception:
        total_count = 0
        total_duration = 0
        total_words = 0

    # Disk usage
    def dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    from audiobench.cli.display.theme import format_size

    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    db_size = db_path.stat().st_size if db_path.exists() else 0
    models_size = dir_size(settings.models_dir)
    voices_size = dir_size(settings.voices_dir) if hasattr(settings, "voices_dir") else 0
    presets_dir = settings.data_dir / "presets"
    presets_count = len(list(presets_dir.glob("*.toml"))) if presets_dir.exists() else 0

    table = make_table(
        "Usage Statistics",
        [
            ("Metric", {"style": BOLD}),
            ("Value", {}),
        ],
    )

    hours = total_duration / 3600
    table.add_row("Transcriptions", str(total_count))
    table.add_row("Total Audio", f"{hours:.1f} hours")
    table.add_row("Total Words", f"{total_words:,}")
    table.add_row("Database Size", format_size(db_size))
    table.add_row("Model Cache", format_size(models_size))
    table.add_row("Voice Cache", format_size(voices_size))
    table.add_row("Saved Presets", str(presets_count))

    console.print(table)


# ── Cleanup Command ─────────────────────────────────────────


@click.command()
@click.option(
    "--older-than",
    default=None,
    help='Delete transcriptions older than (e.g., "30d", "1w", "6m")',
)
@click.option(
    "--cache",
    "clean_cache",
    is_flag=True,
    help="Remove cached model files",
)
@click.option(
    "--temp",
    "clean_temp",
    is_flag=True,
    help="Remove temporary files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting",
)
@click.confirmation_option(prompt="Are you sure?")
def cleanup(
    older_than: str | None,
    clean_cache: bool,
    clean_temp: bool,
    dry_run: bool,
) -> None:
    """Clean up old data, cache, and temp files.

    \b
    Examples:
      audiobench cleanup --older-than 30d        Delete old transcriptions
      audiobench cleanup --cache                  Remove model cache
      audiobench cleanup --temp                   Remove temp files
      audiobench cleanup --older-than 7d --dry-run   Preview what would be deleted
    """
    import tempfile
    from pathlib import Path

    settings = get_settings()
    actions = []

    # Older-than: delete old transcriptions
    if older_than:
        cutoff = _parse_age(older_than)
        if cutoff:
            from audiobench.core.db_engine import init_db
            from audiobench.storage.repository import TranscriptionRepository

            init_db()
            repo = TranscriptionRepository()
            records = repo.get_history(limit=10000)

            from datetime import datetime

            old = [
                r
                for r in records
                if r.get("created_at") and datetime.fromisoformat(r["created_at"]) < cutoff
            ]

            if old:
                if dry_run:
                    actions.append(
                        f"Would delete {len(old)} transcription(s) older than {older_than}"
                    )
                else:
                    for r in old:
                        repo.delete_by_id(r["id"])
                    actions.append(f"Deleted {len(old)} transcription(s) older than {older_than}")

    # Cache cleanup
    if clean_cache:
        cache_path = settings.models_dir
        if cache_path.exists():
            from audiobench.cli.display.theme import format_size

            size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            if dry_run:
                actions.append(f"Would remove model cache ({format_size(size)})")
            else:
                import shutil

                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
                actions.append(f"Removed model cache ({format_size(size)})")

    # Temp cleanup
    if clean_temp:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_files = list(tmp_dir.glob("audiobench_*"))
        if tmp_files:
            size = sum(f.stat().st_size for f in tmp_files if f.is_file())
            if dry_run:
                actions.append(f"Would remove {len(tmp_files)} temp file(s)")
            else:
                import contextlib

                for f in tmp_files:
                    with contextlib.suppress(OSError):
                        f.unlink()
                actions.append(f"Removed {len(tmp_files)} temp file(s)")

    if not actions:
        console.print(f"  [{DIM}]Nothing to clean up.[/]")
        return

    prefix = "[DRY RUN] " if dry_run else ""
    for action in actions:
        icon = "👁" if dry_run else "✓"
        console.print(f"  [{SUCCESS}]{icon}[/] {prefix}{action}")


def _parse_age(age_str: str):
    """Parse an age string into a datetime cutoff."""
    from datetime import datetime, timedelta

    units = {"h": "hours", "d": "days", "w": "weeks", "m": "days"}
    multipliers = {"h": 1, "d": 1, "w": 1, "m": 30}

    if len(age_str) >= 2 and age_str[-1] in units and age_str[:-1].isdigit():
        val = int(age_str[:-1]) * multipliers[age_str[-1]]
        unit = units[age_str[-1]]
        if age_str[-1] == "m":
            unit = "days"
        return datetime.now() - timedelta(**{unit: val})

    return None
