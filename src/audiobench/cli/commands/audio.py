"""Audio utility commands — analyze, convert, merge."""

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
    WARNING,
    console,
    error_panel,
    format_duration,
    format_size,
    make_table,
    stdout,
)

# ── Analyze Command ─────────────────────────────────────────


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--noise",
    default="-35dB",
    show_default=True,
    help="Silence detection threshold (lower = more sensitive)",
)
@click.option(
    "--min-silence",
    default=0.5,
    show_default=True,
    type=float,
    help="Minimum silence duration in seconds",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Output as JSON",
)
def analyze(
    file: str,
    noise: str,
    min_silence: float,
    json_mode: bool,
) -> None:
    """Analyze audio: loudness stats, silence map, quality assessment.

    \b
    Examples:
      audiobench analyze meeting.m4a                  Full analysis
      audiobench analyze recording.wav --json         JSON output
      audiobench analyze noisy.m4a --noise -25dB      Less sensitive silence detection
    """
    import json as json_lib

    from audiobench.transcribe.audio_converter import analyze_audio, probe

    file_path = Path(file).resolve()
    info = probe(file_path)

    if not json_mode:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Audio Analysis")
        console.print(f"  [{'─' * 44}]")
        console.print(
            f"    File:      [{ACCENT}]{file_path.name}[/] "
            f"({format_size(file_path.stat().st_size)})"
        )
        console.print(f"    Duration:  {format_duration(info.duration)}")
        console.print(f"    Codec:     {info.codec} | {info.sample_rate}Hz | {info.channels}ch")
        console.print(f"  [{'─' * 44}]")
        console.print(f"  [{DIM}]Analyzing (silence detection + loudness scan)...[/]")

    analysis = analyze_audio(file_path)

    if json_mode:
        result = {
            "file": str(file_path),
            "duration": analysis.duration,
            "loudness": {
                "integrated_lufs": analysis.integrated_loudness,
                "range_lu": analysis.loudness_range,
                "true_peak_dbtp": analysis.true_peak,
            },
            "silence": {
                "total_seconds": analysis.silence_total,
                "percent": analysis.silence_percent,
                "regions": len(analysis.silence_regions),
                "longest": max((r.duration for r in analysis.silence_regions), default=0),
            },
        }
        stdout.print(json_lib.dumps(result, indent=2), highlight=False)
        return

    # ── Loudness section ──
    console.print()
    console.print(f"  [{BOLD}]Loudness[/]")

    lufs = analysis.integrated_loudness
    if lufs > -10:
        quality = f"[{WARNING}]very loud — may clip[/]"
    elif lufs > -14:
        quality = f"[{SUCCESS}]loud — streaming optimized[/]"
    elif lufs > -18:
        quality = f"[{SUCCESS}]good — speech/podcast range[/]"
    elif lufs > -24:
        quality = f"[{DIM}]quiet — consider --enhance[/]"
    else:
        quality = f"[{WARNING}]very quiet — use --enhance[/]"

    console.print(f"    Integrated:  {lufs:.1f} LUFS  ({quality})")
    console.print(f"    Range:       {analysis.loudness_range:.1f} LU")
    console.print(f"    True Peak:   {analysis.true_peak:.1f} dBTP")

    # ── Silence section ──
    console.print()
    console.print(f"  [{BOLD}]Silence[/]")
    console.print(
        f"    Total:       {format_duration(analysis.silence_total)} "
        f"({analysis.silence_percent:.1f}%)"
    )
    console.print(f"    Speech:      {analysis.speech_percent:.1f}%")
    console.print(f"    Regions:     {len(analysis.silence_regions)} silent passages")

    if analysis.silence_regions:
        longest = max(analysis.silence_regions, key=lambda r: r.duration)
        console.print(
            f"    Longest:     {longest.duration:.1f}s at {format_duration(longest.start)}"
        )

    # ── Top 5 longest silence regions ──
    if len(analysis.silence_regions) > 1:
        console.print()
        table = make_table(
            "Top Silence Regions",
            [
                ("#", {"style": DIM, "width": 4}),
                ("Start", {"width": 10}),
                ("End", {"width": 10}),
                ("Duration", {"justify": "right", "width": 10}),
            ],
        )
        top = sorted(analysis.silence_regions, key=lambda r: r.duration, reverse=True)[:5]
        for i, r in enumerate(top, 1):
            table.add_row(
                str(i),
                format_duration(r.start),
                format_duration(r.end),
                f"{r.duration:.1f}s",
            )
        console.print(table)

    # ── Recommendation ──
    console.print()
    recs = []
    if lufs < -20:
        recs.append("--enhance (normalize loudness)")
    if analysis.silence_percent > 15:
        recs.append("--trim (remove silence)")
    if analysis.loudness_range > 15:
        recs.append("--enhance (compress dynamic range)")

    if recs:
        console.print(f"  [{ACCENT}]Recommendation:[/] Use {', '.join(recs)} for best results")
    else:
        console.print(f"  [{SUCCESS}]✓ Audio quality looks good for transcription[/]")
    console.print()


# ── Convert Command ─────────────────────────────────────────


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path", required=True, help="Output path (format from extension)"
)
@click.option("--bitrate", default=None, help="Override bitrate (e.g., 128k, 320k)")
@click.option("--sample-rate", default=None, type=int, help="Target sample rate in Hz")
@click.option("--channels", default=None, type=int, help="Target channels (1=mono, 2=stereo)")
@click.option(
    "--speed",
    default=None,
    type=float,
    help="Playback speed multiplier (e.g., 1.5 for 50%% faster)",
)
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def convert(
    input_file: str,
    output_path: str,
    bitrate: str | None,
    sample_rate: int | None,
    channels: int | None,
    speed: float | None,
    quiet: bool,
) -> None:
    """Convert audio between formats with optional processing.

    \b
    Examples:
      audiobench convert meeting.m4a -o meeting.mp3           M4A → MP3
      audiobench convert meeting.m4a -o meeting.opus          M4A → Opus (tiny, HD)
      audiobench convert meeting.wav -o meeting.flac          WAV → FLAC (lossless)
      audiobench convert lecture.mp3 -o fast.mp3 --speed 1.5  Speed up 50%
      audiobench convert video.mp4 -o audio.mp3               Extract audio
      audiobench convert meeting.m4a -o out.mp3 --bitrate 320k  Custom bitrate
    """
    import time

    from audiobench.transcribe.audio_converter import convert_audio, probe

    input_p = Path(input_file).resolve()
    output_p = Path(output_path).resolve()

    if not quiet:
        info = probe(input_p)
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Convert")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(
            f"    Input:   [{ACCENT}]{input_p.name}[/] ({format_size(input_p.stat().st_size)})"
        )
        console.print(f"    Output:  [{DIM}]{output_p.name}[/]")
        console.print(f"    Codec:   {info.codec} → {output_p.suffix.lstrip('.')}")
        if speed:
            console.print(f"    Speed:   {speed:.1f}x")
        if bitrate:
            console.print(f"    Bitrate: {bitrate}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"  [{DIM}]Converting...[/]")

    start = time.perf_counter()

    try:
        result_path = convert_audio(
            input_p,
            output_p,
            bitrate=bitrate,
            sample_rate=sample_rate,
            channels=channels,
            speed=speed,
        )
        elapsed = time.perf_counter() - start

        if not quiet:
            out_size = result_path.stat().st_size
            console.print(f"  [{SUCCESS}]✓ Converted in {format_duration(elapsed)}[/]")
            console.print(f"    Output: [{ACCENT}]{result_path}[/] ({format_size(out_size)})")
            console.print()

    except Exception as e:
        console.print(error_panel("Conversion failed", str(e)))
        sys.exit(1)


# ── Merge Command ───────────────────────────────────────────


@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output", "output_path", required=True, help="Output file path")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def merge(
    files: tuple[str, ...],
    output_path: str,
    quiet: bool,
) -> None:
    """Concatenate multiple audio files into one.

    \b
    Examples:
      audiobench merge part1.wav part2.wav -o full.wav
      audiobench merge *.m4a -o combined.m4a
    """
    import time

    from audiobench.transcribe.audio_converter import concat_audio

    if len(files) < 2:
        console.print(error_panel("Need at least 2 files to merge"))
        sys.exit(1)

    if not quiet:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Merge")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        for f in files:
            p = Path(f)
            console.print(f"    [{DIM}]→[/] {p.name} ({format_size(p.stat().st_size)})")
        console.print(f"    Output:  [{DIM}]{output_path}[/]")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"  [{DIM}]Merging {len(files)} files...[/]")

    start = time.perf_counter()

    try:
        result_path = concat_audio(list(files), output_path)
        elapsed = time.perf_counter() - start

        if not quiet:
            out_size = result_path.stat().st_size
            console.print(
                f"  [{SUCCESS}]✓ Merged {len(files)} files in {format_duration(elapsed)}[/]"
            )
            console.print(f"    Output: [{ACCENT}]{result_path}[/] ({format_size(out_size)})")
            console.print()

    except Exception as e:
        console.print(error_panel("Merge failed", str(e)))
        sys.exit(1)
