"""Transcript viewing, formatting, search, and export engine.

Supports rendering transcripts into:
- txt: Readable narrative with optional timestamps and speaker labels
- srt: SubRip subtitles
- vtt: WebVTT tracks
- json: Formatted JSON dump
- tsv / csv: Tabular segment data
- raw: Pre-refinement raw Whisper text
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    format_duration,
)


def format_transcript(
    data: dict[str, Any],
    format_type: str = "text",
    *,
    timestamps: bool = True,
    speakers: bool = True,
    show_raw_whisper: bool = False,
) -> str:
    """Render transcript dictionary into the specified output format string.

    Args:
        data: Transcript dictionary from repository.
        format_type: One of 'text', 'txt', 'srt', 'vtt', 'json', 'tsv', 'csv', 'raw'.
        timestamps: Whether to show timestamp prefixes in text mode.
        speakers: Whether to show speaker badges in text mode.
        show_raw_whisper: If True, display original Whisper text instead of refined.
    """
    fmt = format_type.lower().strip()

    if fmt == "json":
        return json.dumps(data, indent=2, default=str)

    if fmt == "srt":
        return _format_as_srt(data.get("segments", []))

    if fmt == "vtt":
        return _format_as_vtt(data.get("segments", []))

    if fmt in ("tsv", "csv"):
        return _format_as_tabular(data.get("segments", []), delimiter="\t" if fmt == "tsv" else ",")

    if fmt == "raw" or show_raw_whisper:
        raw_whisper = data.get("raw_text") or data.get("full_text", "")
        return raw_whisper if raw_whisper else "(no raw Whisper text stored)"

    # Default: readable text / txt
    return _format_as_text(data, timestamps=timestamps, speakers=speakers)


def _format_as_text(
    data: dict[str, Any],
    *,
    timestamps: bool = True,
    speakers: bool = True,
) -> str:
    """Format transcript data as clean, readable text with optional timestamps and speakers."""
    segments = data.get("segments", [])
    if not segments:
        return data.get("full_text", "").strip()

    chapters = {c["id"]: c["title"] for c in data.get("chapters", []) if "id" in c and "title" in c}
    lines: list[str] = []
    current_chapter_id: int | None = None

    for seg in segments:
        chap_id = seg.get("chapter_id")
        if chap_id is not None and chap_id != current_chapter_id:
            current_chapter_id = chap_id
            chap_title = chapters.get(chap_id, f"Chapter {chap_id}")
            if lines:
                lines.append("")
            lines.append(f"── {chap_title} ──\n")

        parts: list[str] = []

        if timestamps:
            start = seg.get("start", 0)
            if start >= 3600:
                h = int(start // 3600)
                m = int((start % 3600) // 60)
                s = int(start % 60)
                parts.append(f"[{h}:{m:02d}:{s:02d}]")
            else:
                m = int(start // 60)
                s = int(start % 60)
                parts.append(f"[{m:02d}:{s:02d}]")

        if speakers and seg.get("speaker"):
            parts.append(f"<{seg['speaker']}>")

        text = seg.get("text", "").strip()
        if text:
            parts.append(text)
            lines.append(" ".join(parts))

    return "\n".join(lines)


def _format_as_srt(segments: list[dict[str, Any]]) -> str:
    """Format segments into SubRip (SRT) format."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 1.0)
        text = seg.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_as_vtt(segments: list[dict[str, Any]]) -> str:
    """Format segments into WebVTT format."""
    lines: list[str] = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 1.0)
        text = seg.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{_vtt_time(start)} --> {_vtt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_as_tabular(segments: list[dict[str, Any]], delimiter: str = "\t") -> str:
    """Format segments into TSV or CSV string."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(["index", "start", "end", "duration", "speaker", "text"])
    for i, seg in enumerate(segments, 1):
        start = round(seg.get("start", 0.0), 3)
        end = round(seg.get("end", start + 1.0), 3)
        duration = round(max(0.0, end - start), 3)
        speaker = seg.get("speaker") or ""
        text = seg.get("text", "").strip()
        writer.writerow([i, start, end, duration, speaker, text])
    return output.getvalue().rstrip()


def _srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    """Convert seconds to WebVTT time format: HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_transcript_header(data: dict[str, Any]) -> str:
    """Generate a stylized single-line Rich header summary for the transcript."""
    refined_at = data.get("refined_at")
    if refined_at:
        badge = f"[{SUCCESS}]✓ refined {refined_at[:10]}[/]"
    else:
        badge = f"[{DIM}]raw[/]"

    words = f"{data.get('word_count', 0):,} words"
    segs = f"{len(data.get('segments', []))} segs"
    dur = format_duration(data.get("duration", 0))

    header = (
        f"[{BOLD} {ACCENT}]#{data['id']}[/]  "
        f"[{ACCENT}]{data.get('file_name', 'untitled')}[/]  "
        f"[{DIM}]{data.get('language', '?')} • "
        f"{data.get('engine', '?')}/{data.get('model', '?')} • "
        f"{dur} • {words} • {segs}[/]  "
        f"{badge}"
    )
    return header


def search_transcript_segments(data: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Perform a case-insensitive search across segments and return matching hits."""
    q = query.strip().lower()
    if not q:
        return []

    hits: list[dict[str, Any]] = []
    for seg in data.get("segments", []):
        text = seg.get("text", "")
        if q in text.lower():
            start = seg.get("start", 0.0)
            m = int(start // 60)
            s = int(start % 60)
            time_str = f"{m:02d}:{s:02d}"
            hits.append(
                {
                    "index": seg.get("index", len(hits) + 1),
                    "start": start,
                    "time_str": time_str,
                    "speaker": seg.get("speaker"),
                    "text": text,
                }
            )
    return hits


def export_transcript_to_file(
    data: dict[str, Any],
    format_type: str,
    output_path: Path | str,
    *,
    timestamps: bool = True,
    speakers: bool = True,
) -> Path:
    """Export formatted transcript to file on disk."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    content = format_transcript(
        data,
        format_type=format_type,
        timestamps=timestamps,
        speakers=speakers,
    )
    out.write_text(content, encoding="utf-8")
    return out


def copy_to_clipboard(text: str) -> bool:
    """Copy string content to system clipboard via wl-copy, xclip, xsel, or pbcopy."""
    encoded = text.encode("utf-8")

    # 1. Wayland
    try:
        subprocess.run(["wl-copy"], input=encoded, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    # 2. X11 xclip
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=encoded, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    # 3. X11 xsel
    try:
        subprocess.run(["xsel", "--clipboard", "--input"], input=encoded, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    # 4. macOS pbcopy
    try:
        subprocess.run(["pbcopy"], input=encoded, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    return False
