"""Sequence resolution engine for audio files.

Version: 1.0

Resolves a deterministic, lexically-sortable sequence key for an audio file
within its collection/album/work, based on embedded metadata tags (ID3/MP4)
and filename structural patterns.

Keys produced:
  - "01-002"  : Disc 1, Track 2 (for multi-disc sets or Watts disc-in-album albums)
  - "007"     : Global track 7 or extracted filename index 7
  - "012b"    : Sub-chapter 12b (e.g. Ulysses / multi-part tracks)
  - None      : Standalone/unordered audio (callers fall back to file_name sort)
"""

from __future__ import annotations

import re
from pathlib import Path


def _parse_int_prefix(val: str | None) -> int | None:
    """Parse leading integer from a tag value like '9/12', '09', '9'."""
    if not val:
        return None
    val = str(val).strip()
    m = re.match(r"^(\d+)", val)
    return int(m.group(1)) if m else None


def resolve_sequence_key(file_name: str, tags: dict[str, str] | None = None) -> str | None:
    """Resolve a normalized, lexically-sortable sequence key for an audio file.

    Args:
        file_name: The base name or full path of the audio file.
        tags: Dict of lowercase format/metadata tags (e.g. from ffprobe).

    Returns:
        String sort key (e.g. "01-002", "007", "012b") or None if unordered.
    """
    tags = tags or {}
    album = (tags.get("album") or "").strip()
    track_raw = (tags.get("track") or "").strip()
    disc_raw = (tags.get("disc") or "").strip()

    track = _parse_int_prefix(track_raw)
    disc = _parse_int_prefix(disc_raw)

    # Branch 1: Explicit disc tag + track tag
    if disc is not None and track is not None:
        return f"{disc:02d}-{track:03d}"

    # Branch 2: Disc/CD number in album name + track tag (e.g. Alan Watts Out of Your Mind)
    if track is not None:
        m = re.search(r"\b(?:Disc|CD)\s*(\d+)\b", album, re.IGNORECASE)
        if m:
            album_disc = int(m.group(1))
            return f"{album_disc:02d}-{track:03d}"
        # Branch 3: Plain global track
        return f"{track:03d}"

    stem = Path(file_name).stem

    # Branch 4a: Leading-number filename (e.g. "01 - Part 1 Chapter 1", "02_chapter")
    m = re.match(r"^(\d+)\s*[-_\s\.]", stem)
    if m:
        n = int(m.group(1))
        # Guard against 4-digit years at start of filename
        if n < 1900 or n > 2100:
            return f"{n:03d}"

    # Branch 4b: LibriVox / standard audio pattern: prefix_NN_author_kb or prefix_NN[ab]_...
    m = re.match(
        r"^[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*?_(\d+)([a-zA-Z])?(?:_[a-zA-Z0-9]+)*?_\d+kb",
        stem,
        re.IGNORECASE,
    )
    if m:
        num = int(m.group(1))
        sub = (m.group(2) or "").lower()
        return f"{num:03d}{sub}"

    # Branch 4c: Generic "Part NN", "Chapter NN", "Disc/Disk NN", "-NN" in filename
    m = re.search(r"(?:[Pp]art|[Cc]hapter|[Dd]is[ck]|[Cc][Dd]|[-_])\s*0*(\d+)([a-zA-Z])?(?:[-_\s\.]|$)", stem)
    if m:
        num = int(m.group(1))
        sub = (m.group(2) or "").lower()
        if num < 1900 or num > 2100:  # Not a year
            return f"{num:03d}{sub}"

    # Branch 5: No sequence extractable
    return None
