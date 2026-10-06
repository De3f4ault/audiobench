"""Work identity resolution engine for AudioBench.

Provides a unified, deterministic 4-tier hierarchy to resolve the canonical
(title, author) of an audio file, preventing work fragmentation across
multi-file audiobooks, lecture series, and spoken word recordings.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

GENERIC_ALBUM_DENYLIST: set[str] = {
    "",
    "audio cd",
    "audiobook",
    "audiobooks",
    "unknown",
    "unknown album",
}

_DISC_PATTERN = re.compile(r"^(?:disc|cd)\s*\d+$", re.IGNORECASE)
_WATTS_DISC_CLEANER = re.compile(r"\s*-\s*(?:disc|cd)\s*\d+.*$", re.IGNORECASE)
_UNABRIDGED_CLEANER = re.compile(r"\s*\(unabridged\)$", re.IGNORECASE)

_LIBRIVOX_SLUG_MAP: dict[str, tuple[str, str]] = {
    "notes": ("Notes from Underground", "Fyodor Dostoyevsky"),
    "allthesethingsadded": ("All These Things Added", "James Allen"),
}

_MULTIDISC_FILENAME_PATTERNS: list[tuple[re.Pattern[str], tuple[str, str]]] = [
    (re.compile(r"^Genius Makers-Disk\d+\.mp3$", re.IGNORECASE), ("Genius Makers", "Cade Metz")),
    (re.compile(r"^Franz_Kafka_-_The_Trial_\d+\.mp3$", re.IGNORECASE), ("The Trial", "Franz Kafka")),
    (re.compile(r"^(?:\d+mp3\s*—\s*)?Way of Zen CD\d.*\.mp3$", re.IGNORECASE), ("The Way of Zen", "Alan Watts")),
]

_AUTHOR_FILENAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?P<author>.+?)\s*-\s*\d{2}\s*-\s*(?P<title>.+)$"),
    re.compile(r"^(?P<author>.+?)\s*-\s*(?P<title>.+)$"),
    re.compile(r"^(?P<title>.+?)\s*—\s*(?P<author>[^.]+)$"),
]


def clean_string(val: str | None) -> str | None:
    """Strip whitespace and byte-order marks."""
    if not val:
        return None
    cleaned = val.replace("\ufeff", "").strip()
    return cleaned if cleaned else None


def extract_tags_from_file(file_path: str) -> dict[str, str]:
    """Extract format tags from an audio file using ffprobe."""
    p = Path(file_path)
    if not p.exists():
        return {}
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(p)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            return {}
        data = json.loads(res.stdout)
        raw_tags = data.get("format", {}) .get("tags", {})
        return {k.lower(): v for k, v in raw_tags.items()}
    except Exception as exc:
        logger.debug("extract_tags_from_file failed for %s: %s", file_path, exc)
        return {}


def resolve_work_identity(
    file_path: str | None = None,
    file_name: str = "",
    sequence_key: str | None = None,
    tags: dict[str, str] | None = None,
) -> tuple[str, str | None, str]:
    """Resolve the canonical (work_title, author, resolution_source) for an audio file.

    Hierarchy:
    1. Tier 1: Embedded Album Tag (with denylist, disc regex guard, and channel-masking rule)
    2. Tier 2: Sequence Key & Track Metadata
    3. Tier 3: Known LibriVox slugs and multi-disc filename patterns
    4. Tier 4: Cleaned track title or filename stem fallback
    """
    if tags is None and file_path:
        tags = extract_tags_from_file(file_path)
    tags = tags or {}

    alb = clean_string(tags.get("album"))
    art = clean_string(tags.get("artist") or tags.get("album_artist"))
    trk = clean_string(tags.get("title"))

    # 1. Tier 1: Embedded Album Tag
    if alb:
        alb_lower = alb.lower()
        art_lower = art.lower() if art else ""

        is_denied = (
            alb_lower in GENERIC_ALBUM_DENYLIST
            or bool(_DISC_PATTERN.match(alb_lower))
            or (art_lower and alb_lower == art_lower)  # Rule 3: Channel/uploader masking
        )
        if not is_denied:
            clean_title = _WATTS_DISC_CLEANER.sub("", alb).strip()
            clean_title = _UNABRIDGED_CLEANER.sub("", clean_title).strip()
            return clean_title, art, "tier1_album_tag"

    # 2. Tier 2: Sequence Key & Track Metadata
    if sequence_key and trk and art:
        seq_parts = sequence_key.split("-")
        if len(seq_parts) >= 2:
            return trk, art, "tier2_sequence_key"

    # 3. Tier 3: Structural Slugs and Multi-disc Patterns
    lower_fname = file_name.lower()
    for slug, (canonical_title, canonical_author) in _LIBRIVOX_SLUG_MAP.items():
        if lower_fname.startswith(slug + "_"):
            return canonical_title, canonical_author, "tier3_librivox_slug"

    for pattern, (canonical_title, canonical_author) in _MULTIDISC_FILENAME_PATTERNS:
        if pattern.match(file_name):
            return canonical_title, canonical_author, "tier3_multidisc_filename"

    # 4. Tier 4: Track Title or Filename Stem Fallback
    if trk and trk.lower() not in GENERIC_ALBUM_DENYLIST:
        return trk, art, "tier4_track_or_stem"

    stem = Path(file_name).stem if file_name else ""
    for pat in _AUTHOR_FILENAME_PATTERNS:
        m = pat.match(stem)
        if m:
            extracted_author = clean_string(m.group("author"))
            extracted_title = clean_string(m.group("title"))
            if extracted_title:
                return extracted_title, extracted_author or art, "tier4_fallback_stem"

    fallback_title = clean_string(stem) or "Untitled Work"
    return fallback_title, art, "tier4_fallback_stem"
