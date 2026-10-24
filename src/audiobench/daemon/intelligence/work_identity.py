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

_PARTITION_PATTERN = re.compile(
    r"^(?:disc|cd|disk|chapter|appendix|section|part)(?:\.?(?:\s*\d+|[\s_\-]+(?:\d+|[a-z]\b|[ivx]+\b)).*?$|\s*$)",
    re.IGNORECASE,
)
_PARTITION_CLEANER = re.compile(
    r"\s*[\-_,]\s*(?:disc|cd|disk|part|pt|chapter|ch|section|appendix|vol|volume)(?:\.?(?:\s*\d+|[\s_\-]+(?:\d+|[a-z]\b|[ivx]+\b)).*?$|\s*$)",
    re.IGNORECASE,
)
_DISC_PATTERN = _PARTITION_PATTERN
_WATTS_DISC_CLEANER = _PARTITION_CLEANER
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
    re.compile(r"^(?P<author>.+?)\s+-\s+\d{2}\s+-\s+(?P<title>.+)$"),
    re.compile(r"^(?P<author>.+?)\s+-\s+(?P<title>.+)$"),
    re.compile(r"^(?P<title>.+?)\s*—\s*(?P<author>[^.]+)$"),
]


_SLASH_AUTHOR_NARRATOR_RE = re.compile(
    r"^\s*(?P<author>[A-Z](?:[a-zA-Z']+|\.)(?:\s+[A-Z](?:[a-zA-Z']+|\.)){1,3})\s*/\s*"
    r"(?P<narrator>[A-Z](?:[a-zA-Z']+|\.)(?:\s+[A-Z](?:[a-zA-Z']+|\.)){1,3})\s*$"
)


def clean_string(val: str | None) -> str | None:
    """Strip whitespace and byte-order marks."""
    if not val:
        return None
    cleaned = val.replace("\ufeff", "").strip()
    return cleaned if cleaned else None


def clean_author(val: str | None) -> str | None:
    """Clean author string, splitting off slash-delimited narrators if present.

    Audiobook tags frequently format the artist field as 'Author/Narrator'
    (e.g. 'Napoleon Hill/Anthony Rogers', 'Benjamin Graham/Luke Daniels',
    'Paul Strathern/Robert Whitfield'). When exactly two multi-token proper
    names appear separated by a slash, the first token represents the
    intellectual author.
    """
    base = clean_string(val)
    if not base:
        return None
    m = _SLASH_AUTHOR_NARRATOR_RE.match(base)
    if m:
        return m.group("author").strip()
    return base


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
    art = clean_author(tags.get("artist") or tags.get("album_artist"))
    trk = clean_string(tags.get("title"))

    # 1. Tier 1: Embedded Album Tag
    if alb:
        alb_lower = alb.lower()
        art_lower = art.lower() if art else ""

        is_denied = (
            alb_lower in GENERIC_ALBUM_DENYLIST
            or bool(_PARTITION_PATTERN.match(alb_lower))
            or (art_lower and alb_lower == art_lower)  # Rule 3: Channel/uploader masking
        )
        if not is_denied:
            clean_title = _PARTITION_CLEANER.sub("", alb).strip()
            clean_title = _UNABRIDGED_CLEANER.sub("", clean_title).strip()
            if clean_title:
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
    stem = Path(file_name).stem if file_name else ""

    # If author is missing, check known authors from trk or stem
    if not art:
        from audiobench.daemon.intelligence.entity_extractor import _known_author_in
        art = (clean_string(trk) and _known_author_in(trk)) or (clean_string(stem) and _known_author_in(stem))

    if trk and trk.lower() not in GENERIC_ALBUM_DENYLIST:
        clean_trk = _PARTITION_CLEANER.sub("", trk).strip()
        clean_trk = _UNABRIDGED_CLEANER.sub("", clean_trk).strip()
        if clean_trk:
            return clean_trk, art, "tier4_track_or_stem"
        return trk, art, "tier4_track_or_stem"

    for pat in _AUTHOR_FILENAME_PATTERNS:
        m = pat.match(stem)
        if m:
            extracted_author = clean_author(m.group("author"))
            extracted_title = clean_string(m.group("title"))
            if extracted_title:
                clean_extracted = _PARTITION_CLEANER.sub("", extracted_title).strip()
                return clean_extracted or extracted_title, extracted_author or art, "tier4_fallback_stem"

    if art and stem:
        # If author was recognized in the stem, clean author and audiobook tags from title
        cleaned_t = re.sub(re.escape(art), "", stem, flags=re.IGNORECASE).strip(" -_.,|")
        cleaned_t = re.sub(r"\s+Audiobook(?:\s+Full)?.*$", "", cleaned_t, flags=re.IGNORECASE).strip()
        cleaned_t = _PARTITION_CLEANER.sub("", cleaned_t).strip()
        if cleaned_t:
            return cleaned_t, art, "tier4_fallback_stem"

    fallback_title = clean_string(stem) or "Untitled Work"
    cleaned_fallback = _PARTITION_CLEANER.sub("", fallback_title).strip()
    return cleaned_fallback or fallback_title, art, "tier4_fallback_stem"
