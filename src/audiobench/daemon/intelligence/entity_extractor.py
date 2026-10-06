"""Entity extraction from audio file metadata.

Extracts author and title from audio file metadata without requiring any LLM.

Strategy (in order of confidence):
1. ffprobe — reads embedded ID3/MP4/OGG tags (artist, album, title) from the
   actual audio file.  Most reliable — covers files with proper tagging.
2. File name pattern matching — for files without embedded tags or with partial
   metadata.
3. Transcript first-200-words scan — fallback when file name is unstructured
   and no tags are present.

Supported name patterns (case-insensitive):
  Author - NN - Title.ext          ("Alan Watts - 07 - Presence of mind.mp3")
  Author - Title.ext               ("Cameron Hanes - Undeniable.m4b")
  NN - Part N Chapter N - Title — Author.ext   (Dostoyevsky em-dash style)
  Title - Author.ext               ("Never Split the Difference - Chris Voss.m4a")
  Title by Author.ext              (plain pattern without dash)
  Krishnamurti Reflections...      (well-known name prefix, no dash)

No LLM required. All extraction is regex-based or subprocess-based (ffprobe).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Well-known author names and aliases (matched case-insensitively).
# These are used to classify ambiguous "Title - Author" vs "Author - Title".
# ---------------------------------------------------------------------------
_KNOWN_AUTHORS: list[tuple[str, str]] = [
    ("Alan Watts", "Alan Watts"),
    ("Watts", "Alan Watts"),
    ("J. Krishnamurti", "J. Krishnamurti"),
    ("Krishnamurti", "J. Krishnamurti"),
    ("Viktor Frankl", "Viktor Frankl"),
    ("Frankl", "Viktor Frankl"),
    ("Fyodor Dostoyevsky", "Fyodor Dostoyevsky"),
    ("Dostoyevsky", "Fyodor Dostoyevsky"),
    ("David Goggins", "David Goggins"),
    ("Goggins", "David Goggins"),
    ("Cameron Hanes", "Cameron Hanes"),
    ("Hanes", "Cameron Hanes"),
    ("Chris Voss", "Chris Voss"),
    ("Voss", "Chris Voss"),
    ("Napoleon Hill", "Napoleon Hill"),
    ("Eckhart Tolle", "Eckhart Tolle"),
    ("Tolle", "Eckhart Tolle"),
    ("James Joyce", "James Joyce"),
    ("Joyce", "James Joyce"),
    ("F. Scott Fitzgerald", "F. Scott Fitzgerald"),
    ("Fitzgerald", "F. Scott Fitzgerald"),
    ("Herman Melville", "Herman Melville"),
    ("Melville", "Herman Melville"),
    ("Dostoevsky", "Fyodor Dostoyevsky"),  # alternate spelling
    ("Khabib Nurmagomedov", "Khabib Nurmagomedov"),
    ("Khabib", "Khabib Nurmagomedov"),
    ("Bill Gates", "Bill Gates"),
    ("Gates", "Bill Gates"),
    ("Charles Munger", "Charles T. Munger"),
    ("Charlie Munger", "Charles T. Munger"),
    ("Munger", "Charles T. Munger"),
    ("Earl Nightingale", "Earl Nightingale"),
]

# Compiled for fast lookup.
_AUTHOR_RE_LIST: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canonical)
    for alias, canonical in _KNOWN_AUTHORS
]


@dataclass
class ExtractedEntity:
    """Result of entity extraction from a file or transcript."""

    title: str | None
    author: str | None
    confidence: str  # "high" | "medium" | "low"
    source: str  # "filename" | "transcript"


def _known_author_in(text: str) -> str | None:
    """Return canonical author name if any known author is found in text."""
    for pattern, canonical in _AUTHOR_RE_LIST:
        if pattern.search(text):
            return canonical
    return None


def _clean_stem(stem: str) -> str:
    """Remove track numbers, leading zeros, and other noise from a path stem."""
    # Remove leading "NN - " or "NN. " track number prefixes
    stem = re.sub(r"^\d+[\s.-]+", "", stem).strip()
    # Normalise em-dash / en-dash to regular dash
    stem = stem.replace("\u2014", " - ").replace("\u2013", " - ")
    # Collapse multiple spaces
    stem = re.sub(r"\s{2,}", " ", stem).strip()
    return stem


def extract_from_ffprobe(file_path: str) -> ExtractedEntity | None:
    """Extract author and title from embedded audio tags via ffprobe.

    Returns ``None`` if ffprobe is not available, the file does not exist, or
    neither ``artist`` nor ``title`` tags are present.

    Tag priority:
    - author: ``artist`` tag (preferred) or ``album_artist``
    - title:  ``title`` tag (preferred) or ``album`` (series grouping)

    This is Tier 1 in the extraction hierarchy.  Call this first; fall back to
    ``extract_from_filename()`` only if this returns ``None``.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        if not tags:
            return None

        # Normalise tag keys: ffprobe uses lowercase on most systems but
        # some encoders write uppercase (ARTIST, TITLE, ALBUM).
        tags_lower = {k.lower(): v for k, v in tags.items()}

        artist = tags_lower.get("artist") or tags_lower.get("album_artist")
        title = tags_lower.get("title")
        album = tags_lower.get("album")

        if not artist and not title:
            return None

        # Use title if present; fall back to album (series) so we at least
        # group episodes from the same series together.
        resolved_title = title or album

        return ExtractedEntity(
            title=resolved_title,
            author=artist,
            confidence="high",
            source="ffprobe",
        )

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None




def extract_from_filename(file_name: str) -> ExtractedEntity:
    """Attempt to extract author and title from a structured audio file name.

    Returns an ExtractedEntity with confidence="high" if a clear
    Author - Title or similar pattern is found, "medium" if a known
    author name is present but the title is ambiguous, or "low" if
    only a title guess is possible.
    """
    stem = Path(file_name).stem
    stem = _clean_stem(stem)

    # --- Pattern 1: "Author - NN - Title" or "Author - Title" (2+ dash parts) ---
    # Works for: "Alan Watts - 07 - Presence of mind", "Cameron Hanes - Undeniable"
    parts = [p.strip() for p in re.split(r"\s+-\s+", stem)]

    if len(parts) >= 2:
        # Check first part as author
        maybe_author_first = _known_author_in(parts[0])
        if maybe_author_first:
            # "Author - NN - Title"  →  skip numeric-only middle parts
            title_parts = [p for p in parts[1:] if not p.isdigit() and not re.fullmatch(r"\d+", p)]
            title = " - ".join(title_parts) if title_parts else parts[-1]
            return ExtractedEntity(title=title, author=maybe_author_first, confidence="high", source="filename")

        # Check last part as author (Dostoyevsky style: "NN - Part N Chapter N — Author")
        maybe_author_last = _known_author_in(parts[-1])
        if maybe_author_last:
            title_parts = parts[:-1]
            title = " - ".join(p for p in title_parts if not re.fullmatch(r"\d+", p))
            return ExtractedEntity(title=title, author=maybe_author_last, confidence="high", source="filename")

        # "Title - Author" where author is the last single-word token (e.g. "Never Split - Chris Voss")
        maybe_author_last = _known_author_in(parts[-1])
        if maybe_author_last:
            title = " - ".join(parts[:-1])
            return ExtractedEntity(title=title, author=maybe_author_last, confidence="high", source="filename")

    # --- Pattern 2: Any known author found anywhere in the stem ---
    maybe_author = _known_author_in(stem)
    if maybe_author:
        # Remove the author name and use the remainder as title
        title = re.sub(re.escape(maybe_author), "", stem, flags=re.IGNORECASE).strip(" -_.,|")
        return ExtractedEntity(title=title or stem, author=maybe_author, confidence="medium", source="filename")

    # --- Pattern 3: No known author — use the cleaned stem as title only ---
    return ExtractedEntity(title=stem, author=None, confidence="low", source="filename")


def extract_from_transcript(text: str, max_words: int = 200) -> ExtractedEntity:
    """Attempt to extract author / title from the first *max_words* of a transcript.

    Looks for common title/author announcement patterns used in audiobook
    narration openings.
    """
    # Sample only the first max_words words to stay fast.
    words = text.split()
    sample = " ".join(words[:max_words])

    # Pattern: "Title, by Author" or "Title by Author"
    by_match = re.search(
        r"(?P<title>[A-Z][^.!?\n]{5,80}?)\s+by\s+(?P<author>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        sample,
    )
    if by_match:
        return ExtractedEntity(
            title=by_match.group("title").strip(),
            author=by_match.group("author").strip(),
            confidence="medium",
            source="transcript",
        )

    # Pattern: any known author name appears in the sample
    maybe_author = _known_author_in(sample)
    if maybe_author:
        return ExtractedEntity(title=None, author=maybe_author, confidence="low", source="transcript")

    return ExtractedEntity(title=None, author=None, confidence="low", source="transcript")
