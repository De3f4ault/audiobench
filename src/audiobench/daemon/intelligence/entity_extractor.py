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
    ("Og Mandino", "Og Mandino"),
    ("Mandino", "Og Mandino"),
    ("Keshavaswami", "Keshavaswami"),
    ("Radhanath Swami", "Radhanath Swami"),
    ("Billy Graham", "Billy Graham"),
    ("DJ Shipley", "DJ Shipley"),
    ("Shipley", "DJ Shipley"),
    ("Gladys McGarey", "Gladys McGarey"),
    ("Gladys Mcgarly", "Gladys McGarey"),
    ("McGarey", "Gladys McGarey"),
    ("Bill Hicks", "Bill Hicks"),
    ("Terence McKenna", "Terence McKenna"),
    ("McKenna", "Terence McKenna"),
    ("Conor Cummins", "Conor Cummins"),
    ("Sadhguru", "Sadhguru"),
    ("John C. Bogle", "John C. Bogle"),
    ("John C Bogle", "John C. Bogle"),
    ("John Bogle", "John C. Bogle"),
    ("Bogle", "John C. Bogle"),
    ("Benjamin Graham", "Benjamin Graham"),
]

# Compiled for fast lookup.
_AUTHOR_RE_LIST: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canonical)
    for alias, canonical in _KNOWN_AUTHORS
]

_PASSIVE_VERBS_RE = re.compile(
    r"\b(?:told|said|endorsed|echoed|managed|created|generated|made|published|"
    r"initiated|sponsored|quoted|referenced|reviewed|read|narrated|written|"
    r"fooled|performed|accompanied|words|story|definition|endorsement|gains|returns|parable)\s*$",
    re.IGNORECASE,
)

_CLAUSE_STARTERS_RE = re.compile(
    r"^(?:It|There|This|He|She|They|We|You|Listen|Perhaps|Simply|But)\b",
    re.IGNORECASE,
)

_AUTHOR_DENYLIST_RE = re.compile(
    r"\b(?:American|Randomness|Vanguard|Berkshire|Business|Company|Corporation|Fund|Index|Press)\b",
    re.IGNORECASE,
)

_NAME_PATTERN = r"[A-Z](?:[a-zA-Z']+|\.)(?:\s+[A-Z](?:[a-zA-Z']+|\.)){1,3}"


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
        title = re.sub(r"\s+Audiobook(?:\s+Full)?.*$", "", title, flags=re.IGNORECASE).strip()
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
    opening_words = " ".join(words[:40])

    # Pattern 0: Commercial audiobook announcement (case-insensitive)
    # e.g. "Audio Renaissance presents The Little Book of Common Sense Investing ... by John C. Bogle read for you by Tom Pinto"
    ann_match = re.search(
        r"(?:presents|audiobook|unabridged)\s+(?P<title>.+?)\s+by\s+(?P<author>[a-z]+(?:\s+[a-z]\.?)?(?:\s+[a-z]+){1,2})\s+(?:read|narrated)\s+(?:for\s+you\s+)?by",
        sample,
        re.IGNORECASE,
    )
    if ann_match:
        raw_author = ann_match.group("author").strip()
        canonical = _known_author_in(raw_author) or raw_author.title()
        raw_title = ann_match.group("title").strip()
        # Tighten title boundary: strip common marketing subtitles/taglines
        clean_title = re.split(
            r"[:\-—]|\b(?:the only way|how to|a guide to)\b",
            raw_title,
            flags=re.IGNORECASE,
        )[0].strip().title()
        return ExtractedEntity(
            title=clean_title,
            author=canonical,
            confidence="high",
            source="transcript",
        )

    # Pattern 1: Title followed by explicit "Author: Name" or "Written by: Name"
    title_auth_match = re.search(
        r"(?P<title>[A-Z][^.!?\n]{3,80}?)[.!?\n]\s*(?:author|written by)\s*[:\-—]\s*(?P<author>" + _NAME_PATTERN + r")",
        sample,
        re.IGNORECASE,
    )
    if title_auth_match:
        cand_author = title_auth_match.group("author").strip()
        cand_title = title_auth_match.group("title").strip()
        if not re.search(r"\b(?:of|the|in|on|and|a|an)\b", cand_author, re.IGNORECASE):
            return ExtractedEntity(
                title=cand_title,
                author=cand_author,
                confidence="high",
                source="transcript",
            )

    # Pattern 2: Explicit "Author: Name" or "Written by: Name" without preceding title
    auth_match = re.search(
        r"\b(?:author|written by)\s*[:\-—]\s*(?P<author>" + _NAME_PATTERN + r")",
        sample,
        re.IGNORECASE,
    )
    if auth_match:
        cand = auth_match.group("author").strip()
        if not re.search(r"\b(?:of|the|in|on|and|a|an)\b", cand, re.IGNORECASE):
            return ExtractedEntity(
                title=None,
                author=cand,
                confidence="high",
                source="transcript",
            )

    # Pattern 3: "Title by Author" at start of text or sentence boundary within opening window
    by_match = re.search(
        r'(?:^|[.!?\n]\s*|[\"“])(?P<title>[A-Z][^\"”\n.!?]{3,60}?)(?:[\"”]\s*|\s+)by\s+(?P<author>' + _NAME_PATTERN + r")",
        opening_words,
    )
    if by_match:
        cand_title = by_match.group("title").strip()
        cand_author = by_match.group("author").strip()
        if (
            not _PASSIVE_VERBS_RE.search(cand_title)
            and not _CLAUSE_STARTERS_RE.search(cand_title)
            and not _AUTHOR_DENYLIST_RE.search(cand_author)
            and not re.search(r"\b(?:of|the|in|on|and|a|an)\b", cand_author, re.IGNORECASE)
        ):
            return ExtractedEntity(
                title=cand_title,
                author=cand_author,
                confidence="medium",
                source="transcript",
            )

    # Pattern 4: Narrated by or read by
    narr_match = re.search(
        r"\b(?:read by|narrated by)\s*[:\-—]?\s*(?P<author>" + _NAME_PATTERN + r")",
        sample,
        re.IGNORECASE,
    )
    if narr_match:
        cand = narr_match.group("author").strip()
        if not re.search(r"\b(?:of|the|in|on|and|a|an)\b", cand, re.IGNORECASE):
            return ExtractedEntity(
                title=None,
                author=cand,
                confidence="medium",
                source="transcript",
            )

    # Pattern 5: any known author name appears in the sample
    maybe_author = _known_author_in(sample)
    if maybe_author:
        return ExtractedEntity(title=None, author=maybe_author, confidence="low", source="transcript")

    # Pattern 6: Domain heuristics (Scripture, Prayer, Personal, Known Series)
    domain_author = detect_prayer_or_personal("", "", sample) or detect_scripture_author("", sample)
    if domain_author:
        return ExtractedEntity(title=None, author=domain_author, confidence="medium", source="transcript")

    return ExtractedEntity(title=None, author=None, confidence="low", source="transcript")


# ---------------------------------------------------------------------------
# Scripture, Prayer, and Spoken-Audio Domain Heuristics
# ---------------------------------------------------------------------------
_SCRIPTURE_BOOKS_RE = re.compile(
    r"\b(?:"
    r"Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
    r"(?:1|2|First|Second)?\s*Samuel|(?:1|2|First|Second)?\s*Kings|(?:1|2|First|Second)?\s*Chronicles|"
    r"Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs?|Ecclesiastes|Song of Solomon|"
    r"Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|"
    r"Matthew|Mark|Luke|John|Acts|Romans|"
    r"(?:1|2|First|Second)?\s*Corinthians|Galatians|Ephesians|Philippians|Colossians|"
    r"(?:1|2|First|Second)?\s*Thessalonians|(?:1|2|First|Second)?\s*Timothy|Titus|Philemon|"
    r"Hebrews|James|(?:1|2|First|Second)?\s*Peter|(?:1|2|3|First|Second|Third)?\s*John|Jude|Revelation"
    r")\b",
    re.IGNORECASE,
)

_SCRIPTURE_TITLES_RE = re.compile(
    r"\b(?:Scriptural Readings?|Biblical Readings?|Biblical Scriptures?|Biblical Teachings?|Biblical Psalms?|Biblical Lamentations?|First Ten Psalms|Reading of the First Ten Psalms)\b",
    re.IGNORECASE,
)

_SCRIPTURE_TEXT_RE = re.compile(
    r"\b(?:"
    r"The Book of Psalms|"
    r"King James Version|"
    r"English Standard Version|"
    r"New International Version|"
    r"Lord bless you and keep you|"
    r"Blessed is the one who does not walk in step|"
    r"Whoever dwells in the shelter of the Most High|"
    r"I will extol the Lord at all times|"
    r"God is our refuge and strength|"
    r"I lift up my eyes to the mountains|"
    r"Praise the Lord, my soul|"
    r"You have searched me, Lord"
    r")\b",
    re.IGNORECASE,
)

_PRAYER_RE = re.compile(
    r"\b(?:Christian Intercessory|Intercessory Prayers?|Biblical Prayers?|Deliverance Prayers?|Spiritual Warfare Prayers?)\b",
    re.IGNORECASE,
)

_CHURCH_RE = re.compile(
    r"\b(?:Nairobi Chapel|Mamlaka Hill Chapel|Mavuno Church|Youth Fest \d+)\b",
    re.IGNORECASE,
)

_MENS_SERIES_PHRASE = "what's up, brothers? hope everyone's doing well out there"
_MENS_SERIES_TITLES_RE = re.compile(
    r"\b(?:"
    r"Embracing Loner Confidence|"
    r"Winning Dating and Relationship Strategy|"
    r"Calibrating Dating Standards|"
    r"Relaxed Confidence Attracting and Keeping Women|"
    r"Men Must Learn This Important Lesson|"
    r"Most Men Don't Understand How Women Catch|"
    r"Once A Man Learns This Everything Changes|"
    r"Start Having Fun With Women|"
    r"The HARSH Reality of Female Attraction|"
    r"The Main Thing Women Crave From Men|"
    r"When A Woman Has Chosen YOU|"
    r"With Full Certainty|"
    r"Women Become More Attracted To Men|"
    r"You'll Stop Chasing When You Spot THIS|"
    r"This is what gets A Woman OBSESSED"
    r")\b",
    re.IGNORECASE,
)

_PERSONAL_RECORDING_RE = re.compile(
    r"(?:"
    r"\bdaily log\b|\bevening log\b|\bmorning log\b|snippet|mbuzi|machakos|"
    r"brammy|\bbirthday\b|dayz apart|late night conversation|boy from class|"
    r"audiobench system architecture|rag ranking|coding devotion|cognitive poten|"
    r"solitude friend|informal conversation|introspective reflections|friday 6th february|"
    r"day 2 13th january|thursday 26th march|wednesday 25th|tuk ramazan|personal reflections"
    r")",
    re.IGNORECASE,
)

_WATTS_LECTURE_TITLES = [
    "Skin Encapsulated Ego",
    "Meditation and the Illusion of Separate",
    "Time and Causality",
    "Conceptual Models and Patterns",
    "Perceiving Universal Creation",
    "Daoism Unconscious Virtue",
    "God Pretending to Be Human",
]


def detect_prayer_or_personal(title: str = "", file_name: str = "", text_sample: str = "") -> str | None:
    """Identify prayers, church fellowship, personal logs, and specific lectures."""
    comb = f"{title} {file_name} {text_sample[:300]}"
    t_clean = title.strip().lower()

    # 1. Christian Prayers
    if _PRAYER_RE.search(title) or t_clean == "prayers" or t_clean.startswith("prayers "):
        return "Christian Prayers"
    if "mighty god, sanctify and govern" in text_sample.lower() or "woman you're praying for" in text_sample.lower():
        return "Christian Prayers"

    # 2. Church / Chapel Fellowship
    if _CHURCH_RE.search(comb):
        return "Nairobi Chapel"

    # 3. Dostoyevsky
    if "underground man" in title.lower() or "diary and the diary itself are, of course, fictitious" in text_sample.lower():
        return "Fyodor Dostoyevsky"

    # 4. Alan Watts Lectures
    if any(k.lower() in title.lower() for k in _WATTS_LECTURE_TITLES) or "ferryboat vallejo" in text_sample.lower():
        return "Alan Watts"

    # 5. Men's Development / Dating Series
    if _MENS_SERIES_PHRASE in text_sample.lower() or _MENS_SERIES_TITLES_RE.search(title):
        return "Men's Development Series"

    # 6. Personal Audio Memos & Logs
    fn_lower = file_name.lower()
    if fn_lower.startswith("voice_chat_") or fn_lower.startswith("voice_record_"):
        return "Personal Recording"
    if _PERSONAL_RECORDING_RE.search(comb):
        return "Personal Recording"

    # 7. Specific identifiable speakers
    sample_lower = text_sample.lower()
    if "expression of your joy" in sample_lower or "consciousness is intensely focused" in sample_lower:
        return "Sadhguru"
    if "roll the dice" in title.lower() or "if you're going to try, go all the way" in sample_lower:
        return "Charles Bukowski"

    return None


def detect_scripture_author(title: str = "", text_sample: str = "") -> str | None:
    """Identify Biblical texts and chapter recitations, returning 'The Holy Bible'."""
    if _SCRIPTURE_TITLES_RE.search(title):
        return "The Holy Bible"
    if _SCRIPTURE_TEXT_RE.search(text_sample):
        return "The Holy Bible"
    # Book + chapter or number in title (e.g. "Romans 5", "Proverbs 3", "Matthew 6")
    if _SCRIPTURE_BOOKS_RE.search(title) and re.search(r"\b(?:chapter\s*\d+|\d+(?::\d+)?)\b", title, re.IGNORECASE):
        return "The Holy Bible"
    # Book + chapter in text opening (e.g. "Romans chapter 5", "Matthew chapter 6", "John chapter 8")
    if re.search(r"\b(?:" + _SCRIPTURE_BOOKS_RE.pattern[4:-3] + r")\s+(?:chapter\s+\d+|\d+:\d+)", text_sample[:200], re.IGNORECASE):
        return "The Holy Bible"
    return None
