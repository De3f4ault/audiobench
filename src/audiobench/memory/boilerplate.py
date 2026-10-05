"""Audiobook and spoken-word boilerplate detection and stripping.

Covers:
- Opening preambles: publisher stamps, narrator credits, platform announcements.
- Closing outros: end-of-chapter credits, unabridged production notices, copyright stamps.
- Pure boilerplate detection: chunks with no substantive content beyond disclaimers.
- Whisper phonetic variants: LibriVox is commonly transcribed as Libravox, Liebervox,
  Librefox, Librevox, Library Vox, etc.

Public API
----------
strip_preamble(text)        Remove publisher opening formula from the start of text.
strip_outro(text)           Remove closing credits/stamps from the end of text.
clean_audio_disclaimers(text) Run both passes; returns clean text.
is_pure_boilerplate(text)   True if text is substantively empty after cleaning.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Publisher and Platform Name Fragment Pool
#
# Used to build regex patterns dynamically. Each entry is matched
# case-insensitively. Ordering does not affect correctness (all patterns
# are tried), but longer / more specific entries should come first within
# each category so they are tried before their prefixes.
# ---------------------------------------------------------------------------

# ── Major "Big 5" Trade Imprints ───────────────────────────────────────────
_BIG5_PUBLISHERS = [
    # Penguin Random House family
    r"penguin\s+random\s+house\s+audio",
    r"random\s+house\s+audio",
    r"penguin\s+audio",
    r"books\s+on\s+tape",
    r"listening\s+library",
    # HarperCollins family
    r"harperaudio",
    r"harper\s+audio",
    r"harpercollins",
    # Macmillan family
    r"macmillan\s+audio",
    r"st\.?\s+martin'?s?\s+(?:audio|press)",
    r"tor\s+audio",
    r"audio\s+renaissance",
    # Simon & Schuster family
    r"simon\s+&\s+schuster\s+audio",
    r"s(?:imon)?\s*&\s*s(?:chuster)?\s+audio",
    r"pimsleur",
    # Hachette family
    r"hachette\s+audio",
    r"grand\s+central\s+(?:audio|publishing)",
    r"orbit\s+audio",
]

# ── Specialist & Independent Audiobook Publishers ─────────────────────────
_SPECIALIST_PUBLISHERS = [
    r"sounds\s+true",
    r"tantor\s+(?:audio|media)",
    r"recorded\s+books",
    r"rbmedia",
    r"blackstone\s+(?:audio|publishing)",
    r"downpour(?:\.com)?",
    r"highbridge\s+(?:audio|company)",
    r"podium\s+(?:audio|publishing)",
    r"brilliance\s+(?:audio|publishing)",
    r"graphicaudio",
    r"a\s+movie\s+in\s+your\s+mind",      # GraphicAudio tagline
    r"dreamscape\s+(?:media|select)",
    r"lioncrest\s+publishing",
    r"oasis\s+audio",
    r"zondervan",
    r"bolinda\s+(?:audio|publishing)",
    r"w\.?\s*f\.?\s*howes",
    r"pushkin\s+(?:industries|audio)",
    r"scholastic\s+(?:audio|audiobooks?)",
    r"post\s+hypnotic\s+press",
    r"gildan\s+audio",
    r"gildan\s+media",
    r"w\.?\s*w\.?\s*norton\s+(?:audio|&\s+company)",
    r"your\s+coach\s+in\s+a\s+box",
    r"nightingale[-\s]conant",
    r"devorss\s+(?:publications|&\s+company)",
]

# ── Streaming and Digital Distribution Platforms ──────────────────────────
_PLATFORMS = [
    r"audible(?:\s+studios|\s+originals?)?",
    r"storytel(?:\s+original)?",
    r"spotify(?:\s+audiobooks?)?",
    r"apple\s+books",
    r"apple\s+audiobooks?",
    r"kobo(?:\s+audiobooks?)?",
    r"rakuten\s+kobo",
    r"libro\.fm",
    r"audiobooks?\.com",
    r"findaway(?:\s+voices)?",
    r"chirp(?:\s+books)?",
    r"speechify",
    r"everand",
    r"scribd",
]

# ── Public Domain & Volunteer Platforms ───────────────────────────────────
# LibriVox Whisper phonetic variants observed in live DB:
#   "LibriVox", "Libravox", "Liebervox", "Librefox", "Librevox",
#   "Library Vox", "LibreVox", "Libri-Vox"
_LIBRIVOX_RE = re.compile(
    r"libr[aei]?[-\s]?v[oa]x|libravox|liebervox|librefox|librevox|library\s*vox",
    re.IGNORECASE,
)

_PUBLIC_DOMAIN_PLATFORMS = [
    r"librivox\.org",
    r"gutenberg\.org",
    r"project\s+gutenberg",
    r"internet\s+archive",
    r"archive\.org",
    r"legamus",
    r"lit2go",
    r"loyal\s+books",
    r"publicdomainproject\.org",
]

# ── Podcast / YouTube Spoken Word ─────────────────────────────────────────
_PODCAST_PHRASES = [
    r"welcome\s+to\s+(?:the\s+)?[\w\s]{1,50}podcast",
    r"thanks?\s+for\s+tuning\s+in",
    r"don'?t\s+forget\s+to\s+(?:like|subscribe|follow)",
    r"support\s+us\s+on\s+patreon",
    r"join\s+(?:us|me)\s+(?:on|at)\s+patreon",
    r"subscribe\s+(?:to\s+(?:our|my|this)\s+)?(?:channel|podcast|newsletter)",
    r"check\s+(?:us|me)\s+out\s+on\s+(?:youtube|spotify|apple)",
    r"follow\s+(?:us|me)\s+on\s+(?:instagram|twitter|x\.com|tiktok)",
    r"(?:if\s+you\s+)?(?:like|enjoy(?:ed)?)\s+(?:this|that)\s+(?:video|episode|podcast)",
]

# ---------------------------------------------------------------------------
# Build Compiled Regex Patterns
# ---------------------------------------------------------------------------

# All publisher / platform name fragments joined for use in preamble patterns
_ALL_PRODUCERS = (
    _BIG5_PUBLISHERS
    + _SPECIALIST_PUBLISHERS
    + _PLATFORMS
)
_PRODUCER_FRAGMENT = "(?:" + "|".join(_ALL_PRODUCERS) + ")"

# ── Opening Preamble Patterns ─────────────────────────────────────────────
# Ordered from most specific to least specific.
# All patterns are anchored to the start of text (^).
_PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    # "This is Audible. [Producer] presents [Title] by [Author]."
    re.compile(
        r"^this\s+is\s+audible\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "[Producer] presents [Title][, narrated by X][.]"
    re.compile(
        r"^" + _PRODUCER_FRAGMENT + r"\s+presents?\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # Full LibriVox disclaimer block, consuming through the volunteer/org line.
    # Pattern: "[title info?] LibriVox recording. All ... public domain. For more info ... LibriVox.org."
    # The .*? stops at the first sentence ending after seeing one of the terminal anchors.
    re.compile(
        r"^.{0,250}?(?:" + _LIBRIVOX_RE.pattern + r")\s+recording\b.{0,250}?"
        r"(?:volunteer|librivox\.org|library\s*vox\.org)[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # Simpler LibriVox preamble ending at "public domain."
    re.compile(
        r"^.{0,250}?(?:" + _LIBRIVOX_RE.pattern + r")\s+recording\b.{0,250}?public\s+domain[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "This is a LibriVox recording." alone (no trailing volunteer/org line)
    re.compile(
        r"^this\s+is\s+(?:a\s+)?(?:" + _LIBRIVOX_RE.pattern + r")\s+recording\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # Public domain platform preambles
    re.compile(
        r"^(?:" + "|".join(_PUBLIC_DOMAIN_PLATFORMS) + r")\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "A [Producer] production" / "A [Producer] audiobook"
    re.compile(
        r"^a\s+" + _PRODUCER_FRAGMENT + r"\s+(?:production|audiobook|recording|unabridged)\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "[Producer] hopes you enjoy / enjoyed"
    re.compile(
        r"^" + _PRODUCER_FRAGMENT + r"\s+(?:hopes?|brings?\s+you|is\s+proud)\b[^.]{0,250}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # Podcast openers
    re.compile(
        r"^(?:" + "|".join(_PODCAST_PHRASES[:3]) + r")[^.]{0,200}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Brought to you by [Producer]."
    re.compile(
        r"^brought\s+to\s+you\s+by\b[^.]{0,150}\.\s*",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Narrated by [Name]. / Read by [Name]." as opening line
    re.compile(
        r"^(?:narrated|read|performed)\s+by\s+[\w\s.,'-]{1,80}\.\s*",
        re.IGNORECASE,
    ),
    # "Copyright [Year] [entity]." as the very first line
    re.compile(
        r"^copyright\s+(?:\(c\)\s*)?\d{4}\b[^.]{0,150}\.\s*",
        re.IGNORECASE,
    ),
]

# ── Closing Outro Patterns ────────────────────────────────────────────────
# Ordered from most specific to least specific.
# These are searched anywhere in the last 500 chars of the text.
_OUTRO_PATTERNS: list[re.Pattern[str]] = [
    # "Audible hopes you have enjoyed this program."
    re.compile(
        r"(?:audible|" + _PRODUCER_FRAGMENT + r")\s+hopes?\s+you\s+(?:have\s+)?enjoyed\b.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "We hope you have enjoyed this unabridged production of ..."
    re.compile(
        r"we\s+hope\s+you\s+(?:have\s+)?enjoyed\s+this\s+(?:unabridged\s+)?(?:production|recording|audiobook|performance)\b.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "This concludes [Session/Chapter/Part/Volume] N of ..."
    re.compile(
        r"this\s+concludes?\s+(?:session|chapter|part|volume|book|section)\s+\d+\b.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "End of [Chapter/Part/Book/Session] [N] [of Title]"
    re.compile(
        r"end\s+of\s+(?:chapter|part|book|session|volume|disc|disk|side)\s+[\w\d]+\b.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Recording by [Name] for LibriVox."
    re.compile(
        r"recording\s+by\s+[\w\s.,'-]{1,80}(?:for\s+)?(?:" + _LIBRIVOX_RE.pattern + r")[^.]*\.$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "[LibriVox] recording (outro context): "For more information or to volunteer..."
    re.compile(
        r"for\s+more\s+information(?:\s+or\s+to\s+volunteer)?,?\s+(?:please\s+)?(?:visit|see)\s+"
        r"(?:" + _LIBRIVOX_RE.pattern + r")[\w\s/.:]+\.$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "This has been a [Producer] production."
    re.compile(
        r"this\s+has\s+been\s+a\b[^.]{0,150}" + _PRODUCER_FRAGMENT + r"[^.]*\.$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Thanks for listening." / "Thank you for listening."
    re.compile(
        r"thanks?\s+(?:you\s+)?for\s+listening\b[^.]{0,100}[\.\!]*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Don't forget to like and subscribe..."
    re.compile(
        r"(?:don'?t\s+forget\s+to\s+(?:like|subscribe)|if\s+you\s+enjoyed\s+this\s+(?:video|episode))\b.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Copyright [Year] [entity]. All rights reserved."
    re.compile(
        r"copyright\s+(?:\(c\)\s*)?\d{4}\b[^.]{0,200}all\s+rights\s+reserved[^.]*\.$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "All rights reserved." as a terminal line
    re.compile(
        r"all\s+rights\s+reserved\s*\.?\s*$",
        re.IGNORECASE,
    ),
]

# ── Standalone Pure-Disclaimer Signatures ─────────────────────────────────
# Used by is_pure_boilerplate() as a fast pre-check before regex stripping.
_PURE_DISCLAIMER_FAST_CHECKS = [
    "librivox",
    "liebervox",
    "librefox",
    "libravox",
    "library vox",
    "this is audible",
    "public domain",
    "all rights reserved",
    "sounds true presents",
    "tantor audio",
    "brilliance audio",
    "blackstone audio",
    "recorded books",
    "storytel",
]

# Minimum substantive word count after stripping — below this is pure boilerplate.
_MIN_SUBSTANTIVE_WORDS = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_preamble(text: str) -> str:
    """Strip publisher / platform opening preamble from the start of *text*.

    Applies each compiled preamble pattern once in order and removes the first
    match found.  Normalises leading whitespace after stripping.

    Returns the stripped text, or the original text unchanged if no pattern
    matches.
    """
    for pattern in _PREAMBLE_PATTERNS:
        stripped = pattern.sub("", text, count=1)
        if stripped != text:
            return stripped.lstrip()
    return text


def strip_outro(text: str) -> str:
    """Strip closing credits / production stamps from the end of *text*.

    Only searches the last 600 characters of the text to avoid accidentally
    removing substantive mid-text content.  Returns the stripped text with
    trailing whitespace normalised.
    """
    # Quick short-circuit: if the full text is short, scan all of it.
    tail_start = max(0, len(text) - 600)
    head = text[:tail_start]
    tail = text[tail_start:]

    for pattern in _OUTRO_PATTERNS:
        stripped_tail = pattern.sub("", tail)
        if stripped_tail != tail:
            return (head + stripped_tail).rstrip()

    return text


def clean_audio_disclaimers(text: str) -> str:
    """Run preamble stripping followed by outro stripping.

    This is the primary function to call from ``_clean_text()`` in
    ``chunking.py``.  Safe to call on any text; has no effect if no patterns
    match.
    """
    text = strip_preamble(text)
    text = strip_outro(text)
    return text.strip()


def is_pure_boilerplate(text: str) -> bool:
    """Return True if *text* is substantively empty after disclaimer stripping.

    A chunk is considered pure boilerplate when:
    1. It contains at least one known publisher / platform signature, AND
    2. After stripping all preambles and outros the remaining word count is
       fewer than ``_MIN_SUBSTANTIVE_WORDS``.

    Short text with no signature (e.g. a 3-word test sentence) is never
    flagged — this function only fires on text that contains a known
    disclaimer keyword.

    Does **not** modify the text in the database — purely evaluative.
    """
    text_lower = text.lower()

    # Fast exit: no known disclaimer signature means no boilerplate to detect.
    # This is the most common path for real literary / philosophical content.
    has_signature = any(sig in text_lower for sig in _PURE_DISCLAIMER_FAST_CHECKS)
    if not has_signature:
        return False

    cleaned = clean_audio_disclaimers(text)
    return len(cleaned.split()) < _MIN_SUBSTANTIVE_WORDS

