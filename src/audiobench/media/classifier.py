"""Media classifier and privacy firewall.

Asymmetric-loss media classification:
- Class A (Hard Proof): Autonomous biometric bypass (Tier 0 / Public).
  Triggered ONLY by explicit operator intent, verified legal spoken boilerplate
  in the transcript opening, dedicated .m4b container with substantial duration,
  or verified directory provenance corroborated by sibling book artifacts / multi-track packaging.
- Class B (Soft Heuristic): Ambiguous likely-audiobook (author tags, bare path).
  In interactive mode: prompts operator for confirmation.
  In headless/background mode: quarantined under `pending_privacy_review` without
  flooding the ECAPA-TDNN biometric worker queue.
- Class C (Generic Audio): Standard recording, personal memo, or dialogue.
  Proceeds to standard biometric evaluation so personal voice is securely tagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal

from audiobench.core.logger_factory import get_logger

logger = get_logger("media.classifier")

ClassificationCategory = Literal[
    "class_a_hard_proof",
    "class_b_soft_heuristic",
    "class_c_generic_audio",
]

# Audio file extensions
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".m4b", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}

# Companion book and metadata artifacts
BOOK_ARTIFACT_EXTENSIONS = {".epub", ".pdf", ".mobi", ".azw3", ".azw", ".cue", ".opf", ".nfo"}

# Audiobook path keywords
AUDIOBOOK_PATH_KEYWORDS = {
    "audiobook",
    "audiobooks",
    "audio_book",
    "audio_books",
    "audio-book",
    "audio-books",
    "audible",
    "calibre",
    "librivox",
}

# Numbered track pattern for multi-part audiobooks (e.g., "01 - Chapter 1", "Track 02", "Part 1")
_TRACK_PREFIX_RE = re.compile(
    r"^(?:(?:disc|cd|part|track|ch|chapter|lesson|l)\s*[-_.]?\s*)?\d{1,3}(?:\s*[-_.]|\s+)",
    re.IGNORECASE,
)

# LibriVox Whisper phonetic variants
_LIBRIVOX_RE = re.compile(
    r"libr[aei]?[-\s]?v[oa]x|libravox|liebervox|librefox|librevox|library\s*vox",
    re.IGNORECASE,
)

# Great Courses / The Teaching Company specific spoken opening phrases
_TGC_RE = re.compile(
    r"(?:the\s+great\s+courses|the\s+teaching\s+company|professor\s+[\w\s]{2,40}\blectures?)",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    """Result of media classification."""

    category: ClassificationCategory
    confidence: float
    evidence: list[str] = field(default_factory=list)
    is_autonomous_bypass: bool = False
    detected_title: str | None = None
    detected_author: str | None = None
    review_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "is_autonomous_bypass": self.is_autonomous_bypass,
            "detected_title": self.detected_title,
            "detected_author": self.detected_author,
            "review_reason": self.review_reason,
        }


def check_transcript_boilerplate(
    transcript_text: str | None = None,
    segments: list[Any] | None = None,
) -> tuple[bool, str | None]:
    """Check if the opening of the transcript contains verified legal publisher boilerplate.

    Evaluates the first 120 seconds of audio or the first 300 words.
    Returns (True, reason) on match, or (False, None).
    """
    sample_text = ""
    if segments:
        early_texts = []
        for seg in segments:
            # Check segments starting within the first 120 seconds
            start_time = getattr(seg, "start", None)
            if start_time is None:
                start_time = getattr(seg, "start_time", 0.0)
            if float(start_time) <= 120.0:
                text = getattr(seg, "text", "")
                if text:
                    early_texts.append(text.strip())
        sample_text = " ".join(early_texts)

    if not sample_text and transcript_text:
        words = transcript_text.strip().split()
        sample_text = " ".join(words[:300])

    if not sample_text:
        return False, None

    lower_sample = sample_text.lower()
    if "this is audible" in lower_sample:
        return True, "boilerplate:audible"
    if "librivox" in lower_sample or _LIBRIVOX_RE.search(sample_text):
        return True, "boilerplate:librivox"
    if "penguin random house audio" in lower_sample or "random house audio presents" in lower_sample:
        return True, "boilerplate:penguin_random_house"
    if "harperaudio presents" in lower_sample or "harper audio" in lower_sample:
        return True, "boilerplate:harperaudio"
    if "simon & schuster audio" in lower_sample:
        return True, "boilerplate:simon_and_schuster"
    if "recorded books presents" in lower_sample or "recorded books, inc." in lower_sample:
        return True, "boilerplate:recorded_books"
    if "blackstone audio" in lower_sample:
        return True, "boilerplate:blackstone"
    if "tantor audio presents" in lower_sample or "tantor media" in lower_sample:
        return True, "boilerplate:tantor"
    if _TGC_RE.search(sample_text):
        return True, "boilerplate:the_great_courses"

    # Check Audiobench boilerplate preambles as generic fallback
    try:
        from audiobench.memory.boilerplate import _PREAMBLE_PATTERNS

        for pat in _PREAMBLE_PATTERNS:
            if pat.search(sample_text):
                return True, f"preamble_pattern:{pat.pattern[:40]}"
    except ImportError:
        pass

    return False, None


def check_container(
    file_path: Path | str | None,
    duration_seconds: float = 0.0,
) -> tuple[bool, str | None]:
    """Verify if the container is dedicated .m4b format with substantial duration (>= 600s)."""
    if not file_path:
        return False, None
    p = Path(file_path)
    if p.suffix.lower() == ".m4b":
        # .m4b audiobooks are overwhelmingly > 10 minutes.
        # Short test files (<10m) fall through so they aren't blindly exempted.
        if duration_seconds >= 600.0 or duration_seconds == 0.0:
            return True, "container:m4b_audiobook"
    return False, None


def check_provenance_and_artifacts(
    file_path: Path | str | None,
) -> tuple[bool, str | None, bool]:
    """Check directory provenance and companion packaging.

    Returns:
        (is_class_a, reason, has_bare_path_match)
        - is_class_a: True only if path matches AND (sibling book artifacts exist OR multi-track chapter sequence >= 5 tracks).
        - has_bare_path_match: True if path matches keywords even if sibling artifacts are absent (Class B candidate).
    """
    if not file_path:
        return False, None, False

    p = Path(file_path).resolve()
    # Check directory parts against audiobook keywords
    dir_parts = [part.lower() for part in p.parent.parts]
    path_matches = any(kw in part for part in dir_parts for kw in AUDIOBOOK_PATH_KEYWORDS)

    if not path_matches:
        return False, None, False

    parent_dir = p.parent
    check_dirs = [parent_dir]
    # If the parent is a CD/Disc/Volume directory, also check grandparent
    if any(parent_dir.name.lower().startswith(x) for x in ("cd", "disc", "vol", "part")):
        check_dirs.append(parent_dir.parent)

    # 1. Check for companion book artifacts
    for d in check_dirs:
        try:
            for item in d.iterdir():
                if item.is_file() and item.suffix.lower() in BOOK_ARTIFACT_EXTENSIONS:
                    return True, f"provenance:audiobook_dir_with_companion_artifact({item.suffix})", True
        except (PermissionError, OSError):
            continue

    # 2. Check for sequenced multi-track chapter album (>= 5 tracks with track numbering)
    try:
        audio_files = [f for f in parent_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        track_prefixed_files = [f for f in audio_files if _TRACK_PREFIX_RE.match(f.name)]
        if len(audio_files) >= 5 and len(track_prefixed_files) >= (len(audio_files) * 0.7):
            return True, f"provenance:multi_track_chapter_album({len(track_prefixed_files)}_tracks)", True
    except (PermissionError, OSError):
        pass

    # Bare path matched, but lacked verified companion artifacts or multi-track sequence
    return False, None, True


def check_soft_heuristics(
    file_path: Path | str | None = None,
    metadata: dict | None = None,
    duration_seconds: float = 0.0,
    has_bare_path: bool = False,
) -> tuple[bool, str | None, str | None, str | None]:
    """Evaluate soft heuristics (author tag, genre, work identity, bare path).

    Returns:
        (is_class_b, reason, detected_title, detected_author)
    """
    meta = metadata or {}
    tags = meta.get("tags") or {}
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = {}

    genre = str(meta.get("genre") or tags.get("genre") or "").strip().lower()
    artist = str(meta.get("artist") or tags.get("artist") or "").strip()
    album = str(meta.get("album") or tags.get("album") or "").strip()
    title = str(meta.get("track_title") or meta.get("title") or tags.get("title") or "").strip()

    reasons: list[str] = []

    if has_bare_path:
        reasons.append("bare_audiobook_folder_without_artifacts")

    if genre in {"audiobook", "audiobooks", "audio book", "audio books", "spoken word"}:
        reasons.append(f"genre_tag:{genre}")

    if artist and album and duration_seconds >= 900.0:
        reasons.append(f"album_and_artist_metadata({artist} - {album})")

    # Author separator in filename (e.g., "Author - Title" with long duration)
    if file_path:
        stem = Path(file_path).stem
        if " - " in stem and duration_seconds >= 900.0:
            parts = stem.split(" - ", 1)
            if len(parts) == 2 and len(parts[0]) > 2 and len(parts[1]) > 2:
                reasons.append("stem_author_title_split")
                if not artist:
                    artist = parts[0].strip()
                if not title:
                    title = parts[1].strip()

    if reasons:
        return True, ", ".join(reasons), title or album or None, artist or None

    return False, None, None, None


def classify_media(
    file_path: Path | str | None = None,
    transcript: Any | None = None,
    transcript_text: str | None = None,
    segments: list[Any] | None = None,
    metadata: dict | None = None,
    duration_seconds: float = 0.0,
    explicit_flag: bool = False,
) -> ClassificationResult:
    """Classify audio file into Class A, Class B, or Class C.

    Classification order:
    1. Explicit operator flag or persisted user confirmation -> Class A
    2. Spoken legal publisher boilerplate -> Class A
    3. Dedicated .m4b container with duration >= 10m -> Class A
    4. Audiobook directory + companion book artifacts / multi-track sequence -> Class A
    5. Soft heuristics (author tags, bare path, genre, duration >= 15m) -> Class B
    6. Generic audio / Personal memo -> Class C
    """
    evidence: list[str] = []
    meta = metadata or {}
    tags = meta.get("tags") or {}
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = {}

    dur = duration_seconds
    if dur <= 0.0 and transcript:
        dur = getattr(transcript, "duration_seconds", 0.0)

    # ── Gate 0: Explicit Operator Flag or Persisted Confirmation ──────────
    if explicit_flag:
        return ClassificationResult(
            category="class_a_hard_proof",
            confidence=1.0,
            evidence=["explicit_operator_flag"],
            is_autonomous_bypass=True,
            detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
            detected_author=meta.get("artist"),
        )

    if tags.get("audiobook") is True and tags.get("user_confirmed") is True:
        return ClassificationResult(
            category="class_a_hard_proof",
            confidence=1.0,
            evidence=["persisted_user_confirmation"],
            is_autonomous_bypass=True,
            detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
            detected_author=meta.get("artist"),
        )

    # ── Gate 1: Spoken Legal Boilerplate Detector ─────────────────────────
    text_to_check = transcript_text
    segs_to_check = segments
    if transcript:
        if not text_to_check:
            text_to_check = getattr(transcript, "text", "")
        if not segs_to_check:
            segs_to_check = getattr(transcript, "segments", [])

    is_bp, bp_reason = check_transcript_boilerplate(text_to_check, segs_to_check)
    if is_bp and bp_reason:
        evidence.append(bp_reason)
        return ClassificationResult(
            category="class_a_hard_proof",
            confidence=0.99,
            evidence=evidence,
            is_autonomous_bypass=True,
            detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
            detected_author=meta.get("artist"),
        )

    # ── Gate 2: Dedicated .m4b Container Format ───────────────────────────
    is_m4b, m4b_reason = check_container(file_path, dur)
    if is_m4b and m4b_reason:
        evidence.append(m4b_reason)
        return ClassificationResult(
            category="class_a_hard_proof",
            confidence=0.98,
            evidence=evidence,
            is_autonomous_bypass=True,
            detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
            detected_author=meta.get("artist"),
        )

    # ── Gate 3: Directory Provenance + Verified Sibling Artifacts ─────────
    is_prov_a, prov_reason, has_bare_path = check_provenance_and_artifacts(file_path)
    if is_prov_a and prov_reason:
        evidence.append(prov_reason)
        return ClassificationResult(
            category="class_a_hard_proof",
            confidence=0.95,
            evidence=evidence,
            is_autonomous_bypass=True,
            detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
            detected_author=meta.get("artist"),
        )

    # ── Gate 4: Soft Heuristics (Class B) ──────────────────────────────────
    is_soft_b, soft_reason, det_title, det_author = check_soft_heuristics(
        file_path, meta, dur, has_bare_path=has_bare_path
    )
    if is_soft_b and soft_reason:
        evidence.append(soft_reason)
        return ClassificationResult(
            category="class_b_soft_heuristic",
            confidence=0.70,
            evidence=evidence,
            is_autonomous_bypass=False,
            detected_title=det_title or (Path(file_path).stem if file_path else None),
            detected_author=det_author,
            review_reason=soft_reason,
        )

    # ── Default: Generic Audio (Class C) ──────────────────────────────────
    return ClassificationResult(
        category="class_c_generic_audio",
        confidence=0.50,
        evidence=["generic_audio_no_audiobook_signatures"],
        is_autonomous_bypass=False,
        detected_title=meta.get("title") or (Path(file_path).stem if file_path else None),
        detected_author=meta.get("artist"),
    )
