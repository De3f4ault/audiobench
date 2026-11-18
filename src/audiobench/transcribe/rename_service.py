"""Service for generating semantic titles and auto-renaming audio files.

Priority pipeline used by generate_and_apply_title():
    1. Embedded tags   — ffprobe format_tags (title/artist from ID3, MP4, etc.)
    2. CUE disc title  — top-level TITLE line in a .cue sidecar
    3. DB chapter data — real chapter titles already stored on AudioFileRecord
    4. LLM fallback    — Ollama → Gemini, with duration-aware text sampling:
                          • ≤ 30 min  : first 15 min (unchanged behaviour)
                          • > 30 min  : sparse sample (start + middle + end)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

logger = logging.getLogger("audiobench.rename_service")

# ─── Junk-name patterns ────────────────────────────────────────────────────────
# If the current filename already looks like a meaningful human title we skip
# auto-renaming entirely to avoid trashing a well-named file.
_JUNK_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^audiobench[\s_-]",          # our own generated placeholders
        r"^new\s+recording",           # iOS / Android default names
        r"^voice\s*\d+",               # Voice Memos defaults
        r"^record\s*\d+",              # generic recorder defaults
        r"^\d{4}[-_]\d{2}[-_]\d{2}",  # bare timestamp files
        r"^untitled",
        r"^audio\s*\d+",
        r"^[0-9a-f]{8,}_",                          # Hex hashes at start (e.g. 84b29a0b_)
        r"-[a-zA-Z0-9_-]{11}$",                     # YouTube IDs at end
        r"\(?(?:HD\s*)?MP(?:3|4)_[0-9]+K\)?",       # Quality markers (e.g. MP3_160K, HD MP4_128K)
        r"\b\d{2,3}\s*kbps\b",                      # Bitrates (e.g. 128kbps, 320 kbps)
        r"y2mate\.com|yt1s\.com|yt5s\.com",         # Web rip sites
    ]
]

# If a tag-sourced title looks like this it's not useful either
_JUNK_TITLE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^track\s*\d+$",
        r"^untitled$",
        r"^unknown$",
        r"^\s*$",
        r"^[0-9a-f]{8,}_",                          # Hex hashes at start (e.g. 84b29a0b_)
        r"-[a-zA-Z0-9_-]{11}$",                     # YouTube IDs at end
        r"\(?(?:HD\s*)?MP(?:3|4)_[0-9]+K\)?",       # Quality markers (e.g. MP3_160K, HD MP4_128K)
        r"\b\d{2,3}\s*kbps\b",                      # Bitrates (e.g. 128kbps, 320 kbps)
        r"y2mate\.com|yt1s\.com|yt5s\.com",         # Web rip sites
        r"\[?(?:official|lyric)\s+(?:video|audio)\]?", # Official Video/Audio tags
    ]
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _looks_like_junk_filename(name: str) -> bool:
    """Return True if *name* (without extension) matches a junk naming pattern."""
    stem = Path(name).stem
    return any(p.search(stem) for p in _JUNK_PATTERNS)


def _looks_like_junk_title(title: str) -> bool:
    """Return True if *title* from an embedded tag / CUE is not useful."""
    return any(p.search(title.strip()) for p in _JUNK_TITLE_PATTERNS)


def _sanitise_for_fs(title: str) -> str:
    """Strip characters that are illegal in filenames and normalise whitespace."""
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _build_sparse_sample(segments: list, duration_seconds: float) -> str:
    """Return a representative text excerpt from a long recording.

    Grabs ~2 minutes from the start, ~2 minutes from the middle, and ~2 minutes
    from the end — enough for the LLM to understand the overall arc without
    being fooled by a publisher intro or credits.
    """
    def _window(start_s: float, end_s: float) -> str:
        window_segs = [s for s in segments if start_s <= s.start_time < end_s]
        return " ".join(s.text for s in window_segs[:60]).strip()

    mid = duration_seconds / 2
    start_text  = _window(0,            120)
    middle_text = _window(mid - 60,     mid + 60)
    end_text    = _window(duration_seconds - 120, duration_seconds)

    parts = []
    if start_text:
        parts.append(f"[Beginning]\n{start_text}")
    if middle_text:
        parts.append(f"[Middle]\n{middle_text}")
    if end_text:
        parts.append(f"[End]\n{end_text}")

    return "\n\n".join(parts)


def _call_llm(prompt: str, settings, force_gemini: bool = False) -> str:
    """Try Ollama first, fall back to Gemini via ProviderRouter. Returns raw title or ''."""
    from audiobench.chat.providers.router import router_for_settings

    router = router_for_settings(settings)
    if settings.clean_model:
        router.set_primary_model(settings.clean_model)
    if force_gemini:
        router._slots.sort(key=lambda s: 0 if s.name == "gemini" else 1)

    try:
        return router.generate_text(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            digest_mode="silent",
        ).strip()
    except Exception as e:
        logger.warning("LLM call failed in rename service: %s", e)
        return ""


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_and_apply_title(tx_id: int, force_gemini: bool = False, force_disk: bool = False) -> tuple[bool, str]:
    """Generate a semantic title for a transcription and rename the source file.

    Resolution order
    ----------------
    1. **Embedded tags** — ``ffprobe`` format_tags (``title`` / ``album``).
       If the file already carries a proper human-readable title we use it
       immediately and never touch the LLM.
    2. **CUE disc title** — top-level ``TITLE`` from a ``.cue`` sidecar file.
    3. **DB chapter titles** — if real (non-ghost, non-"Untitled") chapter titles
       are already stored, pick the most common / first meaningful one.
    4. **LLM fallback** — for raw voice memos with no metadata:
       - ≤ 30 min: analyse the first 15 min (existing behaviour)
       - > 30 min: sparse sample (start + middle + end excerpt)

    The file is only physically renamed on disk when its current name *looks
    like* an auto-generated junk name (e.g. "Audiobench ideas 01", "New
    Recording 5", etc.).  Files with already-meaningful names are left alone
    on the filesystem; only the DB ``file_name`` / ``file_path`` entries and the
    ``pending_auto_rename`` tag are updated.

    Args:
        tx_id: The ID of the transcription record.
        force_gemini: Skip Ollama and go straight to Gemini.

    Returns:
        tuple[bool, str]: (success, new_filename_or_error_message)
    """
    settings = get_settings()

    # ── Load records ──────────────────────────────────────────────────────────
    with get_session() as session:
        tx_record = session.query(TranscriptionRecord).get(tx_id)
        if not tx_record:
            return False, f"Transcription #{tx_id} not found."

        audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
        if not audio_record:
            return False, f"No audio file linked to transcription #{tx_id}."

        old_path = Path(audio_record.file_path)
        if not old_path.exists():
            return False, f"Original file missing on disk: {old_path}"

        duration_seconds: float = audio_record.duration_seconds or 0.0

        # Snapshot chapter titles while session is open
        real_chapter_titles: list[str] = [
            c.title for c in audio_record.chapters
            if not c.is_ghost and c.title and c.title not in ("Untitled", "Full Recording")
        ]

        # Snapshot segments while session is open
        segments = sorted(tx_record.segments, key=lambda s: s.segment_index)

    new_title: str = ""
    source: str = "unknown"

    # ── 1. Embedded format tags (ffprobe) ─────────────────────────────────────
    try:
        from audiobench.transcribe.audio_converter import probe_tags
        tags = probe_tags(old_path)
        candidate = tags.get("title") or tags.get("album") or ""
        if candidate and not _looks_like_junk_title(candidate):
            # Optionally prepend artist for audiobooks: "No Excuses — Brian Tracy"
            artist = tags.get("artist") or tags.get("album_artist") or ""
            if artist and artist.lower() not in candidate.lower():
                new_title = f"{candidate} — {artist}"
            else:
                new_title = candidate
            source = "embedded_tags"
            logger.info("Using embedded tag title for #%d: %s", tx_id, new_title)
    except Exception as e:
        logger.debug("probe_tags step skipped for #%d: %s", tx_id, e)

    # ── 2. CUE disc title ─────────────────────────────────────────────────────
    if not new_title:
        try:
            from audiobench.chapters.cue_parser import CueParser
            cue_parser = CueParser()
            for cue_path in (
                old_path.with_suffix(".cue"),
                old_path.with_name(old_path.name + ".cue"),
            ):
                disc_title = cue_parser.parse_disc_title(cue_path)
                if disc_title and not _looks_like_junk_title(disc_title):
                    new_title = disc_title
                    source = "cue_disc_title"
                    logger.info("Using CUE disc title for #%d: %s", tx_id, new_title)
                    break
        except Exception as e:
            logger.debug("CUE disc title step skipped for #%d: %s", tx_id, e)

    # ── 3. DB chapter titles ──────────────────────────────────────────────────
    if not new_title and real_chapter_titles:
        # Use the first meaningful chapter title as a proxy for the file title.
        # For a 26-chapter audiobook this is usually the book title or Part 1
        # heading — still far better than asking the LLM about the Audible intro.
        new_title = real_chapter_titles[0]
        source = "chapter_titles"
        logger.info(
            "Using first DB chapter title for #%d: %s (total chapters: %d)",
            tx_id, new_title, len(real_chapter_titles),
        )

    # ── 4. LLM fallback ───────────────────────────────────────────────────────
    if not new_title:
        # Build context text — sparse for long files, linear for short ones
        long_form = duration_seconds > 1800  # > 30 minutes

        if long_form and segments:
            text_context = _build_sparse_sample(segments, duration_seconds)
            prompt_intro = (
                f"You are an expert audio librarian. The following are excerpts "
                f"sampled from the beginning, middle, and end of a {duration_seconds / 60:.0f}-minute "
                f"audio recording (not necessarily contiguous).\n"
                f"Generate a comprehensive, highly semantic title (4–8 words) that captures "
                f"the overarching topic across the whole recording — not just the intro.\n"
                f"Do NOT use quotes, special characters, or prefixes like 'Title:'. "
                f"Output ONLY the raw title text.\n\n"
            )
        else:
            intro_segments = [s for s in segments if s.end_time <= 900][:150]
            text_context = " ".join(s.text for s in intro_segments).strip()
            prompt_intro = (
                "You are an expert audio librarian and archivist. "
                "Analyze the following transcript excerpt (the first 15 minutes of the audio) "
                "to understand the core topic, nuances, and main subject being discussed.\n"
                "Generate a comprehensive, highly semantic, and descriptive title for this audio file. "
                "The title should be 4 to 8 words long and capture the true essence of the content.\n"
                "Do NOT use quotes, special characters, or prefixes like 'Title:'. "
                "Output ONLY the raw title text.\n\n"
            )

        if not text_context:
            _add_pending_tag(tx_id)
            return False, "Not enough text to generate a title."

        prompt = prompt_intro + text_context
        new_title = _call_llm(prompt, settings, force_gemini=force_gemini)
        source = "llm"

        if not new_title:
            _add_pending_tag(tx_id)
            return False, "Failed to connect to LLM APIs. Tagged with pending_auto_rename."

    # ── Sanitise the resolved title for filesystem use ────────────────────────
    new_title = _sanitise_for_fs(new_title)

    if not new_title or len(new_title.split()) > 15:
        return False, f"Resolved title is invalid: '{new_title}' (source={source})"

    new_filename = f"{new_title}{old_path.suffix}"
    new_path = old_path.parent / new_filename

    # Handle collisions
    counter = 1
    while new_path.exists() and new_path != old_path:
        new_path = old_path.parent / f"{new_title}_{counter}{old_path.suffix}"
        new_filename = new_path.name
        counter += 1

    # ── Physical rename (only for junk-named files) ───────────────────────────
    should_rename_on_disk = force_disk or _looks_like_junk_filename(old_path.name)

    if should_rename_on_disk and new_path != old_path:
        try:
            old_path.rename(new_path)
            logger.info(
                "Renamed file on disk: %s → %s (source=%s)", old_path.name, new_filename, source
            )
        except OSError as e:
            return False, f"Filesystem rename failed: {e}"
    else:
        # Keep the existing path — only update display metadata in DB
        new_path = old_path
        new_filename = old_path.name
        logger.info(
            "Kept existing filename '%s'; updated DB display title to '%s' (source=%s)",
            old_path.name, new_title, source,
        )

    # ── Update DB ─────────────────────────────────────────────────────────────
    with get_session() as session:
        tx_record = session.query(TranscriptionRecord).get(tx_id)
        if tx_record:
            tx_record.file_name = new_filename

            if tx_record.audio_file_id:
                audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
                if audio_record:
                    audio_record.file_name = new_filename
                    audio_record.file_path = str(new_path)
                    _remove_pending_tag(audio_record)

        session.commit()

    return True, new_filename


# ─── Tag helpers ──────────────────────────────────────────────────────────────

def _add_pending_tag(tx_id: int) -> None:
    """Add the pending_auto_rename tag to the parent AudioFileRecord."""
    with get_session() as session:
        tx_record = session.get(TranscriptionRecord, tx_id)
        if tx_record and tx_record.audio_file_id:
            audio_record = session.get(AudioFileRecord, tx_record.audio_file_id)
            if audio_record:
                try:
                    data = json.loads(audio_record.tags) if audio_record.tags else []
                except Exception:
                    data = []
                if isinstance(data, dict):
                    labels = data.setdefault("_labels", [])
                    if "pending_auto_rename" not in labels:
                        labels.append("pending_auto_rename")
                    audio_record.tags = json.dumps(data)
                else:
                    tags_list = data if isinstance(data, list) else []
                    if "pending_auto_rename" not in tags_list:
                        tags_list.append("pending_auto_rename")
                    audio_record.tags = json.dumps(tags_list)
        session.commit()


def _remove_pending_tag(audio_record: AudioFileRecord) -> None:
    """Remove the pending_auto_rename tag from an AudioFileRecord in-memory."""
    try:
        data = json.loads(audio_record.tags) if audio_record.tags else []
    except Exception:
        data = []
    if isinstance(data, dict):
        labels = data.get("_labels", [])
        if "pending_auto_rename" in labels:
            labels.remove("pending_auto_rename")
            audio_record.tags = json.dumps(data)
    elif isinstance(data, list):
        if "pending_auto_rename" in data:
            data.remove("pending_auto_rename")
            audio_record.tags = json.dumps(data)


# ─── Background spawn ─────────────────────────────────────────────────────────

def spawn_auto_naming(tx_id: int) -> None:
    """Spawn a background task to generate a title and rename the file."""
    from audiobench.transcribe.transcriber import _POST_PROCESS_POOL

    def _run() -> None:
        try:
            success, msg = generate_and_apply_title(tx_id)
            if not success:
                logger.warning("Background auto-rename failed for #%d: %s", tx_id, msg)
        except Exception as e:
            logger.warning("Background auto-rename thread crashed for #%d: %s", tx_id, e)

    _POST_PROCESS_POOL.submit(_run)
    logger.info("Submitted background auto-rename task for #%d", tx_id)
