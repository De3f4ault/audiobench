"""Canonical Transcript Matcher for AudioBench.

Provides a unified 3-tier matching cascade across external transcript files
and the local database:
  - Tier 1: SHA-256 audio file hash match against audio_files.file_hash
  - Tier 2: Normalized exact filename / stem match
  - Tier 3: Guarded fuzzy substring match (>15 chars)

Used uniformly by `audiobench library upgrade`, `audiobench import`,
and TUI reimport to prevent matching discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class MatchResult:
    """Result of matching an external transcript to a library audio file and transcription."""

    audio_file_id: int
    audio_file_name: str
    audio_file_path: str
    audio_duration: float | None
    existing_transcription_id: int | None
    existing_engine: str | None
    existing_model_name: str | None
    existing_full_text: str | None
    match_tier: str  # "hash" | "stem" | "fuzzy"
    confidence: float  # 1.0 = hash, 0.95 = stem, 0.85 = fuzzy
    matched_key: str


def normalize_stem(name: str) -> str:
    """Normalize a filename or stem for canonical comparison.

    Lowercases, strips audio/json extensions, strips non-alphanumeric chars,
    and collapses whitespace.
    """
    stem = Path(name).stem
    # Strip common secondary extensions if any
    if stem.lower().endswith((".mp3", ".m4a", ".wav", ".flac", ".ogg", ".json")):
        stem = Path(stem).stem
    s = re.sub(r"[^a-z0-9\s]", " ", stem.lower())
    return re.sub(r"\s+", " ", s).strip()


class TranscriptMatcher:
    """Canonical matcher for pairing external transcript files with DB audio files."""

    def __init__(self, session: Session) -> None:
        """Pre-index existing database records into fast lookups."""
        self._by_hash: dict[str, dict[str, Any]] = {}
        self._by_name: dict[str, dict[str, Any]] = {}
        self._by_stem: dict[str, dict[str, Any]] = {}
        self._long_stems: list[tuple[str, dict[str, Any]]] = []

        self._load_records(session)

    def _load_records(self, session: Session) -> None:
        from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

        # Load all audio files with their latest transcription
        audio_records = session.query(AudioFileRecord).all()
        tx_records = session.query(TranscriptionRecord).all()

        # Map audio_file_id -> latest transcription record
        latest_tx_by_audio: dict[int, TranscriptionRecord] = {}
        for tx in tx_records:
            if tx.audio_file_id:
                prev = latest_tx_by_audio.get(tx.audio_file_id)
                if not prev or (tx.id and prev.id and tx.id > prev.id):
                    latest_tx_by_audio[tx.audio_file_id] = tx

        for af in audio_records:
            tx = latest_tx_by_audio.get(af.id)
            entry = {
                "audio_file_id": af.id,
                "audio_file_name": af.file_name,
                "audio_file_path": af.file_path,
                "audio_duration": af.duration_seconds,
                "existing_transcription_id": tx.id if tx else None,
                "existing_engine": tx.engine if tx else None,
                "existing_model_name": tx.model_name if tx else None,
                "existing_full_text": tx.full_text if tx else None,
            }

            if af.file_hash:
                h = af.file_hash.lower().strip()
                self._by_hash[h] = entry

            if af.file_name:
                raw_name = af.file_name.lower().strip()
                self._by_name[raw_name] = entry

                norm_stem = normalize_stem(af.file_name)
                if norm_stem:
                    self._by_stem[norm_stem] = entry
                    if len(norm_stem) > 15:
                        self._long_stems.append((norm_stem, entry))

    def match_transcript_data(
        self,
        data: dict[str, Any],
        filename: str | Path | None = None,
    ) -> MatchResult | None:
        """Match parsed transcript JSON data against library audio files.

        Uses the canonical 3-tier cascade:
          Tier 1: Audio file SHA-256 hash
          Tier 2: Normalized exact stem / name
          Tier 3: Guarded fuzzy substring (>15 chars)
        """
        # --- Tier 1: Hash match ---
        audio_meta = data.get("audio") or data.get("raw_audio") or {}
        raw_hash = audio_meta.get("file_hash") or data.get("file_hash")
        if raw_hash and isinstance(raw_hash, str):
            clean_hash = raw_hash.lower().strip()
            if clean_hash in self._by_hash:
                entry = self._by_hash[clean_hash]
                return MatchResult(
                    audio_file_id=entry["audio_file_id"],
                    audio_file_name=entry["audio_file_name"],
                    audio_file_path=entry["audio_file_path"],
                    audio_duration=entry["audio_duration"],
                    existing_transcription_id=entry["existing_transcription_id"],
                    existing_engine=entry["existing_engine"],
                    existing_model_name=entry["existing_model_name"],
                    existing_full_text=entry["existing_full_text"],
                    match_tier="hash",
                    confidence=1.0,
                    matched_key=clean_hash,
                )

        # --- Tier 2: Normalized Stem / Exact Name ---
        candidate_names: list[str] = []
        if filename:
            candidate_names.append(str(filename))
        audio_name = audio_meta.get("file_name")
        if audio_name and isinstance(audio_name, str):
            candidate_names.append(audio_name)

        # 2a. Direct raw filename check
        for name in candidate_names:
            clean_name = Path(name).name.lower().strip()
            if clean_name in self._by_name:
                entry = self._by_name[clean_name]
                return MatchResult(
                    audio_file_id=entry["audio_file_id"],
                    audio_file_name=entry["audio_file_name"],
                    audio_file_path=entry["audio_file_path"],
                    audio_duration=entry["audio_duration"],
                    existing_transcription_id=entry["existing_transcription_id"],
                    existing_engine=entry["existing_engine"],
                    existing_model_name=entry["existing_model_name"],
                    existing_full_text=entry["existing_full_text"],
                    match_tier="stem",
                    confidence=0.95,
                    matched_key=clean_name,
                )

        # 2b. Normalized stem check
        norm_stems: list[str] = []
        for name in candidate_names:
            ns = normalize_stem(name)
            if ns:
                norm_stems.append(ns)
                if ns in self._by_stem:
                    entry = self._by_stem[ns]
                    return MatchResult(
                        audio_file_id=entry["audio_file_id"],
                        audio_file_name=entry["audio_file_name"],
                        audio_file_path=entry["audio_file_path"],
                        audio_duration=entry["audio_duration"],
                        existing_transcription_id=entry["existing_transcription_id"],
                        existing_engine=entry["existing_engine"],
                        existing_model_name=entry["existing_model_name"],
                        existing_full_text=entry["existing_full_text"],
                        match_tier="stem",
                        confidence=0.95,
                        matched_key=ns,
                    )

        # --- Tier 3: Guarded fuzzy substring (>15 chars) ---
        for ns in norm_stems:
            if len(ns) <= 15:
                continue
            for db_stem, entry in self._long_stems:
                if len(db_stem) > 15 and (db_stem in ns or ns in db_stem):
                    return MatchResult(
                        audio_file_id=entry["audio_file_id"],
                        audio_file_name=entry["audio_file_name"],
                        audio_file_path=entry["audio_file_path"],
                        audio_duration=entry["audio_duration"],
                        existing_transcription_id=entry["existing_transcription_id"],
                        existing_engine=entry["existing_engine"],
                        existing_model_name=entry["existing_model_name"],
                        existing_full_text=entry["existing_full_text"],
                        match_tier="fuzzy",
                        confidence=0.85,
                        matched_key=f"{ns} <=> {db_stem}",
                    )

        return None

    def match_file(self, file_path: Path | str) -> MatchResult | None:
        """Read a JSON transcript file from disk and match it."""
        path = Path(file_path)
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        return self.match_transcript_data(data, filename=path.name)
