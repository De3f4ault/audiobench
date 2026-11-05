"""Unit and integration tests for TranscriptMatcher."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
from audiobench.transcribe.matcher import TranscriptMatcher, normalize_stem


@pytest.fixture
def matcher_session(test_db):
    """Set up database records for matcher testing and yield an active session."""
    from audiobench.core.db_session import get_session

    with get_session() as session:
        af1 = AudioFileRecord(
            file_name="Deep Work - Cal Newport.mp3",
            file_path="/audio/Deep Work - Cal Newport.mp3",
            file_hash="abcdef0123456789abcdef0123456789",
            duration_seconds=3600.0,
        )
        af2 = AudioFileRecord(
            file_name="Why Single and Lonely often Makes You a Better person - Dry Creek Dewayne(MP3_160K).mp3",
            file_path="/audio/Why Single.mp3",
            file_hash=None,
            duration_seconds=1200.0,
        )
        af3 = AudioFileRecord(
            file_name="Track 01.mp3",
            file_path="/audio/Track 01.mp3",
            file_hash=None,
            duration_seconds=180.0,
        )

        session.add_all([af1, af2, af3])
        session.flush()

        tx1 = TranscriptionRecord(
            audio_file_id=af1.id,
            file_name=af1.file_name,
            full_text="Deep work is valuable.",
            engine="whisper-tiny",
            duration_seconds=3600.0,
        )
        session.add(tx1)
        session.commit()

    with get_session() as session:
        yield session


def test_normalize_stem():
    assert normalize_stem("Why Single and Lonely often Makes You a Better Person.json") == "why single and lonely often makes you a better person"
    assert normalize_stem("01-Introduction_to_Economics.mp3") == "01 introduction to economics"
    assert normalize_stem("AudioFile.flac.json") == "audiofile"


def test_tier1_hash_match(matcher_session):
    matcher = TranscriptMatcher(matcher_session)

    data = {
        "audio": {
            "file_name": "DifferentName.mp3",
            "file_hash": "ABCDEF0123456789abcdef0123456789",  # Case difference
        },
        "text": "Deep work text",
        "segments": [{"start": 0.0, "end": 10.0, "text": "Deep work text"}],
    }

    res = matcher.match_transcript_data(data, filename="random_name.json")
    assert res is not None
    assert res.match_tier == "hash"
    assert res.confidence == 1.0
    assert res.audio_file_name == "Deep Work - Cal Newport.mp3"
    assert res.existing_transcription_id is not None


def test_tier2_exact_stem_match(matcher_session):
    matcher = TranscriptMatcher(matcher_session)

    # Name matches stem of af1 even if no hash is provided
    data = {
        "audio": {"file_name": "deep_work___cal_newport.mp3"},
        "text": "Some text",
    }

    res = matcher.match_transcript_data(data, filename="deep work - cal newport.json")
    assert res is not None
    assert res.match_tier == "stem"
    assert res.confidence == 0.95
    assert res.audio_file_name == "Deep Work - Cal Newport.mp3"


def test_tier3_guarded_fuzzy_match(matcher_session):
    matcher = TranscriptMatcher(matcher_session)

    # JSON filename has shorter title without artist suffix, but >15 chars
    json_name = "Why Single and Lonely often Makes You a Better Person.json"
    data = {
        "audio": {"file_name": "unknown.mp3"},
        "text": "Solitude reflection",
    }

    res = matcher.match_transcript_data(data, filename=json_name)
    assert res is not None
    assert res.match_tier == "fuzzy"
    assert res.confidence == 0.85
    assert "Dry Creek Dewayne" in res.audio_file_name


def test_tier3_guard_prevents_short_false_positive(matcher_session):
    matcher = TranscriptMatcher(matcher_session)

    # "Track 01" is <= 15 chars. "Track 010.json" should NOT match via fuzzy substring!
    data = {
        "audio": {"file_name": "Track 010.mp3"},
        "text": "Some other track",
    }

    res = matcher.match_transcript_data(data, filename="Track 010.json")
    assert res is None


def test_match_file(matcher_session, tmp_path):
    matcher = TranscriptMatcher(matcher_session)

    json_file = tmp_path / "deep-work---cal-newport.json"
    json_file.write_text(
        json.dumps({
            "schema_version": "1",
            "audio": {"file_hash": "abcdef0123456789abcdef0123456789"},
            "text": "Deep work text",
            "segments": [],
        }),
        encoding="utf-8",
    )

    res = matcher.match_file(json_file)
    assert res is not None
    assert res.match_tier == "hash"
    assert res.audio_file_name == "Deep Work - Cal Newport.mp3"

