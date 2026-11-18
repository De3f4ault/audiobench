"""Unit tests for transcript_resolver module."""

import pytest
from rich.console import Console

from audiobench.cli.shared.transcript_resolver import (
    list_recent_transcripts,
    resolve_transcript_target,
)
from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord
from audiobench.core.db_session import get_session


@pytest.fixture
def seed_transcriptions(test_db):
    """Seed test database with known transcript records."""
    with get_session() as session:
        # Create audio files
        af1 = AudioFileRecord(
            file_name="way_of_zen_cd1.mp3",
            file_path="/tmp/audio/way_of_zen_cd1.mp3",
            file_hash="hash1",
            duration_seconds=120.0,
        )
        af2 = AudioFileRecord(
            file_name="way_of_zen_cd2.mp3",
            file_path="/tmp/audio/way_of_zen_cd2.mp3",
            file_hash="hash2",
            duration_seconds=180.0,
        )
        af3 = AudioFileRecord(
            file_name="meditation_guide.mp3",
            file_path="/tmp/audio/meditation_guide.mp3",
            file_hash="hash3",
            duration_seconds=60.0,
        )
        session.add_all([af1, af2, af3])
        session.flush()

        tx1 = TranscriptionRecord(
            audio_file_id=af1.id,
            file_name="Way of Zen CD1",
            full_text="This is the way of Zen chapter one.",
            language="en",
            engine="faster-whisper",
            model_name="turbo",
            duration_seconds=120.0,
            word_count=8,
            segment_count=1,
            pipeline_phase="complete",
        )
        tx2 = TranscriptionRecord(
            audio_file_id=af2.id,
            file_name="Way of Zen CD2",
            full_text="Now we explore the history of Taoism.",
            language="en",
            engine="faster-whisper",
            model_name="turbo",
            duration_seconds=180.0,
            word_count=8,
            pipeline_phase="complete",
        )
        tx3 = TranscriptionRecord(
            audio_file_id=af3.id,
            file_name="Mindfulness & Meditation",
            full_text="Breathe in deeply and relax your shoulders.",
            language="en",
            engine="google",
            model_name="gemini-2.0",
            duration_seconds=60.0,
            word_count=7,
            pipeline_phase="complete",
        )
        session.add_all([tx1, tx2, tx3])
        session.flush()

        seg1 = SegmentRecord(
            transcription_id=tx1.id,
            segment_index=1,
            start_time=0.0,
            end_time=5.0,
            text="This is the way of Zen chapter one.",
        )
        session.add(seg1)
        session.commit()

        return {"tx1": tx1.id, "tx2": tx2.id, "tx3": tx3.id}


def test_resolve_direct_id(seed_transcriptions):
    id1 = seed_transcriptions["tx1"]

    # Raw integer
    r1 = resolve_transcript_target(id1)
    assert r1 is not None
    assert r1["id"] == id1

    # String with hash prefix
    r2 = resolve_transcript_target(f"#{id1}")
    assert r2 is not None
    assert r2["id"] == id1

    # Prefix 'tx'
    r3 = resolve_transcript_target(f"tx{id1}")
    assert r3 is not None
    assert r3["id"] == id1


def test_resolve_nonexistent_id(seed_transcriptions):
    c = Console(record=True)
    r = resolve_transcript_target(99999, console=c)
    assert r is None
    output = c.export_text()
    assert "99999 not found" in output


def test_resolve_relative_keywords(seed_transcriptions):
    # tx3 was added last, so it should be latest
    latest_id = seed_transcriptions["tx3"]

    for kw in ("latest", "last", "recent", "-1"):
        r = resolve_transcript_target(kw)
        assert r is not None
        assert r["id"] == latest_id

    # -2 should be tx2
    r_prev = resolve_transcript_target("-2")
    assert r_prev is not None
    assert r_prev["id"] == seed_transcriptions["tx2"]


def test_resolve_unique_substring(seed_transcriptions):
    # "Meditation" matches only tx3
    r = resolve_transcript_target("Meditation")
    assert r is not None
    assert r["id"] == seed_transcriptions["tx3"]

    # "CD1" matches only tx1
    r_cd1 = resolve_transcript_target("CD1")
    assert r_cd1 is not None
    assert r_cd1["id"] == seed_transcriptions["tx1"]


def test_resolve_ambiguous_substring_non_interactive(seed_transcriptions):
    c = Console(record=True, width=120)
    # "Way of Zen" matches tx1 and tx2
    r = resolve_transcript_target("Way of Zen", interactive=False, console=c)
    assert r is None
    output = c.export_text()
    assert "Multiple Transcripts Match" in output
    assert "Way of Zen CD1" in output
    assert "Way of Zen CD2" in output


def test_resolve_ambiguous_substring_interactive(seed_transcriptions, monkeypatch):
    c = Console(record=True)
    # Mock user typing '1' at disambiguation prompt
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    r = resolve_transcript_target("Way of Zen", interactive=True, console=c)
    assert r is not None
    assert "Way of Zen" in r["file_name"]


def test_resolve_no_match_suggests_close(seed_transcriptions):
    c = Console(record=True)
    r = resolve_transcript_target("Meditatn", console=c)
    assert r is None
    output = c.export_text()
    assert "Did you mean" in output or "Not Found" in output


def test_resolve_none_defaults_to_latest(seed_transcriptions):
    r = resolve_transcript_target(None, interactive=False)
    assert r is not None
    assert r["id"] == seed_transcriptions["tx3"]
