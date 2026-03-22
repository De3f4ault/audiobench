"""Tests for TranscriptionRepository — CRUD operations on a test DB."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from audiobench.core.settings import AudioBenchSettings


@pytest.fixture
def repo(tmp_path):
    """Create a fresh repository with a temp database."""
    import audiobench.core.db_engine as db_mod

    db_path = tmp_path / "test_repo.db"
    settings = AudioBenchSettings(
        database_url=f"sqlite:///{db_path}",
        data_dir=tmp_path,
    )

    old_engine = db_mod._engine
    db_mod._engine = None

    with patch("audiobench.core.db_engine.get_settings", return_value=settings):
        from audiobench.core.db_engine import init_db

        init_db()

        from audiobench.storage.repository import TranscriptionRepository

        yield TranscriptionRepository()

    db_mod._engine = old_engine


def _make_transcript(text="Hello world", file_name="test.mp3", duration=10.0):
    """Create a minimal Transcript-like object for testing."""
    from audiobench.transcribe.transcription_result import Segment, Transcript

    return Transcript(
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=5.0,
                text=text,
                words=[],
            )
        ],
        language="en",
        language_probability=0.95,
        duration_seconds=duration,
        model_name="tiny",
        engine="faster-whisper",
    )


class TestSaveAndRetrieve:
    """Test saving and retrieving transcriptions."""

    def test_save_returns_id(self, repo):
        transcript = _make_transcript()
        record_id = repo.save_transcription(transcript)
        assert record_id is not None
        assert isinstance(record_id, int)
        assert record_id > 0

    def test_get_by_id(self, repo):
        transcript = _make_transcript(text="Testing get by ID")
        record_id = repo.save_transcription(transcript)

        result = repo.get_by_id(record_id)
        assert result is not None
        assert result["id"] == record_id
        assert "Testing get by ID" in result["full_text"]

    def test_get_by_id_not_found(self, repo):
        result = repo.get_by_id(99999)
        assert result is None

    def test_history_returns_recent(self, repo):
        for i in range(5):
            t = _make_transcript(text=f"Transcript {i}")
            repo.save_transcription(t)

        records = repo.get_history(limit=3)
        assert len(records) == 3

    def test_history_order_newest_first(self, repo):
        repo.save_transcription(_make_transcript(text="First"))
        repo.save_transcription(_make_transcript(text="Second"))

        records = repo.get_history(limit=2)
        # Newest should be first
        assert records[0]["id"] > records[1]["id"]


class TestSearch:
    """Test text search functionality."""

    def test_search_finds_match(self, repo):
        repo.save_transcription(_make_transcript(text="The quick brown fox"))
        repo.save_transcription(_make_transcript(text="Lazy dog sleeping"))

        results = repo.search("fox")
        assert len(results) >= 1
        assert any("fox" in r.get("text_preview", "").lower() for r in results)

    def test_search_no_match(self, repo):
        repo.save_transcription(_make_transcript(text="Hello world"))

        results = repo.search("xylophone")
        assert len(results) == 0

    def test_search_case_insensitive(self, repo):
        repo.save_transcription(_make_transcript(text="AudioBench ROCKS"))

        results = repo.search("rocks")
        assert len(results) >= 1


class TestUpdateAndDelete:
    """Test update and delete operations."""

    def test_update_text(self, repo):
        record_id = repo.save_transcription(_make_transcript(text="Original text"))

        ok = repo.update_text(record_id, "Updated text")
        assert ok is True

        result = repo.get_by_id(record_id)
        assert "Updated text" in result["full_text"]

    def test_update_text_not_found(self, repo):
        ok = repo.update_text(99999, "No such record")
        assert ok is False

    def test_delete_by_id(self, repo):
        record_id = repo.save_transcription(_make_transcript())

        ok = repo.delete_by_id(record_id)
        assert ok is True

        result = repo.get_by_id(record_id)
        assert result is None

    def test_delete_by_id_not_found(self, repo):
        ok = repo.delete_by_id(99999)
        assert ok is False

    def test_delete_all(self, repo):
        for i in range(3):
            repo.save_transcription(_make_transcript(text=f"Record {i}"))

        count = repo.delete_all()
        assert count >= 3  # May include records from other tests if DB is shared

        records = repo.get_history(limit=100)
        assert len(records) == 0
