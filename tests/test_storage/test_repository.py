"""Tests for TranscriptionRepository — CRUD operations on a test DB."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from audiobench.core.settings import AudioBenchSettings


@pytest.fixture
def repo(test_db):
    """Create a fresh repository using the isolated test_db fixture."""
    from audiobench.storage.repository import TranscriptionRepository

    return TranscriptionRepository()


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


class TestOverwritePath:
    """Regression tests for the overwrite=True / existing audio_record code path.

    Before the fix, `from pathlib import Path` lived inside the
    `if audio_record is None` branch of save_transcription.  Python's
    function-level scoping makes any locally-imported name a local
    throughout the *entire* function, so on the overwrite path (audio_record
    already exists, that branch is skipped) `Path` was an unbound local and
    line 370 raised UnboundLocalError.
    """

    def test_overwrite_existing_audio_record_with_no_chapters_does_not_raise(
        self, repo, tmp_path
    ):
        """The exact crash scenario: second save on an already-known file,
        zero chapters stored, file exists on disk so the elif at L370 is entered."""
        from unittest.mock import MagicMock, patch

        from audiobench.transcribe.transcription_result import AudioMetadata

        audio_file = tmp_path / "regression_overwrite.mp3"
        audio_file.touch()  # must exist so Path(...).exists() is True

        meta = AudioMetadata(
            file_path=str(audio_file),
            file_name=audio_file.name,
            file_size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            file_hash="regression_test_hash_overwrite_path",
        )
        transcript = _make_transcript(text="Regression: overwrite path")

        repo._import_to_library = lambda p: str(p)
        repo._schedule_biometric_pass = MagicMock()

        # Patch ChapterDetector at its definition site — it is imported lazily
        # inside save_transcription, so patching the repository module namespace
        # would not work.
        with patch(
            "audiobench.chapters.detector.ChapterDetector"
        ) as MockDetector:
            MockDetector.return_value.detect.return_value = []

            # First save — creates audio_record (new-file branch)
            repo.save_transcription(transcript, audio_metadata=meta, overwrite=True)

            # Second save — audio_record already exists (overwrite path).
            # File exists on disk → enters elif at line 370 → before the fix
            # this raised UnboundLocalError on Path(...).
            # Must not raise — this is the regression gate.
            repo.save_transcription(
                transcript, audio_metadata=meta, overwrite=True
            )

