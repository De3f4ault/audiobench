"""Tests for Track 1: Database Entity Null Contract Audit.

Verifies that entity IDs (work_id, chapter_id, source_id, expression_id, transcription_id, audio_file_id)
treat 0 and None identically as entity absence across repositories, services, daemon, and UI formatting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from audiobench.core.db_session import get_session
from audiobench.storage.models import ExpressionRecord, ChapterRecord, AudioFileRecord, TranscriptionRecord
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.chapter_repository import ChapterRepository
from audiobench.storage.repository import TranscriptionRepository
from audiobench.storage.bookmark_repository import BookmarkRepository
from audiobench.memory.bookmark_service import generate_smart_bookmark_card
from audiobench.output.text import TextFormatter
from audiobench.playback.service import PlaybackService
from audiobench.daemon.client import DaemonClient
from audiobench.transcribe.transcription_result import Transcript, Segment, Word


@pytest.fixture
def memory_db():
    """Provides a fresh initialized database for testing."""
    from audiobench.core.db_engine import init_db
    init_db()
    yield


class TestWorkIdContract:
    """Verifies work_id = 0 vs None equivalence."""

    def test_register_expression_work_id_zero_does_not_override_existing(self, memory_db):
        """register must not assign work_id=0 or treat 0 as an existing work_id."""
        repo = ExpressionRepository()
        expr = repo.register(
            content="test work_id null contract",
            source_type="audio_transcript",
            work_id=None,
        )
        eid = expr.id

        # Passing work_id=0 should NOT mutate/assign 0
        expr_mut = repo.register(
            content="test work_id null contract",
            source_type="audio_transcript",
            work_id=0,
        )
        assert expr_mut.work_id is None

        # Passing valid work_id=123 should mutate
        expr_valid = repo.register(
            content="test work_id null contract",
            source_type="audio_transcript",
            work_id=123,
        )
        assert expr_valid.work_id == 123

    def test_chapter_repository_work_id_zero_treated_as_absent(self, memory_db):
        """get_ordered_chapters ignores work_id=0 rather than querying AudioFileRecord.work_id == 0."""
        repo = ChapterRepository()
        with patch("audiobench.core.db_session.get_session") as mock_get_s:
            mock_session = MagicMock()
            mock_get_s.return_value.__enter__.return_value = mock_session
            mock_query = mock_session.query.return_value.join.return_value.filter.return_value

            repo.get_ordered_chapters(work_id=0)
            # filter(AudioFileRecord.work_id == work_id) should NOT be called when work_id is 0
            mock_query.filter.assert_not_called()


class TestChapterIdContract:
    """Verifies chapter_id = 0 vs None equivalence."""

    def test_resolve_segment_chapters_zero_treated_as_absent(self, memory_db):
        """_resolve_segment_chapters falls back to interval matching or None when explicit_chapter_id is 0."""
        repo = TranscriptionRepository()
        seg = {"start": 0.0, "end": 10.0, "text": "hello"}
        with get_session() as s:
            res_none = repo._resolve_segment_chapters([seg], audio_file_id=None, session=s, explicit_chapter_id=None)
            res_zero = repo._resolve_segment_chapters([seg], audio_file_id=None, session=s, explicit_chapter_id=0)
            assert res_none == [None]
            assert res_zero == [None]

    def test_text_markdown_formatter_ignores_chapter_id_zero(self):
        """format_transcript_as_markdown does not generate '## Chapter 0' when chapter_id is 0."""
        seg_zero = Segment(
            id=0,
            start=0.0,
            end=5.0,
            text="text under zero",
            words=[Word(start=0.0, end=5.0, word="text under zero", speaker="SPEAKER_1")],
            speaker="SPEAKER_1",
            chapter_id=0,
        )
        tx = Transcript(
            segments=[seg_zero],
        )
        md = TextFormatter().format(tx)
        assert "## Chapter 0" not in md
        assert "text under zero" in md


class TestSourceIdContract:
    """Verifies source_id = 0 vs None equivalence."""

    def test_delete_by_source_zero_returns_zero_immediately(self, memory_db):
        """delete_by_source(..., source_id=0) returns 0 without issuing DB queries."""
        repo = ExpressionRepository()
        with patch.object(repo, "_delete_by_source_impl") as mock_impl:
            count = repo.delete_by_source("audio_transcript", source_id=0)
            assert count == 0
            mock_impl.assert_not_called()

    def test_resolve_pending_relations_zero_skips_query(self, memory_db):
        """_resolve_pending_relations returns immediately when record.source_id is 0."""
        repo = ExpressionRepository()
        mock_rec = MagicMock(spec=ExpressionRecord)
        mock_rec.source_id = 0
        mock_session = MagicMock()

        repo._resolve_pending_relations(mock_session, mock_rec)
        mock_session.query.assert_not_called()


class TestExpressionIdContract:
    """Verifies expression_id = 0 vs None equivalence."""

    def test_generate_smart_bookmark_card_zero_raises_value_error(self, memory_db):
        """generate_smart_bookmark_card(0) raises explicit ValueError."""
        with pytest.raises(ValueError, match="Invalid expression ID"):
            generate_smart_bookmark_card(0)

    def test_generate_smart_bookmark_card_none_raises_value_error(self, memory_db):
        """generate_smart_bookmark_card(None) raises explicit ValueError."""
        with pytest.raises(ValueError, match="Invalid expression ID"):
            generate_smart_bookmark_card(None)  # type: ignore[arg-type]

    def test_play_expression_zero_returns_false_cleanly(self):
        """_play_expression returns False without DB query when expr_id is 0."""
        from audiobench.cli.repl.backslash_commands import _play_expression
        mock_session = MagicMock()
        mock_client = MagicMock()
        with patch("audiobench.core.db_session.get_session") as mock_get_s:
            success = _play_expression(mock_session, mock_client, expr_id=0)
            assert success is False
            mock_get_s.assert_not_called()


class TestConsumerTruthiness:
    """Verifies downstream consumers behave safely with 0 sentinels."""

    def test_load_segment_state_falls_back_on_zero_transcription_id(self):
        """_load_segment_state with transcription_id=0 uses fallback segments instead of DB."""
        svc = PlaybackService.__new__(PlaybackService)
        svc._segments = []
        svc._segment_starts = []

        fallback_segs = [{"start": 1.0, "end": 2.0, "text": "fallback"}]
        with patch.object(svc, "_load_segments_from_db") as mock_db:
            svc._load_segment_state(transcription_id=0, segments=fallback_segs)
            mock_db.assert_not_called()
            assert len(svc._segments) == 1
            assert svc._segments[0]["text"] == "fallback"

    def test_save_resume_position_zero_transcription_id_skips_db(self, memory_db):
        """save_resume_position with transcription_id=0 returns position without DB write."""
        repo = BookmarkRepository()
        with patch("audiobench.core.db_session.get_session") as mock_get_s:
            pos = repo.save_resume_position(transcription_id=0, position=42.5)
            assert pos == 42.5
            mock_get_s.assert_not_called()

    def test_daemon_client_omits_zero_audio_and_transcription_ids_from_payload(self):
        """client.playback_play and sync_session omit 0 IDs from payload to maintain wire contract."""
        client = DaemonClient(socket_path="/tmp/fake.sock")
        with patch.object(client, "_send") as mock_send:
            mock_send.return_value = {"status": "ok"}
            client.playback_play(
                "song.mp3",
                audio_file_id=0,
                transcription_id=0,
            )
            called_args = mock_send.call_args[0][1]
            assert "audio_file_id" not in called_args
            assert "transcription_id" not in called_args

            client.playback_sync_session(
                "song.mp3",
                audio_file_id=0,
                transcription_id=0,
            )
            sync_args = mock_send.call_args[0][1]
            assert "audio_file_id" not in sync_args
            assert "transcription_id" not in sync_args
