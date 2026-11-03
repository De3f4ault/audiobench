"""Unit tests for session-aware playback persistence and audio focus."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch
import pytest
from rich.console import Console

from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, BookmarkRecord, TranscriptionRecord
from audiobench.storage.bookmark_repository import BookmarkRepository
from audiobench.playback.controls import (
    _cmd_play,
    _cmd_toggle,
    handle_playback_command,
)


@pytest.fixture(autouse=True)
def _isolate_db(test_db):
    """Ensure tests run against isolated test_db."""
    pass


def _create_sample_records(tid: int, afid: int | None = None) -> None:
    """Create sample audio file and transcription records in test DB."""
    with get_session() as session:
        af_id = afid or tid
        af = session.query(AudioFileRecord).filter_by(id=af_id).first()
        if not af:
            af = AudioFileRecord(
                id=af_id,
                file_name=f"test_{af_id}.mp3",
                file_path=f"/tmp/test_{af_id}.mp3",
                file_hash=f"hash_{af_id}",
                duration_seconds=300.0,
            )
            session.add(af)
            session.flush()

        tx = session.query(TranscriptionRecord).filter_by(id=tid).first()
        if not tx:
            tx = TranscriptionRecord(
                id=tid,
                audio_file_id=af.id,
                file_name=f"test_{tid}.mp3",
                full_text="Test transcription full text",
                language="en",
                engine="faster-whisper",
                model_name="turbo",
                duration_seconds=300.0,
                pipeline_phase="complete",
            )
            session.add(tx)
            session.flush()
        session.commit()


def test_bookmark_repository_save_and_get_resume_position():
    repo = BookmarkRepository()
    tid = 101
    _create_sample_records(tid)

    # Initially None
    assert repo.get_resume_position(tid) is None

    # Save position 45.0s on 120.0s track
    repo.save_resume_position(transcription_id=tid, position=45.0, duration=120.0)

    # Immediately fetch (< 60s old, so exact resume)
    pos = repo.get_resume_position(tid, apply_courtesy_rollback=True)
    assert pos == 45.0


def test_bookmark_repository_courtesy_rollback():
    repo = BookmarkRepository()
    tid = 102
    _create_sample_records(tid)

    repo.save_resume_position(transcription_id=tid, position=50.0, duration=200.0)

    # Manually update created_at to 2 minutes ago to simulate > 60s pause
    with get_session() as session:
        bm = session.query(BookmarkRecord).filter_by(transcription_id=tid, bookmark_type="resume").first()
        assert bm is not None
        bm.created_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
        session.commit()

    # With courtesy rollback enabled: 50.0 - 2.5 = 47.5
    pos_rolled = repo.get_resume_position(tid, apply_courtesy_rollback=True)
    assert pos_rolled == 47.5

    # With courtesy rollback disabled: exactly 50.0
    pos_exact = repo.get_resume_position(tid, apply_courtesy_rollback=False)
    assert pos_exact == 50.0


def test_bookmark_repository_eof_resets_to_zero():
    repo = BookmarkRepository()
    tid_near_end = 103
    tid_percent = 104
    _create_sample_records(tid_near_end)
    _create_sample_records(tid_percent)

    # Within 3 seconds of end (118.5 of 120.0) -> reset to 0.0
    repo.save_resume_position(transcription_id=tid_near_end, position=118.5, duration=120.0)
    assert repo.get_resume_position(tid_near_end, apply_courtesy_rollback=False) == 0.0

    # >= 98% of duration (99.0 of 100.0) -> reset to 0.0
    repo.save_resume_position(transcription_id=tid_percent, position=99.0, duration=100.0)
    assert repo.get_resume_position(tid_percent, apply_courtesy_rollback=False) == 0.0


def test_bookmark_repository_upsert_single_resume():
    repo = BookmarkRepository()
    tid = 105
    _create_sample_records(tid)

    repo.save_resume_position(transcription_id=tid, position=10.0, duration=100.0)
    repo.save_resume_position(transcription_id=tid, position=25.0, duration=100.0)

    # Verify only 1 record exists in DB for this transcription_id with bookmark_type="resume"
    with get_session() as session:
        records = session.query(BookmarkRecord).filter_by(transcription_id=tid, bookmark_type="resume").all()
        assert len(records) == 1
        assert records[0].timestamp == 25.0


def test_cmd_play_resumes_owned_paused_audio():
    console = MagicMock()
    client = MagicMock()
    tid = 200

    # Daemon has transcript 200 paused
    client.playback_status.return_value = {
        "playing": True,
        "paused": True,
        "state": "paused",
        "transcription_id": tid,
        "position": 33.0,
    }

    # Calling bare /play when 200 is loaded in session
    res = _cmd_play("", client, console, resolver=None, loaded_ids=[tid])
    assert res is True
    client.playback_resume.assert_called_once()
    client.playback_play.assert_not_called()


def test_cmd_play_loads_session_at_bookmark_when_idle_or_foreign():
    console = MagicMock()
    client = MagicMock()
    session_tid = 201
    _create_sample_records(session_tid, afid=99)

    # Save a resume position for session_tid
    BookmarkRepository().save_resume_position(
        transcription_id=session_tid,
        audio_file_id=99,
        position=42.0,
        duration=300.0,
    )

    # Daemon is currently idle (playing is False)
    client.playback_status.return_value = {"playing": False, "state": "idle"}

    # Resolver provides transcript metadata
    resolver = MagicMock(return_value={
        "id": session_tid,
        "file_name": "speech.mp3",
        "file_path": "/path/to/speech.mp3",
        "audio_file_id": 99,
    })

    res = _cmd_play("", client, console, resolver=resolver, loaded_ids=[session_tid])
    assert res is True
    client.playback_play.assert_called_once_with(
        "/path/to/speech.mp3",
        start_pos=42.0,
        audio_file_id=99,
        transcription_id=session_tid,
    )


def test_cmd_play_refuses_timestamp_seek_on_foreign_track():
    console = MagicMock()
    client = MagicMock()
    session_tid = 300
    foreign_tid = 999

    # Daemon is currently playing foreign_tid
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "state": "playing",
        "transcription_id": foreign_tid,
        "position": 10.0,
    }

    # User in session tries /play 01:00
    res = _cmd_play("01:00", client, console, resolver=None, loaded_ids=[session_tid])
    assert res is False
    client.playback_seek.assert_not_called()
    client.playback_play.assert_not_called()


def test_cmd_toggle_saves_bookmark_on_pause():
    console = MagicMock()
    client = MagicMock()
    tid = 400
    _create_sample_records(tid, afid=12)

    # Daemon is playing tid
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "state": "playing",
        "transcription_id": tid,
        "position": 55.0,
        "duration": 200.0,
        "audio_file_id": 12,
    }

    # When toggled, daemon transitions to paused
    client.playback_toggle.return_value = {
        "playing": True,
        "paused": True,
        "state": "paused",
        "transcription_id": tid,
        "position": 55.0,
        "duration": 200.0,
        "audio_file_id": 12,
    }

    _cmd_toggle("/toggle", client, console, loaded_ids=[tid])
    client.playback_toggle.assert_called_once()

    # Check bookmark saved
    pos = BookmarkRepository().get_resume_position(tid, apply_courtesy_rollback=False)
    assert pos == 55.0


def test_chat_repl_pause_session_playback_ownership():
    from audiobench.chat.chat_repl import ChatREPL

    mock_daemon = MagicMock()
    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_daemon):
        repl = ChatREPL(session_type="general")
        repl._daemon = mock_daemon

        # Session owns transcript 501
        repl.session._transcript_ids = {501}

        # Case 1: Daemon is playing transcript 501 -> Should pause and save bookmark
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 501,
            "position": 25.0,
            "duration": 100.0,
            "audio_file_id": 10,
        }
        repl._pause_session_playback(reason="exit")
        mock_daemon.playback_pause.assert_called_once()

        mock_daemon.playback_pause.reset_mock()

        # Case 2: Daemon is playing unowned transcript 999 -> Must NOT touch playback
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 999,
            "position": 70.0,
            "duration": 200.0,
            "audio_file_id": 20,
        }
        repl._pause_session_playback(reason="exit")
        mock_daemon.playback_pause.assert_not_called()

        # Case 3: force_bg is True -> Must NOT touch playback even if owned
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 501,
            "position": 30.0,
            "duration": 100.0,
        }
        repl._pause_session_playback(reason="exit", force_bg=True)
        mock_daemon.playback_pause.assert_not_called()


def test_show_repl_pause_transcript_playback_ownership():
    from audiobench.cli.show_repl import _pause_transcript_playback

    mock_daemon = MagicMock()
    current_record = {"id": 601, "audio_file_id": 15, "file_name": "test.mp3"}

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_daemon):
        # Case 1: Daemon playing current_record -> Pauses and saves bookmark
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 601,
            "position": 62.0,
            "duration": 180.0,
            "audio_file_id": 15,
        }
        _pause_transcript_playback(current_record, reason="exit")
        mock_daemon.playback_pause.assert_called_once()

        mock_daemon.playback_pause.reset_mock()

        # Case 2: Daemon playing different record 777 -> Must NOT pause
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 777,
            "position": 10.0,
            "duration": 100.0,
        }
        _pause_transcript_playback(current_record, reason="exit")
        mock_daemon.playback_pause.assert_not_called()

        # Case 3: force_bg=True -> Must NOT pause
        mock_daemon.playback_status.return_value = {
            "state": "playing",
            "transcription_id": 601,
            "position": 65.0,
            "duration": 180.0,
        }
        _pause_transcript_playback(current_record, reason="exit", force_bg=True)
        mock_daemon.playback_pause.assert_not_called()
