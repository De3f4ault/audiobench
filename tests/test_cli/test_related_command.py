"""Tests for the interconnected \\related command, playback seeking, and chat context transfer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import click
import pytest

from audiobench.cli.repl.session import NavigationFrame, RelatedThought, ReplSession
from audiobench.cli.repl.backslash_commands import (
    cmd_related,
    cmd_play,
    cmd_chat,
    cmd_focus,
    _resolve_thought_segment,
)
from audiobench.core.focused_entity import FocusedEntity


@pytest.fixture
def repl_session():
    group = click.Group("test")
    session = ReplSession(group)
    session.focus = FocusedEntity(type="transcript", id=1501, label="Test Current Transcript")
    return session


@pytest.fixture
def mock_related_thoughts():
    return [
        RelatedThought(
            index=1,
            transcript_id=1502,
            audio_file_id=88,
            file_name="Lecture 1 - Nature of Mind.mp3",
            file_path="/audio/lecture1.mp3",
            start_time=125.5,
            end_time=140.0,
            speaker="Alan Watts",
            content="The world of form is precisely the void.",
            score=2.1,
            match_percent=86,
            work_title="Out of Your Mind",
            work_author="Alan Watts",
        ),
        RelatedThought(
            index=2,
            transcript_id=1503,
            audio_file_id=89,
            file_name="Lecture 2 - The Void.mp3",
            file_path="/audio/lecture2.mp3",
            start_time=250.0,
            end_time=275.0,
            speaker="Alan Watts",
            content="Emptiness is form, and form is emptiness.",
            score=1.85,
            match_percent=75,
            work_title="Out of Your Mind",
            work_author="Alan Watts",
        ),
    ]


def test_related_thought_dataclass():
    thought = RelatedThought(
        index=1,
        transcript_id=1500,
        audio_file_id=80,
        file_name="audio.mp3",
        file_path="/path/audio.mp3",
        start_time=12.0,
        end_time=18.5,
        speaker="Speaker A",
        content="Testing content excerpt.",
        score=2.0,
        match_percent=82,
    )
    assert thought.index == 1
    assert thought.transcript_id == 1500
    assert thought.audio_file_id == 80
    assert thought.start_time == 12.0
    assert thought.match_percent == 82


def test_resolve_thought_segment_match():
    from audiobench.storage.models import ExpressionSegmentMap, SegmentRecord

    seg1 = MagicMock(segment_index=0, start_time=10.0, end_time=20.0, speaker="Speaker A", text="Initial introduction.")
    seg2 = MagicMock(segment_index=1, start_time=20.5, end_time=35.0, speaker="Speaker B", text="Form is emptiness precisely.")

    def query_side_effect(model):
        mock = MagicMock()
        if model is ExpressionSegmentMap:
            mock.filter_by.return_value.first.return_value = None
        elif model is SegmentRecord:
            mock.filter_by.return_value.order_by.return_value.all.return_value = [seg1, seg2]
        return mock

    db = MagicMock()
    db.query.side_effect = query_side_effect

    # content contains seg2.text literally → matches via "s_text in content_clean"
    content = "Form is emptiness precisely. Also some additional context about the void."
    st, et, spk = _resolve_thought_segment(db, expr_id=10, transcription_id=1502, expr_content=content)
    assert st == 20.5
    assert et == 35.0
    assert spk == "Speaker B"


def test_cmd_related_populates_last_related(repl_session):
    mock_repo = MagicMock()
    mock_repo.get_by_id.side_effect = lambda tid, session=None: {
        "id": tid,
        "summary": "Talk about reality and emptiness",
        "audio_file_id": 88 if tid == 1502 else 80,
        "file_name": f"File_{tid}.mp3",
        "file_path": f"/audio/file_{tid}.mp3",
    }
    repl_session._get_repo = lambda: mock_repo

    mock_daemon = MagicMock()
    mock_daemon.search.return_value = [
        {"expression_id": 501, "score": 2.15},
    ]

    mock_expr = MagicMock(
        id=501,
        source_id=1502,
        content="Form is void and void is form.",
        speaker="Alan Watts",
        work_id=None,
    )

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_daemon), \
         patch("audiobench.storage.expression_repository.ExpressionRepository.get_by_id", return_value=mock_expr), \
         patch("audiobench.core.db_session.get_session"), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(45.0, 60.0, "Alan Watts")):
        cmd_related(repl_session, "")

    assert len(repl_session.last_related) == 1
    thought = repl_session.last_related[0]
    assert thought.index == 1
    assert thought.transcript_id == 1502
    assert thought.start_time == 45.0
    assert thought.file_name == "File_1502.mp3"
    assert thought.speaker == "Alan Watts"
    assert thought.match_percent > 80


def test_cmd_play_related_thought_index(repl_session, mock_related_thoughts):
    repl_session.last_related = mock_related_thoughts

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": False}

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_play(repl_session, "1")

        mock_client.playback_play.assert_called_once_with(
            "/audio/lecture1.mp3",
            start_pos=125.5,
            audio_file_id=88,
            transcription_id=1502,
        )


def test_cmd_play_related_thought_seek_when_already_playing(repl_session, mock_related_thoughts):
    repl_session.last_related = mock_related_thoughts

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": True, "transcription_id": 1502}

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client):
        cmd_play(repl_session, "1")

        mock_client.playback_seek.assert_called_once_with(125.5)
        mock_client.playback_play.assert_not_called()


def test_cmd_chat_transfer_related_all(repl_session, mock_related_thoughts):
    repl_session.last_related = mock_related_thoughts

    with patch("audiobench.cli.repl.dispatch.dispatch_command") as mock_dispatch:
        cmd_chat(repl_session, "related")

        # Current transcript is 1501, related are 1502 and 1503
        mock_dispatch.assert_called_once_with(
            repl_session,
            ["chat", "1501", "1502", "1503"],
        )


def test_cmd_chat_transfer_selected_indices(repl_session, mock_related_thoughts):
    repl_session.last_related = mock_related_thoughts

    with patch("audiobench.cli.repl.dispatch.dispatch_command") as mock_dispatch:
        cmd_chat(repl_session, "2")

        # Current transcript is 1501, selected thought 2 has transcript 1503
        mock_dispatch.assert_called_once_with(
            repl_session,
            ["chat", "1501", "1503"],
        )


def test_cmd_focus_related_index(repl_session, mock_related_thoughts):
    repl_session.last_related = mock_related_thoughts

    mock_repo = MagicMock()
    mock_repo.get_latest_transcript_for_file.return_value = {"id": 1502}
    repl_session._get_repo = lambda: mock_repo

    with patch("audiobench.cli.repl.dispatch.print_context_summary"):
        cmd_focus(repl_session, "1")

    assert repl_session.focus is not None
    assert repl_session.focus.type == "file"
    assert repl_session.focus.id == 88
    assert repl_session.focus.label == "Lecture 1 - Nature of Mind.mp3"
    # last_id derives from focus → get_latest_transcript_for_file → returns 1502
    assert repl_session.last_id == 1502
