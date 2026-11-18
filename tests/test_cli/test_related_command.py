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
    mock_db = MagicMock()

    # 1. tx_info
    mock_tx_info = (1501, 80, 10, "Test Work", "Test Author", "File_1501.mp3")
    # 2. chunk_rows
    mock_chunks = [(101,)]
    # 3. sem_edges
    mock_sem_edges = [
        (501, 1502, 20, "Out of Your Mind", "Alan Watts", 0.85, "Form is void and void is form and emptiness is reality through meditation.")
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
    ]

    mock_tx_record = MagicMock(audio_file_id=88)
    mock_af_record = MagicMock(
        id=88,
        file_name="File_1502.mp3",
        file_path="/audio/file_1502.mp3",
        work=MagicMock(title="Out of Your Mind", author="Alan Watts"),
    )

    def query_mock(model):
        from audiobench.storage.models import TranscriptionRecord, AudioFileRecord
        m = MagicMock()
        if model is TranscriptionRecord:
            m.filter_by.return_value.first.return_value = mock_tx_record
        elif model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = mock_af_record
        return m

    mock_db.query.side_effect = query_mock

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(45.0, 60.0, "Alan Watts")):
        cmd_related(repl_session, "")

    assert len(repl_session.last_related) == 1
    thought = repl_session.last_related[0]
    assert thought.index == 1
    assert thought.transcript_id == 1502
    assert thought.start_time == 45.0
    assert thought.file_name == "File_1502.mp3"
    assert thought.speaker == "Alan Watts"
    assert thought.work_title == "Out of Your Mind"
    assert thought.work_author == "Alan Watts"
    assert thought.match_percent >= 80


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


def test_relevance_dots():
    from audiobench.cli.repl.backslash_commands import _relevance_dots

    assert _relevance_dots(1) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan]"
    assert _relevance_dots(2) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan]"
    assert _relevance_dots(3) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim]"
    assert _relevance_dots(4) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim]"
    assert _relevance_dots(5) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim][dim]○[/dim]"
    assert _relevance_dots(6) == "[cyan]●[/cyan][cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim][dim]○[/dim]"
    assert _relevance_dots(7) == "[cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim][dim]○[/dim][dim]○[/dim]"
    assert _relevance_dots(8) == "[cyan]●[/cyan][cyan]●[/cyan][dim]○[/dim][dim]○[/dim][dim]○[/dim]"
    assert _relevance_dots(9) == "[cyan]●[/cyan][dim]○[/dim][dim]○[/dim][dim]○[/dim][dim]○[/dim]"


def test_cmd_related_scales_to_9_items_with_hybrid_threshold(repl_session):
    """Test that cmd_related can populate up to 9 items, filtering out items below max_score * 0.85."""
    mock_db = MagicMock()

    mock_tx_info = (1501, 80, 10, "Source Work", "Source Author", "File_1501.mp3")
    mock_chunks = [(101,)]

    # 12 candidate edges: top is 0.90.
    # Hybrid cutoff = max(0.90 * 0.85, 0.72) = 0.765.
    # Scores: 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.80, 0.75 (below), 0.70 (below)
    # The first 10 are >= 0.765, but cap is 9 items.
    mock_sem_edges = [
        (500 + i, 1600 + i, 20 + i, f"Work {i}", f"Author {i}", score, f"This is substantive test content for passage number {i} across distinct works.")
        for i, score in enumerate([0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.80, 0.75, 0.70], 1)
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
    ]

    def query_mock(model):
        from audiobench.storage.models import TranscriptionRecord, AudioFileRecord
        m = MagicMock()
        if model is TranscriptionRecord:
            m.filter_by.return_value.first.return_value = MagicMock(audio_file_id=88)
        elif model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = MagicMock(
                id=88, file_name="f.mp3", file_path="/f.mp3", work=MagicMock(title="T", author="A")
            )
        return m

    mock_db.query.side_effect = query_mock
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(10.0, 20.0, "Speaker")):
        cmd_related(repl_session, "")

    assert len(repl_session.last_related) == 9
    assert repl_session.last_related[0].score == 0.90
    assert repl_session.last_related[-1].score == 0.82
    for t in repl_session.last_related:
        assert t.score >= 0.765


def test_cmd_related_floor_rejection_when_weak(repl_session):
    """When top candidate score is below the 0.72 absolute floor, reject all and print graceful message."""
    mock_db = MagicMock()

    mock_tx_info = (1501, 80, 10, "Source Work", "Source Author", "File_1501.mp3")
    mock_chunks = [(101,)]

    # Top score is 0.69 (below 0.72)
    mock_sem_edges = [
        (501, 1601, 21, "Weak Work", "Weak Author", 0.69, "This is weak similarity content below floor."),
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
    ]

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_related(repl_session, "")

    # last_related remains empty
    assert len(getattr(repl_session, "last_related", [])) == 0
    printed_texts = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "No cross-work dialectic connections found yet" in printed_texts


def test_cmd_related_book_mode_wide_vs_narrow(repl_session):
    """Verify that wide terminals (width >= 100) render 2-column Book Mode, and narrow render single column."""
    from audiobench.cli.display.theme import console

    mock_db = MagicMock()
    mock_tx_info = (1501, 80, 10, "Source Work", "Source Author", "File_1501.mp3")
    mock_chunks = [(101,)]
    content_a = "This is substantive content for testing passage A that is well over sixty characters long."
    content_b = "This is substantive content for testing passage B that is well over sixty characters long."
    mock_sem_edges = [
        (501, 1601, 21, "Book A", "Author A", 0.88, content_a),
        (502, 1602, 22, "Book B", "Author B", 0.86, content_b),
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
    ]

    def query_mock(model):
        from audiobench.storage.models import TranscriptionRecord, AudioFileRecord
        m = MagicMock()
        if model is TranscriptionRecord:
            m.filter_by.return_value.first.return_value = MagicMock(audio_file_id=88)
        elif model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = MagicMock(
                id=88, file_name="f.mp3", file_path="/f.mp3", work=MagicMock(title="T", author="A")
            )
        return m

    mock_db.query.side_effect = query_mock
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    # Wide terminal: width = 120
    orig_w, orig_h = console._width, console._height
    try:
        console._width, console._height = 120, 40
        with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
             patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(10.0, 20.0, "Speaker")), \
             patch("audiobench.cli.display.theme.console.print") as mock_print_wide:
            cmd_related(repl_session, "")

        wide_prints = " ".join(str(call.args[0]) for call in mock_print_wide.call_args_list if call.args)
        assert "2-column book mode" in wide_prints

        # Narrow terminal: width = 80
        console._width, console._height = 80, 40
        with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
             patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(10.0, 20.0, "Speaker")), \
             patch("audiobench.cli.display.theme.console.print") as mock_print_narrow:
            cmd_related(repl_session, "")

        narrow_prints = " ".join(str(call.args[0]) for call in mock_print_narrow.call_args_list if call.args)
        assert "single column" in narrow_prints
    finally:
        console._width, console._height = orig_w, orig_h


