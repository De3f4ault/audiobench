"""Unit tests for the 360-degree file relations dossier (info command)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
import pytest
from rich.console import Console

from audiobench.cli.commands.info_cmd import render_file_dossier


def test_render_file_dossier_empty_target():
    console = Console(file=io.StringIO())
    res = render_file_dossier("", console)
    assert res is False


@patch("audiobench.cli.commands.info_cmd.get_session")
def test_render_file_dossier_not_found(mock_get_session):
    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_get_session.return_value.__enter__.return_value = mock_session

    console = Console(file=io.StringIO())
    res = render_file_dossier("nonexistent_file_999", console)
    assert res is False


@patch("audiobench.cli.commands.info_cmd.get_session")
def test_render_file_dossier_by_id(mock_get_session):
    mock_audio = MagicMock()
    mock_audio.id = 42
    mock_audio.file_name = "interview_2026.mp3"
    mock_audio.file_path = "/path/to/interview_2026.mp3"
    mock_audio.duration_seconds = 185.0
    mock_audio.format = "mp3"
    mock_audio.sample_rate = 44100
    mock_audio.channels = 2
    mock_audio.file_size_bytes = 4096000
    mock_audio.file_hash = "abcdef0123456789abcdef"

    mock_session = MagicMock()
    def mock_query(model):
        m = MagicMock()
        from audiobench.storage.models import (
            AudioFileRecord,
            BookmarkRecord,
            ChapterRecord,
            ChatConversation,
            TranscriptionRecord,
        )
        if model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = mock_audio
        elif model in (TranscriptionRecord, ChapterRecord, BookmarkRecord, ChatConversation):
            m.filter_by.return_value.order_by.return_value.all.return_value = []
            m.order_by.return_value.all.return_value = []
        return m

    mock_session.query.side_effect = mock_query
    mock_get_session.return_value.__enter__.return_value = mock_session

    buf = io.StringIO()
    console = Console(file=buf, record=True)
    res = render_file_dossier("42", console)
    assert res is True
    output = console.export_text()
    assert "File Dossier: [042] interview_2026.mp3" in output
    assert "[FILE]" in output
    assert "[TRANSCRIPTIONS]" in output
    assert "[CHAPTERS]" in output
    assert "[BOOKMARKS & EXPLAINS]" in output
    assert "[CLIPS]" in output
    assert "[CONVERSATIONS]" in output
