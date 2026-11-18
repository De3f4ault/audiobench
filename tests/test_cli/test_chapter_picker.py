from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_input_returns(monkeypatch):
    def _setter(returns: list[str]):
        it = iter(returns)
        monkeypatch.setattr("builtins.input", lambda *args, **kwargs: next(it))
    return _setter

@pytest.fixture
def audio_file_with_13_chapters(monkeypatch):
    from audiobench.storage.models import AudioFileRecord, ChapterRecord
    mock_file = MagicMock(spec=AudioFileRecord)
    mock_file.id = 1
    mock_chapters = [MagicMock(spec=ChapterRecord, id=i, transcript_length=1000) for i in range(1, 14)]

    # Mock get_session to return a mock session that returns this file
    mock_session = MagicMock()
    mock_session.query().filter_by().first.return_value = mock_file
    mock_session.query().filter_by().order_by().all.return_value = mock_chapters

    # We mock core.db_session because the chat module might use it
    monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: MagicMock(__enter__=lambda self: mock_session, __exit__=lambda self, *args: None))
    return mock_file

@pytest.fixture
def large_audio_file(monkeypatch):
    from audiobench.storage.models import AudioFileRecord, ChapterRecord
    mock_file = MagicMock(spec=AudioFileRecord)
    mock_file.id = 2
    mock_chapters = [MagicMock(spec=ChapterRecord, id=i, transcript_length=40000) for i in range(1, 14)]

    mock_session = MagicMock()
    mock_session.query().filter_by().order_by().all.return_value = mock_chapters
    monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: MagicMock(__enter__=lambda self: mock_session, __exit__=lambda self, *args: None))
    return mock_file

@pytest.fixture
def small_audio_file(monkeypatch):
    from audiobench.storage.models import AudioFileRecord, ChapterRecord
    mock_file = MagicMock(spec=AudioFileRecord)
    mock_file.id = 3
    mock_chapters = [MagicMock(spec=ChapterRecord, id=i, transcript_length=1000) for i in range(1, 14)]

    mock_session = MagicMock()
    mock_session.query().filter_by().order_by().all.return_value = mock_chapters
    monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: MagicMock(__enter__=lambda self: mock_session, __exit__=lambda self, *args: None))
    return mock_file

def test_pick_single_chapter(mock_input_returns, audio_file_with_13_chapters):
    mock_input_returns(["3"])
    from audiobench.cli.tui.chapter_picker import pick_chapters
    result = pick_chapters(audio_file_with_13_chapters.id)
    assert len(result) == 1

def test_pick_comma_separated(mock_input_returns, audio_file_with_13_chapters):
    mock_input_returns(["1,3,5"])
    from audiobench.cli.tui.chapter_picker import pick_chapters
    result = pick_chapters(audio_file_with_13_chapters.id)
    assert len(result) == 3

def test_pick_range(mock_input_returns, audio_file_with_13_chapters):
    mock_input_returns(["2-6"])
    from audiobench.cli.tui.chapter_picker import pick_chapters
    result = pick_chapters(audio_file_with_13_chapters.id)
    assert len(result) == 5

def test_pick_all(mock_input_returns, audio_file_with_13_chapters):
    mock_input_returns(["all"])
    from audiobench.cli.tui.chapter_picker import pick_chapters
    result = pick_chapters(audio_file_with_13_chapters.id)
    assert len(result) == 13

def test_invalid_input_reprompts(mock_input_returns, audio_file_with_13_chapters):
    mock_input_returns(["999", "abc", "1"])  # invalid, invalid, then valid
    from audiobench.cli.tui.chapter_picker import pick_chapters
    result = pick_chapters(audio_file_with_13_chapters.id)
    assert len(result) == 1  # eventually accepted "1"

def test_chapter_picker_triggers_on_large_file(mock_input_returns, large_audio_file, monkeypatch):
    """Files exceeding token threshold must trigger chapter picker automatically."""
    mock_input_returns(["1"])
    from audiobench.cli.commands.chat import _maybe_pick_chapters
    chapters = _maybe_pick_chapters(large_audio_file.id, token_threshold=80_000)
    assert chapters is not None
    assert len(chapters) >= 1

def test_chapter_picker_not_triggered_on_small_file(small_audio_file):
    """Small files must load without the chapter picker."""
    from audiobench.cli.commands.chat import _maybe_pick_chapters
    result = _maybe_pick_chapters(small_audio_file.id, token_threshold=80_000)
    assert result is None  # None means "load all, no picker needed"
