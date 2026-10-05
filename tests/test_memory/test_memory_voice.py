"""Unit tests for voice input and speak synthesis in memory search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from audiobench.cli.commands.memory_cmd import (
    SearchSessionState,
    _parse_slash_command,
    _speak_text,
)
from audiobench.chat.voice_session import (
    capture_voice_query,
    is_internet_connected,
    is_local_stt_available,
)


def test_search_session_state_speak_answer():
    """Verify speak_answer default and toggle in SearchSessionState."""
    state = SearchSessionState()
    assert state.speak_answer is False
    state.speak_answer = True
    assert state.speak_answer is True


def test_parse_slash_command_voice():
    """Test /voice and /voices commands in search REPL."""
    state = SearchSessionState()

    msg = _parse_slash_command("/voice", state)
    assert msg is not None
    assert "/voice" in msg

    # /voice list
    with patch("audiobench.tts.engine.PiperTTSEngine.list_voices", return_value=[]):
        msg = _parse_slash_command("/voice list", state)
        assert msg is not None
        assert "Downloaded voices" in msg

    # /voice set triggers picker
    with patch("audiobench.chat.voice_picker.show_voice_picker") as mock_picker:
        msg = _parse_slash_command("/voice set", state)
        assert msg is None
        mock_picker.assert_called_once()


def test_parse_slash_command_speak():
    """Test /speak command in search REPL."""
    state = SearchSessionState()
    state.last_synthesis = "This is a synthesis answer from memory [1]."

    with patch("audiobench.cli.commands.memory_cmd._speak_text") as mock_speak:
        msg = _parse_slash_command("/speak", state)
        assert msg is None
        mock_speak.assert_called_once_with("This is a synthesis answer from memory [1].")


def test_speak_text_invokes_piper():
    """Test _speak_text cleans text and calls PiperTTSEngine.play."""
    fake_console = MagicMock()
    with patch("audiobench.chat.voice_picker.ensure_voice_downloaded", return_value=True), \
         patch("audiobench.tts.engine.PiperTTSEngine.play") as mock_play:
        _speak_text("Here is the answer [1] with `code`.", console=fake_console)
        mock_play.assert_called_once()
        # Verify markdown/citations were stripped before sending to play
        args, _ = mock_play.call_args
        assert "[1]" not in args[0]


def test_capture_voice_query_offline_fallback():
    """Test capture_voice_query returns offline fallback when internet and local STT are unavailable."""
    fake_console = MagicMock()
    with patch("audiobench.chat.voice_session.is_internet_connected", return_value=False), \
         patch("audiobench.chat.voice_session.is_local_stt_available", return_value=False):
        text, is_fallback = capture_voice_query(console=fake_console)
        assert text == ""
        assert is_fallback is True


def test_search_slash_completer_voice_and_speak():
    """Verify that SearchSlashCompleter yields completions for /voice, /voices, and /speak."""
    from prompt_toolkit.document import Document
    from audiobench.memory.search_completer import SearchSlashCompleter

    completer = SearchSlashCompleter()

    # /v prefix should complete /voice, /voices, /v, /voice-picker
    doc = Document("/v", cursor_position=2)
    completions = [c.text for c in completer.get_completions(doc, None)]
    assert "/voice" in completions
    assert "/voices" in completions
    assert "/v" in completions

    # /sp prefix should complete /speak
    doc = Document("/sp", cursor_position=3)
    completions = [c.text for c in completer.get_completions(doc, None)]
    assert "/speak" in completions

