"""Unit tests for Voice Picker and speech text processing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from audiobench.chat.voice_picker import (
    CURATED_VOICES,
    play_voice_audition,
    show_voice_picker,
    strip_markdown_for_speech,
)


def test_strip_markdown_for_speech():
    """Test that markdown and citation tokens are stripped cleanly for TTS."""
    raw = (
        "## Summary of Results [1]\n"
        "Here is the key finding **bold text** with `inline code` and a [link](https://example.com).\n"
        "```python\ndef test(): pass\n```\n"
        "As noted in [1, 2], the database schema [3] is finalized."
    )
    cleaned = strip_markdown_for_speech(raw)
    assert "[1]" not in cleaned
    assert "[1, 2]" not in cleaned
    assert "[3]" not in cleaned
    assert "```" not in cleaned
    assert "code snippet omitted" in cleaned
    assert "bold text" in cleaned
    assert "inline code" in cleaned
    assert "link" in cleaned
    assert "https://example.com" not in cleaned


def test_play_voice_audition_missing():
    """Test play_voice_audition returns False when voice model does not exist."""
    fake_console = MagicMock()
    with patch("audiobench.tts.engine.PiperTTSEngine._default_voices_dir") as mock_dir:
        mock_dir.return_value = MagicMock()
        mock_dir.return_value.__truediv__.return_value.exists.return_value = False
        res = play_voice_audition("non_existent_voice", console=fake_console)
        assert res is False


def test_play_voice_audition_success():
    """Test play_voice_audition calls engine.play when files exist."""
    fake_console = MagicMock()
    with patch("audiobench.tts.engine.PiperTTSEngine.play") as mock_play, \
         patch("pathlib.Path.exists", return_value=True):
        res = play_voice_audition("en_US-amy-medium", console=fake_console)
        assert res is True
        mock_play.assert_called_once()


def test_show_voice_picker_keep_current():
    """Test entering 0, q, quit, exit, or hitting Enter keeps current voice."""
    fake_console = MagicMock()
    for exit_input in ("0", "q", "Q", "quit", "exit", ""):
        with patch("builtins.input", return_value=exit_input):
            res = show_voice_picker(console=fake_console, current_voice="en_US-amy-medium")
            assert res is None


def test_show_voice_picker_select_downloaded():
    """Test selecting a downloaded voice updates settings."""
    from pathlib import Path
    from audiobench.tts.engine import VoiceInfo

    fake_console = MagicMock()
    mock_settings = MagicMock()
    mock_settings.tts_voice = "en_US-amy-medium"

    sample_voice = VoiceInfo(
        name="en_US-amy-medium",
        language="en_US",
        quality="medium",
        model_path=Path("/tmp/en_US-amy-medium.onnx"),
        config_path=Path("/tmp/en_US-amy-medium.onnx.json"),
    )

    with patch("builtins.input", return_value="1"), \
         patch("audiobench.chat.voice_picker.get_settings", return_value=mock_settings), \
         patch("audiobench.tts.engine.PiperTTSEngine.list_voices", return_value=[sample_voice]), \
         patch("audiobench.core.settings.invalidate_settings_cache"):
        res = show_voice_picker(console=fake_console, current_voice="en_US-lessac-high")
        assert res == "en_US-amy-medium"
        assert mock_settings.tts_voice == "en_US-amy-medium"
        mock_settings.save.assert_called_once()


def test_ensure_voice_downloaded_already_cached():
    """Test ensure_voice_downloaded returns True immediately when files exist."""
    from audiobench.chat.voice_picker import ensure_voice_downloaded

    fake_console = MagicMock()
    with patch("pathlib.Path.exists", return_value=True):
        res = ensure_voice_downloaded("en_US-amy-medium", console=fake_console)
        assert res is True


def test_ensure_voice_downloaded_offline():
    """Test ensure_voice_downloaded returns False when offline and files missing."""
    from audiobench.chat.voice_picker import ensure_voice_downloaded

    fake_console = MagicMock()
    with patch("pathlib.Path.exists", return_value=False), \
         patch("audiobench.chat.voice_session.is_internet_connected", return_value=False):
        res = ensure_voice_downloaded("en_US-lessac-high", console=fake_console)
        assert res is False


def test_ensure_voice_downloaded_prompt_yes():
    """Test ensure_voice_downloaded prompts and downloads voice when user agrees."""
    from audiobench.chat.voice_picker import ensure_voice_downloaded

    fake_console = MagicMock()
    with patch("pathlib.Path.exists", return_value=False), \
         patch("audiobench.chat.voice_session.is_internet_connected", return_value=True), \
         patch("builtins.input", return_value="y"), \
         patch("audiobench.tts.engine.PiperTTSEngine.download_voice") as mock_dl:
        res = ensure_voice_downloaded("en_US-ryan-medium", console=fake_console)
        assert res is True
        mock_dl.assert_called_once_with("en_US-ryan-medium")


def test_ensure_voice_downloaded_prompt_no():
    """Test ensure_voice_downloaded returns False when user declines download."""
    from audiobench.chat.voice_picker import ensure_voice_downloaded

    fake_console = MagicMock()
    with patch("pathlib.Path.exists", return_value=False), \
         patch("audiobench.chat.voice_session.is_internet_connected", return_value=True), \
         patch("builtins.input", return_value="n"), \
         patch("audiobench.tts.engine.PiperTTSEngine.download_voice") as mock_dl:
        res = ensure_voice_downloaded("en_US-ryan-medium", console=fake_console)
        assert res is False
        mock_dl.assert_not_called()


def test_piper_engine_stop():
    """Test PiperTTSEngine.stop invokes sounddevice.stop."""
    from audiobench.tts.engine import PiperTTSEngine

    with patch("sounddevice.stop") as mock_sd_stop:
        PiperTTSEngine.stop()
        mock_sd_stop.assert_called_once()


def test_chat_repl_voices_command():
    """Test /voices command in chat REPL triggers voice picker."""
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL()
    with patch("audiobench.chat.voice_picker.show_voice_picker") as mock_picker:
        res = repl._handle_slash_command("/voices")
        assert res is False
        mock_picker.assert_called_once()


def test_chat_repl_speak_keyboard_interrupt():
    """Test /speak handles KeyboardInterrupt gracefully without crashing."""
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL()
    repl.session._messages.append({"role": "assistant", "content": "Test text to speak"})

    with patch("audiobench.chat.voice_picker.ensure_voice_downloaded", return_value=True), \
         patch("audiobench.tts.engine.PiperTTSEngine.play", side_effect=KeyboardInterrupt), \
         patch("audiobench.tts.engine.PiperTTSEngine.stop") as mock_stop:
        # Should not raise KeyboardInterrupt
        res = repl._handle_slash_command("/speak -f")
        assert res is False
        mock_stop.assert_called_once()


def test_chat_repl_speak_background_daemon():
    """Test /speak delegates to daemon playback in background mode."""
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL()
    repl.session._messages.append({"role": "assistant", "content": "Test text to speak"})
    repl._daemon = MagicMock()

    with patch("audiobench.chat.voice_picker.ensure_voice_downloaded", return_value=True), \
         patch("audiobench.tts.engine.PiperTTSEngine.synthesize_with_timestamps", return_value=("/tmp/test.wav", [{"start": 0.0, "end": 1.0, "text": "Test"}])):
        res = repl._handle_slash_command("/speak")
        assert res is False
        repl._daemon.playback_play.assert_called_once_with(
            "/tmp/test.wav",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test"}],
            auto_resume=True,
        )


def test_memory_speak_text_keyboard_interrupt():
    """Test _speak_text in memory_cmd handles KeyboardInterrupt cleanly."""
    from audiobench.cli.commands.memory_cmd import _speak_text

    fake_console = MagicMock()
    with patch("audiobench.chat.voice_picker.ensure_voice_downloaded", return_value=True), \
         patch("audiobench.tts.engine.PiperTTSEngine.play", side_effect=KeyboardInterrupt), \
         patch("audiobench.tts.engine.PiperTTSEngine.stop") as mock_stop:
        # Should not raise KeyboardInterrupt
        _speak_text("Sample answer from AI synthesis", console=fake_console)
        mock_stop.assert_called_once()


def test_get_voice_download_size_str():
    """Test get_voice_download_size_str returns accurate size string."""
    from audiobench.chat.voice_picker import get_voice_download_size_str

    # Test heuristic fallback when network is mocked or disconnected
    with patch("urllib.request.urlopen", side_effect=Exception("network down")):
        high_size = get_voice_download_size_str("en_US-lessac-high")
        assert "109" in high_size or "110" in high_size

        med_size = get_voice_download_size_str("en_US-amy-medium")
        assert "60" in med_size

        low_size = get_voice_download_size_str("en_US-danny-low")
        assert "18" in low_size or "60" in low_size


def test_piper_engine_voice_caching():
    """Test that PiperTTSEngine caches loaded voice models in _LOADED_VOICES."""
    from pathlib import Path
    from audiobench.tts.engine import PiperTTSEngine, _LOADED_VOICES

    _LOADED_VOICES.clear()
    engine = PiperTTSEngine()

    mock_piper_voice = MagicMock()
    with patch("pathlib.Path.exists", return_value=True), \
         patch("piper.PiperVoice.load", return_value=mock_piper_voice) as mock_load:
        # First load parses from disk
        engine._load_voice("en_US-amy-medium")
        assert mock_load.call_count == 1
        assert "en_US-amy-medium" in _LOADED_VOICES

        # Second load reuses cached model
        engine._load_voice("en_US-amy-medium")
        assert mock_load.call_count == 1  # Not loaded again


def test_piper_engine_play_runs_without_name_errors():
    """Verify PiperTTSEngine.play executes cleanly without NameError for time or numpy."""
    from audiobench.tts.engine import PiperTTSEngine

    engine = PiperTTSEngine()
    mock_voice = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.sample_rate = 22050
    mock_chunk.audio_int16_bytes = b"\x00\x00" * 100
    mock_voice.synthesize.return_value = [mock_chunk]

    mock_stream = MagicMock()
    mock_stream.__enter__.return_value = mock_stream
    mock_stream.active = False

    with patch.object(engine, "_load_voice"), \
         patch.object(engine, "_voice", mock_voice), \
         patch("sounddevice.OutputStream", return_value=mock_stream):
        engine.play("Short speech test", voice="en_US-lessac-high")


def test_split_speech_sentences():
    """Verify sentence boundary segmentation preserves punctuation and handles multiline."""
    from audiobench.tts.engine import split_speech_sentences

    text = "First sentence. Second sentence! Third sentence? Fourth one."
    parts = split_speech_sentences(text)
    assert parts == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
        "Fourth one.",
    ]

    multiline = "Line one.\nLine two without period\nLine three."
    parts_multi = split_speech_sentences(multiline)
    assert parts_multi == ["Line one.", "Line two without period", "Line three."]
    assert split_speech_sentences("") == []


def test_synthesize_with_timestamps(tmp_path):
    """Verify synthesize_with_timestamps calculates exact durations and outputs valid WAV."""
    import numpy as np
    from audiobench.tts.engine import PiperTTSEngine

    engine = PiperTTSEngine()
    mock_voice = MagicMock()

    # 22050 samples at 22050Hz = exactly 1.0s
    def fake_synthesize(sentence):
        chunk = MagicMock()
        chunk.sample_rate = 22050
        chunk.audio_int16_array = np.zeros(22050, dtype=np.int16)
        chunk.audio_int16_bytes = chunk.audio_int16_array.tobytes()
        return [chunk]

    mock_voice.synthesize.side_effect = fake_synthesize

    target_wav = tmp_path / "test_synth.wav"
    with patch.object(engine, "_load_voice"), patch.object(engine, "_voice", mock_voice):
        out_path, segs = engine.synthesize_with_timestamps(
            "Hello world. How are you today?",
            output_path=target_wav,
        )

    assert out_path == target_wav
    assert target_wav.exists()
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.0
    assert segs[1]["text"] == "How are you today?"
    assert segs[1]["start"] == 1.0
    assert segs[1]["end"] == 2.0




