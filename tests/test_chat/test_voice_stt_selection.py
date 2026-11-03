"""Unit tests for STT Provider & Model Selection (Local vs. Cloud), caching, and REPL commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from audiobench.core.settings import AudioBenchSettings, get_settings
from audiobench.chat.voice_session import (
    VoiceChatSession,
    get_whisper_model,
    _LOADED_WHISPER_MODELS,
)
from audiobench.chat.voice_picker import get_voice_download_size_str


@pytest.fixture(autouse=True)
def _isolate_db(test_db):
    """Ensure all tests run against isolated test_db."""
    pass


def test_stt_settings_defaults_and_validation(monkeypatch, tmp_path):
    """Test STT settings defaults (auto, base) and persistence."""
    settings = AudioBenchSettings(stt_provider="auto", stt_model="base")
    assert settings.stt_provider == "auto"
    assert settings.stt_model == "base"

    settings.stt_provider = "local"
    settings.stt_model = "turbo"
    assert settings.stt_provider == "local"
    assert settings.stt_model == "turbo"


def test_voice_download_sizes_accurate():
    """Verify that voice download sizes match actual huggingface distributions."""
    assert get_voice_download_size_str("en_US-ryan-high") == "~115MB"
    assert get_voice_download_size_str("es_ES-carlfm-x_low") == "~27MB"
    assert get_voice_download_size_str("en_US-amy-medium") == "~60MB"
    assert get_voice_download_size_str("fr_FR-siwis-medium") == "~60MB"


def test_whisper_model_caching():
    """Verify get_whisper_model caches loaded models in memory."""
    _LOADED_WHISPER_MODELS.clear()

    with patch("faster_whisper.WhisperModel") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        m1 = get_whisper_model("base")
        assert m1 is instance
        assert "base:cpu:int8" in _LOADED_WHISPER_MODELS
        assert mock_cls.call_count == 1

        # Second call should retrieve from cache without re-instantiating
        m2 = get_whisper_model("base")
        assert m2 is instance
        assert mock_cls.call_count == 1


def test_transcribe_local(monkeypatch):
    """Verify _transcribe_local accurately transcribes raw PCM using cached Whisper."""
    session = VoiceChatSession(conversation_id=1, settings=get_settings())

    fake_segment = MagicMock()
    fake_segment.text = "Hello SIMD instruction set"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([fake_segment], MagicMock())

    with patch("audiobench.chat.voice_session.get_whisper_model", return_value=mock_model):
        fake_pcm = b"\x00" * 32000  # 1s of 16kHz 16-bit audio
        text = session._transcribe_local(fake_pcm)
        assert text == "Hello SIMD instruction set"


def test_transcribe_cloud(monkeypatch):
    """Verify _transcribe_cloud transcribes using gemini-2.5-flash with generate_content."""
    monkeypatch.setattr("audiobench.chat.voice_session.is_internet_connected", lambda timeout=0.8: True)
    settings = get_settings()
    settings.gemini_api_key = "test-api-key"
    session = VoiceChatSession(conversation_id=1, settings=settings)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Vectorization with AVX-512"
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        fake_pcm = b"\x00" * 32000
        text = session._transcribe_cloud(fake_pcm)
        assert text == "Vectorization with AVX-512"
        mock_client.models.generate_content.assert_called_once()
        args, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-2.5-flash"


def test_record_single_turn_provider_routing(monkeypatch):
    """Verify record_single_turn routes correctly according to stt_provider."""
    settings = get_settings()
    session = VoiceChatSession(conversation_id=1, settings=settings, record_audio=False)

    fake_pcm = b"\x00" * 32000
    monkeypatch.setattr(session, "_capture_raw_audio", lambda *args, **kw: (fake_pcm, 1.0))

    # 1. Local routing
    settings.stt_provider = "local"
    with patch.object(session, "_transcribe_local", return_value="Local result") as mock_loc, \
         patch.object(session, "_transcribe_cloud") as mock_cld:
        text, _, _ = session.record_single_turn()
        assert text == "Local result"
        mock_loc.assert_called_once()
        mock_cld.assert_not_called()

    # 2. Cloud routing
    settings.stt_provider = "cloud"
    settings.gemini_api_key = "test-key"
    with patch.object(session, "_transcribe_local") as mock_loc, \
         patch.object(session, "_transcribe_cloud", return_value="Cloud result") as mock_cld:
        text, _, _ = session.record_single_turn()
        assert text == "Cloud result"
        mock_cld.assert_called_once()
        mock_loc.assert_not_called()

    # 3. Cloud fallback to local on failure
    settings.stt_provider = "cloud"
    with patch.object(session, "_transcribe_local", return_value="Fallback local") as mock_loc, \
         patch.object(session, "_transcribe_cloud", side_effect=RuntimeError("Network down")) as mock_cld:
        text, _, _ = session.record_single_turn()
        assert text == "Fallback local"
        mock_cld.assert_called_once()
        mock_loc.assert_called_once()


def test_chat_repl_voice_provider_and_model_commands(monkeypatch):
    """Test /voice provider and /voice model subcommands in ChatREPL."""
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_session import ChatSession

    repl = ChatREPL()
    repl.session = ChatSession(model="gemini")

    # /voice provider local
    repl._handle_slash_command("/voice provider local")
    assert repl.settings.stt_provider == "local"

    # /voice provider cloud
    repl._handle_slash_command("/voice provider cloud")
    assert repl.settings.stt_provider == "cloud"

    # /voice model turbo
    repl._handle_slash_command("/voice model turbo")
    assert repl.settings.stt_model == "turbo"

    # /voice model tiny
    repl._handle_slash_command("/voice model tiny")
    assert repl.settings.stt_model == "tiny"


def test_memory_cmd_voice_provider_and_model(monkeypatch):
    """Test /voice provider and /voice model in memory command handler."""
    from audiobench.cli.commands.memory_cmd import _parse_slash_command, SearchSessionState

    state = SearchSessionState()
    res = _parse_slash_command("/voice provider local", state)
    assert res is not None
    assert "STT Provider updated to:" in res
    assert get_settings().stt_provider == "local"

    res = _parse_slash_command("/voice model small", state)
    assert res is not None
    assert "Whisper STT model updated to:" in res
    assert get_settings().stt_model == "small"


def test_record_single_turn_visual_feedback_and_short_audio(monkeypatch):
    """Test that immediate feedback is printed upon capture completion and <0.1s audio is ignored."""
    from rich.console import Console
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True)
    settings = get_settings()
    session = VoiceChatSession(conversation_id=1, console=console, settings=settings, record_audio=False)

    # 1. Very short audio (< 3200 bytes) -> "No audio captured."
    monkeypatch.setattr(session, "_capture_raw_audio", lambda *args, **kw: (b"\x00" * 500, 0.05))
    text, _, duration = session.record_single_turn()
    assert text == ""
    assert "No audio captured" in buf.getvalue()

    # 2. Valid audio -> Transcribing feedback printed immediately
    buf.seek(0)
    buf.truncate(0)
    settings.stt_provider = "local"
    fake_pcm = b"\x00" * 32000
    monkeypatch.setattr(session, "_capture_raw_audio", lambda *args, **kw: (fake_pcm, 1.0))
    monkeypatch.setattr(session, "_transcribe_local", lambda pcm: "Transcribed result")

    text, _, duration = session.record_single_turn()
    assert text == "Transcribed result"
    output = buf.getvalue()
    assert "Recorded 1.0s. Transcribing with local Whisper" in output


def test_transcribe_cloud_offline_immediate_error(monkeypatch):
    """Verify _transcribe_cloud immediately raises RuntimeError when offline without socket hanging."""
    settings = get_settings()
    settings.gemini_api_key = "test-key"
    session = VoiceChatSession(conversation_id=1, settings=settings)

    # Mock is_internet_connected to False
    monkeypatch.setattr("audiobench.chat.voice_session.is_internet_connected", lambda timeout=0.8: False)

    fake_pcm = b"\x00" * 32000
    with pytest.raises(RuntimeError, match="Internet connection is offline"):
        session._transcribe_cloud(fake_pcm)


def test_chat_repl_voice_keyboard_interrupt_handled(monkeypatch):
    """Verify ChatREPL handles KeyboardInterrupt during /voice cleanly."""
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_session import ChatSession

    repl = ChatREPL()
    repl.session = ChatSession(model="gemini")

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    with patch("audiobench.chat.voice_session.VoiceChatSession.record_single_turn", side_effect=raise_interrupt):
        result = repl._handle_slash_command("/voice")
        assert result is False
