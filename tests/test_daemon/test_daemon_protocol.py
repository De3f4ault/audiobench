"""
Protocol-level tests for the daemon _dispatch async generator.

These tests exercise _dispatch() directly (no socket), but against the live
streaming protocol introduced in 1A. They replace the old synchronous
`json.loads(_dispatch(raw))` pattern with an asyncio helper that drains the
async generator and returns the final terminal frame.

The daemon does NOT need to be running — _dispatch() is called in-process.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from audiobench.daemon.server import _dispatch

# ---------------------------------------------------------------------------
# Helper: drain an async generator, return all frames
# ---------------------------------------------------------------------------

async def _drain(raw: str) -> list[dict]:
    frames = []
    async for line in _dispatch(raw):
        frames.append(json.loads(line))
    return frames


def _run(raw: str) -> list[dict]:
    """Synchronous helper: run _drain in a fresh event loop."""
    return asyncio.run(_drain(raw))


def _terminal(frames: list[dict]) -> dict:
    """Return the last frame (ok or error terminal)."""
    assert frames, "No frames yielded by _dispatch"
    return frames[-1]


# ---------------------------------------------------------------------------
# 1. Ping contract
# ---------------------------------------------------------------------------

def test_daemon_ping_contract() -> None:
    """_dispatch('ping') yields a terminal ok frame with alive=True."""
    import audiobench.daemon.server as _srv
    with patch("audiobench.daemon.server._get_store") as mock_get_store, \
         patch.object(_srv, "_memory_store", new=object()):  # truthy sentinel
        mock_store = MagicMock()
        mock_store.model_version = "mock-model-v1"
        mock_get_store.return_value = mock_store

        request = {"cmd": "ping", "args": {}, "request_id": "test-req-123"}
        frames = _run(json.dumps(request))
        resp = _terminal(frames)

    assert resp["success"] is True
    assert resp["request_id"] == "test-req-123"
    assert resp["data"]["alive"] is True
    assert resp["data"]["embedding_model_version"] == "mock-model-v1"


# ---------------------------------------------------------------------------
# 2. Unknown command → error frame
# ---------------------------------------------------------------------------

def test_daemon_unknown_command() -> None:
    """_dispatch with unknown cmd yields an error frame (not a crash)."""
    request = {"cmd": "does_not_exist", "args": {}, "request_id": "test-err-404"}
    frames = _run(json.dumps(request))
    resp = _terminal(frames)

    assert resp["success"] is False
    assert resp["request_id"] == "test-err-404"
    # Error is now a structured dict per Standard 6; message contains the verb.
    error = resp.get("error", {})
    if isinstance(error, dict):
        assert "does_not_exist" in error.get("message", "") or \
               "Unknown command" in error.get("message", ""), \
               f"Expected command name in error message, got: {error}"
    else:
        # Fallback: some older frames embed error as a string
        assert "Unknown command" in str(error)


# ---------------------------------------------------------------------------
# 3. Search contract
# ---------------------------------------------------------------------------

def test_daemon_search_contract() -> None:
    """_dispatch('search') calls store.search with correct kwargs and returns results."""
    with patch("audiobench.daemon.server._get_store") as mock_get_store:
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"expression_id": 1, "score": 0.99, "content": "hello",
             "source_type": "transcription"}
        ]
        mock_get_store.return_value = mock_store

        request = {
            "cmd": "search",
            "args": {"query": "hello world", "top_k": 3},
            "request_id": "test-search-1",
        }
        frames = _run(json.dumps(request))
        resp = _terminal(frames)

    assert resp["success"] is True
    assert len(resp["data"]["results"]) == 1
    assert resp["data"]["results"][0]["expression_id"] == 1

    mock_store.search.assert_called_once_with(
        query="hello world",
        top_k=3,
        speaker_filter=None,
        audio_file_id=None,
        work_id=None,
        hyde_document=None,
        use_bm25=True,
        use_dense=True,
        use_colbert=True,
    )


# ---------------------------------------------------------------------------
# 4. Whisper STT Lifecycle and Transcription
# ---------------------------------------------------------------------------

def test_daemon_whisper_acquire_release_lifecycle() -> None:
    """_dispatch handles whisper_acquire and whisper_release with reference counting."""
    import audiobench.daemon.server as srv

    # Reset any state
    srv._stt_sessions.clear()
    srv._warm_whisper_model = None
    srv._warm_whisper_key = None

    mock_model = MagicMock()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        # 1. Acquire session 1 (chat)
        req1 = {
            "cmd": "whisper_acquire",
            "args": {"session_id": "chat_session_1", "model": "base"},
            "request_id": "req-acq-1",
        }
        resp1 = _terminal(_run(json.dumps(req1)))
        assert resp1["success"] is True
        assert resp1["data"]["ok"] is True
        assert resp1["data"]["active_sessions"] == 1
        assert srv._warm_whisper_model is mock_model

        # 2. Acquire session 2 (search) with same model -> shared
        req2 = {
            "cmd": "whisper_acquire",
            "args": {"session_id": "search_session_2", "model": "base"},
            "request_id": "req-acq-2",
        }
        resp2 = _terminal(_run(json.dumps(req2)))
        assert resp2["success"] is True
        assert resp2["data"]["active_sessions"] == 2
        assert srv._warm_whisper_model is mock_model

        # 3. Release session 1 -> model remains warm
        req3 = {
            "cmd": "whisper_release",
            "args": {"session_id": "chat_session_1"},
            "request_id": "req-rel-1",
        }
        resp3 = _terminal(_run(json.dumps(req3)))
        assert resp3["success"] is True
        assert resp3["data"]["active_sessions"] == 1
        assert srv._warm_whisper_model is mock_model

        # 4. Release session 2 -> last session exits, model is purged from RAM
        req4 = {
            "cmd": "whisper_release",
            "args": {"session_id": "search_session_2"},
            "request_id": "req-rel-2",
        }
        resp4 = _terminal(_run(json.dumps(req4)))
        assert resp4["success"] is True
        assert resp4["data"]["active_sessions"] == 0
        assert srv._warm_whisper_model is None


def test_daemon_whisper_transcribe() -> None:
    """_dispatch('whisper_transcribe') transcribes audio file using warm model."""
    import audiobench.daemon.server as srv

    mock_segment = MagicMock()
    mock_segment.text = "Hello from warm daemon Whisper"
    mock_info = MagicMock()
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    srv._warm_whisper_model = mock_model
    srv._warm_whisper_key = "base:cpu:int8"

    with patch("os.path.exists", return_value=True), \
         patch("audiobench.daemon.server._ensure_whisper_loaded", return_value=mock_model):
        req = {
            "cmd": "whisper_transcribe",
            "args": {"file_path": "/tmp/test_voice.wav", "model": "base", "language": "en"},
            "request_id": "req-tx-1",
        }
        resp = _terminal(_run(json.dumps(req)))
        assert resp["success"] is True
        assert resp["data"]["ok"] is True
        assert resp["data"]["text"] == "Hello from warm daemon Whisper"
        assert resp["data"]["language"] == "en"
        mock_model.transcribe.assert_called_once_with("/tmp/test_voice.wav", language="en")
