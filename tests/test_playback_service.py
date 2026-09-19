"""Unit tests for PlaybackService state and context window logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from audiobench.playback.service import PlaybackService


def test_playback_service_initial_state():
    svc = PlaybackService()
    st = svc.status()
    assert st == {
        "playing": False,
        "paused": False,
        "position": 0.0,
        "duration": 0.0,
        "speed": 1.0,
        "file": None,
        "audio_file_id": None,
        "transcription_id": None,
    }
    assert svc.get_segment_at(10.0) is None
    ctx = svc.get_context_window(10.0)
    assert ctx["segments"] == []


@patch.object(PlaybackService, "_load_segments_from_db")
@patch("audiobench.playback.service.MpvController")
def test_playback_service_play_and_status(mock_mpv_cls, mock_load_segs):
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (15.5, 1.0, False)
    mock_mpv.get_duration.return_value = 100.0
    mock_mpv_cls.return_value = mock_mpv

    mock_load_segs.return_value = [
        {"start": 0.0, "end": 10.0, "text": "Intro", "speaker": "A"},
        {"start": 10.0, "end": 20.0, "text": "Discussion", "speaker": "B"},
        {"start": 20.0, "end": 30.0, "text": "Conclusion", "speaker": "A"},
    ]

    svc = PlaybackService()
    st = svc.play(
        "/path/to/test.mp3",
        start_pos=5.0,
        speed=1.0,
        audio_file_id=1,
        transcription_id=42,
    )

    assert st["playing"] is True
    assert st["position"] == 15.5
    assert st["duration"] == 100.0
    assert st["transcription_id"] == 42
    assert st["audio_file_id"] == 1

    # Test get_segment_at
    seg = svc.get_segment_at(15.0)
    assert seg is not None
    assert seg["text"] == "Discussion"

    # Test get_context_window
    ctx = svc.get_context_window(15.0, window=1)
    assert len(ctx["segments"]) == 3
    assert ctx["current_idx"] == 1
    assert ctx["segments"][1]["text"] == "Discussion"


@patch("audiobench.playback.service.MpvController")
def test_playback_service_controls(mock_mpv_cls):
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.is_paused.return_value = False
    mock_mpv.get_playback_state.return_value = (50.0, 1.0, False)
    mock_mpv.get_duration.return_value = 100.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc._mpv = mock_mpv
    svc._current_file = "/path/test.mp3"

    # Seek
    svc.seek(60.0)
    mock_mpv.seek_absolute.assert_called_with(60.0)

    # Seek relative
    svc.seek_relative(-10.0)
    mock_mpv.seek.assert_called_with(-10.0)

    # Pause
    svc.pause()
    mock_mpv.toggle_pause.assert_called_once()

    # Speed
    svc.set_speed(1.5)
    mock_mpv.set_speed.assert_called_with(1.5)

    # Stop (returns uniform schema with zeroed fields)
    st = svc.stop()
    assert st == {
        "playing": False,
        "paused": False,
        "position": 0.0,
        "duration": 0.0,
        "speed": 1.0,
        "file": None,
        "audio_file_id": None,
        "transcription_id": None,
    }
    mock_mpv.quit.assert_called_once()
    assert svc._mpv is None


@patch.object(PlaybackService, "_load_segments_from_db")
def test_playback_service_sync_session(mock_load_segs):
    mock_load_segs.return_value = [
        {"start": 0.0, "end": 10.0, "text": "Segment 1", "speaker": "A"},
        {"start": 10.0, "end": 20.0, "text": "Segment 2", "speaker": "B"},
    ]

    svc = PlaybackService()
    st = svc.sync_session(
        "/path/to/external.mp3",
        transcription_id=101,
        position=12.5,
        audio_file_id=5,
    )

    assert st["playing"] is True
    assert st["position"] == 12.5
    assert st["audio_file_id"] == 5
    assert st["transcription_id"] == 101

    # Status call should report external session
    status = svc.status()
    assert status["playing"] is True
    assert status["position"] == 12.5

    # Context window should draw from synced position
    ctx = svc.get_context_window()
    assert len(ctx["segments"]) == 2
    assert ctx["position"] == 12.5
    assert ctx["current_idx"] == 1
