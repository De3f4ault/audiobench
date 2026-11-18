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
        "state": "stopped",
        "position": 0.0,
        "duration": 0.0,
        "speed": 1.0,
        "file": None,
        "audio_file_id": None,
        "transcription_id": None,
        "loop_a": None,
        "loop_b": None,
        "filter": "",
        "last_eof_reason": None,
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
        "state": "stopped",
        "position": 0.0,
        "duration": 0.0,
        "speed": 1.0,
        "file": None,
        "audio_file_id": None,
        "transcription_id": None,
        "loop_a": None,
        "loop_b": None,
        "filter": "",
        "last_eof_reason": "stop",
    }
    # Phase 2A: stop() sends 'stop' command (daemon stays alive), not quit()
    mock_mpv.stop.assert_called_once()
    mock_mpv.quit.assert_not_called()
    # _mpv handle is preserved — daemon remains ready for next play()
    assert svc._mpv is mock_mpv


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


@patch("audiobench.playback.service.MpvController")
def test_playback_service_custom_segments_and_context_window(mock_mpv_cls):
    """Test PlaybackService with in-memory speech segments (transcription_id is None)."""
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (3.5, 1.0, False)
    mock_mpv.get_duration.return_value = 10.0
    mock_mpv_cls.return_value = mock_mpv

    custom_segs = [
        {"start": 0.0, "end": 2.0, "text": "First sentence."},
        {"start": 2.0, "end": 5.0, "text": "Second sentence."},
        {"start": 5.0, "end": 8.0, "text": "Third sentence."},
    ]

    svc = PlaybackService()
    st = svc.play(
        "/path/to/speech.wav",
        start_pos=0.0,
        segments=custom_segs,
    )

    assert st["playing"] is True
    assert st["transcription_id"] is None
    ctx = svc.get_context_window(position=3.5, window=1)
    assert len(ctx["segments"]) == 3
    assert ctx["current_idx"] == 1
    assert ctx["segments"][1]["text"] == "Second sentence."


@patch.object(PlaybackService, "_write_resume_position")
@patch("audiobench.playback.service.MpvController")
def test_playback_service_auto_resume_flow(mock_mpv_cls, mock_write_pos):
    """Test auto-resume stashes track and restores it on EOF."""
    # Phase 2A: one persistent mpv controller throughout
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (42.0, 1.25, False)
    mock_mpv.get_position.return_value = 42.0
    mock_mpv.get_duration.return_value = 200.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    # Play original track
    svc.play("/path/to/track1.mp3", start_pos=10.0, speed=1.25, audio_file_id=7, transcription_id=99)
    assert svc._current_file.endswith("track1.mp3")

    # Speak interrupts track1 with auto_resume=True
    speech_segs = [{"start": 0.0, "end": 3.0, "text": "Speech prompt"}]
    svc.play("/path/to/speech.wav", segments=speech_segs, auto_resume=True)

    assert svc._interrupted_playback is not None
    assert svc._interrupted_playback["file_path"].endswith("track1.mp3")
    assert svc._interrupted_playback["start_pos"] == 42.0
    assert svc._interrupted_playback["speed"] == 1.25
    assert svc._interrupted_playback["transcription_id"] == 99

    # Simulate speech finishing: Phase 2A uses _on_eof("eof") from reader thread
    mock_mpv.get_playback_state.return_value = (3.0, 1.0, False)
    mock_mpv.get_position.return_value = 3.0
    svc._on_eof("eof")

    # Track 1 should be resumed
    assert svc._interrupted_playback is None
    assert svc._current_file.endswith("track1.mp3")
    assert svc._transcription_id == 99


@patch("audiobench.playback.service.MpvController")
def test_playback_service_stop_clears_interrupted(mock_mpv_cls):
    """Test that stop() clears any stashed interrupted track so it does not auto-resume."""
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (10.0, 1.0, False)
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc.play("/path/to/track1.mp3", start_pos=0.0)
    svc.play("/path/to/speech.wav", auto_resume=True)
    assert svc._interrupted_playback is not None

    svc.stop()
    assert svc._interrupted_playback is None


@patch.object(PlaybackService, "_write_resume_position")
@patch("audiobench.playback.service.MpvController")
def test_stop_eof_event_does_not_trigger_auto_resume(mock_mpv_cls, mock_write_pos):
    """stop() sends mpv 'stop' which fires end-file(reason='stop').
    _on_eof('stop') must NOT trigger auto-resume or advance to next track.
    """
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (10.0, 1.0, False)
    mock_mpv.get_position.return_value = 10.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc.play("/path/to/track1.mp3", start_pos=0.0)
    svc.play("/path/to/speech.wav", auto_resume=True)
    assert svc._interrupted_playback is not None

    # User calls stop() — then mpv fires end-file(reason='stop')
    svc.stop()
    svc._on_eof("stop")

    # Auto-resume must NOT have fired: interrupted_playback cleared by stop(),
    # _on_eof("stop") returns early without restoring it or calling play()
    assert svc._interrupted_playback is None
    assert svc._current_file is None   # stop() cleared this
    # play() was called twice above, stop() adds 0 additional play calls
    total_load_calls = mock_mpv.load_file.call_count
    svc._on_eof("stop")   # fire again — still nothing
    assert mock_mpv.load_file.call_count == total_load_calls


@patch.object(PlaybackService, "_schedule_pos_timer")
@patch("audiobench.playback.service.MpvController")
def test_resume_rearms_position_timer(mock_mpv_cls, mock_schedule):
    """resume() must re-arm the 30s position timer when transcription_id is set.
    Without this, a pause of >30s would leave no timer running after resume.
    """
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.is_paused.return_value = True   # currently paused
    mock_mpv.get_playback_state.return_value = (30.0, 1.0, True)
    mock_mpv.get_duration.return_value = 100.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc._mpv = mock_mpv
    svc._current_file = "/path/to/track.mp3"
    svc._transcription_id = 42

    mock_schedule.reset_mock()
    svc.resume()

    mock_schedule.assert_called_once()


@patch.object(PlaybackService, "_write_resume_position")
@patch("audiobench.playback.service.MpvController")
def test_stop_followed_by_trailing_eof_keeps_stop_reason(mock_mpv_cls, mock_write_pos):
    """Verify that stop() followed by a trailing EOF event from mpv reader thread
    does NOT overwrite last_eof_reason='stop' back to 'eof'.
    """
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (10.0, 1.0, False)
    mock_mpv.get_position.return_value = 10.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc.play("/path/to/track1.mp3", start_pos=0.0)
    assert svc.status()["last_eof_reason"] is None

    # User calls stop()
    st = svc.stop()
    assert st["last_eof_reason"] == "stop"
    assert svc.status()["last_eof_reason"] == "stop"

    # Trailing EOF event arrives from reader thread
    svc._on_eof("eof")

    # Reason MUST remain 'stop', NOT 'eof'
    assert svc.status()["last_eof_reason"] == "stop"
    assert svc.status()["playing"] is False
    assert svc.status()["state"] == "stopped"


@patch.object(PlaybackService, "_write_resume_position")
@patch("audiobench.playback.service.MpvController")
def test_playback_service_natural_eof_sets_eof_reason(mock_mpv_cls, mock_write_pos):
    """Natural completion of track sets last_eof_reason='eof' and state='stopped'."""
    mock_mpv = MagicMock()
    mock_mpv.is_running.return_value = True
    mock_mpv.get_playback_state.return_value = (100.0, 1.0, False)
    mock_mpv.get_position.return_value = 100.0
    mock_mpv_cls.return_value = mock_mpv

    svc = PlaybackService()
    svc.play("/path/to/track1.mp3", start_pos=0.0)
    assert svc.status()["last_eof_reason"] is None
    assert svc.status()["playing"] is True

    # Natural EOF arrives
    svc._on_eof("eof")

    st = svc.status()
    assert st["last_eof_reason"] == "eof"
    assert st["playing"] is False
    assert st["state"] == "stopped"
