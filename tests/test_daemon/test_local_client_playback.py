"""Tests verifying LocalRetrievalClient universal playback interface and controls integration."""

from unittest.mock import MagicMock, patch
from rich.console import Console

from audiobench.daemon.local_client import LocalRetrievalClient
from audiobench.playback.controls import handle_playback_command, render_playback_status_line


def test_local_retrieval_client_has_all_playback_methods():
    client = LocalRetrievalClient()
    playback_methods = [
        "playback_play",
        "playback_sync_session",
        "playback_seek",
        "playback_seek_relative",
        "playback_pause",
        "playback_resume",
        "playback_toggle",
        "playback_speed",
        "playback_stop",
        "playback_status",
        "playback_context",
    ]
    for method in playback_methods:
        assert hasattr(client, method), f"LocalRetrievalClient is missing {method}"
        assert callable(getattr(client, method))


def test_local_retrieval_client_delegates_to_playback_service():
    client = LocalRetrievalClient()
    mock_svc = MagicMock()

    with patch.object(client, "_get_playback_service", return_value=mock_svc):
        mock_svc.status.return_value = {"playing": True, "position": 12.0}
        assert client.playback_status() == {"playing": True, "position": 12.0}
        mock_svc.status.assert_called_once()

        mock_svc.play.return_value = {"playing": True}
        client.playback_play("/tmp/test.mp3", start_pos=5.0, speed=1.2, audio_file_id=1, transcription_id=2)
        mock_svc.play.assert_called_once_with(
            "/tmp/test.mp3", start_pos=5.0, speed=1.2, audio_file_id=1, transcription_id=2
        )

        client.playback_seek(30.0)
        mock_svc.seek.assert_called_once_with(30.0)

        client.playback_seek_relative(10.0)
        mock_svc.seek_relative.assert_called_once_with(10.0)

        client.playback_pause()
        mock_svc.pause.assert_called_once()

        client.playback_resume()
        mock_svc.resume.assert_called_once()

        client.playback_toggle()
        mock_svc.toggle.assert_called_once()

        client.playback_speed(1.5)
        mock_svc.set_speed.assert_called_once_with(1.5)

        client.playback_stop()
        mock_svc.stop.assert_called_once()

        client.playback_context(position=10.0, window=2)
        mock_svc.get_context_window.assert_called_once_with(position=10.0, window=2)

        client.playback_sync_session("/tmp/test.mp3", transcription_id=2, position=15.0, audio_file_id=1)
        mock_svc.sync_session.assert_called_once_with(
            "/tmp/test.mp3", transcription_id=2, position=15.0, audio_file_id=1
        )


def test_handle_playback_command_with_local_client_jump_citation():
    """Verify that jumping to a citation does not crash with AttributeError when using LocalRetrievalClient."""
    client = LocalRetrievalClient()
    mock_svc = MagicMock()
    mock_svc.status.return_value = {"playing": False, "position": 0.0}

    with patch.object(client, "_get_playback_service", return_value=mock_svc):
        citations = [("00:11", 11.0), ("00:33", 33.0)]
        default_record = {"id": 1, "file_path": "/tmp/simd.mp3", "audio_file_id": 10}
        console = Console(record=True)

        # Trigger /1 (jumping to first citation)
        handled = handle_playback_command(
            "/1",
            "",
            client,
            target_console=console,
            active_citations=citations,
            default_record=default_record,
        )
        assert handled is True
        mock_svc.status.assert_called_once()
        mock_svc.play.assert_called_once_with(
            "/tmp/simd.mp3", start_pos=11.0, speed=1.0, audio_file_id=10, transcription_id=1
        )
        out = console.export_text()
        assert "Playing [1/2] from 00:11" in out


def test_render_playback_status_line_with_local_client():
    client = LocalRetrievalClient()
    mock_svc = MagicMock()
    mock_svc.status.return_value = {
        "playing": True,
        "paused": False,
        "position": 11.0,
        "duration": 600.0,
        "speed": 1.0,
    }

    with patch.object(client, "_get_playback_service", return_value=mock_svc):
        line = render_playback_status_line(client, citations=[("00:11", 11.0)])
        assert line is not None
        assert "00:11/10:00" in line


def test_handle_playback_command_missing_client_graceful():
    console = Console(record=True)
    handled = handle_playback_command("/play", "", None, target_console=console)
    assert handled is True
    out = console.export_text()
    assert "Playback error: Playback is not available" in out


def test_server_playback_handlers_handle_none_values():
    """Verify daemon server handlers safely coerce None values without TypeError."""
    from audiobench.daemon.server import (
        _handle_playback_play,
        _handle_playback_seek,
        _handle_playback_seek_relative,
        _handle_playback_speed,
        _handle_playback_sync_session,
    )

    mock_svc = MagicMock()
    with patch("audiobench.daemon.server._get_playback_service", return_value=mock_svc):
        # Play with None start_pos and speed
        _handle_playback_play({"file_path": "/tmp/test.mp3", "start_pos": None, "speed": None})
        mock_svc.play.assert_called_once_with(
            "/tmp/test.mp3", start_pos=0.0, speed=1.0, audio_file_id=None, transcription_id=None
        )

        # Sync session with None position
        _handle_playback_sync_session({"file_path": "/tmp/test.mp3", "position": None})
        mock_svc.sync_session.assert_called_once_with(
            "/tmp/test.mp3", transcription_id=None, position=0.0, audio_file_id=None
        )

        # Seek with None position
        _handle_playback_seek({"position": None})
        mock_svc.seek.assert_called_once_with(0.0)

        # Seek relative with None offset
        _handle_playback_seek_relative({"offset": None})
        mock_svc.seek_relative.assert_called_once_with(0.0)

        # Speed with None speed
        _handle_playback_speed({"speed": None})
        mock_svc.set_speed.assert_called_once_with(1.0)


def test_daemon_client_sanitizes_none_values():
    """Verify DaemonClient serializes None positions and speeds as safe floats."""
    from audiobench.daemon.client import DaemonClient

    client = DaemonClient()
    with patch.object(client, "_send", return_value={"status": "ok"}) as mock_send:
        client.playback_play("/tmp/test.mp3", start_pos=None, speed=None)
        mock_send.assert_called_with("playback_play", {
            "file_path": "/tmp/test.mp3",
            "start_pos": 0.0,
            "speed": 1.0,
        })

        client.playback_sync_session("/tmp/test.mp3", position=None)
        mock_send.assert_called_with("playback_sync_session", {
            "file_path": "/tmp/test.mp3",
            "position": 0.0,
        })

        client.playback_seek(None)
        mock_send.assert_called_with("playback_seek", {"position": 0.0})

        client.playback_seek_relative(None)
        mock_send.assert_called_with("playback_seek_relative", {"offset": 0.0})

        client.playback_speed(None)
        mock_send.assert_called_with("playback_speed", {"speed": 1.0})

