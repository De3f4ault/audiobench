"""Unit tests for DaemonPlaybackProxy — Phase 3b daemon-attach mode."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from audiobench.playback.daemon_proxy import DaemonPlaybackProxy


def _make_mock_client(**overrides) -> MagicMock:
    client = MagicMock()
    status = {
        "playing": True,
        "paused": False,
        "state": "playing",
        "position": 10.0,
        "duration": 100.0,
        "speed": 1.0,
        "file": "/path/to/test.mp3",
        "audio_file_id": 1,
        "transcription_id": 42,
        "loop_a": None,
        "loop_b": None,
        "filter": "",
        "last_eof_reason": None,
    }
    status.update(overrides)
    client.playback_status.return_value = status
    client.playback_seek.return_value = status
    client.playback_seek_relative.return_value = status
    client.playback_toggle.return_value = status
    client.playback_pause.return_value = status
    client.playback_resume.return_value = status
    client.playback_speed.return_value = status
    client.playback_loop_segment.return_value = status
    client.playback_clear_loop.return_value = status
    client.playback_set_filter.return_value = status
    client.playback_stop.return_value = status
    return client


class TestDaemonPlaybackProxyStatusAndExtrapolation:
    def test_proxy_initial_status_queries_daemon(self):
        client = _make_mock_client(position=15.0)
        proxy = DaemonPlaybackProxy(client, poll_interval=0.5)
        client.playback_status.assert_called_once()
        st = proxy.status()
        assert st["position"] == 15.0
        assert st["playing"] is True

    def test_proxy_rate_limits_socket_polling(self):
        client = _make_mock_client(position=10.0)
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        assert client.playback_status.call_count == 1

        # 20 consecutive status calls within 50ms
        for _ in range(20):
            proxy.status()
            time.sleep(0.002)

        # Still only 1 IPC call made because poll_interval (1.0s) has not elapsed
        assert client.playback_status.call_count == 1

    def test_proxy_extrapolates_position_while_playing(self):
        client = _make_mock_client(position=10.0, speed=1.0, duration=100.0)
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)

        # Simulate time passing by adjusting _last_sync_time
        proxy._last_sync_time = time.monotonic() - 0.25

        st = proxy.status()
        # Extrapolated ~10.0 + 0.25 = 10.25
        assert 10.20 <= st["position"] <= 10.35

    def test_proxy_does_not_extrapolate_when_paused(self):
        client = _make_mock_client(position=10.0, paused=True)
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)

        proxy._last_sync_time = time.monotonic() - 0.5
        st = proxy.status()
        assert st["position"] == 10.0

    def test_proxy_extrapolation_clamps_to_duration(self):
        client = _make_mock_client(position=99.5, speed=1.0, duration=100.0)
        proxy = DaemonPlaybackProxy(client, poll_interval=5.0)

        proxy._last_sync_time = time.monotonic() - 2.0  # 2s later would be 101.5
        st = proxy.status()
        assert st["position"] == 100.0


class TestDaemonPlaybackProxyActionDelegation:
    def test_seek_forces_immediate_resync(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        assert client.playback_status.call_count == 1

        proxy.seek(45.0)
        client.playback_seek.assert_called_once_with(45.0)
        # Resync status called immediately
        assert client.playback_status.call_count == 2

    def test_seek_relative_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.seek_relative(-5.0)
        client.playback_seek_relative.assert_called_once_with(-5.0)

    def test_toggle_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.toggle()
        client.playback_toggle.assert_called_once()

    def test_set_speed_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.set_speed(1.5)
        client.playback_speed.assert_called_once_with(1.5)

    def test_loop_segment_and_clear_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.loop_segment(12.0, 24.0)
        client.playback_loop_segment.assert_called_once_with(12.0, 24.0)

        proxy.clear_loop()
        client.playback_clear_loop.assert_called_once()

    def test_set_filter_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.set_filter("highpass=f=200")
        client.playback_set_filter.assert_called_once_with("highpass=f=200")

    def test_stop_delegation(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=1.0)
        proxy.stop()
        client.playback_stop.assert_called_once()


class TestDaemonPlaybackProxyEofHandling:
    def test_eof_listener_fires_when_last_eof_reason_is_eof(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=0.1)

        events = []
        unregister = proxy.register_eof_listener(lambda r: events.append(r))

        # Status update arrives with natural EOF
        client.playback_status.return_value = {
            "playing": False,
            "state": "stopped",
            "last_eof_reason": "eof",
        }

        # Force sync
        proxy.seek(0.0)

        assert events == ["eof"]

        # Ensure subsequent status checks do NOT re-fire the same EOF event
        proxy.status()
        assert events == ["eof"]

        unregister()

    def test_eof_listener_does_not_fire_on_stop_reason(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=0.1)

        events = []
        proxy.register_eof_listener(lambda r: events.append(r))

        # Status update arrives with user-initiated stop
        client.playback_status.return_value = {
            "playing": False,
            "state": "stopped",
            "last_eof_reason": "stop",
        }

        proxy.stop()
        assert events == []

    def test_unregister_eof_listener_removes_callback(self):
        client = _make_mock_client()
        proxy = DaemonPlaybackProxy(client, poll_interval=0.1)

        events = []
        unregister = proxy.register_eof_listener(lambda r: events.append(r))
        unregister()
        # Idempotent unregister call
        unregister()

        client.playback_status.return_value = {
            "playing": False,
            "state": "stopped",
            "last_eof_reason": "eof",
        }
        proxy.seek(0.0)
        assert events == []
