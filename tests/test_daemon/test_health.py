from unittest.mock import patch

import pytest

from audiobench.daemon.client import DaemonClient
from audiobench.daemon.server import _handle_ping


@pytest.fixture
def mock_daemon_client(tmp_path):
    return DaemonClient(socket_path=tmp_path / "mock.sock")

def test_daemon_is_healthy_returns_full_dict(mock_daemon_client):
    import json

    mock_response = {
        "status": "ok",
        "data": {
            "alive": True,
            "uptime_seconds": 12.3,
            "embedding_model_version": "v1",
            "memory_mb": 150.5,
            "queue_depth": 5,
            "models": {"embedding": "v1"}
        }
    }

    with patch("socket.socket") as mock_socket:
        instance = mock_socket.return_value
        instance.recv.side_effect = [(json.dumps(mock_response) + "\n").encode("utf-8")]

        result = mock_daemon_client.daemon_is_healthy()

        assert result is not None
        assert result["alive"] is True
        assert result["memory_mb"] == 150.5
        assert result["queue_depth"] == 5

def test_daemon_is_healthy_returns_none_when_unreachable(mock_daemon_client):
    with patch("socket.socket") as mock_socket:
        instance = mock_socket.return_value
        instance.connect.side_effect = ConnectionRefusedError("connection refused")

        result = mock_daemon_client.daemon_is_healthy()
        assert result is None

def test_memory_mb_is_positive_integer_in_server():
    with patch("audiobench.daemon.server._get_store") as mock_store:
        mock_store.return_value.model_version = "mock-v1"

        result = _handle_ping({})

        assert "memory_mb" in result
        assert result["memory_mb"] > 0
        assert "queue_depth" in result
        assert "models" in result
