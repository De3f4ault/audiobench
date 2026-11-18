"""
Tests for 1D — Process Registry.

Three invariants:
1. client.register_process() sends the correct socket command and the daemon
   handler writes the process to managed_processes via upsert_process().
2. client.unregister_process() sends the correct command; handler marks
   the process stopped.
3. @register_with_daemon decorator is a no-op (no exception) when the daemon
   is unreachable — the decorated function still executes normally.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. register_process writes to managed_processes via the socket handler
# ---------------------------------------------------------------------------

def test_handle_register_process_calls_upsert(test_db):
    """
    Calling _handle_register_process({"name": "transcribe", "pid": 99})
    must invoke upsert_process("transcribe", "running", pid=99).
    """
    from audiobench.daemon.server import _handle_register_process

    with patch("audiobench.daemon.server.upsert_process") as mock_upsert:
        result = _handle_register_process({"name": "transcribe", "pid": 99})

    mock_upsert.assert_called_once_with("transcribe", "running", pid=99)
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# 2. unregister_process marks the process stopped
# ---------------------------------------------------------------------------

def test_handle_unregister_process_marks_stopped(test_db):
    """
    Calling _handle_unregister_process({"name": "transcribe", "exit_code": 0})
    must invoke upsert_process("transcribe", "stopped", last_exit_code=0).
    """
    from audiobench.daemon.server import _handle_unregister_process

    with patch("audiobench.daemon.server.upsert_process") as mock_upsert:
        result = _handle_unregister_process({"name": "transcribe", "exit_code": 0})

    mock_upsert.assert_called_once_with("transcribe", "stopped", last_exit_code=0)
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# 3. DaemonClient.register_process sends the correct command
# ---------------------------------------------------------------------------

def test_client_register_process_sends_correct_command():
    """
    DaemonClient.register_process("transcribe", pid=42) must call _send with
    cmd="register_process" and args={"name": "transcribe", "pid": 42}.
    """
    from audiobench.daemon.client import DaemonClient

    client = DaemonClient.__new__(DaemonClient)
    client._socket_path = MagicMock()

    with patch.object(client, "_send", return_value={"ok": True}) as mock_send:
        client.register_process("transcribe", pid=42)

    mock_send.assert_called_once_with(
        "register_process", {"name": "transcribe", "pid": 42}
    )


def test_client_unregister_process_sends_correct_command():
    """
    DaemonClient.unregister_process("transcribe", exit_code=0) must call _send
    with cmd="unregister_process" and args={"name": "transcribe", "exit_code": 0}.
    """
    from audiobench.daemon.client import DaemonClient

    client = DaemonClient.__new__(DaemonClient)
    client._socket_path = MagicMock()

    with patch.object(client, "_send", return_value={"ok": True}) as mock_send:
        client.unregister_process("transcribe", exit_code=0)

    mock_send.assert_called_once_with(
        "unregister_process", {"name": "transcribe", "exit_code": 0}
    )


# ---------------------------------------------------------------------------
# 4. @register_with_daemon is a no-op when daemon is unreachable
# ---------------------------------------------------------------------------

def test_decorator_noop_when_daemon_unreachable():
    """
    @register_with_daemon wraps a callable. If the daemon is unreachable,
    register_process() raises — the decorator must swallow it and the
    wrapped function must still execute and return its value normally.
    """
    from audiobench.observatory.process_client import register_with_daemon

    sentinel = object()

    @register_with_daemon(name="test_proc")
    def _my_fn():
        return sentinel

    # Patch DaemonClient to raise on register
    with patch(
        "audiobench.observatory.process_client.DaemonClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.register_process.side_effect = ConnectionRefusedError("no daemon")
        instance.unregister_process.side_effect = ConnectionRefusedError("no daemon")

        result = _my_fn()

    assert result is sentinel, "wrapped function must still return its value"


def test_decorator_calls_register_and_unregister_on_success():
    """
    When the daemon IS reachable, @register_with_daemon must call
    register_process before calling the wrapped function, and
    unregister_process after it returns — even if it returns normally.
    """
    from audiobench.observatory.process_client import register_with_daemon

    call_log = []

    @register_with_daemon(name="embed")
    def _work():
        call_log.append("work")
        return 42

    with patch(
        "audiobench.observatory.process_client.DaemonClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.register_process.side_effect = lambda *a, **kw: call_log.append("reg")
        instance.unregister_process.side_effect = lambda *a, **kw: call_log.append("unreg")

        result = _work()

    assert result == 42
    assert call_log == ["reg", "work", "unreg"], \
        f"Expected reg → work → unreg, got {call_log}"
