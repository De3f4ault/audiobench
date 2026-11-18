"""
Tests for 1C — ForwardingSubscriber.

Four invariants (per roadmap):
1. test_events_batched_and_forwarded — when the buffer reaches the batch
   threshold, log_events() is called on the DaemonClient.
2. test_flush_on_exit_via_atexit — calling flush() drains the buffer and
   calls log_events even if the batch threshold has not been reached.
3. test_fallback_to_sqlite_when_daemon_unreachable — when log_events raises,
   events are written directly to SQLite via write_events_conn.
4. test_buffer_bounded_at_500 — adding more than 500 events drops the oldest
   so the buffer never grows past 500.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subscriber(**kwargs):
    """Return a ForwardingSubscriber with DaemonClient and write_events_conn patched."""
    from audiobench.observatory.forwarding_subscriber import ForwardingSubscriber

    with patch("audiobench.observatory.forwarding_subscriber.DaemonClient"):
        sub = ForwardingSubscriber(**kwargs)
    return sub


def _payload(n: int = 1) -> dict:
    return {
        "subsystem": "test",
        "event_type": "test_event",
        "message": f"event {n}",
        "level": "INFO",
    }


# ---------------------------------------------------------------------------
# 1. Batch threshold triggers forwarding
# ---------------------------------------------------------------------------

def test_events_batched_and_forwarded():
    """
    When buffer reaches BATCH_SIZE (50), log_events() must be called on the
    DaemonClient with all buffered payloads.
    """
    from audiobench.observatory.forwarding_subscriber import BATCH_SIZE, ForwardingSubscriber

    mock_client = MagicMock()
    with patch("audiobench.observatory.forwarding_subscriber.DaemonClient", return_value=mock_client):
        sub = ForwardingSubscriber()

    # Fill up to batch size - 1 (should not trigger)
    for i in range(BATCH_SIZE - 1):
        sub.record(**_payload(i))
    mock_client.log_events.assert_not_called()

    # The next event crosses the threshold
    sub.record(**_payload(BATCH_SIZE))
    mock_client.log_events.assert_called_once()
    forwarded = mock_client.log_events.call_args[0][0]
    assert len(forwarded) == BATCH_SIZE


# ---------------------------------------------------------------------------
# 2. flush() drains buffer via atexit registration
# ---------------------------------------------------------------------------

def test_flush_on_exit_via_atexit():
    """
    flush() must forward any buffered events even if the batch threshold has
    not been reached, and must clear the buffer afterwards.
    """
    mock_client = MagicMock()
    with patch("audiobench.observatory.forwarding_subscriber.DaemonClient", return_value=mock_client):
        from audiobench.observatory.forwarding_subscriber import ForwardingSubscriber
        sub = ForwardingSubscriber()

    sub.record(**_payload(1))
    sub.record(**_payload(2))

    sub.flush()

    mock_client.log_events.assert_called_once()
    forwarded = mock_client.log_events.call_args[0][0]
    assert len(forwarded) == 2
    # Buffer must be empty after flush
    assert sub._buffer_size() == 0


# ---------------------------------------------------------------------------
# 3. Fallback to SQLite when daemon is unreachable
# ---------------------------------------------------------------------------

def test_fallback_to_sqlite_when_daemon_unreachable():
    """
    When DaemonClient.log_events raises (daemon unreachable), events must be
    written directly to journal.db via write_events_conn.
    """
    mock_client = MagicMock()
    mock_client.log_events.side_effect = ConnectionRefusedError("no daemon")

    with patch("audiobench.observatory.forwarding_subscriber.DaemonClient", return_value=mock_client), \
         patch("audiobench.observatory.forwarding_subscriber.write_events_conn") as mock_write, \
         patch("audiobench.observatory.forwarding_subscriber.get_journal_db_path") as mock_path, \
         patch("audiobench.observatory.forwarding_subscriber.sqlite3") as mock_sqlite:
        from audiobench.observatory.forwarding_subscriber import BATCH_SIZE, ForwardingSubscriber
        sub = ForwardingSubscriber()

        for i in range(BATCH_SIZE):
            sub.record(**_payload(i))

    # Fallback write must have been called
    mock_write.assert_called_once()
    written_payloads = mock_write.call_args[0][1]
    assert len(written_payloads) == BATCH_SIZE


# ---------------------------------------------------------------------------
# 4. Buffer capped at 500
# ---------------------------------------------------------------------------

def test_buffer_bounded_at_500():
    """
    Adding more than 500 events must keep the buffer at exactly 500 by
    dropping the oldest events (FIFO drop from front).
    """
    mock_client = MagicMock()
    # Make log_events a no-op so batched flushes don't drain the mock buffer
    mock_client.log_events.return_value = None

    with patch("audiobench.observatory.forwarding_subscriber.DaemonClient", return_value=mock_client):
        from audiobench.observatory.forwarding_subscriber import BUFFER_MAX, ForwardingSubscriber
        sub = ForwardingSubscriber()

    # Add BUFFER_MAX + 50 events without triggering batch flush
    # by patching _maybe_flush to a no-op
    with patch.object(sub, "_maybe_flush"):
        for i in range(BUFFER_MAX + 50):
            sub.record(**_payload(i))

    assert sub._buffer_size() == BUFFER_MAX, \
        f"Buffer should be capped at {BUFFER_MAX}, got {sub._buffer_size()}"
