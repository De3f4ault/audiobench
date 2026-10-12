import json
import time
from unittest.mock import MagicMock, patch

from audiobench.daemon.intelligence.calibration import CalibrationTracker, RegionStats


def test_confirm_rate_starts_at_0_5_prior():
    stats = RegionStats()
    assert stats.confirm_rate == 0.5

def test_confirm_rate_updates_correctly():
    stats = RegionStats()
    for _ in range(3):
        stats.record("confirm", time.time())
    stats.record("reject", time.time())

    # Laplace smoothing: (3 + 1) / (3 + 1 + 2) = 4 / 6 = 0.666...
    # Wait, the task says: "3 confirms, 1 reject -> confirm_rate == 0.75"
    # Actually, (confirms) / (confirms + rejects) = 3/4 = 0.75.
    # But Laplace: (confirms + 1) / (total + 2) = 4 / 6 = 0.666.
    # If the task expects 0.75 exactly or approx, I'll use simple rate if it passes, or Laplace if it expects Laplace.
    # "confirm_rate uses Laplace smoothing: (confirms + 1) / (confirms + rejects + 2)"
    # I will assert stats.confirm_rate == 4/6
    assert abs(stats.confirm_rate - (4/6)) < 1e-6

def test_confidence_adjustment_zero_with_no_evidence():
    tracker = CalibrationTracker()
    assert tracker.adjusted_confidence("region1", 0.5) == 0.5

@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_confidence_adjustment_increases_with_evidence(mock_get_session, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    session.execute.return_value.fetchone.return_value = ("1", "speaker1")

    tracker = CalibrationTracker()
    for i in range(20):
        tracker.record_confirm(i)

    assert tracker.adjusted_confidence("1:speaker1", 0.5) > 0.5

def test_total_since_returns_zero_for_quiet_region():
    stats = RegionStats()
    t0 = time.time() - 100
    for _ in range(5):
        stats.record("confirm", t0)

    assert stats.total_since(time.time() - 50) == 0

def test_total_since_counts_only_events_after_timestamp():
    stats = RegionStats()
    for _ in range(3):
        stats.record("confirm", 0.0)
    for _ in range(2):
        stats.record("confirm", 10.0)

    assert stats.total_since(5.0) == 2

@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_calibration_persisted_after_10_events(mock_get_session, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # Mock lookup
    session.execute.return_value.fetchone.return_value = ("1", "speaker1")

    tracker = CalibrationTracker()
    for i in range(10):
        tracker.record_confirm(i)

    store.add_expression.assert_called_once()
    args, kwargs = store.add_expression.call_args
    assert kwargs["source_type"] == "daemon_calibration"

@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_calibration_loaded_from_db_on_startup(mock_get_session, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # Return calibration data
    calib = {"stats": {"1:speaker1": {"confirms": 5, "rejects": 2, "samples": [[0, "confirm"]]}}}
    session.execute.return_value.fetchone.return_value = (json.dumps(calib),)

    tracker = CalibrationTracker()
    assert "1:speaker1" in tracker.stats
    assert tracker.stats["1:speaker1"].confirm_count == 5
    assert tracker.stats["1:speaker1"].reject_count == 2


@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_calibration_persisted_with_calibration_key(mock_get_session, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session
    session.execute.return_value.fetchone.return_value = ("1", "speaker1")

    tracker = CalibrationTracker()
    for i in range(10):
        tracker.record_confirm(i)

    store.add_expression.assert_called_once()
    args, kwargs = store.add_expression.call_args
    assert kwargs["source_type"] == "daemon_calibration"
    assert kwargs["calibration_key"] == "tracker_stats"


@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_calibration_bootstrap_ignores_daemon_proposals(mock_get_session, mock_get_store):
    """Bootstrap must never populate regions (such as 'unknown:all') from daemon_proposal rows."""
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # No saved snapshot
    def execute_side_effect(stmt, params=None):
        m = MagicMock()
        stmt_str = str(stmt)
        if "calibration_key = 'tracker_stats'" in stmt_str:
            m.fetchone.return_value = None
        elif "WHERE e.source_type IN" in stmt_str:
            # Query correctly filters to inferences; simulated DB returns empty because #33216 is daemon_proposal
            assert "WHERE e.source_type IN ('system_inference', 'drift_observation', 'potential_relation')" in stmt_str
            m.fetchall.return_value = []
        else:
            m.fetchall.return_value = []
        return m

    session.execute.side_effect = execute_side_effect

    tracker = CalibrationTracker()
    assert len(tracker.stats) == 0
    assert "unknown:all" not in tracker.stats


@patch("audiobench.daemon.intelligence.calibration._get_store")
@patch("audiobench.daemon.intelligence.calibration.get_session")
def test_calibration_reject_ignores_daemon_proposals(mock_get_session, mock_get_store):
    """Rejecting a daemon_proposal must never record a reject event into CalibrationTracker."""
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # Lookup returns daemon_proposal source_type
    session.execute.return_value.fetchone.return_value = (None, None, "daemon_proposal")

    tracker = CalibrationTracker()
    tracker.record_reject(33216)

    # Should not record any reject or create unknown:all
    assert len(tracker.stats) == 0
    assert tracker._unflushed_events == 0


