import json
import time
from unittest.mock import MagicMock, patch

import pytest

from audiobench.daemon.intelligence.operator_registry import OperatorRegistry


@pytest.fixture
def mock_session():
    with patch("audiobench.daemon.intelligence.operator_registry.get_session") as m:
        session = MagicMock()
        m.return_value.__enter__.return_value = session
        yield session

def test_schema_version_mismatch_skipped_with_warning(mock_session):
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "operator_template": "LectureModeOperator",
            "schema_version": 1, # current is 2
            "parameters": {"frequency_multiplier": 1.5}
        }))
    ]

    registry.load_from_db()
    assert len(registry.dynamic_operators) == 0

def test_out_of_bounds_parameter_rejected_by_pydantic(mock_session):
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchone.return_value = (
        json.dumps({
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 500.0} # le=10.0
        }),
    )

    registry.hot_register(1)

    assert len(registry.dynamic_operators) == 0
    # verify update to rejected
    update_call = mock_session.execute.call_args_list[-1]
    assert "inference_status = 'rejected'" in str(update_call[0][0])

def test_revoke_removes_operator_from_registry_immediately(mock_session):
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchone.return_value = (
        json.dumps({
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 1.5}
        }),
    )

    registry.hot_register(1)
    assert 1 in registry.dynamic_operators

    registry.deactivate(1)
    assert 1 not in registry.dynamic_operators
    update_call = mock_session.execute.call_args_list[-1]
    assert "inference_status = 'revoked'" in str(update_call[0][0])

# ----------------- Integration Tests (Marked explicitly) -----------------

@pytest.mark.integration
def test_stale_proposal_expires_on_confirm_rate_drop(mock_session):
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "region_id": "r1",
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 1.5}
        }), time.time() - 3600)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        stats = MagicMock()
        stats.confirm_rate = 0.25
        stats.total_since.return_value = 10
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_proposals()

        assert len(valid) == 0
        update_call = mock_session.execute.call_args_list[-1]
        assert "inference_status = 'expired'" in str(update_call[0][0])

@pytest.mark.integration
def test_stale_proposal_survives_when_evidence_is_fresh(mock_session):
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "region_id": "r1",
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 1.5}
        }), time.time() - 3600)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        stats = MagicMock()
        stats.confirm_rate = 0.85
        stats.total_since.return_value = 10
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_proposals()

        assert len(valid) == 1

@pytest.mark.integration
def test_stale_proposal_expires_on_zero_fresh_samples(mock_session):
    """Proposals older than the grace period (>= 7 days) with < 5 fresh samples MUST expire."""
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "region_id": "r1",
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 1.5}
        }), time.time() - (8 * 86400)) # 8 days old (exceeds 7-day grace period)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        stats = MagicMock()
        stats.confirm_rate = 0.85
        stats.total_since.return_value = 2 # < 5
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_proposals()

        assert len(valid) == 0
        update_call = mock_session.execute.call_args_list[-1]
        assert "inference_status = 'expired'" in str(update_call[0][0])


@pytest.mark.integration
def test_fresh_proposal_survives_zero_fresh_samples_within_grace_period(mock_session):
    """Fresh proposals (< 7 days old) with high confirm rate MUST NOT expire due to low samples."""
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "region_id": "r1",
            "operator_template": "LectureModeOperator",
            "schema_version": 2,
            "parameters": {"frequency_multiplier": 1.5}
        }), time.time() - 3600) # 1 hour old (within 7-day grace period)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        stats = MagicMock()
        stats.confirm_rate = 0.85
        stats.total_since.return_value = 0 # 0 samples since creation
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_proposals()

        assert len(valid) == 1
        # Assert NO status update to 'expired' occurred
        for call in mock_session.execute.call_args_list:
            assert "inference_status = 'expired'" not in str(call[0][0])


@pytest.mark.integration
def test_non_operator_knowledge_gap_never_expires(mock_session):
    """Knowledge gap proposals must never be expired by operator registry staleness check."""
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (1, json.dumps({
            "proposal_type": "knowledge_gap",
            "schema_version": 2,
            "title": "Philosophy Gap",
            "summary": "Distance 0.25",
            "evidence": "distance=0.25",
            "payload": {"distance": 0.25, "threshold": 0.20}
        }), time.time() - 3600)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        # Even if tracker stats would trigger expiry
        stats = MagicMock()
        stats.confirm_rate = 0.10
        stats.total_since.return_value = 0
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_operator_proposals()

        assert len(valid) == 0
        # Assert NO status update to 'expired' occurred
        for call in mock_session.execute.call_args_list:
            assert "inference_status = 'expired'" not in str(call[0][0])


@pytest.mark.integration
def test_plain_text_proposal_never_expires_or_crashes(mock_session):
    """Plain text legacy proposals (such as #33196) must be skipped cleanly without expiring."""
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (33196, "Knowledge gap detected (distance=0.260, threshold=0.2).\nYour search queries...", time.time() - 3600)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        valid = registry.get_pending_operator_proposals()

        assert len(valid) == 0
        for call in mock_session.execute.call_args_list:
            assert "inference_status = 'expired'" not in str(call[0][0])


@pytest.mark.integration
def test_reverse_polarity_stale_envelope_operator_proposal_still_expires(mock_session):
    """Envelope operator proposals (proposal_type='operator') with stale evidence MUST still expire."""
    registry = OperatorRegistry()
    mock_session.execute.return_value.fetchall.return_value = [
        (42, json.dumps({
            "proposal_type": "operator",
            "schema_version": 2,
            "title": "Lecture Mode",
            "summary": "Multiplier 1.5",
            "evidence": "confirm_rate=0.25",
            "payload": {
                "region_id": "r1",
                "operator_template": "LectureModeOperator",
                "parameters": {"frequency_multiplier": 1.5}
            }
        }), time.time() - 3600)
    ]

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker") as mock_get_tracker:
        tracker = MagicMock()
        stats = MagicMock()
        stats.confirm_rate = 0.25 # Stale
        stats.total_since.return_value = 10
        tracker.get_region_stats.return_value = stats
        mock_get_tracker.return_value = tracker

        valid = registry.get_pending_operator_proposals()

        assert len(valid) == 0
        update_call = mock_session.execute.call_args_list[-1]
        assert "inference_status = 'expired'" in str(update_call[0][0])

