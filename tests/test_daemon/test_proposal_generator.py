import time
from unittest.mock import MagicMock, patch

import pytest

from audiobench.daemon.intelligence.calibration import RegionStats
from audiobench.daemon.intelligence.proposal_generator import ProposalGenerator


@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.proposal_generator._get_store")
@patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker")
@patch("audiobench.daemon.intelligence.proposal_generator.get_session")
async def test_proposal_not_generated_with_insufficient_data(mock_get_session, mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # earliest_calibration_timestamp is only 10 days ago
    now = time.time()
    session.execute.return_value.scalar.return_value = now - (10 * 86400)

    tracker = MagicMock()
    # High enough rate and sample size, but not enough days
    tracker.stats = {"region1": RegionStats(confirms=20, rejects=0, inferences=25, samples=[])}
    mock_get_tracker.return_value = tracker

    generator = ProposalGenerator()
    await generator.run()

    store.add_expression.assert_not_called()

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.proposal_generator._get_store")
@patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker")
@patch("audiobench.daemon.intelligence.proposal_generator.get_session")
async def test_proposal_generated_when_criteria_met(mock_get_session, mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # 35 days ago
    now = time.time()
    session.execute.return_value.scalar.return_value = now - (35 * 86400)

    # Mock no pending proposals
    session.execute.return_value.fetchone.return_value = None

    tracker = MagicMock()
    tracker.stats = {"region1": RegionStats(confirms=25, rejects=5, inferences=30, samples=[])}
    mock_get_tracker.return_value = tracker

    generator = ProposalGenerator()
    await generator.run()

    store.add_expression.assert_called_once()

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.proposal_generator._get_store")
@patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker")
@patch("audiobench.daemon.intelligence.proposal_generator.get_session")
async def test_no_duplicate_proposal_for_same_region(mock_get_session, mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    now = time.time()
    # Return 35 days ago for the first query (earliest timestamp)
    # Return a row for the second query (existing proposal)
    def side_effect(stmt, params=None):
        m = MagicMock()
        if "MIN(created_at)" in str(stmt):
            m.scalar.return_value = now - (35 * 86400)
        else:
            m.fetchone.return_value = (1,)
        return m
    session.execute.side_effect = side_effect

    tracker = MagicMock()
    tracker.stats = {"region1": RegionStats(confirms=25, rejects=5, inferences=30, samples=[])}
    mock_get_tracker.return_value = tracker

    generator = ProposalGenerator()
    await generator.run()

    store.add_expression.assert_not_called()

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.proposal_generator._get_store")
@patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker")
@patch("audiobench.daemon.intelligence.proposal_generator.get_session")
async def test_proposal_content_includes_evidence_block(mock_get_session, mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    now = time.time()
    def side_effect(stmt, params=None):
        m = MagicMock()
        if "MIN(created_at)" in str(stmt):
            m.scalar.return_value = now - (35 * 86400)
        else:
            m.fetchone.return_value = None
        return m
    session.execute.side_effect = side_effect

    tracker = MagicMock()
    tracker.stats = {"region1": RegionStats(confirms=25, rejects=5, inferences=30, samples=[])}
    mock_get_tracker.return_value = tracker

    generator = ProposalGenerator()
    await generator.run()

    args, kwargs = store.add_expression.call_args
    content = kwargs["content"]
    assert "confirm_rate=0.8" in content
    assert "across 30 inferences" in content
    assert "over 35 days" in content

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.proposal_generator._get_store")
@patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker")
@patch("audiobench.daemon.intelligence.proposal_generator.get_session")
async def test_duplicate_pending_proposal_suppressed(mock_get_session, mock_get_tracker, mock_get_store):
    # This is essentially the same as test_no_duplicate_proposal_for_same_region
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    now = time.time()
    def side_effect(stmt, params=None):
        m = MagicMock()
        if "MIN(created_at)" in str(stmt):
            m.scalar.return_value = now - (35 * 86400)
        else:
            # mock existing proposal with status IN ("proposed", "deferred")
            m.fetchone.return_value = (1,)
        return m
    session.execute.side_effect = side_effect

    tracker = MagicMock()
    tracker.stats = {"region1": RegionStats(confirms=25, rejects=5, inferences=30, samples=[])}
    mock_get_tracker.return_value = tracker

    generator = ProposalGenerator()
    await generator.run()

    store.add_expression.assert_not_called()
