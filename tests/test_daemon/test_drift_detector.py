import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiobench.daemon.intelligence.drift_detector import DriftDetector, RunningCentroid


def test_running_centroid_matches_batch_mean():
    vectors = [
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [-1.0, 0.0, -0.5]
    ]

    rc = RunningCentroid(dim=3)
    for v in vectors:
        rc.add(v)

    np.testing.assert_allclose(rc.get_centroid(), np.mean(vectors, axis=0))

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.written = []
    def mock_add(source_type, content, inference_status=None, **kwargs):
        expr = MagicMock()
        expr.id = 100 + len(store.written)
        store.written.append({
            "source_type": source_type,
            "content": content,
        })
        return expr
    store.add_expression.side_effect = mock_add
    return store

@pytest.fixture
def mock_session():
    session = MagicMock()

    def create_mock_execute(calib_content=None, window_a_vectors=None, window_b_vectors=None):
        def mock_execute(*args, **kwargs):
            query = str(args[0])
            res = MagicMock()
            if "SELECT content FROM expressions WHERE source_type = 'daemon_calibration'" in query:
                res.fetchone.return_value = (calib_content,) if calib_content else None
            elif "SELECT id FROM expressions" in query:
                params = args[1] if len(args) > 1 else kwargs.get("params", {})
                # We need to return ids that we can map to vectors
                until = params.get("end", 0)

                # very crude mocking: if until is recent, it's Window B. If until is past 15 days, it's Window A.
                import time
                from datetime import datetime
                now = time.time()

                if isinstance(until, str):
                    try:
                        until_ts = datetime.fromisoformat(until.replace('Z', '+00:00')).timestamp()
                    except ValueError:
                        until_ts = now
                else:
                    until_ts = float(until)

                if until_ts > now - 5 * 86400: # recent (Window B)
                    res.all.return_value = [{"id": i} for i in range(len(window_b_vectors or []))]
                else: # older (Window A)
                    res.all.return_value = [{"id": i} for i in range(len(window_a_vectors or []))]
                res.mappings.return_value = res
            return res
        return mock_execute

    session.create_mock_execute = create_mock_execute

    ctx = MagicMock()
    ctx.__enter__.return_value = session
    return ctx

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.drift_detector.get_session")
@patch("audiobench.daemon.intelligence.drift_detector._get_store")
async def test_no_drift_written_when_distance_below_threshold(mock_get_store, mock_get_session, mock_store, mock_session):
    # Setup data where distance < 0.15
    window_a = [[1.0, 0.0, 0.0]] * 15  # centroid [1,0,0]
    window_b = [[0.95, 0.05, 0.0]] * 15 # centroid [0.95, 0.05, 0] -> cosine sim ~0.998 -> dist 0.002

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, window_a, window_b)

    def mock_get_vectors(ids):
        # We assume Window A or B based on len of ids for this mock
        if len(ids) == len(window_a) and window_a:
            return {i: window_a[i] for i in range(len(ids))}
        return {i: window_b[i] for i in range(len(ids))}
    mock_store.get_vectors.side_effect = mock_get_vectors

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    detector = DriftDetector()
    await detector.run()

    drifts = [x for x in mock_store.written if "Semantic centroid shift detected" in x["content"]]
    assert len(drifts) == 0

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.drift_detector.get_session")
@patch("audiobench.daemon.intelligence.drift_detector._get_store")
async def test_drift_written_when_distance_above_threshold(mock_get_store, mock_get_session, mock_store, mock_session):
    # Distance > 0.15 (sim < 0.85)
    window_a = [[1.0, 0.0, 0.0]] * 15
    window_b = [[0.0, 1.0, 0.0]] * 15 # Orthogonal -> sim 0 -> dist 1.0

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, window_a, window_b)

    def mock_get_vectors(ids):
        # The first call is for window A, second for window B.
        # We can just count the calls or look at ids if they were different.
        # Let's track calls:
        if not hasattr(mock_get_vectors, "called"):
            mock_get_vectors.called = True
            return {i: window_a[i] for i in range(len(ids))}
        return {i: window_b[i] for i in range(len(ids))}
    mock_store.get_vectors.side_effect = mock_get_vectors

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    detector = DriftDetector()
    await detector.run()

    drifts = [x for x in mock_store.written if "Semantic centroid shift detected" in x["content"]]
    assert len(drifts) == 1
    assert "Cosine distance: 1.0" in drifts[0]["content"]

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.drift_detector.get_session")
@patch("audiobench.daemon.intelligence.drift_detector._get_store")
async def test_drift_content_is_natural_language(mock_get_store, mock_get_session, mock_store, mock_session):
    window_a = [[1.0, 0.0, 0.0]] * 15
    window_b = [[0.0, 1.0, 0.0]] * 15

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, window_a, window_b)

    def mock_get_vectors2(ids):
        if not hasattr(mock_get_vectors2, "called"):
            mock_get_vectors2.called = True
            return {i: [1.0, 0.0, 0.0] for i in range(len(ids))}
        return {i: [0.0, 1.0, 0.0] for i in range(len(ids))}
    mock_store.get_vectors.side_effect = mock_get_vectors2

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    detector = DriftDetector()
    await detector.run()

    drifts = [x for x in mock_store.written if "Semantic centroid shift detected" in x["content"]]
    assert len(drifts) == 1
    content = drifts[0]["content"]
    assert isinstance(content, str)
    assert len(content) > 0
    # ensure it's not raw json
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.drift_detector.get_session")
@patch("audiobench.daemon.intelligence.drift_detector._get_store")
async def test_drift_detection_skipped_with_insufficient_data(mock_get_store, mock_get_session, mock_store, mock_session):
    window_a = [[1.0, 0.0, 0.0]] * 5  # < 10 items
    window_b = [[0.0, 1.0, 0.0]] * 5

    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, window_a, window_b)

    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session

    detector = DriftDetector()
    await detector.run()

    drifts = [x for x in mock_store.written if "Semantic centroid shift detected" in x["content"]]
    assert len(drifts) == 0
