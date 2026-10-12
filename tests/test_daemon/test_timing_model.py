from unittest.mock import MagicMock, patch

from audiobench.daemon.intelligence.timing_model import TimingModel, VerbTimer


def test_ema_converges_toward_true_mean():
    timer = VerbTimer()
    for _ in range(50):
        timer.record(100.0)

    # EMA starts at 100.0 if empty, or converges to 100.0
    assert abs(timer.ema_ms - 100.0) < 10.0

def test_p95_returns_correct_percentile():
    timer = VerbTimer()
    # 10, 20, 30 ... 1000
    for i in range(1, 101):
        timer.record(i * 10.0)

    assert timer.get_p95() >= 940.0
    assert timer.get_p95() <= 960.0

@patch("audiobench.daemon.intelligence.timing_model._get_store")
@patch("audiobench.daemon.intelligence.timing_model.get_session")
def test_timing_persisted_across_restarts(mock_get_session, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    session = MagicMock()
    session.__enter__.return_value = session
    mock_get_session.return_value = session

    # First instance writes
    model1 = TimingModel()
    for _ in range(20):
        model1.record_latency("test_verb", 150.0)

    store.add_expression.assert_called_once()
    args, kwargs = store.add_expression.call_args
    calib_json = kwargs["content"]

    # Mock lookup
    session.execute.return_value.fetchone.return_value = (calib_json,)

    # Second instance loads
    model2 = TimingModel()
    assert "test_verb" in model2.timers
    assert abs(model2.predict_duration("test_verb") - 150.0) < 15.0
