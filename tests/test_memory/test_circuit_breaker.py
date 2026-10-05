import pytest

from audiobench.memory.llm_caller import CircuitBreaker, RateLimitError, _retry_with_backoff


def test_circuit_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError:
            pass
    assert cb.state == "OPEN"

def test_circuit_open_raises_without_calling_fn():
    cb = CircuitBreaker(failure_threshold=1)
    try:
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
    except RuntimeError:
        pass
    called = []
    try:
        cb.call(lambda: called.append(1))
    except Exception:
        pass
    assert called == [], "Function must not be called when circuit is OPEN"

def test_circuit_half_open_allows_one_probe(monkeypatch):
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    try:
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
    except RuntimeError:
        pass
    import time; time.sleep(0.02)

    # State remains OPEN until `.call()` evaluates the timeout
    # But when we call it and it succeeds, it should go back to CLOSED.
    # To just test HALF_OPEN state, we must cause a fail in `.call()` or intercept it.
    def mock_fn():
        assert cb.state == "HALF_OPEN"
        raise RuntimeError("fail")

    try:
        cb.call(mock_fn)
    except RuntimeError:
        pass

    assert cb.state == "OPEN" # tripped again after the probe failed


def test_backoff_delays_increase_exponentially(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", lambda d: delays.append(d))

    attempts = 0
    def always_fail():
        nonlocal attempts
        attempts += 1
        raise RateLimitError("429")

    try:
        _retry_with_backoff(always_fail, max_retries=3, base_delay=1.0, jitter=False)
    except RateLimitError:
        pass

    assert len(delays) == 3
    assert delays[0] == pytest.approx(1.0, abs=0.01)
    assert delays[1] == pytest.approx(2.0, abs=0.01)
    assert delays[2] == pytest.approx(4.0, abs=0.01)
