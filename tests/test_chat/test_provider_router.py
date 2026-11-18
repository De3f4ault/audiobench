"""Tests for ProviderRouter."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from audiobench.chat.providers.provider_errors import (
    AIError,
    AllProvidersFailedError,
    OllamaFatalError,
    OllamaQuotaError,
    OllamaRateLimitError,
    OllamaUnavailableError,
    ProviderFatalError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.chat.providers.router import ProviderRouter, ProviderSlot
from audiobench.core.settings import AudioBenchSettings
from audiobench.memory.llm_caller import CircuitBreaker


def _make_test_router(
    order: list[str] = ["ollama", "gemini"],
    threshold: int = 3,
    timeout: float = 60.0,
    chain_enabled: bool = True,
    local_enabled: bool = True,
    cloud_enabled: bool = True,
) -> ProviderRouter:
    settings = AudioBenchSettings(
        llm_chain_enabled=chain_enabled,
        llm_local_enabled=local_enabled,
        llm_cloud_enabled=cloud_enabled,
        llm_provider_order=order,
        circuit_breaker_threshold=threshold,
        circuit_breaker_timeout_seconds=timeout,
        gemini_api_key="fake-test-key",
    )
    router = ProviderRouter(settings)
    return router


def test_ollama_primary_succeeds():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.return_value = "hello from ollama"
    gemini_mock = MagicMock()

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.OK
    assert outcome.content == "hello from ollama"
    assert outcome.provider == "ollama"
    gemini_mock.generate.assert_not_called()


def test_ollama_fatal_returns_failed_outcome():
    router = _make_test_router()
    ollama_mock = MagicMock()
    err = OllamaFatalError("Bad prompt")
    ollama_mock.generate.side_effect = err
    gemini_mock = MagicMock()

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "bad"}])
    assert outcome.type == OutcomeType.FAILED
    assert outcome.content == ""
    assert outcome.error is err
    gemini_mock.generate.assert_not_called()


def test_generate_text_reraises_fatal():
    router = _make_test_router()
    ollama_mock = MagicMock()
    err = OllamaFatalError("Bad prompt")
    ollama_mock.generate.side_effect = err

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    with pytest.raises(OllamaFatalError):
        router.generate_text([{"role": "user", "content": "bad"}])


def test_ollama_quota_fallback_gemini():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaQuotaError("Quota exceeded")
    ollama_mock.is_cloud = False  # local model quota (no discovery swap)

    gemini_mock = MagicMock()
    gemini_mock.generate.return_value = "hello from gemini"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.OK
    assert outcome.content == "hello from gemini"
    assert outcome.provider == "gemini"
    assert outcome.fallback_from == "ollama"


def test_ollama_rate_limit_retries_then_gemini(monkeypatch):
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaRateLimitError("429")

    gemini_mock = MagicMock()
    gemini_mock.generate.return_value = "gemini response"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    monkeypatch.setattr("time.sleep", lambda _: None)
    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.OK
    assert outcome.content == "gemini response"
    assert outcome.fallback_from == "ollama"
    assert ollama_mock.generate.call_count >= 3


def test_all_fail_extractive_digest():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")
    gemini_mock = MagicMock()
    gemini_mock.generate.side_effect = ProviderUnavailableError("down")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    fragments = [
        {"author": "Aristotle", "work_title": "Metaphysics", "effective_rel": 0.95, "content": "Being qua being"}
    ]
    outcome = router.generate(
        [{"role": "user", "content": "metaphysics"}],
        digest_mode="extractive",
        context_fragments=fragments,
    )
    assert outcome.type == OutcomeType.DEGRADED
    assert outcome.provider == "digest"
    assert "Dialectic Digest" in outcome.content
    assert "Aristotle" in outcome.content


def test_all_fail_notice_digest():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate(
        [{"role": "user", "content": "hello"}],
        digest_mode="notice",
    )
    assert outcome.type == OutcomeType.DEGRADED
    assert "LLM services unavailable" in outcome.content


def test_all_fail_passthrough_returns_query():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate(
        [{"role": "user", "content": "original query"}],
        digest_mode="passthrough",
    )
    assert outcome.type == OutcomeType.DEGRADED
    assert outcome.content == "original query"


def test_all_fail_silent_empty_content():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate(
        [{"role": "user", "content": "test"}],
        digest_mode="silent",
    )
    assert outcome.type == OutcomeType.DEGRADED
    assert outcome.content == ""


def test_all_fail_generate_text_raises_all_providers_failed():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    with pytest.raises(AllProvidersFailedError):
        router.generate_text([{"role": "user", "content": "test"}], digest_mode="silent")


def test_circuit_breaker_trips_at_threshold():
    router = _make_test_router(threshold=3)
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("down")
    gemini_mock = MagicMock()
    gemini_mock.generate.return_value = "gemini"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker(failure_threshold=3)),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    # 3 failures trip the breaker
    for _ in range(3):
        router.generate([{"role": "user", "content": "hi"}])

    assert router._slots[0].breaker.state == "OPEN"
    ollama_mock.generate.reset_mock()

    # 4th call skips Ollama directly
    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.OK
    assert outcome.provider == "gemini"
    ollama_mock.generate.assert_not_called()


def test_circuit_breaker_half_open_probe():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.state = "OPEN"
    cb.last_failure_time = time.monotonic() - 0.05
    assert cb.can_call() is True
    assert cb.state == "HALF_OPEN"
    cb.record_success()
    assert cb.state == "CLOSED"


def test_local_enabled_false_skips_ollama():
    router = _make_test_router(local_enabled=False)
    assert "ollama" not in [s.name for s in router._slots]


def test_cloud_enabled_false_skips_gemini():
    router = _make_test_router(cloud_enabled=False)
    assert "gemini" not in [s.name for s in router._slots]


def test_chain_disabled_returns_failed_outcome():
    router = _make_test_router(chain_enabled=False)
    outcome = router.generate([{"role": "user", "content": "test"}])
    assert outcome.type == OutcomeType.FAILED
    assert isinstance(outcome.error, ProviderFatalError)


def test_quota_swaps_to_local_and_retries(monkeypatch):
    router = _make_test_router()
    cloud_ollama = MagicMock()
    cloud_ollama.generate.side_effect = OllamaQuotaError("Quota")
    cloud_ollama.is_cloud = True

    gemini_mock = MagicMock()
    gemini_mock.generate.side_effect = ProviderQuotaError("Gemini Quota")

    slot = ProviderSlot("ollama", cloud_ollama, CircuitBreaker())
    router._slots = [slot, ProviderSlot("gemini", gemini_mock, CircuitBreaker())]

    # Mock discovery to return local model
    router._discover_local_ollama = MagicMock(return_value="phi3:3b")  # type: ignore[method-assign]
    fake_client = MagicMock()
    fake_client.chat.return_value = {"content": "local phi3 output"}
    monkeypatch.setattr(
        "audiobench.chat.providers.ollama_provider.OllamaClient",
        lambda **kw: fake_client,
    )

    outcome = router.generate([{"role": "user", "content": "test"}])
    assert outcome.type == OutcomeType.OK
    assert outcome.content == "local phi3 output"
    assert outcome.provider == "ollama-local:phi3:3b"
    gemini_mock.generate.assert_called_once()


def test_local_discovery_prefers_smallest(monkeypatch):
    router = _make_test_router()
    fake_client = MagicMock()
    fake_client.is_available.return_value = True
    fake_client.list_models.return_value = ["llama3:70b", "phi3:3b", "qwen:14b", "deepseek:120b-cloud"]

    monkeypatch.setattr(
        "audiobench.chat.providers.ollama_provider.OllamaClient",
        lambda **kw: fake_client,
    )
    discovered = router._discover_local_ollama()
    assert discovered == "phi3:3b"


def test_set_primary_model_rebuilds_slot():
    router = _make_test_router()
    ollama_mock = MagicMock()
    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker(failure_threshold=5)),
        ProviderSlot("gemini", MagicMock(), CircuitBreaker()),
    ]
    router.set_primary_model("phi3:3b")
    # Slot 0 rebuilt with new client
    assert router._slots[0].name == "ollama"
    assert router._slots[0].provider is not ollama_mock


def test_disclosure_fn_called_on_fallback():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("offline")
    gemini_mock = MagicMock()
    gemini_mock.generate.return_value = "gemini reply"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    disclosed = []
    router.generate([{"role": "user", "content": "hi"}], disclosure_fn=disclosed.append)
    assert len(disclosed) == 1
    assert "ⓘ" in disclosed[0]
    assert "ollama unavailable" in disclosed[0]


def test_disclosure_fn_called_on_fatal():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaFatalError("invalid model")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    disclosed = []
    router.generate([{"role": "user", "content": "hi"}], disclosure_fn=disclosed.append)
    assert len(disclosed) == 1
    assert "✗" in disclosed[0]


def test_disclosure_fn_none_no_error():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaUnavailableError("offline")
    gemini_mock = MagicMock()
    gemini_mock.generate.return_value = "gemini reply"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}], disclosure_fn=None)
    assert outcome.type == OutcomeType.OK


def test_stream_no_tokens_silent_retry():
    router = _make_test_router()
    ollama_mock = MagicMock()

    def bad_stream(*args, **kw):
        raise OllamaUnavailableError("down")
        yield {}

    ollama_mock.generate_stream = bad_stream

    gemini_mock = MagicMock()
    gemini_mock.generate_stream.return_value = iter([{"content": "gemini chunk", "done": True}])

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    chunks = list(router.generate_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0]["content"] == "gemini chunk"


def test_stream_mid_failure_emits_notice():
    router = _make_test_router()
    ollama_mock = MagicMock()

    def failing_stream(*args, **kw):
        yield {"content": "part 1"}
        raise OllamaUnavailableError("cut off")

    ollama_mock.generate_stream = failing_stream

    gemini_mock = MagicMock()
    gemini_mock.generate_stream.return_value = iter([{"content": "gemini full", "done": True}])

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    chunks = list(router.generate_stream([{"role": "user", "content": "hi"}]))
    # Expect: chunk 1 ("part 1"), notice chunk, gemini chunk
    notices = [c.get("notice") for c in chunks if c.get("notice")]
    assert len(notices) == 1
    assert "interrupted" in notices[0]
    assert any(c.get("content") == "gemini full" for c in chunks)


def test_stream_fatal_yields_terminal_chunk():
    router = _make_test_router()
    ollama_mock = MagicMock()

    def fatal_stream(*args, **kw):
        raise OllamaFatalError("fatal bad request")
        yield {}

    ollama_mock.generate_stream = fatal_stream

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    chunks = list(router.generate_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].get("failed") is True
    assert chunks[0].get("done") is True


def test_chat_stream_accepts_model_kwarg():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate_stream.return_value = iter([{"content": "ok", "done": True}])

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    chunks = list(router.chat_stream([{"role": "user", "content": "hi"}], model="some-model", num_ctx=4096))
    assert len(chunks) == 1
    assert chunks[0]["content"] == "ok"


def test_generate_accepts_model_kwarg():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.return_value = "ok"

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}], model="some-model")
    assert outcome.content == "ok"


def test_outcome_failed_has_empty_content():
    router = _make_test_router()
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = OllamaFatalError("fatal")

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.FAILED
    assert outcome.content == ""


def test_outcome_failed_error_is_original():
    router = _make_test_router()
    err = OllamaFatalError("original fatal instance")
    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = err

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.error is err


def test_retry_fatal_during_retry_returns_failed():
    router = _make_test_router()
    fatal_err = ProviderFatalError("auth revoked mid-retry")

    attempts = 0
    def fail_with_fatal_on_retry(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OllamaRateLimitError("429")
        raise fatal_err

    ollama_mock = MagicMock()
    ollama_mock.generate.side_effect = fail_with_fatal_on_retry

    gemini_mock = MagicMock()

    router._slots = [
        ProviderSlot("ollama", ollama_mock, CircuitBreaker()),
        ProviderSlot("gemini", gemini_mock, CircuitBreaker()),
    ]

    outcome = router.generate([{"role": "user", "content": "hi"}])
    assert outcome.type == OutcomeType.FAILED
    assert outcome.error is fatal_err
    gemini_mock.generate.assert_not_called()


def test_router_is_available():
    router = _make_test_router()
    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_prov = MagicMock()
    mock_prov._client = mock_client

    router._slots = [ProviderSlot("ollama", mock_prov, CircuitBreaker())]
    assert router.is_available() is True
    assert router.is_available("ollama") is True
    assert router.is_available("gemini") is False

    mock_client.is_available.return_value = False
    assert router.is_available() is False


def test_router_list_models():
    router = _make_test_router()
    mock_client = MagicMock()
    mock_client.list_models.return_value = ["model1", "model2"]
    mock_prov = MagicMock()
    mock_prov._client = mock_client

    router._slots = [ProviderSlot("ollama", mock_prov, CircuitBreaker())]
    assert router.list_models() == ["model1", "model2"]


def test_router_model_property():
    router = _make_test_router()
    mock_client = MagicMock()
    mock_client.model = "my-test-model"
    mock_prov = MagicMock()
    mock_prov._client = mock_client

    router._slots = [ProviderSlot("ollama", mock_prov, CircuitBreaker())]
    assert router.model == "my-test-model"
    assert router._model == "my-test-model"


def test_set_primary_model_gemini_reorders():
    router = _make_test_router()
    s1 = ProviderSlot("ollama", MagicMock(), CircuitBreaker())
    s2 = ProviderSlot("gemini", MagicMock(), CircuitBreaker())
    router._slots = [s1, s2]

    router.set_primary_model("gemini")
    assert [s.name for s in router._slots] == ["gemini", "ollama"]

