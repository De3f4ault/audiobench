"""Tests for the typed provider exception hierarchy."""
from __future__ import annotations

import pytest

from audiobench.chat.providers.provider_errors import (
    AIError,
    AllProvidersFailedError,
    GeminiError,
    GeminiFatalError,
    GeminiQuotaError,
    GeminiRateLimitError,
    GeminiUnavailableError,
    OllamaError,
    OllamaFatalError,
    OllamaQuotaError,
    OllamaRateLimitError,
    OllamaUnavailableError,
    ProviderError,
    ProviderFatalError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def test_ollama_quota_is_provider_quota():
    assert isinstance(OllamaQuotaError(), ProviderQuotaError)


def test_ollama_quota_is_ollama_error():
    assert isinstance(OllamaQuotaError(), OllamaError)


def test_ollama_quota_is_ai_error():
    assert isinstance(OllamaQuotaError(), AIError)


def test_all_providers_failed_is_ai_error():
    assert isinstance(AllProvidersFailedError(), AIError)


def test_all_providers_failed_stores_last_error():
    cause = ValueError("boom")
    err = AllProvidersFailedError(last_error=cause)
    assert err.last_error is cause


def test_fatal_is_not_quota():
    assert not isinstance(OllamaFatalError(), ProviderQuotaError)


def test_gemini_rate_limit_is_provider_rate_limit():
    assert isinstance(GeminiRateLimitError(), ProviderRateLimitError)


def test_gemini_fatal_is_provider_fatal():
    assert isinstance(GeminiFatalError(), ProviderFatalError)


def test_classify_ollama_402():
    from audiobench.chat.providers.llm_providers import _classify_ollama
    err = AIError("Ollama API error (402)", "credit exhausted")
    err.status_code = 402
    classified = _classify_ollama(err)
    assert isinstance(classified, OllamaQuotaError)


def test_classify_ollama_404():
    from audiobench.chat.providers.llm_providers import _classify_ollama
    err = AIError("Model not found", "not found")
    err.status_code = 404
    classified = _classify_ollama(err)
    assert isinstance(classified, OllamaFatalError)


def test_classify_ollama_503():
    from audiobench.chat.providers.llm_providers import _classify_ollama
    err = AIError("Ollama API error (503)", "service unavailable")
    err.status_code = 503
    classified = _classify_ollama(err)
    assert isinstance(classified, OllamaUnavailableError)


def test_classify_ollama_unknown():
    from audiobench.chat.providers.llm_providers import _classify_ollama
    err = AIError("Something went wrong", "unknown")
    err.status_code = None
    classified = _classify_ollama(err)
    assert isinstance(classified, OllamaUnavailableError)

