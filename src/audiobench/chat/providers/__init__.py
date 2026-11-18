"""Chat providers — AI backend implementations and ProviderRouter."""
from __future__ import annotations

from audiobench.chat.providers.llm_providers import (
    DigestProvider,
    GeminiLLMProvider,
    OllamaLLMProvider,
)
from audiobench.chat.providers.provider_errors import (
    AIError,
    AllProvidersFailedError,
    GeminiError,
    GeminiFatalError,
    GeminiQuotaError,
    GeminiRateLimitError,
    OllamaError,
    OllamaFatalError,
    OllamaQuotaError,
    OllamaRateLimitError,
    ProviderError,
    ProviderFatalError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.chat.providers.router import (
    ProviderRouter,
    ProviderSlot,
    invalidate_router,
    router_for_settings,
)

__all__ = [
    "AIError",
    "AllProvidersFailedError",
    "DigestProvider",
    "GeminiError",
    "GeminiFatalError",
    "GeminiLLMProvider",
    "GeminiQuotaError",
    "GeminiRateLimitError",
    "LlmOutcome",
    "OllamaError",
    "OllamaFatalError",
    "OllamaLLMProvider",
    "OllamaQuotaError",
    "OllamaRateLimitError",
    "OutcomeType",
    "ProviderError",
    "ProviderFatalError",
    "ProviderQuotaError",
    "ProviderRateLimitError",
    "ProviderRouter",
    "ProviderSlot",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "invalidate_router",
    "router_for_settings",
]
