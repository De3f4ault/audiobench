"""Typed exception hierarchy for the ProviderRouter.

All exceptions subclass AIError for backward compatibility — existing
`except AIError` blocks keep working unchanged during the wave-1 migration.
"""
from __future__ import annotations

# Re-export existing base so callers can import everything from one place
from audiobench.core.error_types import AIError  # noqa: F401


class ProviderError(AIError):
    """Base for all typed provider errors."""

    def __init__(self, message: str = "", details: str | None = None) -> None:
        super().__init__(message, details)


class ProviderRateLimitError(ProviderError):
    """429 — retry same provider with backoff; do not advance immediately."""
    retry_after: float | None = None   # from Retry-After header if present


class ProviderQuotaError(ProviderError):
    """402 / credit exhausted — advance immediately, no retry."""


class ProviderUnavailableError(ProviderError):
    """503, connection refused — advance; trip breaker."""


class ProviderTimeoutError(ProviderError):
    """Hard deadline timeout — advance; trip breaker."""


class ProviderFatalError(ProviderError):
    """400, 401, 404 — caught inside generate(); returned as LlmOutcome(type=FAILED).
    Never swallowed silently. Never advanced past. No fallback, no digest."""


class AllProvidersFailedError(AIError):
    """Raised by generate_text() when all slots exhausted and no usable digest configured.
    Subclasses AIError so existing `except AIError` blocks keep working during migration."""

    def __init__(self, last_error: Exception | None = None):
        self.last_error = last_error
        super().__init__(f"All providers exhausted. Last error: {last_error}")


# Provider-specific subtypes — MRO is clean; both parents are non-overlapping mixins.
# OllamaQuotaError is caught by: except ProviderQuotaError (router slot loop),
# except OllamaError (provider tests), except AIError (existing wave-1 callers).

class OllamaError(ProviderError):
    pass


class OllamaRateLimitError(OllamaError, ProviderRateLimitError):
    pass


class OllamaQuotaError(OllamaError, ProviderQuotaError):
    pass


class OllamaUnavailableError(OllamaError, ProviderUnavailableError):
    pass


class OllamaTimeoutError(OllamaError, ProviderTimeoutError):
    pass


class OllamaFatalError(OllamaError, ProviderFatalError):
    pass


class GeminiError(ProviderError):
    pass


class GeminiRateLimitError(GeminiError, ProviderRateLimitError):
    pass


class GeminiQuotaError(GeminiError, ProviderQuotaError):
    pass


class GeminiUnavailableError(GeminiError, ProviderUnavailableError):
    pass


class GeminiTimeoutError(GeminiError, ProviderTimeoutError):
    pass


class GeminiFatalError(GeminiError, ProviderFatalError):
    pass
