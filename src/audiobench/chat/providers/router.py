"""Unified ProviderRouter with multi-layer fallback, per-slot circuit breakers, and degradation."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator

from audiobench.chat.providers.llm_providers import (
    DigestProvider,
    GeminiLLMProvider,
    OllamaLLMProvider,
)
from audiobench.chat.providers.provider_errors import (
    AllProvidersFailedError,
    OllamaQuotaError,
    ProviderFatalError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.core.settings import AudioBenchSettings
from audiobench.memory.llm_caller import CircuitBreaker

logger = logging.getLogger(__name__)


class FallbackPolicy(str, Enum):
    """Controls whether the router may fall back to a local CPU model after all cloud slots fail.

    ``ALLOW_LOCAL`` (default) — after exhausting all configured slots, attempt a local Ollama
    model as a last resort before the Digest degradation path.  Appropriate for interactive
    *chat* where the user expects a response even if it is slow.

    ``NO_LOCAL_CPU`` — skip the local-last-resort step entirely.  Use for search pipeline
    callers (QueryReformulator, HyDE, synthesis) where a 2-minute CPU inference is worse
    than an honest failure: the pipeline has its own graceful degradation path.
    """

    ALLOW_LOCAL = "allow_local"
    NO_LOCAL_CPU = "no_local_cpu"


@dataclass
class ProviderSlot:
    name: str
    provider: OllamaLLMProvider | GeminiLLMProvider
    breaker: CircuitBreaker


class ProviderRouter:
    def __init__(self, settings: AudioBenchSettings) -> None:
        self._settings = settings
        self._slots: list[ProviderSlot] = self._build_slots()
        self._digest = DigestProvider()

    def _build_slots(self) -> list[ProviderSlot]:
        slots = []
        threshold = self._settings.circuit_breaker_threshold
        timeout = self._settings.circuit_breaker_timeout_seconds
        for name in self._settings.llm_provider_order:
            if name == "ollama" and not self._settings.llm_local_enabled:
                continue
            if name == "gemini" and not self._settings.llm_cloud_enabled:
                continue
            provider = self._make_provider(name)
            if provider is None:
                continue
            slots.append(
                ProviderSlot(
                    name=name,
                    provider=provider,
                    breaker=CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout),
                )
            )
        return slots

    def _make_provider(self, name: str) -> OllamaLLMProvider | GeminiLLMProvider | None:
        """Construct a provider adapter. Never probes network availability."""
        if name == "ollama":
            try:
                from audiobench.chat.providers.ollama_provider import OllamaClient

                client = OllamaClient(
                    base_url=self._settings.ollama_base_url,
                    model=self._settings.ollama_model,
                )
                return OllamaLLMProvider(client)
            except Exception:
                logger.exception("Failed to construct Ollama provider")
                return None
        if name == "gemini":
            api_key = self._settings.gemini_api_key
            if not api_key:
                logger.info(
                    "Gemini slot skipped: GEMINI_API_KEY / gemini_api_key not configured. "
                    "If Gemini fallback is not working, this is why."
                )
                return None
            try:
                from audiobench.chat.providers.gemini_provider import GeminiClient

                client = GeminiClient(api_key=api_key, model=self._settings.gemini_model)
                return GeminiLLMProvider(client)
            except Exception:
                logger.exception("Failed to construct Gemini provider")
                return None
        logger.warning("Unknown provider name in llm_provider_order: %s — skipped", name)
        return None

    def _discover_local_ollama(self) -> str | None:
        """Find the smallest non-cloud model on the local Ollama server."""
        import re

        try:
            from audiobench.chat.providers.ollama_provider import OllamaClient

            probe = OllamaClient(base_url=self._settings.ollama_base_url)
            if not probe.is_available():
                return None
            models = probe.list_models()
            local = [m for m in models if "-cloud" not in m]
            if not local:
                return None

            def _param_rank(name: str) -> int:
                m = re.search(r"(\d+)b", name.lower())
                return int(m.group(1)) if m else 99

            return sorted(local, key=_param_rank)[0]
        except Exception:
            return None

    def _swap_ollama_slot(self, slot: ProviderSlot, local_model: str) -> None:
        """Swap slot's provider to a local model in-place. Resets breaker."""
        from audiobench.chat.providers.ollama_provider import OllamaClient

        new_client = OllamaClient(base_url=self._settings.ollama_base_url, model=local_model)
        slot.provider = OllamaLLMProvider(new_client)
        slot.breaker.reset()

    def set_primary_model(self, model: str) -> None:
        """Set primary model and reorder slots so the matching provider is tried first."""
        if model.lower().startswith("gemini"):
            gemini_slots = [s for s in self._slots if s.name == "gemini"]
            other_slots = [s for s in self._slots if s.name != "gemini"]
            if gemini_slots:
                self._slots = gemini_slots + other_slots
            return
        for i, slot in enumerate(self._slots):
            if slot.name == "ollama":
                from audiobench.chat.providers.ollama_provider import OllamaClient

                new_client = OllamaClient(base_url=self._settings.ollama_base_url, model=model)
                self._slots[i] = ProviderSlot(
                    name="ollama",
                    provider=OllamaLLMProvider(new_client),
                    breaker=CircuitBreaker(
                        failure_threshold=self._settings.circuit_breaker_threshold,
                        recovery_timeout=self._settings.circuit_breaker_timeout_seconds,
                    ),
                )
                ollama_slots = [s for s in self._slots if s.name == "ollama"]
                other_slots = [s for s in self._slots if s.name != "ollama"]
                self._slots = ollama_slots + other_slots
                return

    @property
    def model(self) -> str:
        for slot in self._slots:
            client = getattr(slot.provider, "_client", None)
            if client is not None and hasattr(client, "model"):
                return client.model
        return self._settings.ollama_model

    @property
    def _model(self) -> str:
        return self.model

    def is_available(self, provider: str | None = None) -> bool:
        """Check if a specific provider or at least one underlying provider is available."""
        for slot in self._slots:
            if provider and slot.name != provider:
                continue
            if not slot.breaker.can_call():
                continue
            client = getattr(slot.provider, "_client", None)
            if client is not None and hasattr(client, "is_available"):
                try:
                    if client.is_available():
                        return True
                except Exception:
                    continue
            else:
                return True
        return False

    def list_models(self) -> list[str]:
        """List models available across underlying providers (e.g. Ollama)."""
        models: list[str] = []
        for slot in self._slots:
            client = getattr(slot.provider, "_client", None)
            if client is not None and hasattr(client, "list_models"):
                try:
                    for m in client.list_models():
                        if m not in models:
                            models.append(m)
                except Exception:
                    pass
        return models

    def available_providers(self) -> list[str]:
        return [s.name for s in self._slots if s.breaker.can_call()]

    def _router_retry(
        self,
        fn: Callable[[], str],
        retry_after: float | None,
    ) -> tuple[bool, str | Exception]:
        """Wraps _retry_with_backoff to bridge ProviderRateLimitError ↔ legacy RateLimitError."""
        from audiobench.memory.llm_caller import RateLimitError, _retry_with_backoff

        def _adapted():
            try:
                return fn()
            except ProviderRateLimitError as e:
                raise RateLimitError(str(e)) from e

        try:
            content = _retry_with_backoff(
                _adapted,
                max_retries=3,
                base_delay=retry_after or 1.0,
                jitter=True,
            )
            return True, content
        except RateLimitError as e:
            return False, e
        except Exception as e:
            return False, e

    def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        context_fragments: list[Any] | None = None,
        digest_mode: str = "silent",
        disclosure_fn: Callable[[str], None] | None = None,
        fallback_policy: FallbackPolicy = FallbackPolicy.ALLOW_LOCAL,
        **kw,
    ) -> LlmOutcome:
        """Always returns LlmOutcome. Never raises on provider failure."""
        if not self._settings.llm_chain_enabled:
            err = ProviderFatalError("LLM chain disabled (llm_chain_enabled=False)")
            return LlmOutcome(type=OutcomeType.FAILED, content="", error=err)

        prev_name: str | None = None
        prev_err: str | None = None

        idx = 0
        while idx < len(self._slots):
            slot = self._slots[idx]
            retry_this_slot = False

            if not slot.breaker.can_call():
                prev_name, prev_err = slot.name, "circuit_open"
                idx += 1
                continue

            try:
                content = slot.provider.generate(
                    messages,
                    temperature=temperature,
                    context_fragments=context_fragments,
                    digest_mode=digest_mode,
                    **kw,
                )
                slot.breaker.record_success()
                if prev_name and disclosure_fn:
                    disclosure_fn(f"ⓘ {prev_name} unavailable ({prev_err}) — using {slot.name}")
                return LlmOutcome(
                    type=OutcomeType.OK,
                    content=content,
                    provider=slot.name,
                    fallback_from=prev_name,
                    fallback_reason=prev_err,
                )

            except ProviderRateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                ok, result = self._router_retry(
                    lambda: slot.provider.generate(messages, temperature=temperature, **kw),
                    retry_after=retry_after,
                )
                if ok:
                    slot.breaker.record_success()
                    return LlmOutcome(
                        type=OutcomeType.OK,
                        content=result,  # type: ignore[arg-type]
                        provider=slot.name,
                        fallback_from=prev_name,
                        fallback_reason=prev_err,
                    )
                if isinstance(result, ProviderFatalError):
                    if disclosure_fn:
                        disclosure_fn(
                            f"✗ {slot.name} fatal error during retry: {result} — no fallback available"
                        )
                    return LlmOutcome(
                        type=OutcomeType.FAILED,
                        content="",
                        provider=slot.name,
                        fallback_from=prev_name,
                        fallback_reason=prev_err,
                        error=result,
                    )
                # quota or generic rate-limit exhaustion — advance to next slot
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, str(result) if isinstance(result, ProviderQuotaError) else "rate_limit (retries exhausted)"

            except ProviderQuotaError as e:
                # Record the failure and let the loop advance to the next slot
                # (e.g. Gemini). Local CPU discovery is deferred to post-loop so
                # cloud alternatives are always tried first.
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, str(e)

            except (ProviderUnavailableError, ProviderTimeoutError) as e:
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, str(e)

            except ProviderFatalError as e:
                if disclosure_fn:
                    disclosure_fn(f"✗ {slot.name} fatal error: {e} — no fallback available")
                return LlmOutcome(
                    type=OutcomeType.FAILED,
                    content="",
                    provider=slot.name,
                    fallback_from=prev_name,
                    fallback_reason=prev_err,
                    error=e,
                )

            except Exception as e:
                logger.exception("Unexpected error from provider %s", slot.name)
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, f"unexpected: {type(e).__name__}"

            if not retry_this_slot:
                idx += 1

        # ── All configured slots exhausted ────────────────────────────────────
        # Last resort: try a small local Ollama model — but ONLY when the caller
        # permits it AND the failure was quota exhaustion (not a network outage or fatal error).
        # Search-pipeline callers pass NO_LOCAL_CPU because a 2-minute CPU inference
        # is worse than an honest degradation; interactive chat passes ALLOW_LOCAL (the default).
        if (
            fallback_policy == FallbackPolicy.ALLOW_LOCAL
            and prev_err
            and "quota" in prev_err.lower()
        ):
            local = self._discover_local_ollama()
            if local:
                try:
                    from audiobench.chat.providers.ollama_provider import OllamaClient
                    local_client = OllamaClient(
                        base_url=self._settings.ollama_base_url, model=local
                    )
                    # Force think=False — local CPU models with thinking tokens can
                    # stall for minutes; better to get a fast answer without CoT.
                    result = local_client.chat(messages, temperature=temperature, think=False)
                    content = result.get("content", "") if isinstance(result, dict) else str(result)
                    if disclosure_fn:
                        disclosure_fn(
                            f"ⓘ All cloud providers unavailable — using local fallback {local}"
                        )
                    return LlmOutcome(
                        type=OutcomeType.OK,
                        content=content,
                        provider=f"ollama-local:{local}",
                        fallback_from=prev_name,
                        fallback_reason=prev_err,
                    )
                except Exception as e:
                    logger.warning("Local fallback %s also failed: %s", local, e)

        # All slots and local fallback exhausted — degrade
        digest_content = self._digest.generate(
            messages, digest_mode=digest_mode, context_fragments=context_fragments, **kw
        )
        if disclosure_fn and prev_name:
            disclosure_fn(
                f"⚠ All providers unavailable ({prev_err}) — showing degraded response ({digest_mode})"
            )
        return LlmOutcome(
            type=OutcomeType.DEGRADED,
            content=digest_content,
            provider="digest",
            fallback_from=prev_name,
            fallback_reason=prev_err,
            degraded_mode=digest_mode,
        )

    def generate_text(self, messages: list[dict], **kw) -> str:
        """Convenience wrapper. Returns the content string directly."""
        outcome = self.generate(messages, **kw)
        match outcome.type:
            case OutcomeType.FAILED:
                if isinstance(outcome.error, ProviderFatalError):
                    raise outcome.error
                raise AllProvidersFailedError(outcome.error)
            case OutcomeType.DEGRADED:
                if not outcome.content:
                    raise AllProvidersFailedError(
                        outcome.error or Exception(outcome.fallback_reason or "All providers unavailable")
                    )
                return outcome.content
            case OutcomeType.ABSTAINED:
                return ""
            case _:
                return outcome.content

    def generate_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        think: bool = True,
        digest_mode: str = "notice",
        disclosure_fn: Callable[[str], None] | None = None,
        **kw,
    ) -> Iterator[dict]:
        """Streaming variant of generate."""
        prev_name: str | None = None
        prev_err: str | None = None

        idx = 0
        while idx < len(self._slots):
            slot = self._slots[idx]
            retry_this_slot = False

            if not slot.breaker.can_call():
                prev_name, prev_err = slot.name, "circuit_open"
                idx += 1
                continue

            tokens_emitted = False
            try:
                for chunk in slot.provider.generate_stream(
                    messages, temperature=temperature, think=think, **kw
                ):
                    if not tokens_emitted and prev_name and disclosure_fn:
                        disclosure_fn(f"ⓘ {prev_name} unavailable — using {slot.name}")
                    tokens_emitted = True
                    yield chunk
                slot.breaker.record_success()
                return

            except ProviderFatalError as e:
                if disclosure_fn:
                    disclosure_fn(f"✗ {slot.name} fatal error: {e} — no fallback available")
                if tokens_emitted:
                    yield {
                        "notice": f"✗ {slot.name} fatal error — stream aborted",
                        "content": "",
                        "done": False,
                    }
                yield {
                    "content": "",
                    "done": True,
                    "failed": True,
                    "error": str(e),
                    "provider": slot.name,
                }
                return

            except ProviderRateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                ok, result = self._router_retry(
                    lambda: "".join(
                        c.get("content", "")
                        for c in slot.provider.generate_stream(
                            messages, temperature=temperature, **kw
                        )
                    ),
                    retry_after=retry_after,
                )
                if ok:
                    slot.breaker.record_success()
                    yield {"content": result, "done": True}
                    return
                if isinstance(result, ProviderFatalError):
                    if disclosure_fn:
                        disclosure_fn(
                            f"✗ {slot.name} fatal error during retry: {result} — no fallback available"
                        )
                    if tokens_emitted:
                        yield {
                            "notice": f"✗ {slot.name} fatal error — stream aborted",
                            "content": "",
                            "done": False,
                        }
                    yield {
                        "content": "",
                        "done": True,
                        "failed": True,
                        "error": str(result),
                        "provider": slot.name,
                    }
                    return
                if isinstance(result, ProviderQuotaError):
                    slot.breaker.record_failure()
                    prev_name, prev_err = slot.name, str(result)
                    if tokens_emitted:
                        yield {
                            "notice": f"⚡ {slot.name} quota exhausted — restarting on next provider",
                            "content": "",
                            "done": False,
                        }
                    # Advance to next slot (e.g. Gemini). Local swap is deferred to post-loop.
                else:
                    slot.breaker.record_failure()
                    prev_name, prev_err = slot.name, "rate_limit (retries exhausted)"
                    if tokens_emitted:
                        yield {
                            "notice": f"⚡ {slot.name} rate-limited — restarting on next provider",
                            "content": "",
                            "done": False,
                        }

            except ProviderQuotaError as e:
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, str(e)
                if tokens_emitted:
                    yield {
                        "notice": f"⚡ {slot.name} quota exhausted — restarting on next provider",
                        "content": "",
                        "done": False,
                    }
                # Advance to next slot. Local swap deferred to post-loop.

            except Exception as e:
                slot.breaker.record_failure()
                prev_name, prev_err = slot.name, str(e)
                if tokens_emitted:
                    yield {
                        "notice": f"⚡ {slot.name} interrupted — restarting on next provider",
                        "content": "",
                        "done": False,
                    }

            if not retry_this_slot:
                idx += 1

        digest_content = self._digest.generate(messages, digest_mode=digest_mode, **kw)
        if disclosure_fn:
            disclosure_fn(f"⚠ All providers unavailable — showing degraded response ({digest_mode})")
        yield {
            "content": digest_content,
            "done": True,
            "degraded": True,
            "degraded_mode": digest_mode,
        }

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        num_ctx: int | None = None,
        think: bool = True,
    ) -> Iterator[dict]:
        yield from self.generate_stream(messages, temperature=temperature, think=think)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
        think: bool = False,
        **kw,
    ) -> dict[str, Any]:
        """Duck-type alias matching OllamaClient.chat() interface."""
        outcome = self.generate(messages, temperature=temperature, **kw)
        return {"content": outcome.content, "thinking": None}

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        **kw,
    ) -> Iterator[str]:
        """Stream response tokens matching OllamaClient.stream() interface."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        for chunk in self.generate_stream(messages, model=model, temperature=temperature, think=False, **kw):
            if chunk.get("content"):
                yield chunk["content"]


_router_instance: ProviderRouter | None = None
_router_lock = threading.Lock()


def router_for_settings(settings: AudioBenchSettings | None = None) -> ProviderRouter:
    global _router_instance
    if _router_instance is None:
        with _router_lock:
            if _router_instance is None:
                from audiobench.core.settings import get_settings

                _router_instance = ProviderRouter(settings or get_settings())
    return _router_instance


def invalidate_router() -> None:
    global _router_instance
    with _router_lock:
        _router_instance = None
