"""LLM provider adapters and fallback digest provider for ProviderRouter."""
from __future__ import annotations

import logging
from typing import Any, Iterator

from audiobench.chat.providers.provider_errors import (
    AIError,
    GeminiError,
    GeminiFatalError,
    GeminiQuotaError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    OllamaError,
    OllamaFatalError,
    OllamaQuotaError,
    OllamaRateLimitError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

logger = logging.getLogger(__name__)


def _classify_ollama(e: AIError) -> OllamaError:
    """Read .status_code set by _handle_http_error, fall back to string parse."""
    code = getattr(e, "status_code", None)
    msg = str(e).lower()
    if code == 429 or "rate limit" in msg:
        err = OllamaRateLimitError(str(e))
        err.retry_after = None
        return err
    if code == 402 or any(k in msg for k in ("credit", "quota", "payment", "billing")):
        return OllamaQuotaError(str(e))
    if code in (400, 401, 403, 404) or "not found" in msg or "unauthorized" in msg:
        return OllamaFatalError(str(e))
    if code in (503, 502, 504) or any(k in msg for k in ("connection", "refused", "unavailable")):
        return OllamaUnavailableError(str(e))
    return OllamaUnavailableError(str(e))


def _classify_gemini(e: AIError) -> GeminiError:
    code = getattr(e, "status_code", None)
    msg = str(e).lower()
    if code == 429 or "rate limit" in msg or "resource_exhausted" in msg:
        return GeminiRateLimitError(str(e))
    if code == 402 or "quota" in msg or "billing" in msg:
        return GeminiQuotaError(str(e))
    if code in (400, 401, 403, 404) or "invalid" in msg or "permission" in msg:
        return GeminiFatalError(str(e))
    if code in (503, 502, 504) or "unavailable" in msg:
        return GeminiUnavailableError(str(e))
    return GeminiUnavailableError(str(e))


def _render_extractive_digest(context_fragments: list[Any] | None, query: str = "") -> str:
    """Render dialectic digest matching synthesis_engine.py verbatim."""
    if not context_fragments:
        return ""
    title = f"### Dialectic Digest: {query}" if query else "### Dialectic Digest"
    digest_lines = [
        title,
        "",
        "*(LLM service unavailable — displaying grounded comparative excerpts)*",
        "",
    ]
    for idx, f in enumerate(context_fragments, 1):
        author = getattr(f, "author", None) or (f.get("author") if isinstance(f, dict) else "Unknown")
        work_title = getattr(f, "work_title", None) or (f.get("work_title") if isinstance(f, dict) else "Unknown")
        effective_rel = getattr(f, "effective_rel", None)
        if effective_rel is None and isinstance(f, dict):
            effective_rel = f.get("effective_rel", 0.0)
        elif effective_rel is None:
            effective_rel = 0.0
        content = getattr(f, "content", None) or (f.get("content") if isinstance(f, dict) else "")
        digest_lines.append(
            f"**[{idx}] {author}** — *{work_title}* (Relevance: `{effective_rel:.3f}`)\n"
            f"> \"{content}\"\n"
        )
    return "\n".join(digest_lines)


def _last_n_messages(messages: list[dict], n: int = 3) -> str:
    lines = []
    for m in messages[-n:]:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        lines.append(f"**{role}**: {content}")
    return "\n\n".join(lines)


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


class OllamaLLMProvider:
    name = "ollama"

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def is_cloud(self) -> bool:
        return "-cloud" in (self._client.model or "")

    def generate(self, messages: list[dict], temperature: float = 0.3, **kw) -> str:
        try:
            result = self._client.chat(messages, temperature=temperature, think=False)
            return result["content"]
        except AIError as e:
            raise _classify_ollama(e) from e

    def generate_stream(self, messages: list[dict], temperature: float = 0.3, think: bool = True, **kw) -> Iterator[dict]:
        try:
            yield from self._client.chat_stream(messages, temperature=temperature, think=think)
        except AIError as e:
            raise _classify_ollama(e) from e


class GeminiLLMProvider:
    name = "gemini"
    is_cloud = True

    def __init__(self, client: Any, timeout_seconds: float = 45.0) -> None:
        self._client = client
        self._timeout = timeout_seconds

    def generate(self, messages: list[dict], temperature: float = 0.3, **kw) -> str:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._client.chat, messages, temperature=temperature)
            try:
                res = fut.result(timeout=self._timeout)
                return res["content"] if isinstance(res, dict) else str(res)
            except FuturesTimeout:
                raise GeminiTimeoutError(f"Gemini timed out after {self._timeout}s")
            except AIError as e:
                raise _classify_gemini(e) from e

    def generate_stream(self, messages: list[dict], temperature: float = 0.3, **kw) -> Iterator[dict]:
        try:
            yield from self._client.chat_stream(messages, temperature=temperature)
        except AIError as e:
            raise _classify_gemini(e) from e


class DigestProvider:
    name = "digest"
    is_cloud = False

    def generate(
        self,
        messages: list[dict],
        digest_mode: str = "silent",
        context_fragments: list[Any] | None = None,
        **kw,
    ) -> str:
        if digest_mode == "extractive" and context_fragments:
            query = kw.get("query", "")
            return _render_extractive_digest(context_fragments, query=query)
        if digest_mode == "notice":
            last_turns = _last_n_messages(messages, n=3)
            return (
                "*(LLM services unavailable — displaying context from your library)*\n\n"
                + last_turns
            )
        if digest_mode == "passthrough":
            return _last_user_message(messages)
        return ""

    def generate_stream(
        self,
        messages: list[dict],
        digest_mode: str = "silent",
        context_fragments: list[Any] | None = None,
        **kw,
    ) -> Iterator[dict]:
        content = self.generate(
            messages,
            digest_mode=digest_mode,
            context_fragments=context_fragments,
            **kw,
        )
        yield {"content": content, "done": True}
