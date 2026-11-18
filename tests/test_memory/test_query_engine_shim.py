"""Regression test for query_engine _call_llm shim."""
from __future__ import annotations

from unittest.mock import MagicMock

from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.memory import query_engine


def test_call_llm_shim_returns_err_for_degraded_outcome(monkeypatch):
    """When both providers fail, _call_llm returns Err (not Ok with empty content).
    This preserves synthesis_engine.py's extractive digest fallback in wave 1.
    Regression guard: using is_usable instead of OutcomeType.OK would return Ok(""),
    causing synthesis_engine to silently produce a blank synthesis response."""
    fake_router = MagicMock()
    fake_router.generate.return_value = LlmOutcome(
        type=OutcomeType.DEGRADED,
        content="",
        provider="digest",
        degraded_mode="silent",
    )
    monkeypatch.setattr(
        "audiobench.chat.providers.router.router_for_settings",
        lambda settings=None: fake_router,
    )

    result = query_engine._call_llm("test prompt", 0.3, MagicMock(), "key")
    assert isinstance(result, query_engine.Err)
    assert "degraded" in result.error.lower()
