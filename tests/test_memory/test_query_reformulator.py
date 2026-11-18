"""Tests for QueryReformulator and ReformulatedQuery dataclass."""

import dataclasses
import json
from unittest.mock import MagicMock

import pytest

from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.memory.query_reformulator import QueryReformulator, ReformulatedQuery


@pytest.fixture
def mock_llm_returns_valid_json(monkeypatch):
    """Mocks router to return a valid JSON payload with sufficient lengths."""
    payload = {
        "bm25_keywords": "bike incident lesson safety road injury",
        "semantic_query": "The user is looking for reflections or lessons learned from a past incident involving a bicycle.",
        "hyde_anchor": "When I look back at the bike incident, I realize the importance of safety gear. " * 15 # Ensure it's long enough
    }
    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.OK,
        content=json.dumps(payload),
        provider="ollama",
    )
    monkeypatch.setattr("audiobench.memory.query_reformulator.router_for_settings", lambda *a, **kw: mock_router)
    return mock_router


@pytest.fixture
def mock_llm_fails(monkeypatch):
    """Mocks router to return a FAILED outcome."""
    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.FAILED,
        content="",
        error=Exception("Connection refused"),
    )
    monkeypatch.setattr("audiobench.memory.query_reformulator.router_for_settings", lambda *a, **kw: mock_router)
    return mock_router


@pytest.fixture
def mock_llm_returns_garbage(monkeypatch):
    """Mocks router to return invalid JSON."""
    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.OK,
        content="This is definitely not json!",
        provider="ollama",
    )
    monkeypatch.setattr("audiobench.memory.query_reformulator.router_for_settings", lambda *a, **kw: mock_router)
    return mock_router


def test_reformulated_query_is_frozen_dataclass():
    assert dataclasses.is_dataclass(ReformulatedQuery)
    # Frozen — mutation must raise
    rq = ReformulatedQuery(
        original="test",
        bm25_keywords="test keywords",
        semantic_query="test semantic meaning",
        hyde_anchor="test hypothetical passage",
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        rq.original = "mutated"


def test_reformulator_returns_reformulated_query(mock_llm_returns_valid_json):
    rq = QueryReformulator().reformulate("what did I learn from the bike incident")
    assert rq.original == "what did I learn from the bike incident"
    assert len(rq.bm25_keywords.split()) >= 3
    assert len(rq.semantic_query) > 20
    assert len(rq.hyde_anchor) >= 80 * 4  # ~80 words minimum


def test_reformulator_fallback_on_llm_failure(mock_llm_fails):
    """When LLM fails, all sub-queries must fall back to the original query. Must not raise."""
    rq = QueryReformulator().reformulate("test query")
    assert rq.original == "test query"
    assert rq.bm25_keywords == "test query"   # fallback: use original
    assert rq.semantic_query == "test query"
    assert len(rq.hyde_anchor) > 0  # fallback: some non-empty string


def test_reformulator_fallback_on_invalid_json(mock_llm_returns_garbage):
    """Malformed JSON from LLM must degrade gracefully, not crash."""
    rq = QueryReformulator().reformulate("any query")
    assert isinstance(rq.bm25_keywords, str)
    assert len(rq.bm25_keywords) > 0
    assert rq.bm25_keywords == "any query"


def test_reformulator_fallback_on_degraded_outcome(monkeypatch):
    """Gate 2: DEGRADED outcome must trigger fallback to original query, NOT parse digest as JSON."""
    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.DEGRADED,
        content="### Dialectic Digest: test query\n\n*(LLM service unavailable)*\n\n**[1] Author** — *Work*",
        degraded_mode="extractive",
    )
    monkeypatch.setattr("audiobench.memory.query_reformulator.router_for_settings", lambda *a, **kw: mock_router)
    rq = QueryReformulator().reformulate("test query")
    assert rq.original == "test query"
    assert rq.bm25_keywords == "test query"
    assert rq.semantic_query == "test query"
    assert rq.hyde_anchor == "Hypothetical answer to the query: test query"
