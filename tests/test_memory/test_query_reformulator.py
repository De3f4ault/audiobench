"""Tests for QueryReformulator and ReformulatedQuery dataclass."""

import dataclasses
import json

import pytest

from audiobench.memory.query_engine import Err, Ok
from audiobench.memory.query_reformulator import QueryReformulator, ReformulatedQuery


@pytest.fixture
def mock_llm_returns_valid_json(monkeypatch):
    """Mocks _call_llm to return a valid JSON payload with sufficient lengths."""
    def mock_call(*args, **kwargs):
        payload = {
            "bm25_keywords": "bike incident lesson safety road injury",
            "semantic_query": "The user is looking for reflections or lessons learned from a past incident involving a bicycle.",
            "hyde_anchor": "When I look back at the bike incident, I realize the importance of safety gear. " * 15 # Ensure it's long enough
        }
        return Ok(value=json.dumps(payload))
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm", mock_call)
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm_for_model", mock_call)

@pytest.fixture
def mock_llm_fails(monkeypatch):
    """Mocks _call_llm to return an Err."""
    def mock_call(*args, **kwargs):
        return Err(error="Connection refused")
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm", mock_call)
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm_for_model", mock_call)

@pytest.fixture
def mock_llm_returns_garbage(monkeypatch):
    """Mocks _call_llm to return invalid JSON."""
    def mock_call(*args, **kwargs):
        return Ok(value="This is definitely not json!")
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm", mock_call)
    monkeypatch.setattr("audiobench.memory.query_reformulator._call_llm_for_model", mock_call)


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
