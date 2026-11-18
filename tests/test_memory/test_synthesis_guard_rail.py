"""Tests for P1-5: Synthesis Failure Guard Rail."""

import pytest

from audiobench.memory.query_engine import ResearchEngine, ResearchResult
from audiobench.memory.query_types import ReformulatedQuery
from audiobench.memory.rrf_fusion import FusedResult

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fused(segment_id: int, start: float = 10.0, end: float = 20.0) -> FusedResult:
    return FusedResult(
        segment_id=segment_id,
        start_time=start,
        end_time=end,
        text=f"Fragment text for segment {segment_id}.",
        rrf_score=1.0 / 61,
        stream_contributions=(("fts5", 1),),
    )


def _any_rq() -> ReformulatedQuery:
    return ReformulatedQuery(
        original="test",
        bm25_keywords="test",
        semantic_query="test",
        hyde_anchor="test",
    )


def make_failed_query_result(sources: int = 3) -> ResearchResult:
    return ResearchResult(
        query="test query",
        sources=[_make_fused(i + 1, start=float(i * 10 + 1), end=float(i * 10 + 9)) for i in range(sources)],
        synthesis_failed=True,
        synthesis_error="ollama: connection refused | gemini: no api key configured",
        answer=None,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_all_llm_fail(monkeypatch):
    """Make all LLM paths return failure so synthesis always fails."""
    from unittest.mock import MagicMock
    from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType

    # Patch router_for_settings so the router.generate() path returns FAILED
    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.FAILED,
        content="",
        error=Exception("all llm providers failed"),
    )
    monkeypatch.setattr(
        "audiobench.memory.query_engine.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    # Stub reformulator to avoid real LLM call
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.QueryReformulator.reformulate",
        lambda self, q: _any_rq(),
    )
    # Stub FTS5 to return hits so we have sources
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream.retrieve",
        lambda self, rq, top_k, **kw: [
            __import__(
                "audiobench.memory.retrieval_streams",
                fromlist=["SegmentHit"]
            ).SegmentHit(
                segment_id=i,
                start_time=float(i * 10),
                end_time=float(i * 10 + 9),
                text=f"segment {i} text",
            )
            for i in range(1, 4)
        ],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.DenseStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.ColBERTStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )


@pytest.fixture
def mock_input_quit(monkeypatch):
    """Simulate user pressing Q at the panel prompt."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "Q")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_query_result_preserves_sources_on_synthesis_failure(mock_all_llm_fail):
    engine = ResearchEngine()
    result = engine.search("test query with results")
    assert (result.synthesis_failed or result.synthesis_is_fallback) is True
    assert result.synthesis_error is not None
    assert len(result.sources) > 0, "Sources must be preserved even when synthesis fails"


def test_query_result_sources_have_timestamps_on_failure(mock_all_llm_fail):
    engine = ResearchEngine()
    result = engine.search("learning from failure")
    for source in result.sources:
        assert source.start_time >= 0.0
        assert source.end_time > source.start_time


def test_cli_displays_segments_on_synthesis_failure(capsys):
    from audiobench.cli.commands.memory_cmd import _display_results
    result = make_failed_query_result(sources=3)
    _display_results(result)
    captured = capsys.readouterr()
    # Must show synthesis failed banner
    assert "synthesis failed" in captured.out.lower() or "[!]" in captured.out or "synthesis✗" in captured.out
    # Must show fragment markers
    assert "FRAGMENT" in captured.out.upper() or "→" in captured.out or "fragments" in captured.out.lower()
    # Must show timestamps in HH:MM:SS format (contains ":")
    assert ":" in captured.out


def test_cli_shows_panel_prompt_on_synthesis_failure(mock_input_quit, capsys):
    """Even on synthesis failure, the inner loop prompt must appear."""
    from unittest.mock import MagicMock

    from audiobench.cli.commands.memory_cmd import SearchSessionState, _run_search_loop

    result = make_failed_query_result(sources=2)
    mock_engine = MagicMock()
    mock_engine.search.return_value = result

    _run_search_loop(mock_engine, "test", SearchSessionState())  # mock_input_quit makes user "press Q"

    captured = capsys.readouterr()
    assert "[Q]" in captured.out or "quit" in captured.out.lower()

def test_research_result_synthesis_fields_default_to_not_failed():
    """New ResearchResult must default to synthesis_failed=False and answer=None."""
    result = ResearchResult(query="x")
    assert result.synthesis_failed is False
    assert result.answer is None
    assert result.synthesis_error is None
