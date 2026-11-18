"""Tests for ResearchEngine retrieval enhancements (synthesis floor, source cap, pinned fragments, source labeling)."""

from __future__ import annotations

from audiobench.memory.query_engine import ResearchEngine
from audiobench.memory.query_reformulator import ReformulatedQuery
from audiobench.memory.rrf_fusion import FusedResult


def any_rq() -> ReformulatedQuery:
    return ReformulatedQuery(
        original="test query",
        bm25_keywords="test",
        semantic_query="test",
        hyde_anchor="test",
    )


def make_fused(sid: int, score: float, source: str = "podcast_episode.m4a", text: str = "sample text") -> FusedResult:
    return FusedResult(
        segment_id=sid,
        start_time=10.0,
        end_time=20.0,
        text=text,
        source_file=source,
        rrf_score=score,
        stream_contributions=(("fts5", 1),),
    )


def test_research_engine_synthesis_floor_and_source_labeling(monkeypatch):
    """Verify low-score fragments (< 0.020) are excluded from synthesis unless pinned, and source stem is included in header."""
    from unittest.mock import MagicMock
    from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType

    captured_prompt: list[str] = []

    mock_router = MagicMock()

    def _capture_generate(messages, **kwargs):
        # capture the user message content as the "prompt"
        for m in messages:
            if m.get("role") == "user":
                captured_prompt.append(m["content"])
                break
        return LlmOutcome(type=OutcomeType.OK, content="synthesised answer", provider="ollama")

    mock_router.generate.side_effect = _capture_generate

    monkeypatch.setattr(
        "audiobench.memory.query_engine.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.QueryReformulator.reformulate",
        lambda self, q: any_rq(),
    )

    # We mock rrf_merge directly to return our controlled FusedResult list
    high_hit = make_fused(1, 0.035, source="high_source.m4a", text="high score content")
    low_hit = make_fused(2, 0.015, source="low_source.m4a", text="subthreshold unpinned content")
    pinned_hit = make_fused(3, 0.010, source="pinned_source.m4a", text="pinned low score content")

    monkeypatch.setattr(
        "audiobench.memory.query_engine.rrf_merge",
        lambda fts, dense, colbert, top_n, **kwargs: [high_hit, low_hit],
    )
    monkeypatch.setattr(
        "audiobench.memory.query_engine._temporal_dedup",
        lambda hits: hits,
    )

    engine = ResearchEngine()

    # Case 1: Without pinned fragments, low_hit (score 0.015 < 0.020) should be filtered out from synthesis prompt
    res = engine.search("test query")
    assert res.answer == "synthesised answer"
    assert len(captured_prompt) == 1
    prompt_text = captured_prompt[0]
    assert "high score content" in prompt_text
    assert "high_source" in prompt_text  # Source stem labeling
    assert "subthreshold unpinned content" not in prompt_text  # Filtered by SYNTHESIS_RRF_FLOOR

    # Case 2: With pinned_hit passed, even though its score is 0.010, it should be prepended and included in synthesis
    captured_prompt.clear()
    res_pinned = engine.search("test query", pinned_fragments=[pinned_hit])
    assert len(res_pinned.sources) == 3
    assert res_pinned.sources[0].segment_id == 3  # Pinned is prepended
    assert len(captured_prompt) == 1
    prompt_pinned = captured_prompt[0]
    assert "pinned low score content" in prompt_pinned
    assert "pinned_source" in prompt_pinned
    assert "high score content" in prompt_pinned
    assert "subthreshold unpinned content" not in prompt_pinned
