"""Tests for ResearchEngine parallel orchestrator."""

import dataclasses
import time

import pytest

from audiobench.memory.query_engine import ResearchEngine
from audiobench.memory.query_reformulator import ReformulatedQuery
from audiobench.memory.retrieval_streams import SegmentHit

# ── Helpers ───────────────────────────────────────────────────────────────────

def any_rq() -> ReformulatedQuery:
    return ReformulatedQuery(
        original="test",
        bm25_keywords="test",
        semantic_query="test",
        hyde_anchor="test",
    )


def _make_hit(segment_id: int, start: float = 1.0, end: float = 2.0) -> SegmentHit:
    return SegmentHit(
        segment_id=segment_id,
        start_time=start,
        end_time=end,
        text=f"segment {segment_id}",
    )


@pytest.fixture(autouse=True)
def _stub_synthesis_and_llm(monkeypatch):
    """Prevent ResearchEngine tests from hanging on external Ollama or loading models in-process."""
    from unittest.mock import MagicMock
    from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType

    mock_router = MagicMock()
    mock_router.generate.return_value = LlmOutcome(
        type=OutcomeType.OK,
        content="mock answer",
        provider="ollama",
    )
    monkeypatch.setattr(
        "audiobench.memory.query_engine.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.router_for_settings",
        lambda *a, **kw: mock_router,
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.SynthesisStream.retrieve",
        lambda *a, **kw: [],
    )


# ── Parallel test ─────────────────────────────────────────────────────────────

def test_all_three_streams_called_in_parallel(monkeypatch):
    """FTS5, dense, and ColBERT must be called concurrently, not sequentially."""
    call_times: dict[str, float] = {}

    def slow_fts(self, rq, top_k, **kwargs):
        call_times["fts_start"] = time.perf_counter()
        time.sleep(0.1)
        return []

    def slow_dense(self, rq, top_k, **kwargs):
        call_times["dense_start"] = time.perf_counter()
        time.sleep(0.1)
        return []

    def slow_colbert(self, rq, top_k, **kwargs):
        call_times["colbert_start"] = time.perf_counter()
        time.sleep(0.1)
        return []

    monkeypatch.setattr("audiobench.memory.retrieval_streams.FTS5Stream.retrieve", slow_fts)
    monkeypatch.setattr("audiobench.memory.retrieval_streams.DenseStream.retrieve", slow_dense)
    monkeypatch.setattr("audiobench.memory.retrieval_streams.ColBERTStream.retrieve", slow_colbert)
    monkeypatch.setattr("audiobench.memory.retrieval_streams.SynthesisStream.retrieve", lambda *a, **kw: [])

    # Also stub reformulator to avoid real LLM call
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.QueryReformulator.reformulate",
        lambda self, q: any_rq(),
    )

    t0 = time.perf_counter()
    ResearchEngine().search("test query", top_k=5)
    total = time.perf_counter() - t0

    # If sequential: ~0.3s; parallel: ~0.1s + overhead
    assert total < 0.25, f"Streams ran sequentially (took {total:.3f}s)"
    # All three must have started within 50ms of each other
    starts = list(call_times.values())
    assert len(starts) == 3, "Not all streams were called"
    assert max(starts) - min(starts) < 0.05, (
        f"Streams started too far apart: spread={max(starts) - min(starts):.3f}s"
    )


def test_research_engine_result_has_timestamps(monkeypatch):
    """Every FusedResult must have non-zero timestamps when FTS5 finds matches."""
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.QueryReformulator.reformulate",
        lambda self, q: any_rq(),
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream.retrieve",
        lambda self, rq, top_k, **kw: [_make_hit(1, 10.0, 20.0), _make_hit(2, 25.0, 35.0)],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.DenseStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.ColBERTStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )

    result = ResearchEngine().search("learning")
    assert len(result.sources) > 0
    for fr in result.sources:
        assert fr.start_time >= 0.0
        assert fr.end_time > fr.start_time


def test_research_engine_result_is_immutable(monkeypatch):
    """FusedResult objects returned by the engine must be immutable (frozen dataclass)."""
    monkeypatch.setattr(
        "audiobench.memory.query_reformulator.QueryReformulator.reformulate",
        lambda self, q: any_rq(),
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream.retrieve",
        lambda self, rq, top_k, **kw: [_make_hit(1, 1.0, 2.0)],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.DenseStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.ColBERTStream.retrieve",
        lambda self, rq, top_k, **kw: [],
    )

    result = ResearchEngine().search("test")
    if result.sources:
        fr = result.sources[0]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            fr.rrf_score = 0.0  # type: ignore[misc]


def test_fts5_stream_not_using_model_inference(monkeypatch):
    """FTS5 stream must NOT load any ML model. Pure SQL only."""
    from audiobench.memory.retrieval_streams import FTS5Stream

    model_loaded: list[int] = []
    # Guard: if singletons module exists and is imported, patch it
    try:
        monkeypatch.setattr(
            "audiobench.memory.singletons.get_primary_embedder",
            lambda: model_loaded.append(1) or None,
        )
    except AttributeError:
        pass  # module not imported yet — that's fine, it shouldn't be

    FTS5Stream().retrieve(any_rq(), top_k=5)
    assert model_loaded == [], "FTS5 stream must not load ML model"


def test_dense_and_colbert_route_through_daemon_not_singletons(monkeypatch):
    """Dense and ColBERT streams must NOT call local singletons directly."""
    from unittest.mock import MagicMock

    from audiobench.memory.retrieval_streams import ColBERTStream, DenseStream

    singleton_calls: list[str] = []
    try:
        monkeypatch.setattr(
            "audiobench.memory.singletons.get_primary_embedder",
            lambda: singleton_calls.append("embedder") or None,
        )
        monkeypatch.setattr(
            "audiobench.memory.singletons.get_colbert_reranker",
            lambda: singleton_calls.append("colbert") or None,
        )
    except AttributeError:
        pass

    # Mock get_daemon_client so we don't fall back to LocalDaemonClient in tests
    # (LocalDaemonClient intentionally loads singletons)
    mock_client = MagicMock()
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    DenseStream().retrieve(any_rq(), top_k=5)
    ColBERTStream().retrieve(any_rq(), top_k=5)
    assert singleton_calls == [], "Streams must use daemon, not local singletons"


def _no_retrieve(self, rq, top_k, **kwargs):
    """Stub used for retrieve patches that must survive **extra kwargs."""
    return []
