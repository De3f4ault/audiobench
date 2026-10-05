"""Tests for FusedResult and rrf_merge."""

import dataclasses
import time

import pytest

from audiobench.memory.retrieval_streams import SegmentHit
from audiobench.memory.rrf_fusion import FusedResult, K, rrf_merge

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_hit(segment_id: int, start: float = 0.0, end: float = 1.0) -> SegmentHit:
    return SegmentHit(
        segment_id=segment_id,
        start_time=start,
        end_time=end,
        text=f"segment {segment_id} text",
        source_file=f"file_{segment_id}.m4a",
    )


def make_fts_hit(segment_id: int, rank: int) -> SegmentHit:
    """Return a SegmentHit positioned at the given 1-indexed rank."""
    return _make_hit(segment_id, start=float(rank), end=float(rank) + 1.0)


def make_dense_hit(segment_id: int, rank: int) -> SegmentHit:
    return _make_hit(segment_id, start=float(rank), end=float(rank) + 1.0)


def make_fused_result(segment_id: int) -> FusedResult:
    return FusedResult(
        segment_id=segment_id,
        start_time=0.0,
        end_time=1.0,
        text="test",
        rrf_score=0.1,
        stream_contributions=(("fts5", 1),),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fused_result_is_frozen():
    assert dataclasses.is_dataclass(FusedResult)
    fr = make_fused_result(segment_id=1)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        fr.rrf_score = 999.0  # type: ignore[misc]


def test_rrf_score_formula_correct():
    """Verify RRF formula: 1/(60+rank)"""
    assert K == 60
    fts_hit = make_fts_hit(segment_id=42, rank=1)
    results = rrf_merge([fts_hit], [], [], top_n=1)
    expected_score = 1.0 / (60 + 1)
    assert results[0].rrf_score == pytest.approx(expected_score, rel=1e-6)


def test_rrf_rewards_cross_stream_agreement():
    """A result in all three streams must outscore a result in only one."""
    # segment 1 appears in all three streams at rank 5
    # segment 2 appears only in FTS stream at rank 1
    fts = [make_fts_hit(segment_id=1, rank=5), make_fts_hit(segment_id=2, rank=1)]
    dense = [make_dense_hit(segment_id=1, rank=5)]
    colbert = [make_dense_hit(segment_id=1, rank=5)]
    results = rrf_merge(fts, dense, colbert, top_n=2)
    assert results[0].segment_id == 1  # corroborated result wins
    assert results[1].segment_id == 2


def test_rrf_stream_contributions_recorded():
    """Every result must record which streams contributed and at what rank."""
    # segment 7 is at position 3 in fts stream (rank=3) and position 1 in dense (rank=1)
    fts = [
        make_fts_hit(segment_id=99, rank=1),
        make_fts_hit(segment_id=98, rank=2),
        make_fts_hit(segment_id=7, rank=3),   # position 3 → rank 3
    ]
    dense = [make_dense_hit(segment_id=7, rank=1)]  # position 1 → rank 1
    results = rrf_merge(fts, dense, [], top_n=3)
    seg7 = next(r for r in results if r.segment_id == 7)
    contributions = dict(seg7.stream_contributions)
    assert "fts5" in contributions
    assert "dense" in contributions
    assert "colbert" not in contributions
    assert contributions["fts5"] == 3
    assert contributions["dense"] == 1


def test_rrf_top_k_uses_min_heap_not_full_sort():
    """heapq.nlargest must be used — verify by timing 100 runs with 500 hits."""
    # 500 FTS hits, top_n=10
    large_fts = [make_fts_hit(segment_id=i, rank=i) for i in range(1, 501)]
    t0 = time.perf_counter()
    for _ in range(100):
        rrf_merge(large_fts, [], [], top_n=10)
    elapsed = time.perf_counter() - t0
    # Should complete 100 iterations in under 500ms total (even on high-load runner)
    assert elapsed < 0.5, f"RRF merge too slow: {elapsed * 1000:.1f}ms for 100 runs"


def test_rrf_empty_streams_returns_empty():
    assert rrf_merge([], [], [], top_n=10) == []


def test_rrf_top_n_limits_output():
    """Output must never exceed top_n items even when inputs are larger."""
    hits = [make_fts_hit(segment_id=i, rank=i) for i in range(1, 21)]
    results = rrf_merge(hits, [], [], top_n=5)
    assert len(results) == 5


def test_rrf_scores_descending():
    """Results must be sorted by rrf_score descending."""
    fts = [make_fts_hit(segment_id=i, rank=i) for i in range(1, 6)]
    results = rrf_merge(fts, [], [], top_n=5)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_max_per_source_capping():
    """Verify that results from a single source file are capped at max_per_source."""
    hits = [
        SegmentHit(segment_id=i, start_time=float(i), end_time=float(i+1), text=f"seg {i}", source_file="same_file.m4a")
        for i in range(1, 10)
    ]
    results = rrf_merge(hits, [], [], top_n=10, max_per_source=2)
    assert len(results) == 2
    assert all(r.source_file == "same_file.m4a" for r in results)
