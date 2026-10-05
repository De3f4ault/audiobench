"""Reciprocal Rank Fusion (RRF) for blending multi-stream retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

from audiobench.memory.retrieval_streams import SegmentHit

K: int = 60  # RRF constant — must equal 60 per spec


@dataclass(frozen=True)
class FusedResult:
    """A segment hit produced by fusing scores from multiple retrieval streams."""

    segment_id: int
    start_time: float
    end_time: float
    text: str
    rrf_score: float
    source_file: str = ""
    # Tuple of (stream_name, rank) pairs so callers can see which streams contributed
    stream_contributions: tuple[tuple[str, int], ...] = ()
    # Segment IDs removed by temporal deduplication — stored here so the
    # Fragment Reader can still navigate to them via H/L without gaps.
    merged_segment_ids: tuple[int, ...] = ()
    # FK → transcriptions.id; needed by the Fragment Reader for adjacency
    # queries (prev/next segments in the same transcript).
    transcription_id: int = 0


def rrf_merge(
    fts_hits: list[SegmentHit],
    dense_hits: list[SegmentHit],
    colbert_hits: list[SegmentHit],
    top_n: int = 10,
    diversity_weight: float = 0.4,
    max_per_source: int | None = None,
) -> list[FusedResult]:
    """Fuse three ranked lists via RRF with a configurable source-diversity penalty.

    RRF score for a document d = sum over streams of 1 / (K + rank(d, stream))
    where rank is 1-indexed.  Documents appearing in more streams accumulate
    higher scores, rewarding cross-stream agreement.

    After computing raw RRF scores, a diversity penalty is applied so that
    successive fragments from the same source file receive a decaying multiplier::

        penalized_score = raw_rrf_score * 1 / (1 + n * diversity_weight)

    where ``n`` is the number of fragments already seen from that source in
    the raw-score-sorted pass (0-indexed: first fragment gets multiplier 1.0).

    At ``diversity_weight=0.0`` the multiplier is always 1.0 — pure RRF with no
    modification.  At ``diversity_weight=0.4`` (default) the 2nd fragment from a
    source scores at ~71%, the 4th at ~45%, the 6th at ~29% of its raw score.
    A source that still places fragments at those multipliers has genuinely high
    raw RRF scores — i.e. it earned them.  Light over-indexing is pushed down
    without hard-excluding legitimate signal.

    The penalized score is stored in ``FusedResult.rrf_score`` so that all
    downstream consumers (confidence bars, synthesis ranking) see the same signal
    used for ordering.

    O(N log N) for a full sort — important when N is large.
    """
    if not fts_hits and not dense_hits and not colbert_hits:
        return []

    # Map segment_id → {stream: rank, ...} and one representative SegmentHit per id
    contributions: dict[int, dict[str, int]] = {}
    hit_index: dict[int, SegmentHit] = {}

    for stream_name, hits in (
        ("fts5", fts_hits),
        ("dense", dense_hits),
        ("colbert", colbert_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            sid = hit.segment_id
            if sid not in contributions:
                contributions[sid] = {}
                hit_index[sid] = hit
            contributions[sid][stream_name] = rank

    # Compute raw RRF scores
    scores: dict[int, float] = {}
    for sid, stream_ranks in contributions.items():
        scores[sid] = sum(1.0 / (K + r) for r in stream_ranks.values())

    # Apply source-diversity penalty.
    # First pass: iterate in raw-score order to assign per-source occurrence counts;
    # the best fragment from each source always gets multiplier 1.0.
    # Second pass: re-sort by penalized score to determine final ranking.
    source_counts: dict[str, int] = {}
    penalized: dict[int, float] = {}
    for sid in sorted(scores.keys(), key=lambda s: scores[s], reverse=True):
        src = hit_index[sid].source_file
        n = source_counts.get(src, 0)
        multiplier = 1.0 / (1.0 + n * diversity_weight)
        penalized[sid] = scores[sid] * multiplier
        source_counts[src] = n + 1

    # Build final result list in penalized-score order, stopping at top_n.
    results: list[FusedResult] = []
    if max_per_source is None:
        top_sids = heapq.nlargest(top_n, penalized.keys(), key=penalized.__getitem__)
        for sid in top_sids:
            hit = hit_index[sid]
            contribs = contributions[sid]
            results.append(
                FusedResult(
                    segment_id=sid,
                    start_time=hit.start_time,
                    end_time=hit.end_time,
                    text=hit.text,
                    rrf_score=penalized[sid],
                    source_file=hit.source_file,
                    transcription_id=hit.transcription_id,
                    stream_contributions=tuple(sorted(contribs.items())),
                )
            )
    else:
        final_source_counts: dict[str, int] = {}
        for sid in sorted(penalized.keys(), key=penalized.__getitem__, reverse=True):
            hit = hit_index[sid]
            src = hit.source_file
            if final_source_counts.get(src, 0) >= max_per_source:
                continue
            contribs = contributions[sid]
            results.append(
                FusedResult(
                    segment_id=sid,
                    start_time=hit.start_time,
                    end_time=hit.end_time,
                    text=hit.text,
                    rrf_score=penalized[sid],
                    source_file=hit.source_file,
                    transcription_id=hit.transcription_id,
                    stream_contributions=tuple(sorted(contribs.items())),
                )
            )
            final_source_counts[src] = final_source_counts.get(src, 0) + 1
            if len(results) >= top_n:
                break
    return results


def _temporal_dedup(fused: list[FusedResult]) -> list[FusedResult]:
    """Remove overlapping or near-adjacent fragments from the same source.

    Iterates the RRF-sorted list top-to-bottom (highest score first). The
    highest-scoring representative per time window survives. Fragments that
    are deduplicated away are not discarded — their segment IDs are stashed
    in ``merged_segment_ids`` on the surviving fragment so the Fragment
    Reader can still navigate the full transcript without gaps.

    Gap is self-calibrating: 60 % of the mean segment duration in the
    result set. A corpus of 25-second segments gets a ~15 s proximity gap;
    a corpus of 60-second segments gets a ~36 s gap. Zero configuration
    required — the chunker granularity is always the ground truth.

    Two fragments from the *same* ``source_file`` are considered duplicates
    when their time windows overlap or are within ``gap`` seconds of each
    other. Fragments from *different* sources are never deduplicated against
    each other.
    """
    import dataclasses

    if len(fused) < 2:
        return fused

    durations = [fr.end_time - fr.start_time for fr in fused]
    mean_duration = sum(durations) / len(durations)
    gap = mean_duration * 0.6

    kept: list[FusedResult] = []
    # source_file → list of (start_time, end_time, index_into_kept)
    seen: dict[str, list[tuple[float, float, int]]] = {}

    for fr in fused:  # already sorted by rrf_score descending — first = best
        windows = seen.get(fr.source_file, [])
        duplicate_of: int | None = None

        for kept_start, kept_end, kept_idx in windows:
            # Overlap test with proximity gap on both sides
            if fr.start_time < (kept_end + gap) and fr.end_time > (kept_start - gap):
                duplicate_of = kept_idx
                break

        if duplicate_of is None:
            # New unique window — add to kept and register in seen
            idx = len(kept)
            seen.setdefault(fr.source_file, []).append(
                (fr.start_time, fr.end_time, idx)
            )
            kept.append(fr)
        else:
            # Duplicate — stash this segment's id on the surviving representative
            survivor = kept[duplicate_of]
            kept[duplicate_of] = dataclasses.replace(
                survivor,
                merged_segment_ids=survivor.merged_segment_ids + (fr.segment_id,),
            )

    return kept


# ── Thresholds for micro-fragment filtering ───────────────────────────────────
# A fragment must satisfy AT LEAST ONE of the two conditions to pass:
#   • duration ≥ MIN_FRAGMENT_SECONDS
#   • stripped text length ≥ MIN_FRAGMENT_CHARS
#
# The dual-condition OR means a very long transcript chunk that happens to start
# at a chapter boundary (short duration, rich text) is never penalised, and a
# purely instrumental segment (long duration, few words) is never penalised
# either.  Only 1-second clips with 2 words of text get dropped.
#
# These are intentionally conservative: 2 s / 30 chars filters genuine noise
# (single proper nouns, transcription artifacts) without touching meaningful
# short passages.
MIN_FRAGMENT_SECONDS: float = 2.0
MIN_FRAGMENT_CHARS: int = 30


def filter_micro_fragments(
    fused: list[FusedResult],
) -> list[FusedResult]:
    """Drop fragments too small to contribute meaningful synthesis signal.

    A fragment is a *micro-fragment* when BOTH of the following hold:
      - ``end_time - start_time < MIN_FRAGMENT_SECONDS``
      - ``len(text.strip()) < MIN_FRAGMENT_CHARS``

    When a micro-fragment is removed its ``segment_id`` is stashed in
    ``merged_segment_ids`` on the highest-scoring *surviving* fragment from the
    same ``source_file``.  This preserves Fragment Reader navigation continuity
    (H / L adjacency traversal) without polluting synthesis context.

    If no surviving fragment from the same source exists the segment ID is
    simply dropped — that fragment had no meaningful neighbours anyway.

    The list is returned in the same order as the input (RRF score descending).
    """
    import dataclasses

    # Index of the first surviving fragment per source (for merge target lookup)
    first_kept_idx: dict[str, int] = {}
    kept: list[FusedResult] = []
    orphan_sids: dict[str, list[int]] = {}  # source → segment IDs to merge later

    for fr in fused:
        duration = fr.end_time - fr.start_time
        text_len = len(fr.text.strip())

        is_micro = duration < MIN_FRAGMENT_SECONDS and text_len < MIN_FRAGMENT_CHARS
        if is_micro:
            # Stash for merge — we may not have seen the merge target yet
            orphan_sids.setdefault(fr.source_file, []).append(fr.segment_id)
        else:
            idx = len(kept)
            if fr.source_file not in first_kept_idx:
                first_kept_idx[fr.source_file] = idx
            kept.append(fr)

    if not orphan_sids:
        return kept  # fast path — nothing filtered

    # Merge orphan segment IDs onto the first surviving fragment per source
    for src, sids in orphan_sids.items():
        if src in first_kept_idx:
            target_idx = first_kept_idx[src]
            survivor = kept[target_idx]
            kept[target_idx] = dataclasses.replace(
                survivor,
                merged_segment_ids=survivor.merged_segment_ids + tuple(sids),
            )
        # else: no surviving fragment from this source — segment IDs are dropped.
        # These are genuine noise fragments with no meaningful neighbours.

    return kept
