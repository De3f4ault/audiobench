"""Continuous knowledge graph ingestion linker and island eradication substrate.

Ensures newly transcribed and indexed audio expressions are linked into the
cross-work semantic knowledge graph asynchronously without blocking the write path.

Key guarantees:
- Async decoupling: Ingest write path calls enqueue_for_linking() only (<0.01ms).
- HNSW protection: link_expressions_batch() processes chunks in slices <= 50,
  bounding HNSW lock hold to ~50ms to prevent autocomplete latency spikes.
- Strict work isolation: Semantic links are strictly formed between distinct work_ids.
- 3-Check Boilerplate Filter: Rejects disclaimers, bare chapter headers (<60 chars),
  and audio noise/timestamps (<0.40 alphabetic density).
- Concurrency Safe: Serialized via threading.Lock; duplicate insertions safely
  ignored by SQLite unique constraint on (from, to, relation_type).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, UTC
import logging
import queue
import re
import threading
import time
from typing import Any

import numpy as np
from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.daemon.index import get_shared_hnsw_index, InCoreHNSWIndex
from audiobench.memory.boilerplate import is_pure_boilerplate, clean_audio_disclaimers

logger = logging.getLogger("audiobench.memory.graph_linker")

# Bounded queue prevents runaway memory in bulk-import scenarios
_MAX_QUEUE_SIZE: int = 10000
_pending_queue: queue.Queue[int] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)

# Lock protects concurrent batch linking (e.g. background drain vs on-demand seed)
_link_lock: threading.Lock = threading.Lock()

# Bare chapter header pattern (e.g., "Chapter IV", "Part 1", "Book Two")
_BARE_CHAPTER_HEADER_RE = re.compile(
    r"^(?:chapter|part|book|section|volume|disc|session)\s+[\w\dIVXLCDMivxlcdm]+[\.\:\s]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IngestLinkResult:
    """Result summary of a batch linking or island seeding operation."""
    chunks_processed: int
    edges_created: int
    elapsed_ms: float
    linked_work_ids: list[int]


def is_linkable_chunk(content: str) -> bool:
    """Return True if content is substantive and suitable for semantic graph edges.

    Enforces the 3-check spec:
    1. Check 2 (Length threshold): content >= 60 chars.
    2. Check 3 (Density): alphabetic ratio >= 0.40 (filters timestamps, silence, noise).
    3. Check 1 (Boilerplate): Not a known publisher stamp, LibriVox disclaimer,
       Gutenberg notice, or bare chapter-number announcement. Substantive content
       remaining after disclaimer stripping must be >= 60 chars.
    """
    if not content or not isinstance(content, str):
        return False
    stripped = content.strip()
    if len(stripped) < 60:
        return False

    alpha_count = sum(1 for c in stripped if c.isalpha())
    if (alpha_count / len(stripped)) < 0.40:
        return False

    if _BARE_CHAPTER_HEADER_RE.match(stripped):
        return False

    if is_pure_boilerplate(stripped):
        return False

    cleaned = clean_audio_disclaimers(stripped)
    if len(cleaned) < 60:
        return False

    return True


def enqueue_for_linking(expression_ids: list[int]) -> int:
    """Enqueue expression IDs for asynchronous graph linking.

    Called synchronously from the write path / RAG consistency sweep.
    Never calls HNSW or SQLite directly. Fast O(1) in-memory queue operation.

    If the queue is full (e.g. during a massive bulk import), remaining chunks
    are deferred to the periodic seed sweep and logged.

    Returns:
        The number of expression IDs successfully enqueued.
    """
    enqueued = 0
    deferred = 0
    for eid in expression_ids:
        try:
            _pending_queue.put_nowait(eid)
            enqueued += 1
        except queue.Full:
            deferred += 1

    if deferred > 0:
        logger.warning(
            "Pending link queue full (maxsize=%d); %d chunks deferred to periodic seed.",
            _MAX_QUEUE_SIZE,
            deferred,
        )

    return enqueued


def get_pending_queue_size() -> int:
    """Return the current number of expression IDs awaiting linking."""
    return _pending_queue.qsize()


def ensure_hnsw_ready(index: InCoreHNSWIndex | None = None) -> InCoreHNSWIndex:
    """Ensure the InCoreHNSWIndex is loaded into memory and ready for queries."""
    hnsw = index or get_shared_hnsw_index()
    if not hnsw.is_ready():
        # Fast path: try loading the persisted binary snapshot directly
        if hnsw.load_snapshot():
            return hnsw
        logger.info("InCoreHNSWIndex snapshot not available; warming via AutocompleteIndex...")
        from audiobench.daemon.autocomplete import get_autocomplete_index
        auto_idx = get_autocomplete_index()
        if not auto_idx.ready:
            auto_idx.build()
        return auto_idx._shared_index or hnsw
    return hnsw


def link_expressions_batch(
    expression_ids: list[int],
    max_edges_per_chunk: int = 2,
    threshold: float = 0.78,
    slice_size: int = 50,
    index: InCoreHNSWIndex | None = None,
) -> IngestLinkResult:
    """Form cross-work semantic edges for a batch of expression IDs.

    MUST be called in a background task or worker thread, NEVER in the
    interactive write path.

    Args:
        expression_ids: List of expression IDs to link.
        max_edges_per_chunk: Maximum semantic edges to create per expression (default 2).
        threshold: Minimum cosine similarity threshold (default 0.78).
        slice_size: Max vectors per HNSW query_batch call to bound lock duration (default 50).
        index: Optional InCoreHNSWIndex instance (defaults to shared singleton).

    Returns:
        IngestLinkResult with processing stats.
    """
    if not expression_ids:
        return IngestLinkResult(0, 0, 0.0, [])

    t0 = time.perf_counter()
    hnsw = ensure_hnsw_ready(index)
    total_edges = 0
    linked_works_set: set[int] = set()

    with _link_lock:
        # Deduplicate incoming expression IDs preserving order
        unique_eids = list(dict.fromkeys(expression_ids))

        # Process in bounded slices
        for i in range(0, len(unique_eids), slice_size):
            slice_eids = unique_eids[i : i + slice_size]

            # 1. Fetch metadata for the slice from SQLite
            with get_session() as session:
                placeholders = ", ".join(f":id{idx}" for idx in range(len(slice_eids)))
                params = {f"id{idx}": eid for idx, eid in enumerate(slice_eids)}
                rows = session.execute(
                    text(f"""
                        SELECT id, work_id, content
                        FROM expressions
                        WHERE id IN ({placeholders})
                          AND work_id IS NOT NULL
                    """),
                    params,
                ).fetchall()

            if not rows:
                continue

            # 2. Filter linkable sources and ensure they exist in HNSW index
            source_map: dict[int, dict[str, Any]] = {}
            for r in rows:
                eid, wid, content = r[0], r[1], r[2]
                if is_linkable_chunk(content):
                    source_map[eid] = {"work_id": wid, "content": content}

            if not source_map:
                continue

            # Verify presence in in-core HNSW index
            valid_src_eids = [eid for eid in source_map if hnsw.has_item(eid)]
            if not valid_src_eids:
                continue

            # 3. Retrieve vectors and run batch ANN query
            try:
                vectors = hnsw.get_items(valid_src_eids)
            except Exception as e:
                logger.warning("Failed to retrieve vectors from HNSW for batch: %s", e)
                continue

            k_query = 15
            try:
                labels, distances = hnsw.query_batch(vectors, k=k_query)
            except Exception as e:
                logger.warning("HNSW query_batch failed during linking: %s", e)
                continue

            # 4. Gather all unique candidate target IDs from ANN results
            all_candidate_ids: set[int] = set()
            for row in labels:
                for cand_id in row:
                    cid = int(cand_id)
                    if cid not in source_map and cid > 0:
                        all_candidate_ids.add(cid)

            if not all_candidate_ids:
                continue

            # 5. Fetch candidate metadata from SQLite in a single batch
            cand_list = list(all_candidate_ids)
            cand_map: dict[int, dict[str, Any]] = {}
            # Chunk candidate fetch if large
            for c_idx in range(0, len(cand_list), 500):
                c_slice = cand_list[c_idx : c_idx + 500]
                c_placeholders = ", ".join(f":cid{j}" for j in range(len(c_slice)))
                c_params = {f"cid{j}": cid for j, cid in enumerate(c_slice)}
                with get_session() as session:
                    c_rows = session.execute(
                        text(f"""
                            SELECT id, work_id, content
                            FROM expressions
                            WHERE id IN ({c_placeholders})
                              AND work_id IS NOT NULL
                        """),
                        c_params,
                    ).fetchall()
                for cr in c_rows:
                    cid, c_wid, c_content = cr[0], cr[1], cr[2]
                    cand_map[cid] = {"work_id": c_wid, "content": c_content}

            # 6. Evaluate and form cross-work semantic edges
            edges_to_insert: list[dict[str, Any]] = []
            now_dt = datetime.now(UTC)

            for s_idx, src_eid in enumerate(valid_src_eids):
                src_info = source_map[src_eid]
                src_wid = src_info["work_id"]

                matched_candidates: list[tuple[int, float]] = []
                for cand_eid_raw, dist in zip(labels[s_idx], distances[s_idx]):
                    cand_eid = int(cand_eid_raw)
                    if cand_eid == src_eid or cand_eid not in cand_map:
                        continue

                    cand_info = cand_map[cand_eid]
                    cand_wid = cand_info["work_id"]

                    # Enforce strict cross-work isolation
                    if cand_wid == src_wid:
                        continue

                    sim = 1.0 - float(dist)
                    if sim < threshold:
                        continue

                    # Filter candidate content for boilerplate / non-substantive text
                    if not is_linkable_chunk(cand_info["content"]):
                        continue

                    matched_candidates.append((cand_eid, sim))

                # Sort by similarity descending, take top max_edges_per_chunk
                matched_candidates.sort(key=lambda x: x[1], reverse=True)
                for target_eid, sim in matched_candidates[:max_edges_per_chunk]:
                    edges_to_insert.append({
                        "from_id": src_eid,
                        "to_id": target_eid,
                        "relation_type": "semantic",
                        "weight": round(sim, 4),
                        "created_by": "ingest_linker",
                        "created_at": now_dt,
                    })
                    linked_works_set.add(src_wid)
                    linked_works_set.add(cand_map[target_eid]["work_id"])

            # 7. Atomically insert edges into SQLite (INSERT OR IGNORE handles conflicts cleanly)
            if edges_to_insert:
                with get_session() as session:
                    session.execute(
                        text("""
                            INSERT OR IGNORE INTO expression_relations
                            (from_expression_id, to_expression_id, relation_type, weight, created_by, created_at)
                            VALUES (:from_id, :to_id, :relation_type, :weight, :created_by, :created_at)
                        """),
                        edges_to_insert,
                    )
                    session.commit()
                total_edges += len(edges_to_insert)

                # Incremental community assignment for newly linked chunks
                try:
                    import sqlite3
                    from audiobench.community import assign_incremental
                    from audiobench.core.settings import get_settings

                    db_url = get_settings().database_url
                    if db_url.startswith("sqlite:///"):
                        db_p = db_url.replace("sqlite:///", "")
                        with sqlite3.connect(db_p) as c_conn:
                            assign_incremental(c_conn, valid_src_eids)
                except Exception as inc_err:
                    logger.debug("assign_incremental skipped: %s", inc_err)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return IngestLinkResult(
        chunks_processed=len(unique_eids),
        edges_created=total_edges,
        elapsed_ms=round(elapsed_ms, 2),
        linked_work_ids=sorted(linked_works_set),
    )


def drain_pending_links(batch_size: int = 50) -> IngestLinkResult:
    """Drain up to batch_size expressions from the pending link queue and link them.

    Safe to call periodically from background scheduler or worker threads.
    """
    batch_ids: list[int] = []
    while len(batch_ids) < batch_size:
        try:
            eid = _pending_queue.get_nowait()
            batch_ids.append(eid)
        except queue.Empty:
            break

    if not batch_ids:
        return IngestLinkResult(0, 0, 0.0, [])

    return link_expressions_batch(batch_ids)


async def drain_pending_links_loop(poll_interval: float = 2.0) -> None:
    """Async background worker loop that drains the pending link queue.

    Runs in daemon event loop without blocking autocomplete or transcription.
    """
    logger.info("drain_pending_links_loop started (poll_interval=%.1fs)", poll_interval)
    while True:
        try:
            if not _pending_queue.empty():
                await asyncio.to_thread(drain_pending_links, 50)
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info("drain_pending_links_loop cancelled")
            break
        except Exception as e:
            logger.exception("Error in drain_pending_links_loop: %s", e)
            await asyncio.sleep(5.0)


def seed_unlinked_islands(
    limit: int = 2500,
    max_edges_per_chunk: int = 2,
    threshold: float = 0.78,
    index: InCoreHNSWIndex | None = None,
) -> IngestLinkResult:
    """Drain unlinked chunks across isolated works and underlinked classics.

    Queries expressions lacking outbound semantic relations and runs them
    through batch linking to seed the knowledge graph.

    Args:
        limit: Maximum unlinked expressions to process (default 2500).
        max_edges_per_chunk: Maximum semantic edges per expression (default 2).
        threshold: Cosine similarity threshold (default 0.78).
        index: Optional InCoreHNSWIndex instance.

    Returns:
        IngestLinkResult with processing stats.
    """
    logger.info("seed_unlinked_islands: querying up to %d unlinked expressions...", limit)
    with get_session() as session:
        # Fast query selecting unlinked audio transcript chunks with valid works
        rows = session.execute(
            text("""
                SELECT e.id
                FROM expressions e
                WHERE e.source_type = 'audio_transcript'
                  AND e.work_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM expression_relations r
                      WHERE r.from_expression_id = e.id
                        AND r.relation_type = 'semantic'
                  )
                ORDER BY e.work_id, e.id
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

    unlinked_eids = [r[0] for r in rows]
    if not unlinked_eids:
        logger.info("seed_unlinked_islands: no unlinked expressions found.")
        return IngestLinkResult(0, 0, 0.0, [])

    logger.info(
        "seed_unlinked_islands: found %d unlinked expressions. Processing batch...",
        len(unlinked_eids),
    )
    return link_expressions_batch(
        unlinked_eids,
        max_edges_per_chunk=max_edges_per_chunk,
        threshold=threshold,
        slice_size=50,
        index=index,
    )
