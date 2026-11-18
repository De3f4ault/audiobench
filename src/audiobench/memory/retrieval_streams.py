"""Retrieval streams for AudioBench memory."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.query_types import ReformulatedQuery

logger = get_logger("memory.retrieval_streams")


@dataclass
class SegmentHit:
    """A segment retrieved from a stream."""

    segment_id: int
    start_time: float
    end_time: float
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    colbert_score: float = 0.0
    source_file: str = ""
    transcription_id: int = 0  # FK → transcriptions.id; needed for Fragment Reader adjacency queries
    # Embedding vector — only populated by DenseStream when preset='synthesis'
    # so that _mmr_filter can compute cosine similarities.
    embedding: list[float] | None = None


@dataclass
class SynthesisResult:
    """A prior synthesis or session summary retrieved from LanceDB."""

    expression_id: int
    content: str
    source_type: str
    session_id: int
    sequence_num: int
    score: float = 0.0


@runtime_checkable
class RetrievalStream(Protocol):
    """Protocol for all retrieval streams."""

    def retrieve(self, query: ReformulatedQuery, top_k: int = 5) -> list[SegmentHit]:
        """Retrieve segments for a query."""
        ...


class FTS5Stream:
    """Full-text SQLite BM25 retrieval stream."""

    def retrieve(
        self,
        query: ReformulatedQuery,
        top_k: int = 5,
        focus_source: str | None = None,
    ) -> list[SegmentHit]:
        hits: list[SegmentHit] = []
        try:
            if not query.bm25_keywords.strip():
                return []

            # Create FTS MATCH query (e.g., OR between keywords)
            keywords = query.bm25_keywords.split()
            match_query = " OR ".join(f'"{k}"' for k in keywords)

            with get_session() as session:
                focus_clause = "AND LOWER(af.file_path) LIKE :focus" if focus_source else ""
                sql = text(f"""
                    SELECT
                        s.id as segment_id,
                        s.transcription_id,
                        s.start_time,
                        s.end_time,
                        fts.rank as bm25_score,
                        s.text,
                        af.file_path as source_file
                    FROM segments_fts fts
                    JOIN segments s ON fts.rowid = s.id
                    JOIN transcriptions t ON s.transcription_id = t.id
                    LEFT JOIN audio_files af ON t.audio_file_id = af.id
                    WHERE segments_fts MATCH :match_query
                    {focus_clause}
                    ORDER BY bm25_score ASC
                    LIMIT :limit
                """)

                params: dict = {"match_query": match_query, "limit": top_k}
                if focus_source:
                    params["focus"] = f"%{focus_source.lower()}%"

                rows = session.execute(sql, params).mappings().all()
                for r in rows:
                    hits.append(SegmentHit(
                        segment_id=r["segment_id"],
                        transcription_id=r["transcription_id"],
                        start_time=r["start_time"],
                        end_time=r["end_time"],
                        bm25_score=r["bm25_score"],
                        text=r["text"],
                        source_file=r["source_file"] or "",
                    ))
        except Exception as e:
            logger.warning("FTS5Stream retrieval failed: %s", e)
        return hits


def _batch_tid_lookup(seg_ids: list[int]) -> dict[int, int]:
    """Return {segment_id: transcription_id} for a list of segment ids.

    Uses named SQLAlchemy placeholders (:id0, :id1, …) which work correctly
    with SQLAlchemy's ``text()`` construct across all backends.
    """
    if not seg_ids:
        return {}
    placeholders = ", ".join(f":id{i}" for i in range(len(seg_ids)))
    params = {f"id{i}": sid for i, sid in enumerate(seg_ids)}
    with get_session() as session:
        rows = session.execute(
            text(
                f"SELECT id, transcription_id FROM segments "
                f"WHERE id IN ({placeholders})"
            ),
            params,
        ).mappings().all()
    return {int(r["id"]): int(r["transcription_id"]) for r in rows}


def _mmr_filter(
    candidates: list[tuple["SegmentHit", list[float]]],
    query_vector: list[float],
    top_k: int,
    lam: float = 0.5,
) -> list["SegmentHit"]:
    """Greedy Maximum Marginal Relevance selection.

    Selects ``top_k`` items from ``candidates`` to maximise:
        score(d) = λ · sim(d, query) − (1−λ) · max_{s∈selected} sim(d, s)

    A source-diversity bonus of +0.1 is applied if the candidate's
    ``source_file`` has not yet appeared in the selected set, making the
    synthesis preset pull from multiple audiobooks naturally.

    Parameters
    ----------
    candidates : list[(SegmentHit, vector)]
        ANN candidates sorted by relevance (highest first).
    query_vector :
        The query embedding (list[float]).
    top_k :
        Number of items to return.
    lam :
        Trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    import numpy as np

    if not candidates:
        return []

    top_k = min(top_k, len(candidates))

    def cosine(a: list[float], b: list[float]) -> float:
        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = (np.linalg.norm(av) * np.linalg.norm(bv))
        return float(np.dot(av, bv) / denom) if denom > 0 else 0.0

    # Pre-compute query similarity for every candidate
    q_sims = [cosine(query_vector, vec) for _, vec in candidates]

    selected: list[SegmentHit] = []
    selected_vecs: list[list[float]] = []
    selected_sources: set[str] = set()
    remaining = list(range(len(candidates)))

    while len(selected) < top_k and remaining:
        best_idx: int | None = None
        best_score = float("-inf")

        for i in remaining:
            hit, vec = candidates[i]
            relevance = q_sims[i]
            if selected_vecs:
                redundancy = max(cosine(vec, sv) for sv in selected_vecs)
            else:
                redundancy = 0.0
            diversity_bonus = 0.3 if hit.source_file not in selected_sources else -0.15
            score = lam * relevance - (1 - lam) * redundancy + diversity_bonus
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        hit, vec = candidates[best_idx]
        selected.append(hit)
        selected_vecs.append(vec)
        selected_sources.add(hit.source_file)
        remaining.remove(best_idx)

    return selected


class DenseStream:
    """Dense embedding retrieval stream.

    Embeds the query via the daemon's warm Nomic model (no cold load in CLI
    process), then performs ANN search against the segment_vectors LanceDB
    table. Uses hyde_document as the query vector when preset=deep and HyDE
    succeeded; otherwise uses the original query (rq.dense_query).

    When ``preset='synthesis'``, retrieves 3× candidates with their raw
    embeddings and applies Maximum Marginal Relevance to return a
    diverse, cross-source shortlist.
    """

    def retrieve(
        self,
        query: ReformulatedQuery,
        top_k: int = 5,
        preset: str = "balanced",
        mmr_lambda: float = 0.5,
        focus_source: str | None = None,
    ) -> list[SegmentHit]:
        try:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.memory_store import SegmentVectorStore

            daemon = get_daemon_client()

            # Use HyDE document if available (preset=deep), otherwise original query.
            # dense_query is always the original — never the BM25-reformulated version.
            embed_text = query.hyde_document if query.hyde_document else query.dense_query
            if not embed_text:
                embed_text = query.original

            query_vector = daemon.embed_query(embed_text)
            store = SegmentVectorStore()

            if store.count_embedded() == 0:
                logger.debug("DenseStream: segment_vectors table is empty — skipping")
                return []

            use_mmr = preset == "synthesis"
            candidate_k = top_k * 3 if use_mmr else top_k

            # Request more candidates if filtering by focus to ensure we have enough hits
            fetch_k = candidate_k * 5 if focus_source else candidate_k
            rows = store.search(query_vector, fetch_k, return_vectors=use_mmr)

            if focus_source:
                focus_lower = focus_source.lower()
                rows = [r for r in rows if focus_lower in r.get("source_file", "").lower()]

            # Trim back down to the target candidate size
            rows = rows[:candidate_k]

            # Batch-fetch transcription_id from SQLite (LanceDB does not store it)
            seg_ids = [int(r["segment_id"]) for r in rows]
            tid_map = _batch_tid_lookup(seg_ids)

            # Build SegmentHit objects (with optional embedding for MMR)
            candidate_hits: list[tuple[SegmentHit, list[float]]] = []
            plain_hits: list[SegmentHit] = []
            for r in rows:
                sid = int(r["segment_id"])
                hit = SegmentHit(
                    segment_id=sid,
                    transcription_id=tid_map.get(sid, 0),
                    start_time=float(r["start_time"]),
                    end_time=float(r["end_time"]),
                    text=r["text"],
                    dense_score=float(r.get("_distance", 0.0)),
                    source_file=r.get("source_file", ""),
                )
                if use_mmr:
                    vec = r.get("vector", [])
                    candidate_hits.append((hit, vec if isinstance(vec, list) else []))
                else:
                    plain_hits.append(hit)

            if use_mmr:
                logger.debug(
                    "DenseStream: MMR λ=%.2f on %d candidates → top %d",
                    mmr_lambda, len(candidate_hits), top_k,
                )
                return _mmr_filter(candidate_hits, query_vector, top_k, lam=mmr_lambda)

            return plain_hits

        except Exception as e:
            logger.warning("DenseStream retrieval failed: %s", e)
        return []


class ColBERTStream:
    """Late-interaction ColBERT retrieval stream.

    Strategy:
      1. Fetch top_k * 3 candidates from segment_vectors via ANN (same as DenseStream).
      2. Send all candidate texts + query to the daemon's CrossEncoder reranker.
      3. Return the top_k candidates sorted by rerank score.

    A 5-second timeout is applied to the reranker call. If it times out or
    fails, an empty list is returned so the caller can show a colbert✗ badge
    and RRF continues with the FTS5 + Dense results.

    Uses hyde_document as query vector when available, otherwise dense_query.
    """

    _RERANK_TIMEOUT: float = 15.0

    def retrieve(
        self,
        query: ReformulatedQuery,
        top_k: int = 5,
        preset: str = "balanced",
    ) -> list[SegmentHit]:
        if preset == "synthesis":
            # Partial ColBERT: only return top 5 so RRF validates anchors
            # but leaves the tail open for MMR diversity.
            top_k = min(top_k, 5)

        try:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.memory_store import SegmentVectorStore

            daemon = get_daemon_client()

            embed_text = query.hyde_document if query.hyde_document else query.dense_query
            if not embed_text:
                embed_text = query.original

            query_vector = daemon.embed_query(embed_text)
            store = SegmentVectorStore()

            if store.count_embedded() == 0:
                logger.debug("ColBERTStream: segment_vectors table is empty — skipping")
                return []

            # Retrieve 3× candidates for reranking headroom
            candidates = store.search(query_vector, top_k * 3)
            if not candidates:
                return []

            # CrossEncoder rerank via daemon (warm model, 5s timeout)
            import concurrent.futures
            texts = [r["text"] for r in candidates]
            rerank_query = query.dense_query if query.dense_query else query.original

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(daemon.rerank, rerank_query, texts)
                try:
                    scores = future.result(timeout=self._RERANK_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "ColBERTStream: reranker timed out after %.1fs — skipping", self._RERANK_TIMEOUT
                    )
                    return []

            # Sort by descending rerank score, return top_k
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

            # Batch-fetch transcription_id via named-param lookup (fixes SQLAlchemy ? bug)
            all_seg_ids = [int(r["segment_id"]) for r in candidates]
            tid_map = _batch_tid_lookup(all_seg_ids)

            hits = []
            for r, score in ranked[:top_k]:
                sid = int(r["segment_id"])
                hits.append(
                    SegmentHit(
                        segment_id=sid,
                        transcription_id=tid_map.get(sid, 0),
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        text=r["text"],
                        colbert_score=float(score),
                        source_file=r.get("source_file", ""),
                    )
                )
            return hits

        except Exception as e:
            logger.warning("ColBERTStream retrieval failed: %s", e)
        return []

class SynthesisStream:
    """Retrieves prior search synthesis and session summaries from LanceDB.

    Only surfaces synthesis from the *current session* so results from
    unrelated past sessions never bleed into the active search thread.
    """

    def retrieve(
        self,
        query: ReformulatedQuery,
        top_k: int = 3,
        session_id: int | None = None,
    ) -> list[SynthesisResult]:
        try:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.memory_store import MemoryStore

            daemon = get_daemon_client()

            embed_text = query.hyde_document if query.hyde_document else query.dense_query
            if not embed_text:
                embed_text = query.original

            query_vector = daemon.embed_query(embed_text)
            store = MemoryStore()

            try:
                table = store.table
            except Exception:
                logger.debug("SynthesisStream: table not initialized")
                return []

            fetch_k = top_k * 3
            raw_results = (
                table.search(query_vector)
                .where("source_type IN ('search_synthesis', 'search_session_summary')")
                .limit(fetch_k)
                .to_list()
            )

            if not raw_results:
                return []

            import numpy as np
            def cosine_dist(a, b):
                a_arr, b_arr = np.array(a), np.array(b)
                return 1 - (np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

            deduped_results = []
            for r in raw_results:
                if len(deduped_results) >= top_k:
                    break
                vec = r.get("vector")
                if vec is None:
                    deduped_results.append(r)
                    continue
                
                is_duplicate = False
                for existing in deduped_results:
                    e_vec = existing.get("vector")
                    if e_vec is not None:
                        dist = cosine_dist(vec, e_vec)
                        if dist < 0.15:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    deduped_results.append(r)

            raw_results = deduped_results

            if not raw_results:
                return []

            expr_ids = [int(r["expression_id"]) for r in raw_results]

            from audiobench.core.db_session import get_session
            from sqlalchemy import text

            placeholders = ", ".join(f":id{i}" for i in range(len(expr_ids)))
            params = {f"id{i}": eid for i, eid in enumerate(expr_ids)}

            # session_id filter: only return synthesis from the active session.
            # Without this, every search draws prior-knowledge hits from all past
            # sessions regardless of topic, which is the "spilling" behaviour.
            session_filter = ""
            if session_id is not None and session_id >= 0:
                params["active_session_id"] = session_id
                session_filter = (
                    "AND ("
                    "  (e.source_type = 'search_synthesis'       AND sq.session_id = :active_session_id)"
                    "  OR"
                    "  (e.source_type = 'search_session_summary' AND ss.id        = :active_session_id)"
                    ")"
                )

            sql = text(f"""
                SELECT e.id, e.source_type,
                       CASE WHEN e.source_type = 'search_synthesis'
                            THEN sq.session_id ELSE ss.id END AS session_id,
                       CASE WHEN e.source_type = 'search_synthesis'
                            THEN sq.sequence_num ELSE 0 END AS sequence_num
                FROM expressions e
                LEFT JOIN search_queries sq
                       ON e.source_type = 'search_synthesis'       AND e.source_id = sq.id
                LEFT JOIN search_sessions ss
                       ON e.source_type = 'search_session_summary' AND e.source_id = ss.id
                WHERE e.id IN ({placeholders})
                {session_filter}
            """)

            with get_session() as session:
                rows = session.execute(sql, params).mappings().all()
                meta_map = {r["id"]: r for r in rows}

            hits = []
            for r in raw_results:
                eid = int(r["expression_id"])
                meta = meta_map.get(eid)
                if not meta:
                    continue  # filtered out (wrong session or missing join)
                hits.append(SynthesisResult(
                    expression_id=eid,
                    content=r["content"],
                    source_type=meta["source_type"],
                    session_id=meta["session_id"] or 0,
                    sequence_num=meta["sequence_num"] or 0,
                    score=float(r.get("_distance", 0.0)),
                ))

            # LanceDB returns ascending L2 distance → most relevant first.
            return hits

        except Exception as e:
            logger.warning("SynthesisStream retrieval failed: %s", e)
        return []
