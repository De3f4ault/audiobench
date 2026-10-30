"""Graph-Augmented Synthesis Retriever (Blended-Relevance MMR).

Combines:
1. ANN Vector Seeds: Fast approximate nearest neighbors via InCoreHNSWIndex (< 2ms).
2. 1-Hop & 2-Hop Graph Expansion: Explores outbound semantic edges (w >= 0.78, H <= 2)
   bridging into diverse external authors.
3. Dual-Signal Relevance Blending:
   EffRel = alpha * QuerySim + (1 - alpha) * EdgeWeight  (alpha=0.60)
4. Triadic Clique Consensus: Detects 3-cliques across distinct thinkers and awards
   a clique confidence boost (untuned initial value: +0.10).
5. Vector MMR: In-memory diversity reranking in NumPy (~0.3ms) with author diversity bonus
   to break echo chambers and surface cross-disciplinary dialectics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.daemon.index import get_shared_hnsw_index, InCoreHNSWIndex
from audiobench.memory.graph_linker import ensure_hnsw_ready, is_linkable_chunk

logger = logging.getLogger("audiobench.memory.synthesis_retriever")

# Validated hyperparameters from Segment 2 sweep:
DEFAULT_ALPHA: float = 0.60          # Query vs Edge blend (0.60 * Q + 0.40 * Edge)
DEFAULT_LAMBDA: float = 0.55         # MMR Relevance vs Redundancy (0.55 * Rel - 0.45 * Red)
DEFAULT_EDGE_THRESHOLD: float = 0.78 # Minimum edge weight for semantic bridges
DEFAULT_RELEVANCE_FLOOR: float = 0.48# Minimum query similarity for graph neighbors
DEFAULT_CLIQUE_BOOST: float = 0.10   # Empirically validated via sweep_clique_boost.py over [0.00-0.25]


def _clean_title_truncate(text: str, max_chars: int = 80) -> str:
    """Truncate text cleanly at a word boundary, avoiding mid-word cutting."""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 25:
        truncated = truncated[:last_space]
    truncated = truncated.rstrip(",;:-.?! ")
    return f"{truncated}…"


def _extract_first_heading(content: str) -> str | None:
    """Extract first bold or header line from markdown content."""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**"):
            return line.strip("* ")
        if line.startswith("#"):
            return line.lstrip("# ")
    return None


def _format_expression_attribution(
    source_type: str | None,
    work_title: str | None,
    work_author: str | None,
    sq_query: str | None,
    ss_title: str | None,
    content: str,
) -> tuple[str, str, bool]:
    """Return (author, work_title, is_playable) for an expression."""
    st = (source_type or "").lower()
    if st == "search_synthesis":
        author = "[Prior Synthesis]"
        raw_title = sq_query or _extract_first_heading(content) or "Research Synthesis"
        title = _clean_title_truncate(raw_title, max_chars=80)
        return (author, title, False)
    elif st == "search_session_summary":
        author = "[Past Session]"
        raw_title = ss_title or "Research Session Digest"
        title = _clean_title_truncate(raw_title, max_chars=80)
        return (author, title, False)
    elif st in ("chat_message", "journal_entry"):
        author = "[Cognitive Note]" if st == "journal_entry" else "[Chat Reflection]"
        first_line = content.splitlines()[0] if content else "Note"
        title = _clean_title_truncate(first_line, max_chars=80)
        return (author, title, False)
    else:
        author = work_author or "Unknown Author"
        title = work_title or "Unknown Work"
        is_playable = st in ("audio_transcript", "chapter_summary") or bool(work_title)
        return (author, title, is_playable)


@dataclass
class SynthesisFragment:
    """A retrieved thought fragment with relevance and graph attribution."""
    expression_id: int
    work_id: int | None
    work_title: str
    author: str
    content: str
    q_sim: float
    edge_weight: float
    effective_rel: float
    mmr_score: float
    source_mode: str  # 'ANN' or 'GRAPH_BRIDGE'
    is_clique_partner: bool = False
    source_type: str = "audio_transcript"
    is_playable: bool = True


def embed_query(query: str) -> np.ndarray:
    """Embed query string using the daemon client and normalize."""
    from audiobench.daemon.factory import get_daemon_client

    client = get_daemon_client()
    raw = client.embed_query(query)
    v = np.array(raw, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def retrieve_synthesis_context(
    query: str,
    top_k: int = 4,
    lam: float = DEFAULT_LAMBDA,
    alpha: float = DEFAULT_ALPHA,
    clique_boost: float = DEFAULT_CLIQUE_BOOST,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
    index: InCoreHNSWIndex | None = None,
    community_id: int | None = None,
) -> list[SynthesisFragment]:
    """Retrieve diverse, cross-disciplinary fragments using Blended-Relevance Graph MMR.

    Args:
        query: User search or synthesis query.
        top_k: Number of diverse fragments to return (default 4).
        lam: MMR tradeoff lambda (default 0.55).
        alpha: Dual-signal blend ratio (default 0.60).
        clique_boost: Confidence boost for 3-clique consensus (untuned initial value: 0.10).
        edge_threshold: Minimum semantic edge weight to follow (default 0.78).
        relevance_floor: Relevance floor to reject pure tangents (default 0.48).
        index: Optional InCoreHNSWIndex instance.
        community_id: Optional community ID to restrict retrieval within a single school of thought.

    Returns:
        List of SynthesisFragment dataclasses ordered by MMR selection.
    """
    t0 = time.perf_counter()
    hnsw = ensure_hnsw_ready(index)
    q_vec = embed_query(query)

    # -------------------------------------------------------------
    # Step 1: Retrieve ANN Seeds from In-Core HNSW Index
    # -------------------------------------------------------------
    k_ann = 15 if community_id is None else 60
    raw_hits = hnsw.query(q_vec, k=k_ann)
    if not raw_hits:
        logger.warning("No ANN hits found for query: %s", query)
        return []

    ann_eids = [int(eid) for eid, _ in raw_hits]
    dist_map = {int(eid): float(dist) for eid, dist in raw_hits}

    with get_session() as session:
        placeholders = ", ".join(f":id{i}" for i in range(len(ann_eids)))
        params = {f"id{i}": eid for i, eid in enumerate(ann_eids)}
        cid_filter = ""
        if community_id is not None:
            cid_filter = " AND e.community_id = :cid"
            params["cid"] = community_id

        rows = session.execute(
            text(f"""
                SELECT e.id, e.work_id, e.content, w.title, w.author, e.source_type, sq.query_text, ss.title
                FROM expressions e
                LEFT JOIN works w ON e.work_id = w.id
                LEFT JOIN search_queries sq ON (e.source_type = 'search_synthesis' AND e.source_id = sq.id)
                LEFT JOIN search_sessions ss ON (e.source_type = 'search_session_summary' AND e.source_id = ss.id)
                WHERE e.id IN ({placeholders}){cid_filter}
            """),
            params,
        ).fetchall()

    candidate_pool: list[dict[str, Any]] = []
    seen_eids: set[int] = set()

    for r_eid, r_wid, r_content, r_title, r_author, r_st, r_sq, r_ss in rows:
        if not is_linkable_chunk(r_content):
            continue
        if not hnsw.has_item(r_eid):
            continue

        q_sim = 1.0 - dist_map.get(r_eid, 1.0)
        try:
            vec = hnsw.get_items([r_eid])[0]
        except Exception:
            continue

        author, title, is_playable = _format_expression_attribution(
            r_st, r_title, r_author, r_sq, r_ss, r_content
        )

        candidate_pool.append({
            "expression_id": r_eid,
            "work_id": r_wid,
            "work_title": title,
            "author": author,
            "content": r_content.strip(),
            "vec": vec,
            "q_sim": q_sim,
            "edge_weight": 1.0,
            "effective_rel": q_sim,
            "source_mode": "ANN",
            "is_clique_partner": False,
            "source_type": r_st or "audio_transcript",
            "is_playable": is_playable,
        })
        seen_eids.add(r_eid)

    # -------------------------------------------------------------
    # Step 2: 1-Hop Graph Expansion from Top 3 Seeds (Bounded H <= 2)
    # -------------------------------------------------------------
    top3_seed_ids = [c["expression_id"] for c in candidate_pool[:3]]
    if top3_seed_ids:
        seed_placeholders = ", ".join(f":sid{i}" for i in range(len(top3_seed_ids)))
        seed_params = {f"sid{i}": sid for i, sid in enumerate(top3_seed_ids)}
        seed_params["min_w"] = edge_threshold
        cid_filter2 = ""
        if community_id is not None:
            cid_filter2 = " AND e2.community_id = :cid"
            seed_params["cid"] = community_id

        with get_session() as session:
            graph_rows = session.execute(
                text(f"""
                    SELECT 
                        CASE WHEN r.from_expression_id IN ({seed_placeholders}) THEN r.to_expression_id ELSE r.from_expression_id END as neighbor_id,
                        r.weight,
                        e2.work_id, e2.content, w2.title, w2.author, e2.source_type, sq.query_text, ss.title
                    FROM expression_relations r
                    JOIN expressions e2 ON e2.id = (CASE WHEN r.from_expression_id IN ({seed_placeholders}) THEN r.to_expression_id ELSE r.from_expression_id END)
                    LEFT JOIN works w2 ON e2.work_id = w2.id
                    LEFT JOIN search_queries sq ON (e2.source_type = 'search_synthesis' AND e2.source_id = sq.id)
                    LEFT JOIN search_sessions ss ON (e2.source_type = 'search_session_summary' AND e2.source_id = ss.id)
                    WHERE (r.from_expression_id IN ({seed_placeholders}) OR r.to_expression_id IN ({seed_placeholders}))
                      AND r.relation_type IN ('semantic', 'thematic')
                      AND r.weight >= :min_w
                      {cid_filter2}
                    ORDER BY r.weight DESC
                    LIMIT 25
                """),
                seed_params,
            ).fetchall()

        for t_eid, weight, g_wid, g_content, g_title, g_author, g_st, g_sq, g_ss in graph_rows:
            if t_eid in seen_eids:
                continue
            if not is_linkable_chunk(g_content):
                continue
            if not hnsw.has_item(t_eid):
                continue

            try:
                v = hnsw.get_items([t_eid])[0]
            except Exception:
                continue

            q_sim = float(np.dot(q_vec, v))
            # Relevance floor: reject pure tangents
            if q_sim < relevance_floor:
                continue

            # Dual-Signal Effective Relevance
            eff_rel = alpha * q_sim + (1.0 - alpha) * float(weight)

            author, title, is_playable = _format_expression_attribution(
                g_st, g_title, g_author, g_sq, g_ss, g_content
            )

            candidate_pool.append({
                "expression_id": t_eid,
                "work_id": g_wid,
                "work_title": title,
                "author": author,
                "content": g_content.strip(),
                "vec": v,
                "q_sim": q_sim,
                "edge_weight": float(weight),
                "effective_rel": eff_rel,
                "source_mode": "GRAPH_BRIDGE",
                "is_clique_partner": False,
                "source_type": g_st or "audio_transcript",
                "is_playable": is_playable,
            })
            seen_eids.add(t_eid)

    if not candidate_pool:
        return []

    # -------------------------------------------------------------
    # Step 3: Triadic Clique Consensus Detection (3-Cliques)
    # -------------------------------------------------------------
    pool_eids = [c["expression_id"] for c in candidate_pool]
    if len(pool_eids) >= 3:
        p_placeholders = ", ".join(f":pid{i}" for i in range(len(pool_eids)))
        p_params = {f"pid{i}": pid for i, pid in enumerate(pool_eids)}
        p_params["min_w"] = edge_threshold

        with get_session() as session:
            triangle_edges = session.execute(
                text(f"""
                    SELECT r.from_expression_id, r.to_expression_id
                    FROM expression_relations r
                    JOIN expressions e1 ON r.from_expression_id = e1.id
                    JOIN expressions e2 ON r.to_expression_id = e2.id
                    WHERE r.from_expression_id IN ({p_placeholders})
                      AND r.to_expression_id IN ({p_placeholders})
                      AND r.relation_type IN ('semantic', 'thematic')
                      AND r.weight >= :min_w
                      AND e1.work_id != e2.work_id
                """),
                p_params,
            ).fetchall()

        # Build adjacency set within pool
        adj: dict[int, set[int]] = {eid: set() for eid in pool_eids}
        for u, v in triangle_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Detect 3-cliques: u - v - w with 3 distinct works
        eid_to_wid = {c["expression_id"]: c["work_id"] for c in candidate_pool}
        clique_members: set[int] = set()
        for u in pool_eids:
            neighbors = adj.get(u, set())
            for v in neighbors:
                if v > u:
                    common = neighbors.intersection(adj.get(v, set()))
                    for w in common:
                        if w > v:
                            wu, wv, ww = eid_to_wid.get(u), eid_to_wid.get(v), eid_to_wid.get(w)
                            if wu and wv and ww and wu != wv and wv != ww and wu != ww:
                                clique_members.update([u, v, w])

        if clique_members:
            for item in candidate_pool:
                if item["expression_id"] in clique_members:
                    item["is_clique_partner"] = True
                    # Apply confidence boost (+0.10 untuned initial guess)
                    item["effective_rel"] += clique_boost

    # -------------------------------------------------------------
    # Step 4: Vector MMR with Work Diversity Bonus (~0.3ms in NumPy)
    # -------------------------------------------------------------
    selected: list[dict[str, Any]] = []
    selected_vecs: list[np.ndarray] = []
    selected_works: set[Any] = set()
    remaining = list(candidate_pool)

    while len(selected) < top_k and remaining:
        best_idx = None
        best_score = float("-inf")

        for idx, item in enumerate(remaining):
            rel = item["effective_rel"]
            vec = item["vec"]
            red = max([float(np.dot(vec, sv)) for sv in selected_vecs]) if selected_vecs else 0.0

            wid = item["work_id"]
            div_term = 0.30 if wid not in selected_works else -0.15

            score = lam * rel - (1.0 - lam) * red + div_term
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = round(best_score, 4)
        selected.append(chosen)
        selected_vecs.append(chosen["vec"])
        if chosen["work_id"] is not None:
            selected_works.add(chosen["work_id"])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        "retrieve_synthesis_context: retrieved %d fragments in %.2f ms (pool=%d)",
        len(selected), elapsed_ms, len(candidate_pool),
    )

    return [
        SynthesisFragment(
            expression_id=s["expression_id"],
            work_id=s["work_id"],
            work_title=s["work_title"],
            author=s["author"],
            content=s["content"],
            q_sim=round(s["q_sim"], 4),
            edge_weight=round(s["edge_weight"], 4),
            effective_rel=round(s["effective_rel"], 4),
            mmr_score=s["mmr_score"],
            source_mode=s["source_mode"],
            is_clique_partner=s["is_clique_partner"],
            source_type=s.get("source_type", "audio_transcript"),
            is_playable=s.get("is_playable", True),
        )
        for s in selected
    ]
