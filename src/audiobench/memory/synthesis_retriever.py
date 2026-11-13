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
import re
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

# Per-author MMR cap — prevents single-author monopolisation of synthesis results.
# Empirically swept at k_ann=40 against three benchmark queries. cap=6 preserves
# diversity across multi-author queries while allowing adequate single-author depth.
# See: implementation_plan.md → Empirical Sweep Results.
DEFAULT_MAX_PER_AUTHOR: int = 6

# Pseudo-author buckets — internal metadata sources that share sub-caps independently
# of named-author cap. PRIOR and CHAT buckets have lower caps because these sources
# are non-playable and should not crowd out primary audio content.
PRIOR_BUCKET: frozenset[str] = frozenset({"[Prior Synthesis]", "[Past Session]"})
CHAT_BUCKET: frozenset[str] = frozenset({"[Chat Reflection]", "[Cognitive Note]"})
UNKNOWN_BUCKET_SENTINEL: frozenset[str | None] = frozenset({None, "Unknown", "Unknown Author"})

DEFAULT_PRIOR_CAP: int = 3     # max fragments from PRIOR_BUCKET (non-playable, pre-synthesised)
DEFAULT_UNKNOWN_CAP: int = 5   # max fragments from UNKNOWN_BUCKET (unattributed content)


class AmbiguousAuthorError(ValueError):
    """Raised when an author search pattern matches multiple distinct authors in the candidate pool."""

    def __init__(self, requested: str, candidates: list[str]):
        cand_str = ", ".join(sorted(candidates))
        super().__init__(f"Author '{requested}' matches multiple in pool: {cand_str}. Be more specific.")
        self.requested = requested
        self.candidates = sorted(candidates)


@dataclass
class RetrievalMeta:
    """Metadata about a retrieve_synthesis_context call, returned alongside fragments.

    Enables downstream callers (footer rendering, tests) to inspect retrieval internals
    without coupling to the MMR loop implementation. Note: `focused`, `focused_author`,
    and `author_match_status` are ephemeral call metadata, not persisted in synthesis history.
    """
    pool_size: int                          # total candidates before MMR selection
    k_ann: int                              # ANN seed count used for this call
    author_counts_pool: dict[str, int]      # per-author fragment count in pool (pre-cap)
    author_counts_selected: dict[str, int]  # per-author fragment count in selected set
    capped_authors: set[str]               # authors where pool_n > cap AND selected == cap
    prior_count: int                        # PRIOR_BUCKET fragments selected
    unknown_count: int                      # UNKNOWN_BUCKET fragments selected
    focused: bool = False                   # True if per-author cap was bypassed
    focused_author: str | None = None       # Resolved author if author filter/focus was used
    author_match_status: str | None = None  # None | "resolved" | "zero_match" | "multi_match"


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
        author = (work_author or "").strip() or "Unknown Author"
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
    focused: bool = False,
    author: str | None = None,
    max_per_author: int | None = DEFAULT_MAX_PER_AUTHOR,
) -> tuple[list[SynthesisFragment], RetrievalMeta]:
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
        focused: When True, bypasses the per-author diversity cap across named authors.
        author: Optional author name to filter candidate pool before MMR selection.
        max_per_author: Maximum fragments per named author when not in focused mode (default 6).

    Returns:
        Tuple of (fragments, meta) where fragments is a list of SynthesisFragment ordered
        by MMR selection, and meta is a RetrievalMeta with pool-level accounting for
        footer rendering and diagnostics.

    BREAKING: Changed from list[SynthesisFragment] to tuple[list[SynthesisFragment], RetrievalMeta].
    All six call sites updated. See commit message for full audit.
    """
    t0 = time.perf_counter()
    hnsw = ensure_hnsw_ready(index)
    q_vec = embed_query(query)

    # -------------------------------------------------------------
    # Step 1: Retrieve ANN Seeds from In-Core HNSW Index
    # -------------------------------------------------------------
    # k_ann=max(40, top_k*2) is empirically faster than k_ann=15 on warm daemon:
    # ANN: 2.4→0.6ms, SQL: 22→2.6ms, clique: 11.6→3.8ms. Pool grows ~65%.
    # This is a behavior change — seed composition may differ from prior versions.
    k_ann = max(40, top_k * 2) if community_id is None else max(60, top_k * 3)
    raw_hits = hnsw.query(q_vec, k=k_ann)
    if not raw_hits:
        logger.warning("No ANN hits found for query: %s", query)
        empty_meta = RetrievalMeta(
            pool_size=0, k_ann=k_ann,
            author_counts_pool={}, author_counts_selected={},
            capped_authors=set(), prior_count=0, unknown_count=0,
            focused=focused, focused_author=author,
            author_match_status=None,
        )
        return [], empty_meta

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

        f_author, f_title, is_playable = _format_expression_attribution(
            r_st, r_title, r_author, r_sq, r_ss, r_content
        )

        candidate_pool.append({
            "expression_id": r_eid,
            "work_id": r_wid,
            "work_title": f_title,
            "author": f_author,
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
    # Step 2: 1-Hop Graph Expansion from Top-6 Work-Backed Seeds
    # -------------------------------------------------------------
    # Only use expressions with work_id IS NOT NULL as seeds.
    # search_synthesis and search_query expressions (work_id = NULL) are
    # LLM-compressed and score high in ANN, but graph_linker.py explicitly
    # excludes them from edge creation — so they have zero neighbors and
    # silently kill graph expansion. This compounds with use: each \synthesize
    # run creates another ghost that crowds out book-backed seeds. Empirically:
    # filtering to work-backed seeds + raising count from 3→6 raises triangle
    # discovery from 40% to 70% across 20 test queries (no quality loss).
    _TOP_K_SEEDS = 6
    top_seed_ids = [
        c["expression_id"]
        for c in candidate_pool
        if c.get("work_id") is not None
    ][:_TOP_K_SEEDS]
    if top_seed_ids:
        seed_placeholders = ", ".join(f":sid{i}" for i in range(len(top_seed_ids)))
        seed_params = {f"sid{i}": sid for i, sid in enumerate(top_seed_ids)}
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

            f_author, f_title, is_playable = _format_expression_attribution(
                g_st, g_title, g_author, g_sq, g_ss, g_content
            )

            candidate_pool.append({
                "expression_id": t_eid,
                "work_id": g_wid,
                "work_title": f_title,
                "author": f_author,
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
        empty_meta = RetrievalMeta(
            pool_size=0, k_ann=k_ann,
            author_counts_pool={}, author_counts_selected={},
            capped_authors=set(), prior_count=0, unknown_count=0,
            focused=focused, focused_author=author,
            author_match_status=None,
        )
        return [], empty_meta

    # Count every candidate's author in the pool before selection begins.
    # This establishes the pool-level author distribution for the footer
    # and for capped_authors computation at exit.
    author_counts_pool: dict[str, int] = {}
    for item in candidate_pool:
        a = item["author"]
        author_counts_pool[a] = author_counts_pool.get(a, 0) + 1

    # Author filtering and validation
    focused_author: str | None = None
    author_match_status: str | None = None

    if author:
        # Match using word boundaries to avoid false substring collisions
        author_pattern = r"\b" + re.escape(author.strip()) + r"\b"
        unique_authors = [
            a for a in author_counts_pool.keys()
            if a and a not in PRIOR_BUCKET and a not in UNKNOWN_BUCKET_SENTINEL
        ]
        matches = [a for a in unique_authors if re.search(author_pattern, a, re.IGNORECASE)]

        if len(matches) > 1:
            raise AmbiguousAuthorError(author, matches)
        elif len(matches) == 1:
            focused_author = matches[0]
            author_match_status = "resolved"
            # Filter pool to this author
            candidate_pool = [c for c in candidate_pool if c["author"] == focused_author]
        else:
            author_match_status = "zero_match"
            if focused:
                # Fall back to general focused mode across all authors
                focused_author = author
            else:
                # Author filter requested alone with 0 matches -> empty pool
                empty_meta = RetrievalMeta(
                    pool_size=0, k_ann=k_ann,
                    author_counts_pool=author_counts_pool, author_counts_selected={},
                    capped_authors=set(), prior_count=0, unknown_count=0,
                    focused=focused, focused_author=author, author_match_status=author_match_status,
                )
                return [], empty_meta

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
    # Step 4: Vector MMR with Work Diversity Bonus and Author Cap
    # -------------------------------------------------------------
    selected: list[dict[str, Any]] = []
    selected_vecs: list[np.ndarray] = []
    selected_works: set[Any] = set()
    author_counts_selected: dict[str, int] = {}  # tracks selected per author/bucket
    prior_count: int = 0
    unknown_count: int = 0
    remaining = list(candidate_pool)

    while len(selected) < top_k and remaining:
        best_idx = None
        best_score = float("-inf")

        for idx, item in enumerate(remaining):
            a = item["author"]

            # PRIOR and UNKNOWN caps fire regardless of focused mode. `--focused`
            # only bypasses caps for named authors. Rationale: prior syntheses and
            # unknown-author fragments are noise sources, not legit depth targets.
            if a in PRIOR_BUCKET:
                if prior_count >= DEFAULT_PRIOR_CAP:
                    continue
            elif a in UNKNOWN_BUCKET_SENTINEL or a is None:
                if unknown_count >= DEFAULT_UNKNOWN_CAP:
                    continue
            else:
                # Named author cap enforcement
                effective_cap = None if focused else max_per_author
                if effective_cap is not None and author_counts_selected.get(a, 0) >= effective_cap:
                    continue

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

        # Update per-author selection counts for cap enforcement
        a = chosen["author"]
        author_counts_selected[a] = author_counts_selected.get(a, 0) + 1
        if a in PRIOR_BUCKET:
            prior_count += 1
        elif a in UNKNOWN_BUCKET_SENTINEL or a is None:
            unknown_count += 1

    # Build capped_authors: authors where pool_n > cap AND selected == cap.
    # pool_n == cap does not indicate capping — the pool was exhausted naturally.
    capped_authors: set[str] = set()
    if not focused and max_per_author is not None:
        capped_authors = {
            a for a, pool_n in author_counts_pool.items()
            if a not in PRIOR_BUCKET
            and a not in UNKNOWN_BUCKET_SENTINEL
            and a is not None
            and pool_n > max_per_author
            and author_counts_selected.get(a, 0) == max_per_author
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        "retrieve_synthesis_context: retrieved %d fragments in %.2f ms (pool=%d, capped=%s)",
        len(selected), elapsed_ms, len(candidate_pool), capped_authors or "none",
    )

    meta = RetrievalMeta(
        pool_size=len(candidate_pool),
        k_ann=k_ann,
        author_counts_pool=author_counts_pool,
        author_counts_selected=author_counts_selected,
        capped_authors=capped_authors,
        prior_count=prior_count,
        unknown_count=unknown_count,
        focused=focused,
        focused_author=focused_author,
        author_match_status=author_match_status,
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
    ], meta
