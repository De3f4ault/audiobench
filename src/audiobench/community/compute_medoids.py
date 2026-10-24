"""Standalone production utility for community medoid computation.

Implements Soft-Weighted medoid selection:
    Score(n) = InternalStrength(n) * sqrt(AuthorCount(Author(n)) / CommunitySize)

This balances topological hubness (sum of internal edge weights) with author representation,
preventing cross-work bridge nodes from minority authors (<5% share) from hijacking cluster identity.
"""

from __future__ import annotations

import collections
import math
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from audiobench.memory.boilerplate import is_pure_boilerplate
from audiobench.memory.graph_linker import is_linkable_chunk


def is_clean_candidate_chunk(content: Optional[str], min_length: int = 80) -> bool:
    """Return True if content is a substantive, linkable chunk free of boilerplate."""
    if not content:
        return False
    text = content.strip()
    if len(text) < min_length:
        return False
    return is_linkable_chunk(text) and not is_pure_boilerplate(text)


def format_sample_passage(content: Optional[str], max_chars: int = 280) -> str:
    """Format and clean a sample passage, trimming at sentence boundaries."""
    if not content:
        return ""
    import re
    from audiobench.memory.boilerplate import clean_audio_disclaimers

    text = clean_audio_disclaimers(content).strip().replace("\r", "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    min_len = min(120, max(20, int(max_chars * 0.4)))
    last_stop = -1
    for punct in (". ", "! ", "? ", '." ', '!" ', '?" '):
        pos = cut.rfind(punct)
        if pos >= min_len and pos > last_stop:
            last_stop = pos + len(punct.rstrip())

    if last_stop > 0:
        return cut[:last_stop].strip()

    last_space = cut.rfind(" ")
    if last_space > min_len:
        return cut[:last_space].strip() + "..."
    return cut.strip() + "..."


def compute_soft_weighted_medoid(
    conn: sqlite3.Connection,
    community_id: int,
) -> Optional[int]:
    """Compute the Soft-Weighted medoid expression ID for a single community.

    Returns the expression ID of the winning medoid, or None if the community is empty.
    """
    cur = conn.cursor()

    # 1. Fetch all expressions belonging to this community
    cur.execute(
        """
        SELECT e.id, w.author, w.title, e.content
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        """,
        (community_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    node_ids: List[int] = [r[0] for r in rows]
    content_by_node: Dict[int, str] = {r[0]: (r[3] or "") for r in rows}

    # Author key: use author if present and non-generic; otherwise fall back to title.
    # This prevents distinct unauthored files from pooling into an artificial 'Unknown Author' monopoly.
    author_by_node: Dict[int, str] = {}
    for r in rows:
        author = r[1]
        title = r[2]
        if author and author.strip() and author.lower() not in ("unknown", "unknown author"):
            author_by_node[r[0]] = author.strip()
        elif title and title.strip():
            author_by_node[r[0]] = title.strip()
        else:
            author_by_node[r[0]] = "Unknown"

    community_size = len(node_ids)
    author_counts: collections.Counter = collections.Counter(author_by_node.values())

    if community_size == 1:
        return node_ids[0]

    # 2. Fetch all internal edges within this community (canonical dedup)
    placeholders = ",".join("?" * community_size)
    cur.execute(
        f"""
        SELECT 
            MIN(from_expression_id, to_expression_id),
            MAX(from_expression_id, to_expression_id),
            MAX(weight)
        FROM expression_relations
        WHERE from_expression_id IN ({placeholders})
          AND to_expression_id IN ({placeholders})
          AND relation_type = 'semantic'
          AND weight >= 0.78
        GROUP BY 1, 2
        """,
        node_ids + node_ids,
    )
    edge_rows = cur.fetchall()

    # 3. Compute internal weighted strength per node
    internal_strengths: Dict[int, float] = collections.defaultdict(float)
    for u, v, w in edge_rows:
        internal_strengths[u] += w
        internal_strengths[v] += w

    # 4. Soft-weighted score: Strength(n) * sqrt(AuthorShare(n))
    # Candidates must be substantive, linkable chunks and not pure boilerplate.
    best_node: Optional[int] = None
    best_score: float = -1.0

    for nid in node_ids:
        content = content_by_node[nid]
        if not is_clean_candidate_chunk(content, min_length=80):
            continue

        strength = internal_strengths.get(nid, 0.0)
        author = author_by_node[nid]
        author_share = author_counts[author] / community_size
        score = strength * math.sqrt(author_share)

        if score > best_score:
            best_score = score
            best_node = nid

    # If no filtered candidate had positive strength, fall back to any eligible node or first node
    if best_node is None or best_score <= 0.0:
        for nid in node_ids:
            content = content_by_node[nid]
            if is_clean_candidate_chunk(content, min_length=80):
                return nid
        return node_ids[0]

    return best_node


def select_representative_samples(
    conn: sqlite3.Connection,
    community_id: int,
    medoid_id: Optional[int] = None,
    max_samples: int = 6,
) -> List[Dict[str, Any]]:
    """Select up to `max_samples` substantive representative expressions for a community.

    Selection rules:
      1. Ranks top works within the community by chunk share.
      2. For each work (distinct works first), selects the highest internal-strength
         expression satisfying `is_clean_candidate_chunk(content, min_length=80)` that is not
         the medoid.
      3. If fewer than `max_samples` are found across distinct works, fills from remaining
         clean candidate expressions ordered by internal strength descending.
      4. Degrades gracefully to whatever number of clean candidates exist (0, 1, 2, ...).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id, e.work_id, COALESCE(w.title, 'Unknown Title'), COALESCE(w.author, 'Unknown Author'), e.content
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        """,
        (community_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return []

    node_ids = [r[0] for r in rows]
    node_data = {r[0]: {"work_id": r[1], "title": r[2], "author": r[3], "content": r[4] or ""} for r in rows}

    community_size = len(node_ids)
    placeholders = ",".join("?" * community_size)
    cur.execute(
        f"""
        SELECT 
            MIN(from_expression_id, to_expression_id),
            MAX(from_expression_id, to_expression_id),
            MAX(weight)
        FROM expression_relations
        WHERE from_expression_id IN ({placeholders})
          AND to_expression_id IN ({placeholders})
          AND relation_type = 'semantic'
          AND weight >= 0.78
        GROUP BY 1, 2
        """,
        node_ids + node_ids,
    )
    edge_rows = cur.fetchall()

    internal_strengths: Dict[int, float] = collections.defaultdict(float)
    for u, v, w in edge_rows:
        internal_strengths[u] += w
        internal_strengths[v] += w

    # Group clean candidate nodes by work_id
    work_nodes: Dict[Any, List[int]] = collections.defaultdict(list)
    for nid in node_ids:
        if medoid_id is not None and nid == medoid_id:
            continue
        if is_clean_candidate_chunk(node_data[nid]["content"], min_length=80):
            work_nodes[node_data[nid]["work_id"]].append(nid)

    # Sort each work's candidates by internal strength descending
    for wid in work_nodes:
        work_nodes[wid].sort(key=lambda n: internal_strengths[n], reverse=True)

    # Order works by total chunk count descending
    work_counts = collections.Counter(node_data[nid]["work_id"] for nid in node_ids)
    sorted_works = [w for w, _ in work_counts.most_common()]

    samples: List[Dict[str, Any]] = []
    chosen_nodes: Set[int] = set()

    # Pass 1: Top candidate per top work
    for wid in sorted_works:
        candidates = work_nodes.get(wid, [])
        for c in candidates:
            if c not in chosen_nodes:
                chosen_nodes.add(c)
                d = node_data[c]
                raw_c = d["content"].strip()
                samples.append({
                    "id": c,
                    "content": raw_c,
                    "formatted_content": format_sample_passage(raw_c, max_chars=280),
                    "title": d["title"],
                    "author": d["author"],
                    "strength": internal_strengths[c],
                })
                break
        if len(samples) >= max_samples:
            break

    # Pass 2: Fill remaining slots by strength across all remaining clean candidates
    if len(samples) < max_samples:
        remaining_candidates = []
        for wid, cands in work_nodes.items():
            for c in cands:
                if c not in chosen_nodes:
                    remaining_candidates.append(c)
        remaining_candidates.sort(key=lambda n: internal_strengths[n], reverse=True)
        for c in remaining_candidates:
            chosen_nodes.add(c)
            d = node_data[c]
            raw_c = d["content"].strip()
            samples.append({
                "id": c,
                "content": raw_c,
                "formatted_content": format_sample_passage(raw_c, max_chars=280),
                "title": d["title"],
                "author": d["author"],
                "strength": internal_strengths[c],
            })
            if len(samples) >= max_samples:
                break

    return samples


def populate_medoids(
    conn: sqlite3.Connection,
    community_ids: Optional[List[int]] = None,
) -> Dict[int, Optional[int]]:
    """Compute and persist medoid_expr_id for one or more communities.

    If community_ids is None, recomputes for all active communities.
    Returns mapping of community_id -> medoid_expr_id.
    """
    cur = conn.cursor()

    if community_ids is None:
        cur.execute("SELECT community_id FROM expression_communities WHERE is_active = 1")
        community_ids = [r[0] for r in cur.fetchall()]

    medoid_map: Dict[int, Optional[int]] = {}

    for cid in community_ids:
        medoid_id = compute_soft_weighted_medoid(conn, cid)
        medoid_map[cid] = medoid_id

        cur.execute(
            """
            UPDATE expression_communities
            SET medoid_expr_id = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE community_id = ?
            """,
            (medoid_id, cid),
        )

    conn.commit()
    return medoid_map


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from audiobench.core.settings import get_settings

    parser = argparse.ArgumentParser(description="Recompute community medoids.")
    parser.add_argument("--verify", action="store_true", help="Print verified medoids after computation")
    parser.add_argument("--community", type=int, default=None, help="Specific community_id to recompute")
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(settings.database_url.replace("sqlite:///", "")).resolve()
    print(f"[compute_medoids] Connecting to {db_path}...")

    conn = sqlite3.connect(str(db_path))
    targets = [args.community] if args.community is not None else None
    results = populate_medoids(conn, targets)
    print(f"[compute_medoids] Computed medoids for {len(results)} communities.")

    if args.verify:
        cur = conn.cursor()
        for cid, mid in sorted(results.items()):
            if mid is not None:
                cur.execute(
                    """
                    SELECT e.id, COALESCE(w.author, 'Unknown'), COALESCE(w.title, 'Unknown'), e.content
                    FROM expressions e
                    LEFT JOIN works w ON e.work_id = w.id
                    WHERE e.id = ?
                    """,
                    (mid,),
                )
                r = cur.fetchone()
                if r:
                    print(f"  • Community #{cid:02d}: Expr #{r[0]} | {r[1]} — {r[2][:35]} | \"{r[3][:60].strip()}...\"")
    conn.close()
