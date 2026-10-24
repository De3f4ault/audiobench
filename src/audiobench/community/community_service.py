"""Core Community Detection ("Schools of Thought") service layer.

Provides:
    1. run_canonical_clustering: Stable base partition via Louvain with Jaccard-reconciliation.
    2. assign_incremental: Plurality-vote neighbor assignment for new ingest chunks with staleness refresh.
    3. get_community_summary: Formatted summary for CLI exploration and scoping.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from scipy.optimize import linear_sum_assignment

from audiobench.community.compute_medoids import populate_medoids
from audiobench.community.naming import generate_fallback_name
from audiobench.core.logger_factory import get_logger
from audiobench.memory.boilerplate import is_pure_boilerplate
from audiobench.memory.graph_linker import is_linkable_chunk

logger = get_logger("community.service")

# Minimum community size to be indexed as a persistent School of Thought.
# Micro-islands (<50 nodes) remain unclustered (community_id = NULL).
MACRO_COMMUNITY_MIN_SIZE = 50

# Minimum Jaccard similarity to preserve an existing community_id across re-clustering.
# Jaccard >= 0.50 mathematically guarantees majority overlap and prevents sister-cluster hijacking.
RECONCILIATION_JACCARD_THRESHOLD = 0.50

# Heuristic staleness thresholds: growth of >=20% and >=25 new nodes triggers local medoid/naming refresh.
STALENESS_GROWTH_RATIO = 0.20
STALENESS_MIN_NEW_NODES = 25

# Floating point epsilon for weighted vote comparison
FLOAT_TIE_EPSILON = 1e-6


def run_canonical_clustering(
    conn: Any,
    resolution: float = 0.6,
    seed: int = 42,
    min_size: int = MACRO_COMMUNITY_MIN_SIZE,
) -> Dict[str, Any]:
    """Execute canonical community detection across the semantic graph.

    Performs:
        1. Canonical edge deduplication (eliminating reciprocal double-counting).
        2. Louvain modularity optimization (resolution=0.6, seed=42).
        3. Size filtering (>= 50 nodes).
        4. Bipartite Maximum Jaccard Reconciliation against existing communities (if any).
        5. Atomic persistence to expression_communities and expressions.community_id.
        6. Soft-Weighted medoid calculation.
        7. Initial fallback naming.

    Returns:
        Summary dict: {n_communities, n_assigned, n_unassigned, elapsed_ms}
    """
    t0 = time.perf_counter()
    cur = conn.cursor()

    # 1. Total expressions in library
    cur.execute("SELECT COUNT(*) FROM expressions")
    total_expressions = cur.fetchone()[0]

    # 2. Load canonical deduped graph edges
    cur.execute(
        """
        SELECT 
            MIN(from_expression_id, to_expression_id) AS u,
            MAX(from_expression_id, to_expression_id) AS v,
            MAX(weight) AS max_w
        FROM expression_relations
        WHERE relation_type = 'semantic' AND weight >= 0.78
        GROUP BY 1, 2
        """
    )
    edge_rows = cur.fetchall()

    if not edge_rows:
        return {
            "n_communities": 0,
            "n_assigned": 0,
            "n_unassigned": total_expressions,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    G = nx.Graph()
    for u, v, w in edge_rows:
        G.add_edge(u, v, weight=w)

    # 3. Louvain partitioning
    raw_communities = nx_comm.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed
    )

    # Filter to macro-communities (>= min_size nodes), sorted by size descending
    new_macro_clusters: List[Set[int]] = sorted(
        [c for c in raw_communities if len(c) >= min_size],
        key=len,
        reverse=True,
    )

    # 4. Check for existing active communities for reconciliation
    cur.execute(
        """
        SELECT community_id
        FROM expression_communities
        WHERE is_active = 1
        ORDER BY community_id
        """
    )
    existing_community_ids = [r[0] for r in cur.fetchall()]

    existing_clusters: Dict[int, Set[int]] = {}
    for cid in existing_community_ids:
        cur.execute("SELECT id FROM expressions WHERE community_id = ?", (cid,))
        node_set = {r[0] for r in cur.fetchall()}
        if node_set:
            existing_clusters[cid] = node_set

    # Reconciliation mapping: new_cluster_index -> assigned_community_id
    cluster_id_assignment: Dict[int, int] = {}
    preserved_old_ids: Set[int] = set()

    next_available_id = (
        max(existing_community_ids) + 1 if existing_community_ids else 1
    )

    if existing_clusters and new_macro_clusters:
        old_id_list = list(existing_clusters.keys())
        n_old = len(old_id_list)
        n_new = len(new_macro_clusters)

        # Build Jaccard matrix
        cost_matrix = np.zeros((n_old, n_new))
        jaccard_matrix = np.zeros((n_old, n_new))

        for i, old_id in enumerate(old_id_list):
            old_set = existing_clusters[old_id]
            for j, new_set in enumerate(new_macro_clusters):
                inter = len(old_set & new_set)
                union = len(old_set | new_set)
                jacc = inter / union if union > 0 else 0.0
                jaccard_matrix[i, j] = jacc
                cost_matrix[i, j] = -jacc

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            jacc = jaccard_matrix[i, j]
            if jacc >= RECONCILIATION_JACCARD_THRESHOLD:
                assigned_id = old_id_list[i]
                cluster_id_assignment[j] = assigned_id
                preserved_old_ids.add(assigned_id)
                logger.debug(
                    "Preserved community_id %d (Jaccard = %.4f)", assigned_id, jacc
                )

    # Assign IDs to newly emerged clusters without match
    for j in range(len(new_macro_clusters)):
        if j not in cluster_id_assignment:
            cluster_id_assignment[j] = next_available_id
            next_available_id += 1

    # Mark dissolved old communities as archived (is_active = 0, node_count = 0)
    for old_id in existing_community_ids:
        if old_id not in preserved_old_ids:
            cur.execute(
                """
                UPDATE expression_communities
                SET is_active = 0,
                    node_count = 0,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE community_id = ?
                """,
                (old_id,),
            )

    # 5. Reset all community_id assignments on expressions
    cur.execute("UPDATE expressions SET community_id = NULL")

    # 6. Insert or update expression_communities and assign expressions
    total_assigned = 0
    assigned_cids: List[int] = []

    for j, cluster_nodes in enumerate(new_macro_clusters):
        cid = cluster_id_assignment[j]
        size = len(cluster_nodes)
        assigned_cids.append(cid)
        total_assigned += size

        # Upsert community row
        cur.execute(
            """
            INSERT INTO expression_communities (
                community_id, name, name_source, is_active, node_count,
                medoid_at_node_count, resolution, seed, indexed_at, updated_at
            ) VALUES (?, NULL, 'fallback', 1, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            ON CONFLICT(community_id) DO UPDATE SET
                is_active = 1,
                node_count = excluded.node_count,
                medoid_at_node_count = excluded.medoid_at_node_count,
                resolution = excluded.resolution,
                seed = excluded.seed,
                indexed_at = excluded.indexed_at,
                updated_at = excluded.updated_at
            """,
            (cid, size, size, resolution, seed),
        )

        # Bulk update expressions.community_id for this cluster
        node_list = list(cluster_nodes)
        # Process in batches of 500
        batch_size = 500
        for b_start in range(0, len(node_list), batch_size):
            batch = node_list[b_start : b_start + batch_size]
            cur.execute(
                f"UPDATE expressions SET community_id = ? WHERE id IN ({','.join('?' * len(batch))})",
                [cid] + batch,
            )

    conn.commit()

    # 7. Compute Soft-Weighted Medoids
    populate_medoids(conn, assigned_cids)

    # 8. Generate fallback names
    for cid in assigned_cids:
        cur.execute("SELECT name FROM expression_communities WHERE community_id = ?", (cid,))
        row = cur.fetchone()
        if not row or not row[0]:
            fallback_title = generate_fallback_name(conn, cid)
            cur.execute(
                """
                UPDATE expression_communities
                SET name = ?, name_source = 'fallback'
                WHERE community_id = ?
                """,
                (fallback_title, cid),
            )

    conn.commit()

    total_unassigned = total_expressions - total_assigned
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Canonical clustering complete: %d communities, %d assigned, %d unassigned (%.1f ms)",
        len(new_macro_clusters),
        total_assigned,
        total_unassigned,
        elapsed_ms,
    )

    return {
        "n_communities": len(new_macro_clusters),
        "n_assigned": total_assigned,
        "n_unassigned": total_unassigned,
        "elapsed_ms": elapsed_ms,
    }


def assign_incremental(
    conn: Any,
    expression_ids: List[int],
) -> int:
    """Assign newly ingested expressions to communities using weighted neighbor-voting.

    Plurality is determined by the sum of edge weights to neighboring communities:
        Vote(c) = sum_{v in Neighbors(u), comm(v) = c} w(u, v)

    Ambiguous ties (diff < 1e-6) or expressions with no community-assigned neighbors stay NULL.
    Monitors community growth and triggers targeted local medoid refresh if size increases by >= 20%.

    Returns:
        Number of expressions successfully assigned to an active community.
    """
    if not expression_ids:
        return 0

    cur = conn.cursor()
    assigned_count = 0
    touched_communities: Set[int] = set()

    for expr_id in expression_ids:
        # Fetch all semantic neighbors and their edge weights
        cur.execute(
            """
            SELECT 
                (CASE WHEN from_expression_id = ? THEN to_expression_id ELSE from_expression_id END) AS nbr_id,
                weight
            FROM expression_relations
            WHERE (from_expression_id = ? OR to_expression_id = ?)
              AND relation_type = 'semantic'
              AND weight >= 0.78
            """,
            (expr_id, expr_id, expr_id),
        )
        edges = cur.fetchall()
        if not edges:
            continue

        nbr_ids = [r[0] for r in edges]
        weights_by_nbr = {r[0]: r[1] for r in edges}

        # Fetch communities of neighbors
        placeholders = ",".join("?" * len(nbr_ids))
        cur.execute(
            f"""
            SELECT id, community_id
            FROM expressions
            WHERE id IN ({placeholders}) AND community_id IS NOT NULL
            """,
            nbr_ids,
        )
        nbr_comms = cur.fetchall()
        if not nbr_comms:
            continue

        # Sum weights per community
        votes: Dict[int, float] = collections.defaultdict(float)
        for n_id, c_id in nbr_comms:
            votes[c_id] += weights_by_nbr[n_id]

        if not votes:
            continue

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_cid, top_weight = sorted_votes[0]

        # Check for tie with epsilon
        if len(sorted_votes) > 1:
            second_weight = sorted_votes[1][1]
            if abs(top_weight - second_weight) < FLOAT_TIE_EPSILON:
                # Ambiguous tie: leave community_id NULL
                continue

        # Assign expression to winning community
        cur.execute(
            "UPDATE expressions SET community_id = ? WHERE id = ?",
            (top_cid, expr_id),
        )
        cur.execute(
            """
            UPDATE expression_communities
            SET node_count = node_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE community_id = ?
            """,
            (top_cid,),
        )
        assigned_count += 1
        touched_communities.add(top_cid)

    # Heuristic staleness check for touched communities
    for cid in touched_communities:
        cur.execute(
            """
            SELECT node_count, medoid_at_node_count
            FROM expression_communities
            WHERE community_id = ?
            """,
            (cid,),
        )
        row = cur.fetchone()
        if row:
            node_cnt, med_cnt = row[0], row[1]
            threshold = max(STALENESS_MIN_NEW_NODES, int(STALENESS_GROWTH_RATIO * med_cnt))
            if (node_cnt - med_cnt) >= threshold:
                logger.info(
                    "Community #%d grew by %d nodes (>= %d threshold); refreshing medoid",
                    cid,
                    node_cnt - med_cnt,
                    threshold,
                )
                populate_medoids(conn, [cid])
                cur.execute(
                    """
                    UPDATE expression_communities
                    SET medoid_at_node_count = ?,
                        name_source = 'fallback',
                        naming_attempted_at = NULL,
                        naming_failure_count = 0,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    WHERE community_id = ?
                    """,
                    (node_cnt, cid),
                )

    if assigned_count > 0:
        conn.commit()

    return assigned_count


def get_community_summary(conn: Any, community_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve full summary of a single community for exploration or scoping.

    Guarantees a clean human-readable name:
    If name is None or pending, returns: "School #<id> (<authors>) [Naming in progress...]".
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            c.community_id, c.name, c.name_source, c.is_active, c.node_count,
            c.medoid_expr_id, e.content, e.work_id
        FROM expression_communities c
        LEFT JOIN expressions e ON c.medoid_expr_id = e.id
        WHERE c.community_id = ?
        """,
        (community_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    cid, name, name_source, is_active, node_count, med_id, med_content, med_work_id = row

    # Fetch top authors and works
    cur.execute(
        """
        SELECT COALESCE(w.author, 'Unknown'), COUNT(*)
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        GROUP BY w.author
        ORDER BY COUNT(*) DESC
        LIMIT 3
        """,
        (community_id,),
    )
    top_authors = [f"{r[0]} ({r[1]})" for r in cur.fetchall()]

    authors_str = ", ".join(r.split(" (")[0] for r in top_authors if "Unknown" not in r)
    if not authors_str:
        authors_str = "Library Authors"

    # Display name fallback
    display_name = name
    if not display_name or name_source == "pending":
        display_name = f"School #{cid} ({authors_str}) [Naming in progress...]"

    return {
        "community_id": cid,
        "name": display_name,
        "name_source": name_source,
        "is_active": bool(is_active),
        "node_count": node_count,
        "medoid_expr_id": med_id,
        "medoid_content": med_content or "",
        "top_authors": top_authors,
    }


def list_active_schools(conn: Any) -> List[Dict[str, Any]]:
    """List all active communities formatted for table display.

    Returns:
        List of dicts: [
            {
                "community_id": int,
                "name": str,
                "node_count": int,
                "core_sources": str,
            },
            ...
        ]
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT community_id, name, node_count
        FROM expression_communities
        WHERE is_active = 1
        ORDER BY community_id ASC
        """
    )
    rows = cur.fetchall()
    results = []
    for cid, name, node_count in rows:
        cur.execute(
            """
            SELECT COALESCE(NULLIF(TRIM(w.author), ''), NULLIF(TRIM(w.title), ''), 'Unknown') AS src,
                   COUNT(*) as cnt
            FROM expressions e
            LEFT JOIN works w ON e.work_id = w.id
            WHERE e.community_id = ?
            GROUP BY src
            ORDER BY cnt DESC
            LIMIT 3
            """,
            (cid,),
        )
        top = cur.fetchall()
        core_sources = ", ".join(f"{s} ({c})" for s, c in top)
        results.append({
            "community_id": cid,
            "name": name or f"School #{cid}",
            "node_count": node_count,
            "core_sources": core_sources,
        })
    return results


def get_school_details(conn: Any, community_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve comprehensive details for a single School of Thought dossier card.

    Includes:
        - Metadata (name, node_count, resolution, seed)
        - Medoid passage with author/work attribution
        - Top 5 works breakdown with chunk counts and percentages
        - 3 key sample expressions across top works filtered against boilerplate
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            c.community_id, c.name, c.node_count, c.resolution, c.seed,
            c.medoid_expr_id, e.content, w.title, w.author
        FROM expression_communities c
        LEFT JOIN expressions e ON c.medoid_expr_id = e.id
        LEFT JOIN works w ON e.work_id = w.id
        WHERE c.community_id = ? AND c.is_active = 1
        """,
        (community_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    cid, name, node_count, resolution, seed, med_id, med_content, med_work_title, med_work_author = row

    # Top 5 works breakdown
    cur.execute(
        """
        SELECT COALESCE(w.title, 'Unknown Title'),
               COALESCE(w.author, 'Unknown Author'),
               COUNT(*) as cnt
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        GROUP BY e.work_id
        ORDER BY cnt DESC
        LIMIT 5
        """,
        (community_id,),
    )
    top_works = []
    for w_title, w_author, cnt in cur.fetchall():
        pct = (cnt / node_count * 100.0) if node_count > 0 else 0.0
        top_works.append({
            "title": w_title,
            "author": w_author,
            "count": cnt,
            "pct": pct,
        })

    # Select representative sample expressions using canonical medoid filtering
    from audiobench.community.compute_medoids import select_representative_samples

    samples = select_representative_samples(conn, cid, medoid_id=med_id, max_samples=6)

    return {
        "community_id": cid,
        "name": name or f"School #{cid}",
        "node_count": node_count,
        "resolution": resolution,
        "seed": seed,
        "medoid": {
            "id": med_id,
            "content": (med_content or "").strip(),
            "work_title": med_work_title or "Unknown Work",
            "work_author": med_work_author,
        },
        "top_works": top_works,
        "sample_expressions": samples,
    }


def get_active_school_ids(conn: Any) -> List[Tuple[int, str]]:
    """Return list of (community_id, name) tuples for all active schools."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT community_id, COALESCE(name, 'School #' || community_id)
        FROM expression_communities
        WHERE is_active = 1
        ORDER BY community_id ASC
        """
    )
    return cur.fetchall()


def get_related_schools(conn: Any, community_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Query macro inter-community edge bundles connecting community_id to neighboring schools."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            CASE WHEN e1.community_id = ? THEN e2.community_id ELSE e1.community_id END as other_cid,
            ec.name,
            COUNT(*) as edge_count,
            SUM(er.weight) as total_weight,
            AVG(er.weight) as avg_weight
        FROM expression_relations er
        JOIN expressions e1 ON er.from_expression_id = e1.id
        JOIN expressions e2 ON er.to_expression_id = e2.id
        JOIN expression_communities ec ON (CASE WHEN e1.community_id = ? THEN e2.community_id ELSE e1.community_id END) = ec.community_id
        WHERE ((e1.community_id = ? AND e2.community_id != ?) OR (e2.community_id = ? AND e1.community_id != ?))
          AND er.relation_type = 'semantic'
          AND er.weight >= 0.78
          AND ec.is_active = 1
        GROUP BY other_cid
        ORDER BY total_weight DESC
        LIMIT ?
        """,
        (community_id, community_id, community_id, community_id, community_id, community_id, limit),
    )
    return [
        {
            "community_id": r[0],
            "name": r[1] or f"School #{r[0]}",
            "edge_count": r[2],
            "coupling_weight": float(r[3]),
            "avg_weight": float(r[4]),
        }
        for r in cur.fetchall()
    ]


def get_cross_school_bridge_pairs(conn: Any, community_id: int, limit: int = 4) -> List[Dict[str, Any]]:
    """Query top individual semantic bridge edges connecting inside expressions to outside expressions."""
    from audiobench.community.compute_medoids import is_clean_candidate_chunk, format_sample_passage

    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            e_in.id, w_in.title, w_in.author, e_in.content, e_in.community_id, e_in.source_id,
            e_ext.id, w_ext.title, w_ext.author, e_ext.content, e_ext.community_id, e_ext.source_id,
            ec_ext.name,
            er.weight
        FROM expression_relations er
        JOIN expressions e_in ON (er.from_expression_id = e_in.id OR er.to_expression_id = e_in.id)
        JOIN expressions e_ext ON (
            CASE WHEN er.from_expression_id = e_in.id THEN er.to_expression_id ELSE er.from_expression_id END = e_ext.id
        )
        LEFT JOIN works w_in ON e_in.work_id = w_in.id
        LEFT JOIN works w_ext ON e_ext.work_id = w_ext.id
        LEFT JOIN expression_communities ec_ext ON e_ext.community_id = ec_ext.community_id
        WHERE e_in.community_id = ?
          AND (e_ext.community_id IS NULL OR e_ext.community_id != ?)
          AND er.relation_type = 'semantic'
          AND er.weight >= 0.78
        ORDER BY er.weight DESC
        """,
        (community_id, community_id),
    )

    pairs = []
    seen_in_nodes = set()
    seen_ext_nodes = set()
    for row in cur.fetchall():
        in_id, in_title, in_author, in_content, in_cid, in_src, ext_id, ext_title, ext_author, ext_content, ext_cid, ext_src, ext_cname, weight = row
        if in_id in seen_in_nodes or ext_id in seen_ext_nodes:
            continue
        if not is_clean_candidate_chunk(in_content, min_length=80):
            continue
        if not is_clean_candidate_chunk(ext_content, min_length=80):
            continue
        seen_in_nodes.add(in_id)
        seen_ext_nodes.add(ext_id)
        pairs.append({
            "inside": {
                "id": in_id,
                "title": in_title or "Unknown Work",
                "author": in_author or "Unknown Author",
                "content": in_content.strip(),
                "formatted_content": format_sample_passage(in_content, max_chars=280),
                "community_id": in_cid,
                "source_id": in_src,
            },
            "outside": {
                "id": ext_id,
                "title": ext_title or "Unknown Work",
                "author": ext_author or "Unknown Author",
                "content": ext_content.strip(),
                "formatted_content": format_sample_passage(ext_content, max_chars=280),
                "community_id": ext_cid,
                "source_id": ext_src,
                "community_name": ext_cname or "General Corpus",
            },
            "weight": float(weight),
        })
        if len(pairs) >= limit:
            break

    return pairs


