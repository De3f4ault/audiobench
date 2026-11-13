"""Smart Contextual Bookmarking Service.

Transforms bookmarks from static timestamps into dynamic conceptual study cards
featuring:
1. Tier 1 (Narrative Arc): Contextual build-up and resolution within the same book.
   Uses source relations or temporal offset |t1 - t2| >= 90s (no upper bound)
   to capture cross-chapter callbacks and thematic developments.
2. Tier 2 (Dialectic Connection): Cross-library conceptual bridges with work isolation.
   Falls back to in-memory HNSW (21.6k vectors in ~2ms) if node has < 2 edges.
3. Triadic Consensus Detection: Identifies 3-cliques (A-B-C across 3 distinct authors)
   and renders the Consensus Triangle Badge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.daemon.index import InCoreHNSWIndex
from audiobench.memory.graph_linker import ensure_hnsw_ready, is_linkable_chunk

logger = logging.getLogger("audiobench.memory.bookmark_service")


@dataclass
class DialecticBridge:
    """A cross-work conceptual connection from the bookmarked passage."""
    target_expression_id: int
    work_title: str
    author: str
    cosine_similarity: float
    snippet: str
    is_clique_partner: bool = False


@dataclass
class SmartBookmarkCard:
    """Two-tiered dynamic conceptual study card for a bookmarked passage."""
    expression_id: int
    work_id: int | None
    work_title: str
    author: str
    timestamp_display: str
    quote: str
    # Tier 1: Local Narrative Arc
    preceding_context: str | None = None
    following_resolution: str | None = None
    # Tier 2: Cross-Library Dialectic
    dialectic_bridges: list[DialecticBridge] = field(default_factory=list)
    consensus_triangle: list[str] | None = None  # e.g. ["Dostoyevsky", "Frankl", "Watts"]
    rendered_markdown: str = ""
    community_id: int | None = None
    community_name: str | None = None


def _format_timestamp(seconds: float | None) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def resolve_bookmark_expression(
    expression_id: int | None = None,
    audio_file_id: int | None = None,
    timestamp_seconds: float | None = None,
) -> int | None:
    """Resolve an expression ID either directly or from an audio_file_id + timestamp.

    Args:
        expression_id: Direct expression ID if known.
        audio_file_id: Audio file ID to search within.
        timestamp_seconds: Timestamp in seconds.

    Returns:
        Resolved expression_id, or None if not found.
    """
    if expression_id is not None:
        return expression_id

    if audio_file_id is None or timestamp_seconds is None:
        return None

    with get_session() as session:
        # 1. Match segment overlapping timestamp
        seg = session.execute(
            text("""
                SELECT s.id, tx.id
                FROM segments s
                JOIN transcriptions tx ON s.transcription_id = tx.id
                WHERE tx.audio_file_id = :af_id
                  AND :ts >= s.start_time AND :ts <= s.end_time
                LIMIT 1
            """),
            {"af_id": audio_file_id, "ts": timestamp_seconds},
        ).fetchone()

        if not seg:
            # Fallback to closest segment in audio file
            seg = session.execute(
                text("""
                    SELECT s.id, tx.id
                    FROM segments s
                    JOIN transcriptions tx ON s.transcription_id = tx.id
                    WHERE tx.audio_file_id = :af_id
                    ORDER BY ABS(s.start_time - :ts) ASC
                    LIMIT 1
                """),
                {"af_id": audio_file_id, "ts": timestamp_seconds},
            ).fetchone()

        if not seg:
            return None

        seg_id, tx_id = seg[0], seg[1]

        # 2. Check expression_segment_map
        expr = session.execute(
            text("""
                SELECT expression_id
                FROM expression_segment_map
                WHERE segment_id = :seg_id
                LIMIT 1
            """),
            {"seg_id": seg_id},
        ).fetchone()

        if expr:
            return int(expr[0])

        # 3. Fallback to sweep_chunk expression on transcription
        expr = session.execute(
            text("""
                SELECT id
                FROM expressions
                WHERE source_id = :tx_id AND graph_role = 'sweep_chunk'
                LIMIT 1
            """),
            {"tx_id": tx_id},
        ).fetchone()

        if expr:
            return int(expr[0])

    return None


def generate_smart_bookmark_card(
    expression_id: int,
    index: InCoreHNSWIndex | None = None,
) -> SmartBookmarkCard:
    """Generate a rich, two-tiered SmartBookmarkCard for the specified expression.

    Args:
        expression_id: Expression ID to inspect.
        index: Optional InCoreHNSWIndex instance.

    Returns:
        Populated SmartBookmarkCard with rendered markdown.
    """
    if not expression_id:
        raise ValueError("Invalid expression ID (must be positive non-zero integer).")

    with get_session() as session:
        # 1. Fetch source expression details
        row = session.execute(
            text("""
                SELECT e.id, e.work_id, e.content, w.title, w.author,
                       COALESCE(tx.audio_file_id, c.audio_file_id, c2.audio_file_id) as audio_file_id,
                       COALESCE(s.start_time, c.start_time, c2.start_time) as start_time,
                       COALESCE(s.end_time, c.end_time, c2.end_time) as end_time,
                       e.community_id, ec.name as community_name,
                       e.source_type, sq.query_text as sq_query, ss.title as ss_title
                FROM expressions e
                LEFT JOIN works w ON e.work_id = w.id
                LEFT JOIN transcriptions tx ON e.source_id = tx.id AND e.source_type = 'audio_transcript'
                LEFT JOIN chapters c ON (e.source_type = 'chapter_summary' AND e.source_id = c.id)
                LEFT JOIN chapters c2 ON (e.chapter_id IS NOT NULL AND e.chapter_id = c2.id)
                LEFT JOIN expression_segment_map esm ON e.id = esm.expression_id
                LEFT JOIN segments s ON esm.segment_id = s.id
                LEFT JOIN expression_communities ec ON e.community_id = ec.community_id
                LEFT JOIN search_queries sq ON (e.source_type = 'search_synthesis' AND e.source_id = sq.id)
                LEFT JOIN search_sessions ss ON (e.source_type = 'search_session_summary' AND e.source_id = ss.id)
                WHERE e.id = :eid
                LIMIT 1
            """),
            {"eid": expression_id},
        ).fetchone()

        if not row:
            raise ValueError(f"Expression #{expression_id} not found in database.")

        eid = row[0]
        work_id = row[1]
        content = row[2]
        r_title = row[3]
        r_author = row[4]
        af_id = row[5]
        start_time = row[6]
        end_time = row[7]
        community_id = row[8] if len(row) > 8 else None
        community_name = row[9] if len(row) > 9 else None
        source_type = row[10] if len(row) > 10 and row[10] else ""
        sq_query = row[11] if len(row) > 11 else None
        ss_title = row[12] if len(row) > 12 else None

        from audiobench.memory.synthesis_retriever import _format_expression_attribution
        author, work_title, is_playable = _format_expression_attribution(
            source_type, r_title, r_author, sq_query, ss_title, content
        )

        timestamp_str = _format_timestamp(start_time) if is_playable and start_time is not None else "Text Artifact"

        # -------------------------------------------------------------
        # Tier 1: Local Narrative Arc (Premise & Resolution)
        # -------------------------------------------------------------
        preceding_context: str | None = None
        following_resolution: str | None = None

        if work_id is not None and start_time is not None:
            # Query same-work context:
            # Prior premise (|t1 - t2| >= 90s prior in work)
            prior = session.execute(
                text("""
                    SELECT e.content
                    FROM expressions e
                    JOIN expression_segment_map esm ON e.id = esm.expression_id
                    JOIN segments s ON esm.segment_id = s.id
                    WHERE e.work_id = :wid
                      AND s.end_time <= :curr_start - 90
                      AND e.id != :eid
                    ORDER BY s.end_time DESC
                    LIMIT 1
                """),
                {"wid": work_id, "curr_start": start_time, "eid": eid},
            ).fetchone()
            if prior and is_linkable_chunk(prior[0]):
                preceding_context = prior[0].strip()

            # Following resolution (|t1 - t2| >= 90s later in work, no upper bound)
            later = session.execute(
                text("""
                    SELECT e.content
                    FROM expressions e
                    JOIN expression_segment_map esm ON e.id = esm.expression_id
                    JOIN segments s ON esm.segment_id = s.id
                    WHERE e.work_id = :wid
                      AND s.start_time >= :curr_end + 90
                      AND e.id != :eid
                    ORDER BY s.start_time ASC
                    LIMIT 1
                """),
                {"wid": work_id, "curr_end": end_time or start_time, "eid": eid},
            ).fetchone()
            if later and is_linkable_chunk(later[0]):
                following_resolution = later[0].strip()

        # Fallback to source-linked parent/child expressions if temporal query returned None
        if not preceding_context or not following_resolution:
            source_links = session.execute(
                text("""
                    SELECT e.id, e.content, r.from_expression_id
                    FROM expression_relations r
                    JOIN expressions e ON (
                        (r.from_expression_id = :eid AND r.to_expression_id = e.id)
                        OR (r.to_expression_id = :eid AND r.from_expression_id = e.id)
                    )
                    WHERE r.relation_type = 'source'
                    LIMIT 4
                """),
                {"eid": eid},
            ).fetchall()
            for s_id, s_content, f_id in source_links:
                if not preceding_context and f_id == eid and is_linkable_chunk(s_content):
                    preceding_context = s_content.strip()
                elif not following_resolution and f_id != eid and is_linkable_chunk(s_content):
                    following_resolution = s_content.strip()

        # -------------------------------------------------------------
        # Tier 2: Cross-Library Semantic Bridges
        # -------------------------------------------------------------
        bridges_raw = session.execute(
            text("""
                SELECT e2.id, e2.work_id, w2.title, w2.author, r.weight, e2.content
                FROM expression_relations r
                JOIN expressions e2 ON r.to_expression_id = e2.id
                LEFT JOIN works w2 ON e2.work_id = w2.id
                WHERE r.from_expression_id = :eid
                  AND r.relation_type = 'semantic'
                  AND (:wid IS NULL OR e2.work_id IS NULL OR e2.work_id != :wid)
                ORDER BY r.weight DESC
                LIMIT 5
            """),
            {"eid": eid, "wid": work_id},
        ).fetchall()

        dialectic_bridges: list[DialecticBridge] = []
        bridge_target_ids: list[int] = []

        for b_eid, b_wid, b_title, b_author, b_weight, b_content in bridges_raw:
            if is_linkable_chunk(b_content):
                dialectic_bridges.append(
                    DialecticBridge(
                        target_expression_id=b_eid,
                        work_title=b_title or "Unknown Work",
                        author=b_author or "Unknown Author",
                        cosine_similarity=round(float(b_weight), 3),
                        snippet=b_content.strip(),
                    )
                )
                bridge_target_ids.append(b_eid)

        # Dynamic HNSW Fallback if < 2 bridges in SQLite
        if len(dialectic_bridges) < 2:
            try:
                hnsw = ensure_hnsw_ready(index)
                if hnsw.has_item(eid):
                    q_vec = hnsw.get_items([eid])[0]
                    hits = hnsw.query(q_vec, k=30)
                    cand_ids = [h_id for h_id, _ in hits if h_id != eid and h_id not in bridge_target_ids]
                    if cand_ids:
                        c_placeholders = ", ".join(f":c{i}" for i in range(len(cand_ids)))
                        c_params = {f"c{i}": cid for i, cid in enumerate(cand_ids)}
                        c_params["wid"] = work_id
                        c_rows = session.execute(
                            text(f"""
                                SELECT e.id, w.title, w.author, e.content
                                FROM expressions e
                                LEFT JOIN works w ON e.work_id = w.id
                                WHERE e.id IN ({c_placeholders})
                                  AND (:wid IS NULL OR e.work_id IS NULL OR e.work_id != :wid)
                            """),
                            c_params,
                        ).fetchall()
                        dist_map = dict(hits)
                        for cr_id, cr_title, cr_author, cr_content in c_rows:
                            if is_linkable_chunk(cr_content):
                                sim = round(1.0 - dist_map.get(cr_id, 0.0), 3)
                                if sim >= 0.78:
                                    dialectic_bridges.append(
                                        DialecticBridge(
                                            target_expression_id=cr_id,
                                            work_title=cr_title or "Unknown Work",
                                            author=cr_author or "Unknown Author",
                                            cosine_similarity=sim,
                                            snippet=cr_content.strip(),
                                        )
                                    )
                                    bridge_target_ids.append(cr_id)
                                    if len(dialectic_bridges) >= 3:
                                        break
            except Exception as e:
                logger.warning("Dynamic HNSW fallback failed in bookmark service: %s", e)

        # -------------------------------------------------------------
        # Triadic Consensus Detection (3-Cliques)
        # -------------------------------------------------------------
        consensus_triangle: list[str] | None = None
        if len(bridge_target_ids) >= 2:
            # Check if any two bridge partners also connect to each other
            b_placeholders = ", ".join(f":b{i}" for i in range(len(bridge_target_ids)))
            b_params = {f"b{i}": bid for i, bid in enumerate(bridge_target_ids)}
            inter_edges = session.execute(
                text(f"""
                    SELECT r.from_expression_id, r.to_expression_id,
                           w1.author as a1, w2.author as a2
                    FROM expression_relations r
                    JOIN expressions e1 ON r.from_expression_id = e1.id
                    LEFT JOIN works w1 ON e1.work_id = w1.id
                    JOIN expressions e2 ON r.to_expression_id = e2.id
                    LEFT JOIN works w2 ON e2.work_id = w2.id
                    WHERE r.from_expression_id IN ({b_placeholders})
                      AND r.to_expression_id IN ({b_placeholders})
                      AND r.relation_type IN ('semantic', 'thematic')
                      AND r.weight >= 0.78
                      AND e1.work_id != e2.work_id
                    LIMIT 1
                """),
                b_params,
            ).fetchone()

            if inter_edges:
                b1_id, b2_id, b1_author, b2_author = inter_edges
                if b1_author and b2_author and b1_author != author and b2_author != author:
                    consensus_triangle = [author, b1_author, b2_author]
                    for b in dialectic_bridges:
                        if b.target_expression_id in (b1_id, b2_id):
                            b.is_clique_partner = True

    # -------------------------------------------------------------
    # Render Markdown Card
    # -------------------------------------------------------------
    md_lines: list[str] = [
        f"### Smart Bookmark Card: {work_title}",
        f"**Author**: {author} | **Timestamp**: `{timestamp_str}` (Expression #{expression_id})",
        "",
        f"> \"{content.strip()}\"",
        "",
        "---",
        "#### Tier 1: Local Narrative Arc",
    ]

    if preceding_context:
        md_lines.append(f"* **Premise (Context)**: \"{preceding_context[:220]}...\"")
    else:
        md_lines.append("* **Premise**: *(Opening or standalone passage)*")

    if following_resolution:
        md_lines.append(f"* **Resolution (Follow-up)**: \"{following_resolution[:220]}...\"")
    else:
        md_lines.append("* **Resolution**: *(Concluding or open-ended chapter segment)*")

    md_lines.extend([
        "",
        "---",
        "#### Tier 2: Cross-Library Semantic Bridges",
    ])

    if dialectic_bridges:
        for b in dialectic_bridges[:3]:
            clique_badge = " *(Consensus Partner)*" if b.is_clique_partner else ""
            md_lines.append(
                f"* **{b.author}** — *{b.work_title}* (Cosine: `{b.cosine_similarity:.3f}`){clique_badge}\n"
                f"  > \"{b.snippet[:200]}...\""
            )
    else:
        md_lines.append("*No cross-library semantic bridges detected above threshold.*")

    if consensus_triangle:
        md_lines.extend([
            "",
            f"**Consensus Triangle Badge**: Closed dialectic between **{consensus_triangle[0]}**, **{consensus_triangle[1]}**, and **{consensus_triangle[2]}**.",
        ])

    md_lines.extend([
        "",
        "---",
        f"*Type `\\synthesize \"{work_title}\"` to synthesize these perspectives.*",
    ])

    rendered_markdown = "\n".join(md_lines)

    return SmartBookmarkCard(
        expression_id=expression_id,
        work_id=work_id,
        work_title=work_title,
        author=author,
        timestamp_display=timestamp_str,
        quote=content.strip(),
        preceding_context=preceding_context,
        following_resolution=following_resolution,
        dialectic_bridges=dialectic_bridges,
        consensus_triangle=consensus_triangle,
        rendered_markdown=rendered_markdown,
        community_id=community_id,
        community_name=community_name,
    )
