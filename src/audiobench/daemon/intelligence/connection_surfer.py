"""ConnectionSurfer — daemon intelligence task for building the knowledge graph.

Finds semantically similar sweep_chunk expressions across different audio files
and writes permanent SEMANTIC edges between them in ``expression_relations``.

Algorithm
---------
1. Sample up to ``surfer_sample_size`` sweep_chunk expressions that have vectors
   in LanceDB, joining to audio_files to resolve file titles at query time.
2. Retrieve their vectors from LanceDB in one batch call.
3. Run pairwise cosine similarity over all sampled pairs (O(N^2) over 2,000
   vectors; each vector is 768-dim floats, takes ~2 s on CPU).
4. For each pair above ``surfer_sim_threshold`` (default 0.78):
   - Skip if both expressions come from the same audio_file_id (same source).
   - Skip if either node already has >= ``surfer_max_edges_per_node`` outbound
     SEMANTIC edges (prevents super-attractor collapse).
   - Write a permanent ``ExpressionRelation`` with ``relation_type='semantic'``
     via ``ExpressionRepository.link()`` (idempotent ON CONFLICT DO NOTHING).
5. On the first-ever run, promote all 39 existing ``pending_relations`` rows
   to real SEMANTIC edges before sampling begins.
6. Calibrate: store the total SEMANTIC edge count in a ``daemon_calibration``
   expression so the next cycle can log progress.
"""

from __future__ import annotations

import json

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.server import _get_store
from audiobench.memory.enums import RelationType
from audiobench.storage.expression_repository import ExpressionRepository

from .math_utils import pairwise_cosine_above_threshold
from .scheduler import IntelligenceTask

logger = get_logger("daemon.intelligence.connection_surfer")


class ConnectionSurfer(IntelligenceTask):
    INTERVAL_SECONDS = 3600 * 2  # Every 2 hours (tuned from 6 hours for responsive cross-linking)
    IS_INFERENCE_TASK = False

    # ------------------------------------------------------------------
    # Backfill — promote pending_relations on first run
    # ------------------------------------------------------------------

    def _backfill_pending_relations(self, expr_repo: ExpressionRepository) -> int:
        """Promote all existing ``pending_relations`` rows to real SEMANTIC edges.

        These stubs were created by the old broken surfer but never written to
        ``expression_relations``.  We resolve them once on the first cycle.
        """
        promoted = 0
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                    SELECT pr.from_expression_id, pr.to_expression_id_hint
                    FROM pending_relations pr
                    JOIN expressions ea ON ea.id = pr.from_expression_id
                    JOIN expressions eb ON eb.id = pr.to_expression_id_hint
                    WHERE pr.to_expression_id_hint IS NOT NULL
                """)
            ).fetchall()

        for from_id, to_id in rows:
            try:
                expr_repo.link(
                    from_id=from_id,
                    to_id=to_id,
                    relation_type=RelationType.SEMANTIC.value,
                    weight=1.0,
                    created_by="backfill_pending",
                )
                promoted += 1
            except Exception as e:
                logger.debug("backfill: could not link %d -> %d: %s", from_id, to_id, e)

        if promoted:
            with get_session() as session:
                session.execute(sql_text("DELETE FROM pending_relations"))
                session.commit()
            logger.info("Backfilled %d pending_relations as SEMANTIC edges", promoted)

        return promoted

    # ------------------------------------------------------------------
    # Edge cap check
    # ------------------------------------------------------------------

    def _semantic_degree(self, expr_id: int) -> int:
        """Return the total number of existing SEMANTIC edges (inbound or outbound) for *expr_id*."""
        with get_session() as session:
            count = session.execute(
                sql_text("""
                    SELECT COUNT(*) FROM expression_relations
                    WHERE (from_expression_id = :id OR to_expression_id = :id)
                    AND relation_type = :rel
                """),
                {"id": expr_id, "rel": RelationType.SEMANTIC.value},
            ).scalar()
        return int(count or 0)

    def _outbound_semantic_count(self, expr_id: int) -> int:
        """Backward-compatible alias for _semantic_degree."""
        return self._semantic_degree(expr_id)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("ConnectionSurfer: starting cycle")
        settings = get_settings()
        store = _get_store()
        expr_repo = ExpressionRepository()

        # --- Step 0: backfill pending_relations once (idempotent) --------
        with get_session() as session:
            pending_count = session.execute(
                sql_text("SELECT COUNT(*) FROM pending_relations")
            ).scalar() or 0

        if pending_count > 0:
            self._backfill_pending_relations(expr_repo)

        # --- Step 1: sample sweep_chunk expressions ----------------------
        # CRITICAL: filter to graph_role='sweep_chunk' ONLY.
        # sweep_document and sweep_passage have 0 LanceDB vectors — sampling
        # them causes silently dropped pairs and wastes 30% of the budget.
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                    SELECT e.id, e.work_id, af.id AS audio_file_id, af.file_name, e.content
                    FROM expressions e
                    JOIN transcriptions tx ON e.source_id = tx.id AND e.source_type = 'audio_transcript'
                    JOIN audio_files af ON tx.audio_file_id = af.id
                    WHERE e.graph_role = 'sweep_chunk'
                      AND LENGTH(TRIM(e.content)) >= :min_len
                    ORDER BY (
                        CASE WHEN EXISTS (
                            SELECT 1 FROM expression_relations er 
                            WHERE er.from_expression_id = e.id AND er.relation_type = 'semantic'
                        ) THEN 1 ELSE 0 END
                    ) ASC, e.id DESC
                    LIMIT :limit
                """),
                {
                    "limit": settings.surfer_sample_size,
                    "min_len": settings.surfer_min_content_length,
                },
            ).mappings().all()

        if len(rows) < 2:
            logger.info("ConnectionSurfer: fewer than 2 sweep_chunk expressions — skipping")
            return

        logger.info("ConnectionSurfer: sampled %d sweep_chunks", len(rows))

        # --- Step 2: fetch vectors from LanceDB --------------------------
        expr_ids = [r["id"] for r in rows]
        vectors = store.get_vectors(expr_ids)

        valid_rows = [r for r in rows if r["id"] in vectors]
        if len(valid_rows) < 2:
            logger.warning(
                "ConnectionSurfer: only %d of %d sampled rows have LanceDB vectors",
                len(valid_rows), len(rows),
            )
            return

        logger.info("ConnectionSurfer: %d rows have vectors, computing pairwise similarities", len(valid_rows))

        vec_list = [vectors[r["id"]] for r in valid_rows]
        id_list = [r["id"] for r in valid_rows]
        row_map = {r["id"]: r for r in valid_rows}

        # --- Step 3: compute similarities using InCoreHNSWIndex if ready ---
        from audiobench.daemon.index import get_shared_hnsw_index
        shared_index = get_shared_hnsw_index()

        pairs: list[tuple[int, int, float]] = []
        if shared_index.is_ready():
            k_query = min(11, shared_index.item_count())
            max_dist = 1.0 - settings.surfer_sim_threshold
            labels, dists = shared_index.query_batch(vec_list, k=k_query, chunk_size=25)

            # Look up any neighbor IDs that aren't already in row_map
            raw_neighbor_ids = set()
            for row_idx in range(labels.shape[0]):
                for col_idx in range(labels.shape[1]):
                    nb_id = int(labels[row_idx, col_idx])
                    nb_dist = float(dists[row_idx, col_idx])
                    query_id = id_list[row_idx]
                    if nb_id != query_id and nb_dist <= max_dist:
                        raw_neighbor_ids.add(nb_id)

            missing_neighbor_ids = [int(nid) for nid in raw_neighbor_ids if nid not in row_map]
            if missing_neighbor_ids:
                with get_session() as session:
                    for chunk_start in range(0, len(missing_neighbor_ids), 500):
                        chunk_ids = missing_neighbor_ids[chunk_start : chunk_start + 500]
                        ids_clause = ", ".join(str(i) for i in chunk_ids)
                        extra_rows = session.execute(
                            sql_text(f"""
                                SELECT e.id, e.work_id, af.id AS audio_file_id, af.file_name, e.content
                                FROM expressions e
                                JOIN transcriptions tx ON e.source_id = tx.id AND e.source_type = 'audio_transcript'
                                JOIN audio_files af ON tx.audio_file_id = af.id
                                WHERE e.id IN ({ids_clause})
                                  AND LENGTH(TRIM(e.content)) >= :min_len
                            """),
                            {"min_len": settings.surfer_min_content_length},
                        ).mappings().all()
                        for r in extra_rows:
                            row_map[r["id"]] = r

            seen_pairs: set[tuple[int, int]] = set()
            for row_idx, id_a in enumerate(id_list):
                for col_idx in range(labels.shape[1]):
                    id_b = int(labels[row_idx, col_idx])
                    dist = float(dists[row_idx, col_idx])
                    if id_b == id_a or dist > max_dist:
                        continue
                    if id_b not in row_map:
                        continue
                    sim = 1.0 - dist
                    pair_key = (min(id_a, id_b), max(id_a, id_b))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pairs.append((pair_key[0], pair_key[1], sim))
        else:
            logger.info("ConnectionSurfer: shared HNSW index not ready — using pairwise cosine")
            pairs = pairwise_cosine_above_threshold(
                vec_list, id_list, settings.surfer_sim_threshold
            )

        logger.info(
            "ConnectionSurfer: %d pairs above threshold %.2f",
            len(pairs), settings.surfer_sim_threshold,
        )

        # --- Step 4: write edges -----------------------------------------
        # Per-node outbound edge count for this cycle (in-memory tracking to
        # avoid one DB query per pair; refreshed from DB at start of cycle).
        outbound_this_cycle: dict[int, int] = {}

        edges_written = 0
        pairs_skipped_same_file = 0
        pairs_skipped_cap = 0
        pairs_skipped_boilerplate = 0
        pairs_skipped_short = 0
        newly_linked_expr_ids: set[int] = set()

        from audiobench.memory.boilerplate import is_pure_boilerplate

        min_content_len = settings.surfer_min_content_length

        for id_a, id_b, sim in sorted(pairs, key=lambda t: t[2], reverse=True):
            row_a = row_map[id_a]
            row_b = row_map[id_b]

            # Guard 0: defense-in-depth length floor (prevents short filler/ticks from forming edges)
            c_a = (row_a.get("content") or "").strip()
            c_b = (row_b.get("content") or "").strip()
            if len(c_a) < min_content_len or len(c_b) < min_content_len:
                pairs_skipped_short += 1
                continue

            # Guard 1: must be genuinely different works (or different files if work_id is None)
            wid_a = row_a.get("work_id")
            wid_b = row_b.get("work_id")
            if wid_a is not None and wid_b is not None:
                if wid_a == wid_b:
                    pairs_skipped_same_file += 1
                    continue
            elif row_a["audio_file_id"] == row_b["audio_file_id"]:
                pairs_skipped_same_file += 1
                continue

            # Guard 2: boilerplate nodes must not form edges
            # (SQL migration 039 should have removed them; this is a safety net)
            if is_pure_boilerplate(row_a["content"]) or is_pure_boilerplate(row_b["content"]):
                pairs_skipped_boilerplate += 1
                continue

            # Guard 3: per-node edge cap
            cap = settings.surfer_max_edges_per_node
            a_out = outbound_this_cycle.get(id_a)
            b_out = outbound_this_cycle.get(id_b)
            if a_out is None:
                a_out = self._outbound_semantic_count(id_a)
                outbound_this_cycle[id_a] = a_out
            if b_out is None:
                b_out = self._outbound_semantic_count(id_b)
                outbound_this_cycle[id_b] = b_out

            if a_out >= cap or b_out >= cap:
                pairs_skipped_cap += 1
                continue

            # Write permanent SEMANTIC edge (idempotent)
            try:
                expr_repo.link(
                    from_id=id_a,
                    to_id=id_b,
                    relation_type=RelationType.SEMANTIC.value,
                    weight=round(float(sim), 4),
                    created_by="connection_surfer",
                )
                outbound_this_cycle[id_a] = a_out + 1
                outbound_this_cycle[id_b] = b_out + 1
                newly_linked_expr_ids.add(id_a)
                newly_linked_expr_ids.add(id_b)
                edges_written += 1
            except ValueError as e:
                logger.debug("ConnectionSurfer: self-link or invalid pair %d/%d: %s", id_a, id_b, e)
                continue
            except Exception as e:
                logger.warning("ConnectionSurfer: failed to link %d -> %d: %s", id_a, id_b, e)
                continue

        # --- Step 4b: assign incremental community membership -------------
        if newly_linked_expr_ids:
            try:
                import sqlite3
                from audiobench.community import assign_incremental

                db_url = settings.database_url
                if db_url.startswith("sqlite:///"):
                    db_p = db_url.replace("sqlite:///", "")
                    with sqlite3.connect(db_p) as c_conn:
                        assigned = assign_incremental(c_conn, list(newly_linked_expr_ids))
                        logger.info(
                            "ConnectionSurfer: assigned %d chunks to communities out of %d newly linked",
                            assigned,
                            len(newly_linked_expr_ids),
                        )
            except Exception as c_err:
                logger.warning("ConnectionSurfer: community assignment failed: %s", c_err)

        # --- Step 5: log calibration stats --------------------------------
        logger.info(
            "ConnectionSurfer: cycle complete — edges_written=%d, "
            "skipped_same_file=%d, skipped_cap=%d, skipped_boilerplate=%d, skipped_short=%d",
            edges_written,
            pairs_skipped_same_file,
            pairs_skipped_cap,
            pairs_skipped_boilerplate,
            pairs_skipped_short,
        )

        with get_session() as session:
            total_semantic = session.execute(
                sql_text(
                    "SELECT COUNT(*) FROM expression_relations WHERE relation_type = :rel"
                ),
                {"rel": RelationType.SEMANTIC.value},
            ).scalar() or 0

        calib = {
            "surfer_cycle_edges_written": edges_written,
            "surfer_total_semantic_edges": total_semantic,
            "surfer_threshold": settings.surfer_sim_threshold,
            "surfer_sample_size": settings.surfer_sample_size,
        }
        try:
            store.add_expression(
                content=json.dumps(calib),
                source_type="daemon_calibration",
                calibration_key="surfer_telemetry",
            )
        except Exception as e:
            logger.debug("ConnectionSurfer: could not write calibration: %s", e)

        logger.info(
            "ConnectionSurfer: total SEMANTIC edges in DB: %d", total_semantic
        )
