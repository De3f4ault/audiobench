"""
SearchSessionSweep — intelligence task for search-world enrichment.

Runs every 300 seconds (5 minutes, same cadence as the RAG sweep) when the
system is idle (CPU < 80%, > 120 s since last client request).

Responsibilities
----------------
1. Auto-summarize: find sessions with query_count >= 5 and no summary, generate
   one via LLM, save to session_store, and push to the summary ingestion deque so
   the RAG sweep picks it up on the next tick.

2. Session stitching: detect pairs of sessions whose synthesis-expression centroids
   are highly similar (cosine > 0.82) and emit RESUMES relations between their
   SEARCH_SESSION_SUMMARY expressions, encoding multi-session research continuity.

3. Knowledge gap detection: compare the centroid of recent SEARCH_QUERY expression
   embeddings (what you are asking about) against the corpus centroid of recent
   AUDIO_TRANSCRIPT expression embeddings (what your library covers).  If the
   distance exceeds DRIFT_THRESHOLD, emit a daemon_proposal suggesting the user
   ingest more content on the under-covered topic cluster.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.scheduler import IntelligenceTask
from audiobench.memory.enums import RelationType, SourceType

logger = logging.getLogger("audiobench.daemon.intelligence.search_session_sweep")


class SearchSessionSweep(IntelligenceTask):
    """Idle-time enrichment task for the search knowledge layer."""

    INTERVAL_SECONDS: int = 300  # same cadence as the RAG sweep
    IS_INFERENCE_TASK: bool = True

    # ── Auto-summarize ─────────────────────────────────────────────────
    MIN_QUERIES_FOR_AUTO_SUMMARY: int = 5

    # ── Session stitching ──────────────────────────────────────────────
    STITCH_SIM_THRESHOLD: float = 0.82   # cosine similarity for RESUMES relation
    STITCH_MAX_PAIRS_PER_RUN: int = 10   # guard against runaway on large corpora

    # ── Knowledge gap ──────────────────────────────────────────────────
    GAP_DRIFT_THRESHOLD: float = 0.20    # centroid distance to trigger proposal
    GAP_MIN_SEARCH_QUERIES: int = 5      # need at least this many search queries
    GAP_WINDOW_DAYS: int = 14            # look at last 14 days of activity

    # ── Embedding dim ──────────────────────────────────────────────────
    EMBED_DIM: int = 768

    async def run(self) -> int:
        logger.info("SearchSessionSweep: starting run")
        await self._repair_missing_thematic_links()
        summaries = await self._auto_summarize_sessions()
        await self._detect_session_stitching()
        await self._detect_knowledge_gap()
        logger.info("SearchSessionSweep: run complete")
        return summaries

    # ------------------------------------------------------------------
    # 0. Self-healing: repair THEMATIC links missed by prior bug
    # ------------------------------------------------------------------

    async def _repair_missing_thematic_links(self) -> None:
        """
        Find SEARCH_SYNTHESIS expressions that have an ELABORATES link (so they
        were ingested) but zero THEMATIC links (meaning the resolver failed).
        Re-resolve using the corrected two-step lookup and emit the missing links.

        This becomes a permanent no-op once all THEMATIC links are established.
        The query is O(expressions) so it is cheap after the first run.
        """
        import asyncio

        await asyncio.get_event_loop().run_in_executor(
            None, self._repair_thematic_sync
        )

    def _repair_thematic_sync(self) -> None:
        from audiobench.memory.search_ingester import SearchIngester
        from audiobench.storage.expression_repository import ExpressionRepository

        with get_session() as session:
            rows = session.execute(
                sql_text(
                    "SELECT e.id as synth_id, e.source_id as query_id "
                    "FROM expressions e "
                    "WHERE e.source_type = 'search_synthesis' "
                    "  AND NOT EXISTS ( "
                    "      SELECT 1 FROM expression_relations er "
                    "      WHERE er.from_expression_id = e.id "
                    "        AND er.relation_type = 'thematic' "
                    "  ) "
                    "ORDER BY e.id"
                )
            ).mappings().all()

        if not rows:
            logger.debug("SearchSessionSweep: no THEMATIC links to repair")
            return

        logger.info(
            "SearchSessionSweep: repairing THEMATIC links for %d synthesis expressions",
            len(rows),
        )

        ingester = SearchIngester()
        expr_repo = ExpressionRepository()
        repaired = 0
        deferred = 0
        deferred_tx_ids: set[int] = set()
        tx_cache: dict[str, int | None] = {}

        for row in rows:
            synth_id = int(row["synth_id"])
            query_id = int(row["query_id"])

            # Load the full fragment context for resilient healing
            with get_session() as session:
                frag_rows = session.execute(
                    sql_text(
                        "SELECT segment_id, source_file, start_time, fragment_text "
                        "FROM search_query_fragments "
                        "WHERE query_id = :qid ORDER BY rank"
                    ),
                    {"qid": query_id},
                ).mappings().all()

            linked_for_synth = 0

            for frag in frag_rows:
                old_seg_id = int(frag["segment_id"])
                frag_expr_id, healed_seg_id = ingester.resolve_fragment_expression(
                    segment_id=old_seg_id,
                    source_file=frag.get("source_file"),
                    start_time=frag.get("start_time"),
                    fragment_text=frag.get("fragment_text"),
                    deferred_tx_ids=deferred_tx_ids,
                    tx_cache=tx_cache,
                )
                if frag_expr_id is not None:
                    try:
                        expr_repo.link(
                            from_id=synth_id,
                            to_id=frag_expr_id,
                            relation_type=RelationType.THEMATIC.value,
                        )
                        repaired += 1
                        linked_for_synth += 1

                        # Self-heal search_query_fragments with the new segment_id
                        if healed_seg_id and healed_seg_id != old_seg_id:
                            try:
                                with get_session() as s:
                                    s.execute(
                                        sql_text(
                                            "UPDATE OR IGNORE search_query_fragments "
                                            "SET segment_id = :nid "
                                            "WHERE query_id = :qid AND segment_id = :oid"
                                        ),
                                        {"nid": healed_seg_id, "qid": query_id, "oid": old_seg_id},
                                    )
                                    s.commit()
                            except Exception as update_err:
                                logger.debug("Failed to update healed fragment: %s", update_err)
                    except Exception as exc:
                        logger.debug(
                            "SearchSessionSweep: THEMATIC link %d→%d failed: %s",
                            synth_id, frag_expr_id, exc,
                        )
                else:
                    deferred += 1

            # If all fragments could not be linked and none are deferred waiting for RAG sweep:
            # Anchor synthesis to its parent query expression so it does not loop infinitely.
            query_deferred = False
            for frag in frag_rows:
                path = frag.get("source_file")
                if path and tx_cache.get(path) in deferred_tx_ids:
                    query_deferred = True
                    break

            if linked_for_synth == 0 and not query_deferred:
                with get_session() as session:
                    query_expr = session.execute(
                        sql_text(
                            "SELECT id FROM expressions "
                            "WHERE source_type = 'search_query' AND source_id = :qid "
                            "LIMIT 1"
                        ),
                        {"qid": query_id},
                    ).fetchone()
                if query_expr is not None:
                    try:
                        expr_repo.link(
                            from_id=synth_id,
                            to_id=int(query_expr[0]),
                            relation_type=RelationType.THEMATIC.value,
                        )
                        logger.info(
                            "SearchSessionSweep: synthesis #%d orphaned (all sources extinct); "
                            "anchored THEMATIC link to query expr #%d",
                            synth_id, query_expr[0],
                        )
                    except Exception as exc:
                        logger.debug(
                            "SearchSessionSweep: fallback anchor link %d→%d failed: %s",
                            synth_id, query_expr[0], exc,
                        )

        if deferred_tx_ids:
            logger.info(
                "SearchSessionSweep: THEMATIC repair complete — %d links created, "
                "%d still deferred across %d unswept transcriptions",
                repaired, deferred, len(deferred_tx_ids),
            )
        else:
            logger.info(
                "SearchSessionSweep: THEMATIC repair complete — %d links created, "
                "%d still deferred",
                repaired, deferred,
            )



    async def _auto_summarize_sessions(self) -> None:
        """
        Find sessions with query_count >= MIN_QUERIES_FOR_AUTO_SUMMARY and
        no existing session_summary.  Generate a summary via LLM, save it,
        and push the session ID to the sweep deque for expression ingestion.
        """
        import asyncio

        with get_session() as session:
            rows = session.execute(
                sql_text(
                    "SELECT id FROM search_sessions "
                    "WHERE query_count >= :min_q "
                    "  AND (session_summary IS NULL OR session_summary = '') "
                    "ORDER BY updated_at DESC "
                    "LIMIT 5"  # cap per run — LLM calls are expensive
                ),
                {"min_q": self.MIN_QUERIES_FOR_AUTO_SUMMARY},
            ).scalars().all()

        if not rows:
            logger.debug("SearchSessionSweep: no sessions need auto-summary")
            return 0

        logger.info(
            "SearchSessionSweep: auto-summarizing %d sessions", len(rows)
        )
        summarized_count = 0

        for session_id in rows:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._generate_and_save_summary, session_id
                )
                summarized_count += 1
            except Exception as exc:
                logger.error(
                    "SearchSessionSweep: auto-summary failed for session %d: %s",
                    session_id, exc,
                )

        return summarized_count

    def _generate_and_save_summary(self, session_id: int) -> None:
        """Synchronous helper: generate summary and save to session_store."""
        try:
            # session_store is a module of free functions — no SessionStore class
            from audiobench.memory import session_store as ss_mod

            detail = ss_mod.get_session(session_id)
            if not detail or not detail.queries:
                return

            # Build prompt the same way the /summary REPL command does
            lines: list[str] = []
            for q in detail.queries:
                lines.append(f"Q: {q.query_text}")
                if q.synthesis_text:
                    lines.append(f"A: {q.synthesis_text}")
                lines.append("")
            transcript = "\n".join(lines).strip()

            if not transcript:
                return

            from audiobench.chat.providers.ollama_provider import OllamaClient

            from audiobench.core.prompts import SESSION_SUMMARY_PROMPT
            from audiobench.core.settings import get_settings
            from audiobench.memory.query_engine import Ok, _call_llm

            prompt = SESSION_SUMMARY_PROMPT.format(transcript=transcript)
            settings = get_settings()
            llm = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )

            match _call_llm(prompt, 0.3, llm, settings.gemini_api_key):
                case Ok(value=summary_text):
                    summary_text = summary_text.strip()
                    if summary_text:
                        ss_mod.save_session_summary(session_id, summary_text)
                        logger.info(
                            "SearchSessionSweep: auto-summary saved for session %d", session_id
                        )

                        # Push to sweep deque so next RAG sweep ingests it as an expression
                        from audiobench.daemon.sweep_state import get_sweep_state
                        get_sweep_state().push_summary_session(session_id)

        except Exception as exc:
            logger.error(
                "SearchSessionSweep._generate_and_save_summary(%d): %s",
                session_id, exc,
            )
            raise

    # ------------------------------------------------------------------
    # 2. Session stitching (RESUMES relations)
    # ------------------------------------------------------------------

    async def _detect_session_stitching(self) -> None:
        """
        Find pairs of SEARCH_SESSION_SUMMARY expressions with high cosine
        similarity (> STITCH_SIM_THRESHOLD) that do not yet have a RESUMES
        relation.  Emit RESUMES: later_summary_expr → earlier_summary_expr.
        """
        import asyncio

        await asyncio.get_event_loop().run_in_executor(
            None, self._stitch_sessions_sync
        )

    def _stitch_sessions_sync(self) -> None:
        from audiobench.storage.expression_repository import ExpressionRepository

        expr_repo = ExpressionRepository()

        # Load all SEARCH_SESSION_SUMMARY expressions with their embeddings
        with get_session() as session:
            rows = session.execute(
                sql_text(
                    "SELECT e.id, e.session_id, e.created_at "
                    "FROM expressions e "
                    "WHERE e.source_type = :st "
                    "ORDER BY e.created_at ASC",
                ),
                {"st": SourceType.SEARCH_SESSION_SUMMARY.value},
            ).mappings().all()

        if len(rows) < 2:
            logger.debug("SearchSessionSweep: < 2 session summaries, skipping stitching")
            return

        expr_ids = [int(r["id"]) for r in rows]

        # Fetch embeddings from LanceDB
        try:
            from audiobench.memory.memory_store import MemoryStore
            store = MemoryStore()
            vectors = store.get_vectors(expr_ids)  # dict[int, list[float]]
        except Exception as exc:
            logger.warning("SearchSessionSweep: could not fetch summary vectors: %s", exc)
            return

        # Load existing RESUMES pairs to avoid duplicating
        with get_session() as session:
            existing = session.execute(
                sql_text(
                    "SELECT from_expression_id, to_expression_id "
                    "FROM expression_relations "
                    "WHERE relation_type = :rt",
                ),
                {"rt": RelationType.RESUMES.value},
            ).fetchall()
        existing_pairs: set[tuple[int, int]] = {(r[0], r[1]) for r in existing}

        pairs_created = 0
        for i, ei in enumerate(expr_ids):
            if ei not in vectors:
                continue
            vi = np.array(vectors[ei], dtype=np.float32)
            vi /= np.linalg.norm(vi) + 1e-8

            for ej in expr_ids[i + 1:]:
                if ej not in vectors:
                    continue
                vj = np.array(vectors[ej], dtype=np.float32)
                vj /= np.linalg.norm(vj) + 1e-8

                sim = float(np.dot(vi, vj))
                if sim >= self.STITCH_SIM_THRESHOLD:
                    # ej was created after ei (rows sorted ASC) → ej RESUMES ei
                    pair = (ej, ei)
                    if pair not in existing_pairs:
                        try:
                            expr_repo.link(
                                from_id=ej,
                                to_id=ei,
                                relation_type=RelationType.RESUMES.value,
                            )
                            existing_pairs.add(pair)
                            pairs_created += 1
                            logger.info(
                                "SearchSessionSweep: RESUMES %d → %d (sim=%.3f)",
                                ej, ei, sim,
                            )
                        except Exception as exc:
                            logger.warning(
                                "SearchSessionSweep: could not link RESUMES %d→%d: %s",
                                ej, ei, exc,
                            )

                if pairs_created >= self.STITCH_MAX_PAIRS_PER_RUN:
                    logger.info(
                        "SearchSessionSweep: RESUMES cap reached (%d pairs this run)",
                        pairs_created,
                    )
                    return

        logger.info(
            "SearchSessionSweep: session stitching complete (%d new RESUMES relations)",
            pairs_created,
        )

    # ------------------------------------------------------------------
    # 3. Knowledge gap detection
    # ------------------------------------------------------------------

    async def _detect_knowledge_gap(self) -> None:
        """
        Compare centroid of recent SEARCH_QUERY expression embeddings (what you
        are asking about) vs centroid of recent AUDIO_TRANSCRIPT expression
        embeddings (what the corpus covers).

        If centroid distance > GAP_DRIFT_THRESHOLD, emit a daemon_proposal
        expression noting the under-covered topic cluster.
        """
        import asyncio

        await asyncio.get_event_loop().run_in_executor(
            None, self._knowledge_gap_sync
        )

    def _knowledge_gap_sync(self) -> None:
        cutoff = time.time() - self.GAP_WINDOW_DAYS * 86400

        with get_session() as session:
            search_ids = session.execute(
                sql_text(
                    "SELECT id FROM expressions "
                    "WHERE source_type = :st "
                    "  AND created_at >= datetime(:cutoff, 'unixepoch') "
                    "ORDER BY created_at DESC LIMIT 200"
                ),
                {"st": SourceType.SEARCH_QUERY.value, "cutoff": cutoff},
            ).scalars().all()

            corpus_ids = session.execute(
                sql_text(
                    "SELECT id FROM expressions "
                    "WHERE source_type = :st "
                    "  AND graph_role = 'sweep_chunk' "
                    "  AND created_at >= datetime(:cutoff, 'unixepoch') "
                    "ORDER BY created_at DESC LIMIT 200"
                ),
                {"st": SourceType.AUDIO_TRANSCRIPT.value, "cutoff": cutoff},
            ).scalars().all()

        if len(search_ids) < self.GAP_MIN_SEARCH_QUERIES:
            logger.debug(
                "SearchSessionSweep: only %d search queries in window — skipping gap detection",
                len(search_ids),
            )
            return

        if not corpus_ids:
            logger.debug("SearchSessionSweep: no corpus expressions in window")
            return

        try:
            from audiobench.memory.memory_store import MemoryStore
            store = MemoryStore()
            search_vecs = store.get_vectors(list(search_ids))
            corpus_vecs = store.get_vectors(list(corpus_ids))
        except Exception as exc:
            logger.warning(
                "SearchSessionSweep: could not fetch vectors for gap detection: %s", exc
            )
            return

        def centroid(vecs: dict[int, list[float]]) -> np.ndarray:
            arr = np.array(list(vecs.values()), dtype=np.float32)
            c = arr.mean(axis=0)
            n = np.linalg.norm(c)
            return c / n if n > 1e-8 else c

        if not search_vecs or not corpus_vecs:
            return

        search_centroid = centroid(search_vecs)
        corpus_centroid = centroid(corpus_vecs)
        distance = float(1.0 - np.dot(search_centroid, corpus_centroid))

        logger.info(
            "SearchSessionSweep: knowledge gap distance=%.4f (threshold=%.2f)",
            distance, self.GAP_DRIFT_THRESHOLD,
        )

        if distance <= self.GAP_DRIFT_THRESHOLD:
            return

        # Deduplication check: do not emit duplicate if a knowledge_gap proposal is already pending or recently confirmed
        cooldown_cutoff = time.time() - 7 * 86400  # 7-day cooldown for confirmed proposals
        with get_session() as session:
            existing = session.execute(
                sql_text("""
                SELECT id FROM expressions
                WHERE source_type = 'daemon_proposal'
                  AND created_at >= datetime(:cutoff, 'unixepoch')
                  AND (
                      inference_status IN ('proposed', 'deferred')
                      OR (inference_status = 'confirmed' AND created_at >= datetime(:cooldown, 'unixepoch'))
                  )
                  AND (
                      content LIKE '%"proposal_type": "knowledge_gap"%'
                      OR content LIKE 'Knowledge gap detected%'
                  )
                LIMIT 1
                """),
                {"cutoff": cutoff, "cooldown": cooldown_cutoff}
            ).fetchone()
            if existing:
                logger.debug(
                    "SearchSessionSweep: recent gap proposal already pending or confirmed (#%d) — skipping emission",
                    existing[0]
                )
                return

            # Fetch query texts for top search queries
            from sqlalchemy import bindparam
            stmt = sql_text("""
                SELECT id, content FROM expressions
                WHERE id IN :ids
                ORDER BY created_at DESC
            """).bindparams(bindparam("ids", expanding=True))
            query_rows = session.execute(stmt, {"ids": list(search_ids[:30])}).fetchall()

        id_to_content = {r[0]: (r[1] or "").strip() for r in query_rows if r[1]}
        # Find medoid query: the search query whose embedding is closest to search_centroid
        medoid_id = min(
            search_vecs.keys(),
            key=lambda eid: 1.0 - float(np.dot(search_vecs[eid], search_centroid))
        )
        medoid_text = id_to_content.get(medoid_id, "")
        query_texts = [id_to_content[eid] for eid in search_ids if eid in id_to_content and id_to_content[eid]]

        from audiobench.daemon.intelligence.gap_naming import name_knowledge_gap
        title, diagnostic, rec, name_source = name_knowledge_gap(query_texts, medoid_text)

        # Emit DaemonProposalEnvelope
        envelope = {
            "proposal_type": "knowledge_gap",
            "schema_version": 2,
            "title": title,
            "summary": f"Query cluster diverges from audio corpus (distance {distance:.3f} > {self.GAP_DRIFT_THRESHOLD:.2f})",
            "evidence": f"Centroid distance {distance:.3f} vs threshold {self.GAP_DRIFT_THRESHOLD:.2f} over {self.GAP_WINDOW_DAYS} days",
            "payload": {
                "distance": distance,
                "threshold": self.GAP_DRIFT_THRESHOLD,
                "window_days": self.GAP_WINDOW_DAYS,
                "diagnostic": diagnostic,
                "recommended_ingestion": rec,
                "stranded_queries": query_texts,
                "name_source": name_source,
                "medoid_query": medoid_text,
            },
        }
        content = json.dumps(envelope)

        try:
            from audiobench.daemon.server import _get_store
            store_obj = _get_store()
            if store_obj:
                store_obj.add_expression(
                    source_type="daemon_proposal",
                    content=content,
                    inference_status="proposed",
                )
                logger.info(
                    "SearchSessionSweep: knowledge gap proposal emitted (distance=%.4f, title='%s', source=%s)",
                    distance, title, name_source
                )
        except Exception as exc:
            logger.warning(
                "SearchSessionSweep: failed to emit gap proposal: %s", exc
            )
