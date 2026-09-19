"""
SearchIngester — bridges search session data into the expression graph and LanceDB.

This module is the ONLY point where search session data crosses into the expression
system.  It is called exclusively from the daemon sweep (never from the REPL directly),
ensuring:

  - Non-blocking UI: the REPL writes to session_store and returns immediately.
  - Idempotent:      content_hash deduplication in ExpressionRepository.register()
                     prevents duplicate expressions on repeated sweep runs.
  - Fault-tolerant:  failures are logged per-query; the sweep requeues failed IDs.
  - Retroactive:     the first sweep pass detects ALL existing uningested queries.

Signal taxonomy produced per search query
-----------------------------------------
1. SEARCH_QUERY expression          — the user's research question (intent signal)
2. SEARCH_SYNTHESIS expression      — the AI synthesis answer  (distilled knowledge)
3. ELABORATES relation              — synthesis_expr → query_expr
4. THEMATIC relations               — synthesis_expr → each retrieved fragment's expr
5. TEMPORAL relation                — prior_query_expr → this_query_expr  (thought arc)
6. INSPIRED_BY relations (summary)  — summary_expr → each query_expr in session

All expressions carry:
  session_type = 'memory_search'
  session_id   = search_sessions.id   (the search REPL session, not a chat session)
"""

from __future__ import annotations

import logging
from typing import Any

from audiobench.core.db_session import get_session
from audiobench.memory.enums import RelationType, SessionType, SourceType
from audiobench.storage.expression_repository import ExpressionRepository

logger = logging.getLogger("audiobench.memory.search_ingester")


class SearchIngester:
    """Bridges completed search queries and session summaries into the expression graph."""

    def __init__(self, store: Any | None = None) -> None:
        self._expr_repo = ExpressionRepository()
        self._store = store

    @property
    def store(self) -> Any:
        if self._store is None:
            try:
                from audiobench.daemon.server import _get_store
                self._store = _get_store()
            except Exception:
                from audiobench.memory.memory_store import MemoryStore
                self._store = MemoryStore()
        return self._store

    # ------------------------------------------------------------------
    # Public API (called by the daemon sweep)
    # ------------------------------------------------------------------

    def ingest_query(self, query_id: int) -> tuple[int, int | None]:
        """
        Ingest one ``search_queries`` row into the expression graph.

        Reads
        -----
        - ``search_queries``       — query_text, synthesis_text, session_id, sequence_num
        - ``search_query_fragments``— segment_ids retrieved for this query
        - ``expressions``          — resolves segment → expression via source_id lookup

        Creates
        -------
        1. SEARCH_QUERY expression     (always)
        2. SEARCH_SYNTHESIS expression (only when synthesis_text is not None)
        3. ELABORATES relation         synthesis → query
        4. THEMATIC relations          synthesis → each resolved fragment expression
        5. TEMPORAL relation           prior_query_expr → this_query_expr (if prior exists)
        6. LanceDB embedding           both expressions via MemoryStore.write_node()

        Returns
        -------
        (query_expr_id, synthesis_expr_id | None)
        """
        from sqlalchemy import text as sql_text

        # ── Fetch query row ────────────────────────────────────────────
        with get_session() as session:
            row = session.execute(
                sql_text(
                    "SELECT sq.id, sq.session_id, sq.sequence_num, "
                    "       sq.query_text, sq.synthesis_text "
                    "FROM search_queries sq "
                    "WHERE sq.id = :qid"
                ),
                {"qid": query_id},
            ).mappings().fetchone()

        if row is None:
            logger.warning("SearchIngester.ingest_query: query_id=%d not found", query_id)
            return -1, None

        session_id   = row["session_id"]
        sequence_num = row["sequence_num"]
        query_text   = row["query_text"] or ""
        synthesis    = row["synthesis_text"]  # may be None

        # ── Fetch retrieved fragment segment_ids ───────────────────────
        with get_session() as session:
            frag_rows = session.execute(
                sql_text(
                    "SELECT segment_id FROM search_query_fragments "
                    "WHERE query_id = :qid ORDER BY rank"
                ),
                {"qid": query_id},
            ).scalars().all()
        segment_ids: list[int] = list(frag_rows)

        # ── 1. Create SEARCH_QUERY expression ─────────────────────────
        query_expr = self._expr_repo.register(
            content=query_text,
            source_type=SourceType.SEARCH_QUERY.value,
            source_id=query_id,          # enables dedup: source_type+source_id is unique
            session_type=SessionType.MEMORY_SEARCH.value,
            session_id=session_id,
        )
        logger.debug(
            "SearchIngester: SEARCH_QUERY expr #%d for query_id=%d session=%d seq=%d",
            query_expr.id, query_id, session_id, sequence_num,
        )

        # ── 2. Create SEARCH_SYNTHESIS expression ──────────────────────
        synthesis_expr_id: int | None = None
        if synthesis and synthesis.strip():
            synth_expr = self._expr_repo.register(
                content=synthesis,
                source_type=SourceType.SEARCH_SYNTHESIS.value,
                source_id=query_id,      # same source_id; source_type differentiates
                session_type=SessionType.MEMORY_SEARCH.value,
                session_id=session_id,
            )
            synthesis_expr_id = synth_expr.id
            logger.debug(
                "SearchIngester: SEARCH_SYNTHESIS expr #%d for query_id=%d",
                synth_expr.id, query_id,
            )

            # ── 3. ELABORATES: synthesis → query ───────────────────────
            self._expr_repo.link(
                from_id=synth_expr.id,
                to_id=query_expr.id,
                relation_type=RelationType.ELABORATES.value,
            )

            # ── 4. THEMATIC: synthesis → retrieved fragment expressions ─
            for seg_id in segment_ids:
                frag_expr_id = self._resolve_fragment_expression_id(seg_id)
                if frag_expr_id is not None:
                    self._expr_repo.link(
                        from_id=synth_expr.id,
                        to_id=frag_expr_id,
                        relation_type=RelationType.THEMATIC.value,
                    )

        # ── 5. TEMPORAL: prior query → this query (thought arc) ────────
        prior_expr_id = self._resolve_prior_query_expression(
            session_id=session_id,
            sequence_num=sequence_num,
        )
        if prior_expr_id is not None:
            self._expr_repo.link(
                from_id=prior_expr_id,
                to_id=query_expr.id,
                relation_type=RelationType.TEMPORAL.value,
            )

        # ── 6. Embed both expressions into LanceDB ─────────────────────
        self._embed_expression(query_expr.id, query_text, SourceType.SEARCH_QUERY.value)
        if synthesis_expr_id is not None and synthesis:
            self._embed_expression(synthesis_expr_id, synthesis, SourceType.SEARCH_SYNTHESIS.value)

        return query_expr.id, synthesis_expr_id

    def ingest_session_summary(self, session_id: int) -> int:
        """
        Ingest a session's AI-generated summary into the expression graph.

        Reads
        -----
        - ``search_sessions``  — session_summary text
        - ``expressions``      — finds all SEARCH_QUERY exprs for this session

        Creates
        -------
        1. SEARCH_SESSION_SUMMARY expression
        2. INSPIRED_BY relations: summary_expr → each SEARCH_QUERY expr in session
        3. LanceDB embedding via MemoryStore.write_node()

        Returns
        -------
        summary_expr_id
        """
        from sqlalchemy import text as sql_text

        # ── Fetch summary text ─────────────────────────────────────────
        with get_session() as session:
            row = session.execute(
                sql_text(
                    "SELECT session_summary FROM search_sessions WHERE id = :sid"
                ),
                {"sid": session_id},
            ).fetchone()

        if row is None or not row[0]:
            logger.debug(
                "SearchIngester.ingest_session_summary: session_id=%d has no summary",
                session_id,
            )
            return -1

        summary_text: str = row[0]

        # ── Create SEARCH_SESSION_SUMMARY expression ───────────────────
        summary_expr = self._expr_repo.register(
            content=summary_text,
            source_type=SourceType.SEARCH_SESSION_SUMMARY.value,
            source_id=session_id,
            session_type=SessionType.MEMORY_SEARCH.value,
            session_id=session_id,
        )
        logger.debug(
            "SearchIngester: SEARCH_SESSION_SUMMARY expr #%d for session_id=%d",
            summary_expr.id, session_id,
        )

        # ── INSPIRED_BY: summary → all query exprs in session ──────────
        with get_session() as session:
            query_expr_ids = session.execute(
                sql_text(
                    "SELECT id FROM expressions "
                    "WHERE source_type = :st "
                    "  AND session_type = :stype "
                    "  AND session_id = :sid "
                    "ORDER BY id"
                ),
                {
                    "st":    SourceType.SEARCH_QUERY.value,
                    "stype": SessionType.MEMORY_SEARCH.value,
                    "sid":   session_id,
                },
            ).scalars().all()

        for qeid in query_expr_ids:
            self._expr_repo.link(
                from_id=summary_expr.id,
                to_id=qeid,
                relation_type=RelationType.INSPIRED_BY.value,
            )

        # ── Embed into LanceDB ─────────────────────────────────────────
        self._embed_expression(
            summary_expr.id, summary_text, SourceType.SEARCH_SESSION_SUMMARY.value
        )

        return summary_expr.id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_fragment_expression_id(
        self,
        segment_id: int,
        deferred_tx_ids: set[int] | None = None,
    ) -> int | None:
        """
        Resolve a ``segment_id`` from ``search_query_fragments`` to the
        corresponding expression in the expression graph.

        Why this is non-trivial
        -----------------------
        The RAG sweep creates ``sweep_chunk`` expressions with:
          - source_type = 'audio_transcript'
          - source_id   = transcription.id      ← NOT segment.id
          - graph_role  = 'sweep_chunk'

        There is no column joining segment_id to any expression.  Instead we
        use content proximity:

          Step 1: segment_id → (transcription_id, text) from the segments table.
          Step 2: Find the sweep_chunk expression where
                  source_id = transcription_id  AND
                  content LIKE '%{first_40_chars_of_segment_text}%'

        The 40-character prefix is enough to be distinctive while short enough
        to survive minor cleaning differences between WhisperX output and the
        content-aware chunker's cleaned text.

        Falls back to the sweep_document (tier-1 full-transcript) expression
        if no chunk matches — a less precise but always-valid anchor.

        Returns None only if the transcription has never been swept at all.
        """
        from sqlalchemy import text as sql_text

        # ── Step 1: resolve segment → transcription_id + raw text ─────
        with get_session() as session:
            seg_row = session.execute(
                sql_text(
                    "SELECT transcription_id, text "
                    "FROM segments WHERE id = :sid"
                ),
                {"sid": segment_id},
            ).fetchone()

        if seg_row is None:
            logger.debug(
                "SearchIngester: segment_id=%d not found in segments table",
                segment_id,
            )
            return None

        transcription_id: int = seg_row[0]
        if deferred_tx_ids is not None and transcription_id in deferred_tx_ids:
            # Short-circuit: already known to be unswept in this cycle
            return None

        seg_text: str = (seg_row[1] or "").strip()

        # ── Step 2a: match sweep_chunk by content substring ────────────
        # Use first 40 chars of the segment's raw WhisperX text as a LIKE
        # probe against the chunker-assembled sweep_chunk content.
        if seg_text:
            probe = seg_text[:40].replace("%", "%%").replace("_", "\\_")
            with get_session() as session:
                chunk_row = session.execute(
                    sql_text(
                        "SELECT id FROM expressions "
                        "WHERE source_id    = :tx_id "
                        "  AND graph_role   = 'sweep_chunk' "
                        "  AND content LIKE :probe ESCAPE '\\' "
                        "LIMIT 1"
                    ),
                    {"tx_id": transcription_id, "probe": f"%{probe}%"},
                ).fetchone()

            if chunk_row is not None:
                logger.debug(
                    "SearchIngester: segment_id=%d → sweep_chunk expr #%d "
                    "(via content match, tx=%d)",
                    segment_id, chunk_row[0], transcription_id,
                )
                return int(chunk_row[0])

        # ── Step 2b: fall back to sweep_document (tier-1) ─────────────
        with get_session() as session:
            doc_row = session.execute(
                sql_text(
                    "SELECT id FROM expressions "
                    "WHERE source_id  = :tx_id "
                    "  AND graph_role = 'sweep_document' "
                    "LIMIT 1"
                ),
                {"tx_id": transcription_id},
            ).fetchone()

        if doc_row is not None:
            logger.debug(
                "SearchIngester: segment_id=%d → sweep_document expr #%d "
                "(chunk match failed, tx=%d)",
                segment_id, doc_row[0], transcription_id,
            )
            return int(doc_row[0])

        # Transcription has never been swept — defer
        if deferred_tx_ids is not None:
            if transcription_id not in deferred_tx_ids:
                logger.debug(
                    "SearchIngester: segment_id=%d transcription_id=%d has no "
                    "expression yet (not swept) — THEMATIC link deferred",
                    segment_id, transcription_id,
                )
                deferred_tx_ids.add(transcription_id)
        else:
            logger.debug(
                "SearchIngester: segment_id=%d transcription_id=%d has no "
                "expression yet (not swept) — THEMATIC link deferred",
                segment_id, transcription_id,
            )
        return None

    def _resolve_prior_query_expression(
        self, session_id: int, sequence_num: int
    ) -> int | None:
        """
        Find the SEARCH_QUERY expression for sequence_num - 1 in this session.
        Returns None if this is the first query (seq=1) or prior not yet ingested.
        """
        if sequence_num <= 1:
            return None

        from sqlalchemy import text as sql_text

        with get_session() as session:
            row = session.execute(
                sql_text(
                    "SELECT e.id FROM expressions e "
                    "JOIN search_queries sq ON e.source_id = sq.id "
                    "WHERE e.source_type = :st "
                    "  AND sq.session_id = :sid "
                    "  AND sq.sequence_num = :prev_seq "
                    "LIMIT 1"
                ),
                {
                    "st":       SourceType.SEARCH_QUERY.value,
                    "sid":      session_id,
                    "prev_seq": sequence_num - 1,
                },
            ).fetchone()

        return int(row[0]) if row else None

    def _embed_expression(
        self, expression_id: int, content: str, source_type: str
    ) -> None:
        """
        Write expression to LanceDB via MemoryStore.write_node().
        Non-fatal — logs warning and continues if embedding fails.
        """
        try:
            self.store.write_node(
                expression_id=expression_id,
                content=content,
                source_type=source_type,
            )
            logger.debug(
                "SearchIngester: embedded expression #%d into LanceDB", expression_id
            )
        except Exception as exc:
            logger.warning(
                "SearchIngester: failed to embed expression #%d: %s",
                expression_id, exc,
            )
