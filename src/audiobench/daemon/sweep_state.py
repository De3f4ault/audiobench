"""Sweep state — in-memory O(1) data structures for the RAG consistency sweep.

The daemon's sweep loop performs four classes of lookup during each pass:

1. Content-hash deduplication — "have we already embedded this text?"
2. Expression-indexed check  — "is this expression_id already in LanceDB?"
3. Pending-segment queue     — "which segment IDs still need vectorization?"
4. Unindexed-transcript queue — "which transcript IDs still need chunking?"

Without this module, all four are answered with SQL queries (O(log n) with
index, plus disk I/O).  With this module, (1) and (2) are O(1) hash-set
lookups in RAM, and (3)/(4) are O(1) deque pops — no SQL round-trips at all
during a sweep tick.

Usage
-----
    from audiobench.daemon.sweep_state import init_sweep_state, get_sweep_state

    # Once at daemon startup:
    state = init_sweep_state()
    state.load_from_db()

    # During each sweep tick (O(1) checks):
    if state.has_hash(h):
        ...
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from audiobench.core.logger_factory import get_logger

logger = get_logger("daemon.sweep_state")


@dataclass
class SweepState:
    """In-memory O(1) state store for the RAG consistency sweep.

    All mutating methods are protected by an RLock so they are safe to call
    from the background sweep thread AND from asyncio executor threads that
    handle incoming socket commands concurrently.
    """

    # O(1) content-hash dedup: contains every content_hash already in SQLite
    known_hashes: set[str] = field(default_factory=set)

    # O(1) LanceDB write optimiser: contains every expression_id already in
    # the LanceDB expressions table — lets us skip speculative deletes
    indexed_expression_ids: set[int] = field(default_factory=set)

    # O(1) segment queue: segment IDs waiting for vectorization
    pending_segment_ids: deque[int] = field(default_factory=deque)

    # O(1) transcript queue: transcript IDs waiting for chunking + indexing
    unindexed_transcript_ids: deque[int] = field(default_factory=deque)

    # Circuit breaker — consecutive sweep failures before backoff kicks in.
    # Written only from the sweep thread; read by _handle_ping for CLI observability.
    # Not protected by _lock: written by one thread, read for display only.
    consecutive_sweep_failures: int = 0

    # Diagnostic counter: total segment sub-batches processed since startup.
    # Used to number each sub-batch in RSS log lines so we can track
    # cumulative memory growth across batches 1 → N in a single run.
    seg_batch_count: int = 0

    # O(1) search query queue: search_queries.id values that have not yet been
    # ingested as expressions.  Populated from DB on startup and incrementally
    # refreshed every sweep tick.  The first sweep tick processes the full
    # backlog of all existing sessions retroactively.
    uningested_query_ids: deque[int] = field(default_factory=deque)

    # Session IDs whose session_summary is not yet an expression.
    # Populated on startup and refreshed incrementally.
    uningested_summary_session_ids: deque[int] = field(default_factory=deque)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------
    # Bulk loader
    # ------------------------------------------------------------------

    def load_from_db(self) -> None:
        """Populate all structures from the current DB state.

        Call once at daemon startup, after the DB engine is ready.
        Also safe to call again to re-sync after a crash/restart.
        """
        from sqlalchemy import text as sql_text

        from audiobench.core.db_session import get_session

        with get_session() as session:
            # 1. Known content hashes
            hashes = session.execute(
                sql_text(
                    "SELECT content_hash FROM expressions "
                    "WHERE content_hash IS NOT NULL"
                )
            ).scalars().all()

            # 2. Un-vectorized segment IDs (ordered so we process oldest first)
            seg_ids = session.execute(
                sql_text(
                    "SELECT s.id FROM segments s "
                    "WHERE s.vector_indexed = 0 "
                    "ORDER BY s.id"
                )
            ).scalars().all()

            # 3. Unindexed transcript IDs
            tx_ids = session.execute(
                sql_text(
                    "SELECT id FROM transcriptions "
                    "WHERE is_indexed = 0 "
                    "ORDER BY id"
                )
            ).scalars().all()

        with self._lock:
            self.known_hashes = set(hashes)
            self.pending_segment_ids = deque(seg_ids)
            self.unindexed_transcript_ids = deque(tx_ids)

        # 4. Uningested search queries: search_queries that have no corresponding
        #    expression with source_type='search_query' AND source_id=search_queries.id
        from sqlalchemy import text as sql_text

        from audiobench.core.db_session import get_session
        with get_session() as session:
            query_ids = session.execute(
                sql_text(
                    "SELECT sq.id FROM search_queries sq "
                    "WHERE NOT EXISTS ( "
                    "    SELECT 1 FROM expressions e "
                    "    WHERE e.source_type = 'search_query' "
                    "      AND e.source_id = sq.id "
                    ") ORDER BY sq.id"
                )
            ).scalars().all()

            # 5. Sessions with summaries not yet ingested as expressions
            summary_ids = session.execute(
                sql_text(
                    "SELECT ss.id FROM search_sessions ss "
                    "WHERE ss.session_summary IS NOT NULL "
                    "  AND NOT EXISTS ( "
                    "      SELECT 1 FROM expressions e "
                    "      WHERE e.source_type = 'search_session_summary' "
                    "        AND e.source_id = ss.id "
                    "  ) ORDER BY ss.id"
                )
            ).scalars().all()

        with self._lock:
            self.uningested_query_ids = deque(query_ids)
            self.uningested_summary_session_ids = deque(summary_ids)

        logger.info(
            "SweepState loaded — %d known hashes | %d pending segments | "
            "%d unindexed transcripts | %d uningested search queries | "
            "%d uningested summaries",
            len(self.known_hashes),
            len(self.pending_segment_ids),
            len(self.unindexed_transcript_ids),
            len(self.uningested_query_ids),
            len(self.uningested_summary_session_ids),
        )

    def sync_indexed_expression_ids(self) -> None:
        """Load expression IDs already present in LanceDB.

        Requires the MemoryStore to be initialised first.
        """
        try:
            from audiobench.memory.memory_store import MemoryStore

            store = MemoryStore.__new__(MemoryStore)  # reuse existing table
            from pathlib import Path

            import lancedb

            from audiobench.core.settings import get_settings

            settings = get_settings()
            lancedb_dir = Path(settings.data_dir) / "lancedb"
            db = lancedb.connect(str(lancedb_dir))
            if "expressions" in db.table_names():
                table = db.open_table("expressions")
                # Use to_arrow() for a full, unlimited scan of the table.
                # table.search() has a default 7000-row limit that would cause
                # indexed_expression_ids to be incomplete for large tables.
                ids = [
                    int(x) for x in table.to_arrow()["expression_id"].to_pylist()
                ]
                with self._lock:
                    self.indexed_expression_ids = set(ids)
                logger.info(
                    "SweepState: %d expression IDs loaded from LanceDB",
                    len(self.indexed_expression_ids),
                )
        except Exception as exc:
            logger.warning("SweepState: could not load indexed_expression_ids: %s", exc)

    # ------------------------------------------------------------------
    # Hash set — O(1) dedup
    # ------------------------------------------------------------------

    def has_hash(self, h: str) -> bool:
        """O(1) membership check. GIL makes reads safe without lock."""
        return h in self.known_hashes

    def add_hash(self, h: str) -> None:
        """Register a newly inserted expression hash."""
        with self._lock:
            self.known_hashes.add(h)

    def add_hashes(self, hs: list[str]) -> None:
        """Register multiple hashes at once (after a batch insert)."""
        with self._lock:
            self.known_hashes.update(hs)

    # ------------------------------------------------------------------
    # Indexed expression IDs — O(1) LanceDB write optimiser
    # ------------------------------------------------------------------

    def is_expression_indexed(self, expression_id: int) -> bool:
        """O(1) check — True if expression_id is already in LanceDB."""
        return expression_id in self.indexed_expression_ids

    def mark_expression_indexed(self, expression_id: int) -> None:
        with self._lock:
            self.indexed_expression_ids.add(expression_id)

    def mark_expressions_indexed(self, ids: list[int]) -> None:
        with self._lock:
            self.indexed_expression_ids.update(ids)

    # ------------------------------------------------------------------
    # Segment deque — O(1) pop
    # ------------------------------------------------------------------

    def push_segment(self, segment_id: int) -> None:
        """Enqueue a segment ID for vectorization."""
        with self._lock:
            self.pending_segment_ids.append(segment_id)

    def push_segments(self, ids: list[int]) -> None:
        """Enqueue multiple segment IDs."""
        with self._lock:
            self.pending_segment_ids.extend(ids)

    def pop_segment_batch(self, size: int = 64) -> list[int]:
        """Pop up to *size* segment IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.pending_segment_ids))
            return [self.pending_segment_ids.popleft() for _ in range(n)]

    def pending_segment_count(self) -> int:
        return len(self.pending_segment_ids)

    # ------------------------------------------------------------------
    # Transcript deque — O(1) pop
    # ------------------------------------------------------------------

    def push_transcript(self, transcript_id: int) -> None:
        """Enqueue a transcript ID for chunking/indexing."""
        with self._lock:
            self.unindexed_transcript_ids.append(transcript_id)

    def push_transcripts(self, ids: list[int]) -> None:
        with self._lock:
            self.unindexed_transcript_ids.extend(ids)

    def pop_transcript_batch(self, size: int = 20) -> list[int]:
        """Pop up to *size* transcript IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.unindexed_transcript_ids))
            return [self.unindexed_transcript_ids.popleft() for _ in range(n)]

    def unindexed_transcript_count(self) -> int:
        return len(self.unindexed_transcript_ids)

    # ------------------------------------------------------------------
    # Re-enqueue on failure (rollback semantics)
    # ------------------------------------------------------------------

    def requeue_segments(self, ids: list[int]) -> None:
        """Put failed segment IDs back at the front of the deque."""
        with self._lock:
            self.pending_segment_ids.extendleft(reversed(ids))

    def requeue_transcripts(self, ids: list[int]) -> None:
        """Put failed transcript IDs back at the front of the deque."""
        with self._lock:
            self.unindexed_transcript_ids.extendleft(reversed(ids))

    # ------------------------------------------------------------------
    # Search query deque — O(1) pop
    # ------------------------------------------------------------------

    def push_query(self, query_id: int) -> None:
        """Enqueue a search_query ID for expression ingestion."""
        with self._lock:
            self.uningested_query_ids.append(query_id)

    def push_queries(self, ids: list[int]) -> None:
        """Enqueue multiple search_query IDs."""
        with self._lock:
            self.uningested_query_ids.extend(ids)

    def pop_query_batch(self, size: int = 50) -> list[int]:
        """Pop up to *size* search_query IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.uningested_query_ids))
            return [self.uningested_query_ids.popleft() for _ in range(n)]

    def requeue_queries(self, ids: list[int]) -> None:
        """Put failed query IDs back at the front of the deque."""
        with self._lock:
            self.uningested_query_ids.extendleft(reversed(ids))

    def uningested_query_count(self) -> int:
        return len(self.uningested_query_ids)

    # ------------------------------------------------------------------
    # Summary session deque — O(1) pop
    # ------------------------------------------------------------------

    def push_summary_session(self, session_id: int) -> None:
        """Enqueue a session ID whose summary needs expression ingestion."""
        with self._lock:
            self.uningested_summary_session_ids.append(session_id)

    def pop_summary_session_batch(self, size: int = 20) -> list[int]:
        """Pop up to *size* session IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.uningested_summary_session_ids))
            return [self.uningested_summary_session_ids.popleft() for _ in range(n)]

    def requeue_summary_sessions(self, ids: list[int]) -> None:
        """Put failed session IDs back at the front of the deque."""
        with self._lock:
            self.uningested_summary_session_ids.extendleft(reversed(ids))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_state: SweepState | None = None


def init_sweep_state() -> SweepState:
    """Create and return the global SweepState singleton.

    Must be called once at daemon startup before get_sweep_state().
    """
    global _state
    _state = SweepState()
    return _state


def get_sweep_state() -> SweepState:
    """Return the global SweepState singleton."""
    if _state is None:
        raise RuntimeError(
            "SweepState not initialised — call init_sweep_state() first."
        )
    return _state
