"""Search session persistence layer.

Provides CRUD operations for search_sessions, search_queries, and
search_query_fragments tables created by migration 030_search_sessions.sql.

Design notes:
- Sessions are always created on REPL entry, always persisted — no explicit
  /save command. Each REPL invocation is exactly one session.
- Fragment text is stored denormalized (fragment_text column) because segment_id
  is not a stable reference: re-transcription can replace or delete segment rows
  entirely. Session records must show what the user actually saw, not what the
  current segment says after a later re-index.
- Segment-ID overlap detection is the offline-safe base layer: pure set
  intersection on segment_id across searches within a session. No LLM required.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

if TYPE_CHECKING:
    from audiobench.memory.query_engine import ResearchResult
    from audiobench.memory.rrf_fusion import FusedResult

logger = get_logger("memory.session_store")


# ── DB connection helper ─────────────────────────────────────────────────────


def _get_conn() -> sqlite3.Connection:
    """Return a sqlite3 connection to the main AudioBench database."""
    db_path = Path(get_settings().database_url.replace("sqlite:///", ""))
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class SessionSummary:
    """Minimal representation for session listing."""
    session_id: int
    title: str | None
    created_at: str
    query_count: int
    preset: str


@dataclass
class QueryRecord:
    """One search within a session."""
    query_id: int
    session_id: int
    sequence_num: int
    query_text: str
    preset: str
    synthesis_text: str | None
    synthesis_failed: bool
    created_at: str
    segment_ids: list[int]  # populated by get_session()
    prior_synthesis_hits: list = field(default_factory=list)  # list of dicts with content, source_type, sequence_num


@dataclass
class SessionDetail:
    """Full session record with all queries."""
    session_id: int
    title: str | None
    created_at: str
    updated_at: str
    query_count: int
    preset: str
    queries: list[QueryRecord]
    session_summary: str | None = None
    summary_generated_at: str | None = None


# ── Write operations ─────────────────────────────────────────────────────────


def create_session(preset: str = "balanced") -> int:
    """Create a new search session row and return its integer ID.

    Called on REPL entry before any search is run.
    """
    try:
        conn = _get_conn()
        with conn:
            cursor = conn.execute(
                "INSERT INTO search_sessions (preset) VALUES (?)",
                (preset,),
            )
            session_id = cursor.lastrowid
        conn.close()
        logger.debug("Created search session %s (preset=%s)", session_id, preset)
        return session_id
    except sqlite3.Error as e:
        logger.warning("Failed to create search session: %s", e)
        return -1  # sentinel — callers must tolerate this


def set_session_title(session_id: int, title: str) -> None:
    """Update the session title (called after first query completes)."""
    if session_id < 0:
        return
    # Truncate to 80 chars for display sanity
    title = title[:80].strip()
    try:
        conn = _get_conn()
        with conn:
            conn.execute(
                "UPDATE search_sessions SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (title, session_id),
            )
        conn.close()
    except sqlite3.Error as e:
        logger.warning("Failed to set session title: %s", e)


def create_query_record(
    session_id: int,
    sequence_num: int,
    query_text: str,
    preset: str,
) -> int:
    """Create a new search query row before retrieval/synthesis begins."""
    if session_id < 0:
        return -1
    try:
        conn = _get_conn()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO search_queries (
                    session_id, sequence_num, query_text, preset
                ) VALUES (?, ?, ?, ?)
                """,
                (session_id, sequence_num, query_text, preset),
            )
            query_id = cursor.lastrowid
            conn.execute(
                """UPDATE search_sessions
                   SET query_count = query_count + 1,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (session_id,),
            )
        conn.close()
        logger.debug("Created query record %s (session=%s, seq=%s)", query_id, session_id, sequence_num)
        return query_id
    except sqlite3.Error as e:
        logger.warning("Failed to create query record: %s", e)
        return -1


def update_query_synthesis(
    query_id: int,
    result: ResearchResult,
) -> None:
    """Update a search query row with the final synthesis results."""
    if query_id < 0:
        return
    try:
        conn = _get_conn()
        with conn:
            conn.execute(
                """
                UPDATE search_queries
                SET synthesis_text = ?,
                    synthesis_failed = ?,
                    synthesis_error = ?,
                    hyde_document = ?,
                    retrieval_time_seconds = ?,
                    synthesis_time_seconds = ?,
                    total_time_seconds = ?
                WHERE id = ?
                """,
                (
                    result.answer,
                    1 if result.synthesis_failed else 0,
                    result.synthesis_error,
                    result.hyde_document,
                    result.retrieval_time_seconds,
                    result.synthesis_time_seconds,
                    result.query_time_seconds,
                    query_id,
                ),
            )
        conn.close()
    except sqlite3.Error as e:
        logger.warning("Failed to update query synthesis: %s", e)


def persist_synthesis_context(query_id: int, prior_hits: list) -> None:
    """Persist the prior synthesis hits used as context for this query."""
    if query_id < 0 or not prior_hits:
        return
    rows = []
    for rank, h in enumerate(prior_hits, start=1):
        rows.append((
            query_id,
            rank,
            h.source_type,
            getattr(h, "sequence_num", None),
            h.content,
        ))
    try:
        conn = _get_conn()
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO search_query_synthesis_context (
                    query_id, rank, source_type, source_query_id, content
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.close()
    except sqlite3.Error as e:
        logger.warning("Failed to persist synthesis context: %s", e)


def persist_fragments(query_id: int, fragments: list[FusedResult]) -> None:
    """Persist retrieved fragments for a query.

    Fragment text is stored denormalized — see module docstring for rationale.
    """
    if query_id < 0 or not fragments:
        return
    rows = []
    for rank, fr in enumerate(fragments, start=1):
        stream_json = json.dumps(list(fr.stream_contributions)) if fr.stream_contributions else None
        rows.append((
            query_id,
            fr.segment_id,
            fr.source_file or None,
            rank,
            fr.rrf_score,
            stream_json,
            fr.start_time,
            fr.end_time,
            fr.text,
        ))
    try:
        conn = _get_conn()
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO search_query_fragments (
                    query_id, segment_id, source_file, rank, rrf_score,
                    stream_contributions, start_time, end_time, fragment_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.close()
    except sqlite3.Error as e:
        logger.warning("Failed to persist search fragments: %s", e)


def close_session(session_id: int) -> None:
    """Mark a session as closed (update updated_at timestamp).

    Called on REPL exit. Lightweight — just a timestamp update.
    """
    if session_id < 0:
        return
    try:
        conn = _get_conn()
        with conn:
            conn.execute(
                "UPDATE search_sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (session_id,),
            )
        conn.close()
        logger.debug("Closed search session %s", session_id)
    except sqlite3.Error as e:
        logger.warning("Failed to close session: %s", e)


def save_session_summary(session_id: int, summary: str) -> None:
    """Persist an AI-generated summary for a session.

    Overwrites any previous summary — the latest always wins.
    Called by the /summary REPL command after LLM generation succeeds.
    """
    if session_id < 0:
        return
    try:
        conn = _get_conn()
        with conn:
            conn.execute(
                """UPDATE search_sessions
                   SET session_summary=?,
                       summary_generated_at=CURRENT_TIMESTAMP,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (summary.strip(), session_id),
            )
        conn.close()
        logger.debug("Saved summary for session %s", session_id)
    except sqlite3.Error as e:
        logger.warning("Failed to save session summary: %s", e)


# ── Read operations ──────────────────────────────────────────────────────────


def list_sessions(limit: int = 20) -> list[SessionSummary]:
    """Return recent sessions ordered by creation time descending."""
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """SELECT s.id, s.title, s.created_at, s.query_count, 
                      GROUP_CONCAT(DISTINCT q.preset) as presets
               FROM search_sessions s
               LEFT JOIN search_queries q ON s.id = q.session_id
               GROUP BY s.id
               ORDER BY s.created_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            SessionSummary(
                session_id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                query_count=r["query_count"],
                preset=(r["presets"] or "").replace(",", ", "),
            )
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.warning("Failed to list sessions: %s", e)
        return []


def get_session(session_id: int) -> SessionDetail | None:
    """Return full session detail with all queries and their segment IDs."""
    try:
        conn = _get_conn()

        session_row = conn.execute(
            "SELECT * FROM search_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not session_row:
            conn.close()
            return None

        query_rows = conn.execute(
            """SELECT * FROM search_queries WHERE session_id=? ORDER BY sequence_num""",
            (session_id,),
        ).fetchall()

        queries: list[QueryRecord] = []
        for qr in query_rows:
            seg_rows = conn.execute(
                "SELECT segment_id FROM search_query_fragments WHERE query_id=? ORDER BY rank",
                (qr["id"],),
            ).fetchall()
            
            ctx_rows = conn.execute(
                "SELECT content, source_type, source_query_id FROM search_query_synthesis_context WHERE query_id=? ORDER BY rank",
                (qr["id"],),
            ).fetchall()
            prior_hits = [
                {
                    "content": r["content"],
                    "source_type": r["source_type"],
                    "sequence_num": r["source_query_id"],
                }
                for r in ctx_rows
            ]
            # Fallback for old sessions that only have prior_synthesis_json (if any)
            if not prior_hits and "prior_synthesis_json" in qr.keys() and qr["prior_synthesis_json"]:
                try:
                    prior_hits = json.loads(qr["prior_synthesis_json"])
                except Exception:
                    pass

            queries.append(QueryRecord(
                query_id=qr["id"],
                session_id=session_id,
                sequence_num=qr["sequence_num"],
                query_text=qr["query_text"],
                preset=qr["preset"],
                synthesis_text=qr["synthesis_text"],
                synthesis_failed=bool(qr["synthesis_failed"]),
                created_at=qr["created_at"],
                segment_ids=[r["segment_id"] for r in seg_rows],
                prior_synthesis_hits=prior_hits,
            ))

        conn.close()
        return SessionDetail(
            session_id=session_row["id"],
            title=session_row["title"],
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            query_count=session_row["query_count"],
            preset=session_row["preset"],
            queries=queries,
            session_summary=session_row["session_summary"] if "session_summary" in session_row.keys() else None,
            summary_generated_at=session_row["summary_generated_at"] if "summary_generated_at" in session_row.keys() else None,
        )
    except sqlite3.Error as e:
        logger.warning("Failed to get session %s: %s", session_id, e)
        return None


def get_session_segment_ids(session_id: int) -> dict[int, list[int]]:
    """Return a mapping of segment_id → [search sequence numbers it appeared in].

    Used for offline-safe overlap detection: pure set intersection, no LLM.
    Example: {4821: [1, 3], 5012: [2]} means segment 4821 appeared in searches
    1 and 3 of this session, and segment 5012 appeared only in search 2.
    """
    result: dict[int, list[int]] = {}
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT sqf.segment_id, sq.sequence_num
            FROM search_query_fragments sqf
            JOIN search_queries sq ON sqf.query_id = sq.id
            WHERE sq.session_id = ?
            ORDER BY sq.sequence_num
            """,
            (session_id,),
        ).fetchall()
        conn.close()
        for row in rows:
            seg = row["segment_id"]
            seq = row["sequence_num"]
            result.setdefault(seg, [])
            if seq not in result[seg]:
                result[seg].append(seq)
    except sqlite3.Error as e:
        logger.warning("Failed to get segment IDs for session %s: %s", session_id, e)
    return result


def get_session_source_files(session_id: int) -> dict[str, list[int]]:
    """Return a mapping of source_file → [search sequence numbers it appeared in].

    Coarser than segment-level overlap: detects when the same *source* appears
    across searches even when different segments were retrieved. Also offline-safe.
    """
    result: dict[str, list[int]] = {}
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT sqf.source_file, sq.sequence_num
            FROM search_query_fragments sqf
            JOIN search_queries sq ON sqf.query_id = sq.id
            WHERE sq.session_id = ? AND sqf.source_file IS NOT NULL
            ORDER BY sq.sequence_num
            """,
            (session_id,),
        ).fetchall()
        conn.close()
        for row in rows:
            src = row["source_file"]
            seq = row["sequence_num"]
            result.setdefault(src, [])
            if seq not in result[src]:
                result[src].append(seq)
    except sqlite3.Error as e:
        logger.warning("Failed to get source files for session %s: %s", session_id, e)
    return result


def resume_session_state(session_id: int) -> dict | None:
    """Load everything needed to resume a prior search session.

    Returns a dict with keys:
        session_id        int          — the DB primary key
        search_count      int          — number of prior queries (next will be count+1)
        search_segment_ids dict[int, set[int]]
                                       — {sequence_num: set(segment_ids)} per prior search
        search_source_files dict[str, list[int]]
                                       — {source_file: [seq_nums]} per prior search
        last_synthesis    str | None   — synthesis text of the most recent query
        title             str | None   — session title for display
        preset            str          — preset used in this session

    Returns None if the session does not exist.
    """
    detail = get_session(session_id)
    if detail is None:
        return None

    # Rebuild segment-level map: sequence_num → set of segment_ids
    seg_map: dict[int, set[int]] = {}
    for q in detail.queries:
        seg_map[q.sequence_num] = set(q.segment_ids)

    # Query text map: sequence_num → query text
    query_texts: dict[int, str] = {q.sequence_num: q.query_text for q in detail.queries}

    # Source-level map comes from the dedicated helper (already handles dedup)
    src_map = get_session_source_files(session_id)

    # Last synthesis for prior-context carryforward
    last_syn: str | None = None
    if detail.queries:
        last_syn = detail.queries[-1].synthesis_text

    return {
        "session_id": session_id,
        "search_count": detail.query_count,
        "search_segment_ids": seg_map,
        "search_source_files": src_map,
        "search_query_texts": query_texts,
        "last_synthesis": last_syn,
        "title": detail.title,
        "preset": detail.preset,
    }


def get_query_text(query_id: int) -> str | None:
    """Return the query_text for a search query by its primary key ID."""
    try:
        conn = _get_conn()
        row = conn.execute("SELECT query_text FROM search_queries WHERE id = ?", (query_id,)).fetchone()
        conn.close()
        if row:
            return row["query_text"]
    except sqlite3.Error as e:
        logger.warning("Failed to get query text for query %s: %s", query_id, e)
    return None


def get_query_id_by_sequence(session_id: int, sequence_num: int) -> int | None:
    """Find a search query ID by its session ID and sequence number."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM search_queries WHERE session_id = ? AND sequence_num = ?",
            (session_id, sequence_num),
        ).fetchone()
        conn.close()
        if row:
            return row["id"]
    except sqlite3.Error as e:
        logger.warning(
            "Failed to get query ID for session %s sequence %s: %s", session_id, sequence_num, e
        )
    return None


def resolve_query_id_for_session(
    session_id: int, sequence_num: int | None = None, title: str | None = None
) -> int | None:
    """Attempt to resolve a search query ID from session ID, sequence num, or title."""
    try:
        conn = _get_conn()
        if sequence_num is not None:
            row = conn.execute(
                "SELECT id FROM search_queries WHERE session_id = ? AND sequence_num = ?",
                (session_id, sequence_num),
            ).fetchone()
            if row:
                conn.close()
                return row["id"]

        rows = conn.execute(
            "SELECT id, sequence_num, query_text FROM search_queries WHERE session_id = ? ORDER BY sequence_num",
            (session_id,),
        ).fetchall()
        conn.close()

        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]["id"]

        if title:
            norm_title = title.lower()
            for r in rows:
                q_text = (r["query_text"] or "").lower()
                if q_text and (q_text in norm_title or norm_title in q_text):
                    return r["id"]
    except sqlite3.Error as e:
        logger.warning("Failed to resolve query ID for session %s: %s", session_id, e)
    return None


def get_query_fragments(query_id: int) -> list[dict]:
    """Retrieve fragments persisted for a search query, ordered by rank.

    Returns a list of dicts with keys:
        source_file, transcription_id, start_time, end_time, score, text, rank, segment_id
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT sqf.segment_id, sqf.source_file, sqf.rank, sqf.rrf_score,
                   sqf.start_time, sqf.end_time, sqf.fragment_text,
                   seg.transcription_id
            FROM search_query_fragments sqf
            LEFT JOIN segments seg ON sqf.segment_id = seg.id
            WHERE sqf.query_id = ?
            ORDER BY sqf.rank ASC
            """,
            (query_id,),
        ).fetchall()
        conn.close()

        fragments = []
        for r in rows:
            fragments.append({
                "segment_id": r["segment_id"],
                "source_file": r["source_file"] or (f"Transcript #{r['transcription_id']}" if r["transcription_id"] else "Unknown"),
                "transcription_id": r["transcription_id"],
                "rank": r["rank"],
                "score": r["rrf_score"] or 0.0,
                "start_time": r["start_time"] or 0.0,
                "end_time": r["end_time"] or 0.0,
                "text": (r["fragment_text"] or "").strip(),
            })
        return fragments
    except sqlite3.Error as e:
        logger.warning("Failed to get fragments for query %s: %s", query_id, e)
        return []


