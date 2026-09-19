"""
startup_recovery.py — RecoveryStep contract and StartupRecovery container.

This module defines the typed step contract that all 1G recovery steps
must satisfy, and the container that runs them in registration order,
returns structured results, and logs each result.

The three concrete steps registered in get_startup_recovery() are:

  1. GhostJobRecovery     — marks running jobs with dead PIDs as 'failed'
  2. UnindexedExpressionRecovery — queues expressions absent from LanceDB
  3. WorkAssignedBackfill — backfills reconciliation_queue from system_events

Each step is idempotent. run() returns {step_name: recovered_count}.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("audiobench.daemon.startup_recovery")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class RecoveryStep:
    """Named, callable startup recovery step that returns a recovered count.

    Every step registered into StartupRecovery must satisfy this contract:
      - name: str          — unique identifier; appears in run() results and logs
      - fn: Callable[[], int] — idempotent function returning number of items recovered

    The callable interface (`__call__`) delegates to fn so instances can be
    used as callables directly in tests.
    """
    name: str
    fn: Callable[[], int]

    def __call__(self) -> int:
        return self.fn()


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class StartupRecovery:
    """Runs registered RecoveryStep instances in registration order.

    run() returns {step_name: recovered_count} and logs each result.
    A failing step is caught and logged; subsequent steps still run.
    The failed step contributes 0 to the results dict.
    """

    def __init__(self) -> None:
        self._steps: list[RecoveryStep] = []

    def register(self, step: RecoveryStep) -> None:
        self._steps.append(step)

    def run(self) -> dict[str, int]:
        logger.info("Running %d startup recovery steps...", len(self._steps))
        results: dict[str, int] = {}
        for step in self._steps:
            try:
                count = step()
                results[step.name] = count
                logger.info("Recovery step '%s' complete: %d item(s) recovered.", step.name, count)
            except Exception as exc:
                logger.error("Recovery step '%s' failed: %s", step.name, exc)
                results[step.name] = 0
        logger.info("Startup recovery complete — results: %s", results)
        return results


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------

def GhostJobRecovery() -> int:
    """Mark running jobs whose PID is no longer alive as 'failed'.

    Uses JobRepository which opens its own sqlite3 connection so that this
    step works correctly even before the SQLAlchemy session factory is ready.
    Returns the count of jobs marked failed.
    """
    import os

    from audiobench.jobs.repository import JobRepository

    repo = JobRepository()
    running = repo.get_running_jobs()
    count = 0
    for job in running:
        pid = job.get("pid")
        if pid and pid > 0:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                # pid 0 → dead; PermissionError on some systems means the
                # process exists but is owned by another user — treat as alive.
                repo.mark_job_failed(job["id"], exit_code=-1)
                count += 1
                logger.info("Ghost job #%d (PID %d) marked failed.", job["id"], pid)
    return count


def UnindexedExpressionRecovery(sweep_state=None) -> int:
    """Queue expressions present in SQLite but absent from LanceDB into SweepState.

    If sweep_state is None (normal daemon startup) the global singleton is used.
    Returns the count of expression IDs enqueued.
    """
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session

    if sweep_state is None:
        from audiobench.daemon.sweep_state import get_sweep_state
        sweep_state = get_sweep_state()

    indexed_ids = sweep_state.indexed_expression_ids

    with get_session() as session:
        rows = session.execute(
            sql_text("""
            SELECT e.id, e.source_id
            FROM expressions e
            JOIN transcriptions t ON e.source_id = t.id
            WHERE e.source_type = 'audio_transcript'
              AND t.status = 'complete'
              AND t.full_text IS NOT NULL
              AND TRIM(t.full_text) != ''
              AND (
                  e.graph_role = 'sweep_chunk'
                  OR (
                      e.graph_role IS NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM expressions c
                          WHERE c.source_id = e.source_id
                            AND c.source_type = 'audio_transcript'
                            AND c.graph_role = 'sweep_chunk'
                      )
                  )
              )
            """)
            # graph_role filter is intentional and load-bearing:
            # Only sweep_chunk nodes (T3) are written to LanceDB. T1 (sweep_document)
            # and T2 (sweep_passage) are graph-structure nodes that exist only in SQLite.
            #
            # NULL rows are included ONLY for transcripts that have no sweep_chunk
            # children — i.e. genuine pre-Track-4-only transcripts whose flat rows
            # were the sole LanceDB representation.  If a transcript already has
            # sweep_chunk nodes, its NULL rows are orphaned residue from the old flat
            # path that was superseded by Track 4 tiering; they were never written to
            # LanceDB and including them would trigger a permanent boot loop where
            # recovery resets is_indexed=0 every restart.
        ).fetchall()

    missing_tx = {r[1] for r in rows if r[0] not in indexed_ids and r[1] is not None}

    # ── Pass 2: catch transcripts that are marked indexed but have NO expressions ──
    # This happens when expressions were purged by migration, test teardown, or
    # the is_indexed flag was set before the sweep could write expressions.
    with get_session() as session:
        orphan_tx_ids = set(
            session.execute(
                sql_text("""
                    SELECT t.id FROM transcriptions t
                    WHERE t.status = 'complete'
                      AND t.is_indexed = 1
                      AND t.full_text IS NOT NULL
                      AND TRIM(t.full_text) != ''
                      AND NOT EXISTS (
                          SELECT 1 FROM expressions e
                          WHERE e.source_type = 'audio_transcript'
                            AND e.source_id = t.id
                      )
                """)
            ).scalars().all()
        )

    all_to_reset = missing_tx | orphan_tx_ids
    if all_to_reset:
        # Reset is_indexed=0 in SQLite so these transcripts are picked up by
        # the sweep loop's incremental refresh (which queries WHERE is_indexed=0)
        # on EVERY tick, not just once in the startup deque.  Without this reset
        # the transcripts stay marked is_indexed=1 and are silently skipped on
        # any restart that happens before the sweep finishes writing them.
        from audiobench.core.db_session import get_session as _gs2
        with _gs2() as s2:
            ids_list = ", ".join(str(i) for i in all_to_reset)
            s2.execute(
                sql_text(
                    f"UPDATE transcriptions SET is_indexed=0 WHERE id IN ({ids_list})"
                )
            )
            s2.commit()
        sweep_state.push_transcripts(list(all_to_reset))
        logger.info(
            "UnindexedExpressionRecovery: reset is_indexed=0 and queued %d transcript(s) "
            "(%d with missing LanceDB expressions, %d with zero SQLite expressions).",
            len(all_to_reset),
            len(missing_tx),
            len(orphan_tx_ids),
        )
    return len(all_to_reset)


def WorkAssignedBackfill() -> int:
    """Backfill reconciliation_queue from system_events for work_assigned events.

    Idempotent — only inserts entries whose expression_id is not already queued.
    Returns the count of queue rows inserted.
    """
    import json
    import sqlite3

    from audiobench.core.db_session import get_session
    from audiobench.observatory.db import get_journal_db_path

    journal_path = get_journal_db_path()
    conn = sqlite3.connect(str(journal_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    inserted = 0
    try:
        row = conn.execute(
            "SELECT MAX(queued_at) as max_ts FROM reconciliation_queue"
        ).fetchone()
        latest_ts = row["max_ts"] if row and row["max_ts"] else "1970-01-01T00:00:00"

        events = conn.execute(
            "SELECT id, metadata FROM system_events "
            "WHERE event_type = 'work_assigned' AND ts > ?",
            (latest_ts,)
        ).fetchall()

        if not events:
            return 0

        from audiobench.memory.enums import SourceType
        from audiobench.storage.models import ExpressionRecord, TranscriptionRecord

        for event in events:
            try:
                metadata = json.loads(event["metadata"])
                audio_file_id = metadata.get("audio_file_id")
                work_id = metadata.get("work_id")
                if not audio_file_id or not work_id:
                    continue

                with get_session() as session:
                    exprs = session.query(ExpressionRecord.id).filter(
                        ExpressionRecord.source_id.in_(
                            session.query(TranscriptionRecord.id)
                            .filter_by(audio_file_id=audio_file_id)
                        ),
                        ExpressionRecord.source_type == SourceType.AUDIO_TRANSCRIPT.value
                    ).all()
                    expr_ids = [e.id for e in exprs]

                if not expr_ids:
                    continue

                existing = conn.execute(
                    f"SELECT expression_id FROM reconciliation_queue "
                    f"WHERE expression_id IN ({','.join('?' * len(expr_ids))})",
                    expr_ids
                ).fetchall()
                existing_eids = {r["expression_id"] for r in existing}

                to_insert = [
                    (eid, work_id) for eid in expr_ids if eid not in existing_eids
                ]
                if to_insert:
                    conn.executemany(
                        "INSERT INTO reconciliation_queue (expression_id, work_id) VALUES (?, ?)",
                        to_insert
                    )
                    inserted += len(to_insert)
            except Exception as exc:
                logger.error("WorkAssignedBackfill: failed event #%s: %s", event["id"], exc)

        conn.commit()
    finally:
        conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Factory — returns the fully-registered container used in _serve()
# ---------------------------------------------------------------------------

def get_startup_recovery(sweep_state=None) -> StartupRecovery:
    """Return a StartupRecovery with all three steps registered in order."""
    recovery = StartupRecovery()
    recovery.register(RecoveryStep(name="ghost_jobs",   fn=GhostJobRecovery))
    recovery.register(RecoveryStep(name="unindexed_expressions",
                                   fn=lambda: UnindexedExpressionRecovery(sweep_state)))
    recovery.register(RecoveryStep(name="work_assigned_backfill", fn=WorkAssignedBackfill))
    return recovery
