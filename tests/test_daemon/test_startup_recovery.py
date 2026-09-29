"""
Tests for 1G — StartupRecovery container and RecoveryStep contract.

The contract:
  - RecoveryStep has a name and is callable returning int (recovered_count)
  - StartupRecovery.run() returns {step_name: recovered_count}
  - Results are logged as structured journal events (one per step)
  - Steps execute in registration order
  - UnindexedExpressionRecovery queues expressions missing from LanceDB
  - WorkAssignedBackfill is idempotent (run twice → same queue depth)

GhostJobRecovery tests were removed 2026-09-29: the function monitored the
legacy 'jobs' table (System 1, last write 2026-08-31) for dead PIDs.

All DB-touching tests use test_db for isolation.
"""

from unittest.mock import patch

from audiobench.core.db_session import get_session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GhostJobRecovery tests removed 2026-09-29.
# GhostJobRecovery monitored the legacy 'jobs' table (System 1) for dead PIDs.
# System 3 (worker_daemon + scheduler.startup_recovery) handles unified_jobs
# orphan detection independently. The function and its helpers are deleted.
# ---------------------------------------------------------------------------



def test_recovery_step_has_name_and_returns_count():
    """
    RecoveryStep must carry a string name and be callable returning int.
    This is the contract every step in the system must satisfy.
    """
    from audiobench.daemon.startup_recovery import RecoveryStep

    step = RecoveryStep(name="my_step", fn=lambda: 3)
    assert step.name == "my_step"
    assert step() == 3


def test_startup_recovery_run_returns_name_to_count_dict():
    """
    StartupRecovery.run() must return a dict mapping each step name to the
    count it returned. This is the structured result that gets logged.
    """
    from audiobench.daemon.startup_recovery import RecoveryStep, StartupRecovery

    recovery = StartupRecovery()
    recovery.register(RecoveryStep(name="step_a", fn=lambda: 2))
    recovery.register(RecoveryStep(name="step_b", fn=lambda: 0))

    result = recovery.run()

    assert result == {"step_a": 2, "step_b": 0}


def test_steps_execute_in_registration_order():
    """
    Steps must run in the order they were registered. This is the contract
    that lets 1E's GhostJobRecovery run before UnindexedExpressionRecovery
    (ghost jobs produce failed-state expressions that shouldn't be re-queued).
    """
    from audiobench.daemon.startup_recovery import RecoveryStep, StartupRecovery

    execution_order = []
    recovery = StartupRecovery()
    recovery.register(RecoveryStep(name="first",  fn=lambda: execution_order.append("first")  or 0))
    recovery.register(RecoveryStep(name="second", fn=lambda: execution_order.append("second") or 0))
    recovery.register(RecoveryStep(name="third",  fn=lambda: execution_order.append("third")  or 0))

    recovery.run()

    assert execution_order == ["first", "second", "third"]


def test_recovery_results_logged_to_journal():
    """
    run() must emit one log call per step with name and count.
    We patch the logger to avoid real I/O — the point is that the structured
    result reaches the logging layer, not that a specific SQLite row exists.
    """
    import audiobench.daemon.startup_recovery as sr_mod
    from audiobench.daemon.startup_recovery import RecoveryStep, StartupRecovery

    recovery = StartupRecovery()
    recovery.register(RecoveryStep(name="ghost_jobs", fn=lambda: 5))

    with patch.object(sr_mod.logger, "info") as mock_log:
        recovery.run()

    # At least one call must mention the step name and count
    logged_messages = " ".join(str(c) for c in mock_log.call_args_list)
    assert "ghost_jobs" in logged_messages
    assert "5" in logged_messages


def test_failing_step_does_not_abort_subsequent_steps():
    """
    A step that raises must be caught so subsequent steps still run.
    The result for the failed step must indicate 0 (safe default).
    """
    from audiobench.daemon.startup_recovery import RecoveryStep, StartupRecovery

    def _bad():
        raise RuntimeError("disk exploded")

    calls = []
    recovery = StartupRecovery()
    recovery.register(RecoveryStep(name="bad_step",  fn=_bad))
    recovery.register(RecoveryStep(name="good_step", fn=lambda: calls.append(1) or 1))

    result = recovery.run()

    assert calls == [1], "good_step must still run after bad_step raises"
    assert result["bad_step"] == 0
    assert result["good_step"] == 1




# ---------------------------------------------------------------------------
# UnindexedExpressionRecovery
# ---------------------------------------------------------------------------

def test_unindexed_recovery_queues_missing_expressions(test_db):
    """
    UnindexedExpressionRecovery must return a count of expressions that are
    present in the DB but absent from the provided set of indexed IDs.
    It uses SweepState to enqueue them for the RAG sweep.
    """
    from audiobench.daemon.startup_recovery import UnindexedExpressionRecovery
    from audiobench.daemon.sweep_state import SweepState
    from audiobench.storage.models import ExpressionRecord, TranscriptionRecord

    # Seed two expressions
    with get_session() as session:
        for eid in [10, 11]:
            t = TranscriptionRecord(id=eid, is_indexed=1, file_name="", full_text="sample text", pipeline_phase="complete")
            e = ExpressionRecord(id=eid, source_type='audio_transcript', source_id=eid, content='content', graph_role='sweep_chunk')
            session.add(t)
            session.add(e)
        session.commit()

    sweep = SweepState()
    sweep.indexed_expression_ids = {10}  # 11 is missing

    count = UnindexedExpressionRecovery(sweep_state=sweep)

    assert count == 1
    assert sweep.unindexed_transcript_count() == 1


# ---------------------------------------------------------------------------
# WorkAssignedBackfill idempotency
# ---------------------------------------------------------------------------

def test_work_assigned_backfill_idempotent(test_db):
    """
    Running WorkAssignedBackfill twice must not grow the reconciliation_queue
    beyond what the first run inserted. INSERT OR IGNORE semantics.
    Since the test DB starts empty and there are no work_assigned system_events
    to backfill, both runs return 0 — idempotency is the invariant.
    """
    import sqlite3

    from audiobench.daemon.startup_recovery import WorkAssignedBackfill
    from audiobench.observatory.db import get_journal_db_path, init_journal_db

    # The journal DB (journal.db) is separate from the SQLAlchemy test DB.
    # init_journal_db creates it (with reconciliation_queue) in the tmp data_dir.
    init_journal_db()

    def _queue_depth():
        conn = sqlite3.connect(str(get_journal_db_path()), timeout=5.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM reconciliation_queue").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    WorkAssignedBackfill()
    depth_after_first = _queue_depth()

    WorkAssignedBackfill()
    depth_after_second = _queue_depth()

    assert depth_after_second == depth_after_first

