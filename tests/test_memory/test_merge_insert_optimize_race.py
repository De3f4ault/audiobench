"""Verification test: confirm optimize() does not lose expression IDs after merge_insert.

Bug 2b from root cause analysis: the previous delete()+add() write path had a race
window where a concurrent optimize() call could compact-out deleted rows before the
add() committed, permanently losing expressions from LanceDB.

The fix (merge_insert) eliminates the window by construction \u2014 each write is a single
LanceDB transaction. This test verifies:
  1. merge_insert + optimize() running concurrently causes zero ID loss (20 iterations).
  2. After a forced optimize(), the exact pre-optimize ID set is preserved.

Both checks use ID-set equality (not row count), matching the precision of the
forensic analysis that identified the bug (899 consecutive runs of missing IDs, not
one large gap \u2014 a count-only check would not have caught this).
"""

from __future__ import annotations

import datetime
import threading

import pyarrow as pa
import pytest


@pytest.fixture
def fresh_lance_db(tmp_path):
    """Isolated LanceDB instance with pre-seeded expressions table."""
    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    class TestNode(LanceModel):
        expression_id: int
        content: str
        source_type: str
        vector: Vector(4)  # tiny vector for speed  # type: ignore[valid-type]

    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = db.create_table("expressions", schema=TestNode)
    table.create_scalar_index("expression_id")

    seed = pa.table({
        "expression_id": list(range(1, 501)),
        "content": [f"seed content {i}" for i in range(1, 501)],
        "source_type": ["audio_transcript"] * 500,
        "vector": [[float(i % 4)] * 4 for i in range(500)],
    })
    table.add(seed)
    return table


def _snapshot_ids(table) -> set[int]:
    """Read the exact set of expression_ids currently in the table."""
    return set(int(x) for x in table.to_arrow()["expression_id"].to_pylist())


def test_optimize_preserves_all_ids_after_merge_insert(fresh_lance_db):
    """Verify that a forced optimize() preserves the exact ID set."""
    table = fresh_lance_db

    # Write 20 new expressions via merge_insert
    new_ids = list(range(501, 521))
    new_data = pa.Table.from_pylist(
        [
            {"expression_id": i, "content": f"new content {i}",
             "source_type": "audio_transcript", "vector": [float(i % 4)] * 4}
            for i in new_ids
        ],
        schema=table.schema,
    )
    table.merge_insert("expression_id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(new_data)

    # Snapshot BEFORE optimize (exact ID set, not count)
    before_ids = _snapshot_ids(table)
    assert new_ids[0] in before_ids, "New IDs should be present before optimize"
    assert len(before_ids) == 520

    # Force optimize
    table.optimize(cleanup_older_than=datetime.timedelta(minutes=30))

    # Snapshot AFTER optimize — must be identical
    after_ids = _snapshot_ids(table)
    lost = before_ids - after_ids
    gained = after_ids - before_ids

    assert not lost, (
        f"optimize() permanently lost {len(lost)} expression IDs: {sorted(lost)[:20]}"
    )
    assert not gained, (
        f"optimize() gained unexpected IDs (internal state inconsistency): {sorted(gained)[:10]}"
    )


def test_concurrent_merge_insert_and_optimize_no_id_loss(fresh_lance_db):
    """Directly test the original race condition: merge_insert racing optimize().

    Runs 20 iterations where both operations fire simultaneously from separate
    threads. Uses ID-set equality (not count) to detect any loss.

    This is an explicit test of LanceDB's MVCC transaction guarantee: merge_insert
    is a single committed version, so optimize() cannot catch it in a partial state.
    If this test starts failing, it means LanceDB's transaction semantics have
    changed and the fix needs re-evaluation.
    """
    table = fresh_lance_db
    ITERATIONS = 20
    BATCH = 16

    for i in range(ITERATIONS):
        existing = list(range(i * 2 + 1, i * 2 + BATCH // 2 + 1))
        new = list(range(500 + i * BATCH + 1, 500 + i * BATCH + BATCH // 2 + 1))
        all_ids = existing + new

        batch = pa.Table.from_pylist(
            [
                {"expression_id": eid, "content": f"iter-{i}-content-{eid}",
                 "source_type": "audio_transcript", "vector": [float(eid % 4)] * 4}
                for eid in all_ids
            ],
            schema=table.schema,
        )

        # Snapshot BEFORE (exact IDs)
        before_ids = _snapshot_ids(table)

        merge_done = threading.Event()
        opt_done = threading.Event()
        errors: list[str] = []

        def do_merge():
            try:
                table.merge_insert("expression_id") \
                    .when_matched_update_all() \
                    .when_not_matched_insert_all() \
                    .execute(batch)
            except Exception as exc:
                errors.append(f"merge error: {exc}")
            finally:
                merge_done.set()

        def do_optimize():
            try:
                table.optimize(cleanup_older_than=datetime.timedelta(minutes=30))
            except Exception as exc:
                errors.append(f"optimize error: {exc}")
            finally:
                opt_done.set()

        t1 = threading.Thread(target=do_merge)
        t2 = threading.Thread(target=do_optimize)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Classify errors. LanceDB's MVCC raises a *retryable* conflict when
        # optimize() tries to commit a Rewrite at a version already taken by a
        # concurrent merge_insert Update. This is correct MVCC behavior:
        # merge_insert won the commit race; optimize is preempted and told to
        # retry. In production, lancedb_optimizer.py catches all optimize()
        # exceptions (line 189) and logs them — the next cycle succeeds cleanly.
        # Only merge errors and non-retryable optimize errors are genuine failures.
        merge_errors = [e for e in errors if e.startswith("merge error")]
        hard_opt_errors = [
            e for e in errors
            if e.startswith("optimize error") and "Retryable" not in e
        ]
        assert not merge_errors, f"Iteration {i}: merge_insert failed: {merge_errors}"
        assert not hard_opt_errors, f"Iteration {i}: non-retryable optimize error: {hard_opt_errors}"

        after_ids = _snapshot_ids(table)
        expected = before_ids | set(all_ids)
        lost = expected - after_ids

        assert not lost, (
            f"Iteration {i}: concurrent merge_insert+optimize() lost {len(lost)} IDs: "
            f"{sorted(lost)[:10]}\n"
            f"LanceDB MVCC guarantee failed — re-evaluate the merge_insert fix."
        )
