"""tests/test_daemon/test_hnsw_index.py

Unit and integration tests for InCoreHNSWIndex:
- Lifecycle (build, point query, batch query, query_similar)
- Chunked query_batch output shape (N, k) and lock interleaving
- Snapshot save & load round-trip
- Drift validation (row_count, max_id, checksum) falling back to full rebuild
- Dynamic auto-resize on capacity exhaustion
- mark_deleted and slot reuse with replace_deleted=True
- Early-boot buffering and drain order
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audiobench.daemon.index import InCoreHNSWIndex


@pytest.fixture
def sample_vectors() -> list[tuple[int, np.ndarray]]:
    """Generate 50 deterministic 64-dim vectors with distinct expression IDs."""
    np.random.seed(42)
    rows = []
    for eid in range(1, 51):
        v = np.random.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)  # Unit normalized for cosine
        rows.append((eid, v))
    return rows


def test_hnsw_lifecycle_and_point_query(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test initial state, load_or_build, and single point query."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)

    # 1. Not ready initially
    assert not index.is_ready()
    assert index.item_count() == 0
    with pytest.raises(RuntimeError, match="INDEX_NOT_READY"):
        index.query(sample_vectors[0][1], k=5)

    # 2. Build index
    index.load_or_build(sample_vectors)
    assert index.is_ready()
    assert index.item_count() == 50

    # 3. Query exact vector (should match itself as top-1 with dist ~0)
    top_hit = index.query(sample_vectors[0][1], k=3)
    assert len(top_hit) == 3
    assert top_hit[0][0] == sample_vectors[0][0]
    assert top_hit[0][1] < 1e-4


def test_hnsw_query_batch_shapes_and_chunking(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Verify query_batch returns strictly (N, k) arrays and chunking produces identical results."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)

    with pytest.raises(RuntimeError, match="INDEX_NOT_READY"):
        index.query_batch(np.array([sample_vectors[0][1]]), k=5)

    index.load_or_build(sample_vectors)

    query_vecs = np.array([r[1] for r in sample_vectors[:25]], dtype=np.float32)

    # Query with chunk_size=7 (uneven chunks)
    labels_chunked, dists_chunked = index.query_batch(query_vecs, k=5, chunk_size=7)
    assert labels_chunked.shape == (25, 5)
    assert dists_chunked.shape == (25, 5)

    # Query with chunk_size=25 (single chunk)
    labels_single, dists_single = index.query_batch(query_vecs, k=5, chunk_size=25)
    assert labels_single.shape == (25, 5)
    assert dists_single.shape == (25, 5)

    # Both chunk sizes must yield identical neighbors
    np.testing.assert_array_equal(labels_chunked, labels_single)
    np.testing.assert_allclose(dists_chunked, dists_single, atol=1e-5)

    # Empty batch returns empty (0, 0) arrays without crashing
    empty_labels, empty_dists = index.query_batch(np.empty((0, 64), dtype=np.float32), k=5)
    assert empty_labels.shape == (0, 0)
    assert empty_dists.shape == (0, 0)


def test_hnsw_query_similar_excludes_self(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """query_similar must filter out the target expression_id and respect max_distance."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors)

    target_eid = sample_vectors[5][0]
    similar = index.query_similar(target_eid, k=5, max_distance=2.0)

    assert len(similar) > 0
    neighbor_ids = [eid for eid, _ in similar]
    assert target_eid not in neighbor_ids, "Self-match was not filtered out!"

    # Non-existent ID returns empty list
    assert index.query_similar(99999, k=5) == []


def test_hnsw_find_duplicate(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """find_duplicate returns matched expression_id when cosine distance <= max_distance."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)

    # Empty index returns None
    assert index.find_duplicate(sample_vectors[0][1]) is None

    index.load_or_build(sample_vectors)

    # Exact match distance ~0 <= 0.05
    exact_match = index.find_duplicate(sample_vectors[0][1], max_distance=0.05)
    assert exact_match == sample_vectors[0][0]

    # Perturbed vector with tiny noise (distance < 0.05)
    noisy_vec = sample_vectors[0][1] + np.random.randn(64).astype(np.float32) * 0.01
    noisy_vec /= np.linalg.norm(noisy_vec)
    dup = index.find_duplicate(noisy_vec, max_distance=0.05)
    assert dup == sample_vectors[0][0]

    # Orthogonal/far vector returns None
    orthogonal_vec = np.random.randn(64).astype(np.float32)
    orthogonal_vec /= np.linalg.norm(orthogonal_vec)
    assert index.find_duplicate(orthogonal_vec, max_distance=0.001) is None



def test_hnsw_snapshot_save_and_load_roundtrip(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test saving snapshot to disk and reloading in a cold index instance."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors)

    # Save snapshot
    assert index.save() is True
    assert (tmp_path / "expression_index.bin").exists()
    assert (tmp_path / "expression_index.meta.json").exists()

    # Cold instance loads from snapshot
    index_reloaded = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index_reloaded.load_or_build(sample_vectors)

    assert index_reloaded.is_ready()
    assert index_reloaded.item_count() == 50

    # Ensure search results match between original and reloaded index
    q = sample_vectors[3][1]
    res_orig = index.query(q, k=5)
    res_loaded = index_reloaded.query(q, k=5)
    assert res_orig == res_loaded


def test_hnsw_snapshot_drift_detection(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test that row count, max_id, or checksum drift triggers rebuild instead of stale load."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors)
    assert index.save() is True

    meta_file = tmp_path / "expression_index.meta.json"

    # Case 1: row_count drift
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["row_count"] = 9999
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    index2 = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index2.load_or_build(sample_vectors)
    # Drift detected -> rebuild occurred -> count is still 50
    assert index2.item_count() == 50

    # Case 2: max_id drift
    assert index2.save() is True
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["max_expression_id"] = 99999
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    index3 = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index3.load_or_build(sample_vectors)
    assert index3.item_count() == 50

    # Case 3: checksum drift
    assert index3.save() is True
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["id_checksum"] = 12345
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    index4 = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index4.load_or_build(sample_vectors)
    assert index4.item_count() == 50


def test_hnsw_auto_resize_on_overflow(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test dynamic capacity expansion when adding items beyond max_elements."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors[:10])

    # Manually constrain max_elements to test resize trigger
    with index._lock:
        index._max_elements = 12

    # Add 5 more items (count 10 + 5 = 15 >= 12, triggers resize)
    extra_items = sample_vectors[10:15]
    index.add_batch(extra_items)

    assert index.item_count() == 15
    assert index._max_elements > 12

    # Verify newly added items are searchable
    hit = index.query(extra_items[0][1], k=1)
    assert hit[0][0] == extra_items[0][0]


def test_hnsw_mark_deleted_and_replace(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test deleting an element and verifying its slot is reused."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors[:10])

    target_eid = sample_vectors[0][0]
    target_vec = sample_vectors[0][1]

    # Verify present
    assert index.query(target_vec, k=1)[0][0] == target_eid
    assert index.item_count() == 10

    # Mark deleted
    index.mark_deleted(target_eid)
    assert index.item_count() == 9

    # Verify query no longer returns target_eid
    top_after_del = index.query(target_vec, k=1)
    assert top_after_del[0][0] != target_eid

    # Add new item (reusing slots)
    new_vec = np.random.randn(64).astype(np.float32)
    new_vec /= np.linalg.norm(new_vec)
    index.add(500, new_vec)
    assert index.item_count() == 10
    assert index.query(new_vec, k=1)[0][0] == 500


def test_hnsw_early_boot_buffering(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test that add() calls prior to load_or_build are buffered and drained under lock."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)

    # Buffer writes while index is not ready
    early_item_1 = (1001, np.random.randn(64).astype(np.float32))
    early_item_2 = (1002, np.random.randn(64).astype(np.float32))
    index.add(early_item_1[0], early_item_1[1])
    index.add_batch([early_item_2])

    assert len(index._pending_queue) == 2
    assert not index.is_ready()

    # Now call load_or_build
    index.load_or_build(sample_vectors[:10])

    # Queue must be drained and items indexed
    assert len(index._pending_queue) == 0
    assert index.is_ready()
    assert index.item_count() == 12  # 10 base + 2 buffered
    assert index.query(early_item_1[1], k=1)[0][0] == 1001
    assert index.query(early_item_2[1], k=1)[0][0] == 1002


def test_hnsw_early_boot_concurrent_race(tmp_path: Path) -> None:
    """Stress test: concurrent add() and add_batch() calls arriving WHILE load_or_build is executing.

    Verifies the atomic contract: load_or_build -> drain -> ready=True under self._lock.
    No writes are lost, no queue entries are orphaned, and no deadlocks or race conditions occur.
    """
    import threading
    import time

    index = InCoreHNSWIndex(data_dir=tmp_path, dim=32, m=16, ef_construction=100, ef_search=50)

    # 50 base vectors for load_or_build
    np.random.seed(99)
    base_rows = [
        (i, np.random.randn(32).astype(np.float32))
        for i in range(1, 51)
    ]

    total_added_ids = set()
    errors = []
    stop_signal = threading.Event()

    # Worker 1 & 2: single add calls
    def single_adder(worker_id: int):
        for seq in range(50):
            eid = 10_000 + worker_id * 1_000 + seq
            v = np.random.randn(32).astype(np.float32)
            try:
                index.add(eid, v)
                total_added_ids.add(eid)
            except Exception as e:
                errors.append(e)
            time.sleep(0.001)

    # Worker 3 & 4: batch add calls
    def batch_adder(worker_id: int):
        for seq in range(10):
            batch = [
                (20_000 + worker_id * 1_000 + seq * 10 + j, np.random.randn(32).astype(np.float32))
                for j in range(5)
            ]
            try:
                index.add_batch(batch)
                for eid, _ in batch:
                    total_added_ids.add(eid)
            except Exception as e:
                errors.append(e)
            time.sleep(0.003)

    threads = [
        threading.Thread(target=single_adder, args=(1,)),
        threading.Thread(target=single_adder, args=(2,)),
        threading.Thread(target=batch_adder, args=(3,)),
        threading.Thread(target=batch_adder, args=(4,)),
    ]

    # Start writer threads hammering the index while it is NOT ready
    for t in threads:
        t.start()

    # Sleep briefly to ensure some items hit the pending queue before build starts
    time.sleep(0.005)

    # Invoke load_or_build while threads are actively pushing writes
    index.load_or_build(base_rows)

    # Wait for all concurrent writers to finish
    for t in threads:
        t.join()

    # Assertions
    assert len(errors) == 0, f"Errors occurred during concurrent race: {errors}"
    assert len(index._pending_queue) == 0, "Pending queue was not completely drained!"
    assert index.is_ready()

    # Expected count: 50 base + all concurrent additions
    expected_count = 50 + len(total_added_ids)
    assert index.item_count() == expected_count, (
        f"Item count mismatch: expected {expected_count}, got {index.item_count()}"
    )


def test_hnsw_fast_forward_incremental_sync(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Test that incremental delta rows are fast-forwarded without full rebuild."""
    # 1. Build initial base snapshot of 40 items
    base_rows = sample_vectors[:40]
    index_base = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index_base.load_or_build(base_rows)
    assert index_base.item_count() == 40
    assert index_base.save() is True

    meta_file = tmp_path / "expression_index.meta.json"
    meta_initial = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_initial["row_count"] == 40

    # 2. Cold instance loads all 50 items (40 base + 10 delta)
    index_ff = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index_ff.load_or_build(sample_vectors)

    # 3. Assert fast-forwarded successfully to 50 items
    assert index_ff.is_ready()
    assert index_ff.item_count() == 50
    assert index_ff.has_item(50)

    # Newly added delta item (eid=45) must be searchable and match itself
    top_hit = index_ff.query(sample_vectors[44][1], k=1)
    assert top_hit[0][0] == sample_vectors[44][0]

    # 4. Verify snapshot on disk was updated atomically
    meta_updated = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_updated["row_count"] == 50
    assert meta_updated["max_expression_id"] == 50

    # 5. Cold load_snapshot loads the updated snapshot directly
    index_cold = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    assert index_cold.load_snapshot(validate_against_lancedb=False) is True
    assert index_cold.item_count() == 50


def test_hnsw_fast_forward_recall_parity(tmp_path: Path) -> None:
    """Verify search recall parity between full rebuild and incremental fast-forward."""
    np.random.seed(123)
    n_total = 100
    rows = []
    for eid in range(1, n_total + 1):
        v = np.random.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)
        rows.append((eid, v))

    dir_full = tmp_path / "full"
    dir_ff = tmp_path / "fast_forward"
    dir_full.mkdir()
    dir_ff.mkdir()

    # Index A: Full rebuild
    idx_full = InCoreHNSWIndex(data_dir=dir_full, dim=64, m=16, ef_construction=100, ef_search=50)
    idx_full.load_or_build(rows)

    # Index B: 75 base items -> save -> fast forward 25 delta items
    idx_ff = InCoreHNSWIndex(data_dir=dir_ff, dim=64, m=16, ef_construction=100, ef_search=50)
    idx_ff.load_or_build(rows[:75])
    assert idx_ff.save() is True

    idx_ff_reloaded = InCoreHNSWIndex(data_dir=dir_ff, dim=64, m=16, ef_construction=100, ef_search=50)
    idx_ff_reloaded.load_or_build(rows)

    # Query 20 test vectors for top-5 neighbors
    test_queries = [r[1] for r in rows[:20]]
    overlaps = []
    for q in test_queries:
        res_full = [eid for eid, _ in idx_full.query(q, k=5)]
        res_ff = [eid for eid, _ in idx_ff_reloaded.query(q, k=5)]
        overlap = len(set(res_full) & set(res_ff)) / 5.0
        overlaps.append(overlap)

    avg_parity = float(np.mean(overlaps))
    assert avg_parity >= 0.95, f"Recall parity {avg_parity} fell below 0.95"


def test_hnsw_fast_forward_exceeds_threshold_falls_back_to_rebuild(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Verify that drift exceeding the conservative threshold falls back to a full rebuild."""
    # Base snapshot with 20 items
    index_base = InCoreHNSWIndex(
        data_dir=tmp_path,
        dim=64,
        m=16,
        ef_construction=100,
        ef_search=50,
        max_fast_forward_delta_pct=0.10,  # 10% max delta
        min_fast_forward_delta_floor=2,    # Floor of 2 items
    )
    index_base.load_or_build(sample_vectors[:20])
    assert index_base.save() is True

    # Attempt load_or_build with 30 items (10 new items > threshold 2)
    index_drifted = InCoreHNSWIndex(
        data_dir=tmp_path,
        dim=64,
        m=16,
        ef_construction=100,
        ef_search=50,
        max_fast_forward_delta_pct=0.10,
        min_fast_forward_delta_floor=2,
    )
    index_drifted.load_or_build(sample_vectors[:30])

    # Rebuild succeeded, full 30 items indexed
    assert index_drifted.is_ready()
    assert index_drifted.item_count() == 30
    assert index_drifted.has_item(30)


def test_hnsw_fast_forward_handles_deletions(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Verify that fast-forward handles deleted expressions and maintains accurate count."""
    index_base = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index_base.load_or_build(sample_vectors[:30])
    assert index_base.save() is True

    # Drop items 5 and 10, add items 31 and 32
    modified_rows = [r for r in sample_vectors[:30] if r[0] not in (5, 10)]
    modified_rows.extend(sample_vectors[30:32])
    assert len(modified_rows) == 30

    index_ff = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index_ff.load_or_build(modified_rows)

    assert index_ff.item_count() == 30
    assert not index_ff.has_item(5)
    assert not index_ff.has_item(10)
    assert index_ff.has_item(31)
    assert index_ff.has_item(32)


def test_hnsw_save_locked_avoids_deadlock(tmp_path: Path, sample_vectors: list[tuple[int, np.ndarray]]) -> None:
    """Verify that _save_locked executes safely when caller already holds self._lock."""
    index = InCoreHNSWIndex(data_dir=tmp_path, dim=64, m=16, ef_construction=100, ef_search=50)
    index.load_or_build(sample_vectors[:20])

    # 1. Direct invocation under acquired self._lock
    with index._lock:
        # Non-reentrant lock: calling index.save() here would deadlock; _save_locked must succeed
        success = index._save_locked()
        assert success is True

    # 2. Public save() outside lock succeeds
    assert index.save() is True

    # 3. Concurrent thread safety: multiple threads triggering save and query
    import concurrent.futures

    def worker_save():
        return index.save()

    def worker_query():
        return index.query(sample_vectors[0][1], k=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_saves = [executor.submit(worker_save) for _ in range(5)]
        f_queries = [executor.submit(worker_query) for _ in range(10)]
        for f in f_saves:
            assert f.result(timeout=5.0) is True
        for f in f_queries:
            assert len(f.result(timeout=5.0)) == 3



