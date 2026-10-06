"""src/audiobench/daemon/index.py

Process-wide in-memory HNSW index (InCoreHNSWIndex) for AudioBench.
Shared by Autocomplete, ConnectionSurfer, PatternDetector, and SearchSessionSweep.

Key properties:
- Space: cosine (hnswlib computes 1 - (u.v)/(||u||*||v||)).
- Direct 64-bit expression_id labels (no fragile secondary mapping arrays).
- Coarse threading.Lock guarding both reads and writes.
- Chunked query_batch() releasing the lock between chunks so concurrent
  autocomplete queries and incremental writes can interleave with <2ms wait time.
- Atomic snapshot persistence (expression_index.bin + expression_index.meta.json)
  with row count, max ID, and 64-bit ID sum checksum validation.
- Auto-resizing via resize_index() when nearing max_elements.
- Memory slot reuse via allow_replace_deleted=True and replace_deleted=True.
- Early-boot buffer (_pending_queue) protected under the critical section.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np

logger = logging.getLogger("audiobench.daemon.index")

# Default HNSW hyperparameters
_DIM: int = 768
_SPACE: str = "cosine"
_M: int = 32               # Bidirectional links per element (higher = better recall on high-dim)
_EF_CONSTRUCTION: int = 200  # Build quality
_EF_SEARCH: int = 100      # Query recall
_CHUNK_SIZE: int = 25      # Vectors per lock acquisition in query_batch (~15ms per chunk)


class InCoreHNSWIndex:
    """Process-wide in-memory HNSW index shared across all daemon consumers."""

    def __init__(
        self,
        data_dir: Path | None = None,
        dim: int = _DIM,
        space: str = _SPACE,
        m: int = _M,
        ef_construction: int = _EF_CONSTRUCTION,
        ef_search: int = _EF_SEARCH,
    ) -> None:
        if data_dir is None:
            from audiobench.core.settings import get_settings
            data_dir = get_settings().data_dir / "hnsw"
        self._data_dir = Path(data_dir)
        self._dim = dim
        self._space = space
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._lock = threading.Lock()
        self._ready: bool = False
        self._index: hnswlib.Index | None = None
        self._count: int = 0
        self._max_elements: int = 0
        self._last_max_id: int = 0
        self._id_checksum: int = 0  # sum(expression_ids) % (2**64)

        # Early boot buffer for write_node() calls arriving before load_or_build() completes
        self._pending_queue: collections.deque[tuple[int, np.ndarray]] = collections.deque()
        self._indexed_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle: Load, Build, Save
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the index is built/loaded and ready for queries."""
        with self._lock:
            return self._ready

    def item_count(self) -> int:
        """Return number of active elements in the index."""
        with self._lock:
            return self._count

    def count(self) -> int:
        """Alias for item_count."""
        return self.item_count()

    def __len__(self) -> int:
        return self.item_count()

    def has_item(self, expression_id: int) -> bool:
        """Return True if expression_id is present in the index."""
        with self._lock:
            return expression_id in self._indexed_ids

    def __contains__(self, expression_id: int) -> bool:
        return self.has_item(expression_id)

    def get_items(self, expression_ids: list[int]) -> np.ndarray:
        """Return the vector embeddings for the specified expression IDs."""
        with self._lock:
            if not self._ready or self._index is None:
                raise RuntimeError("INDEX_NOT_READY")
            valid_ids = [int(eid) for eid in expression_ids if eid in self._indexed_ids]
            if not valid_ids:
                return np.empty((0, self._dim), dtype=np.float32)
            return np.array(self._index.get_items(valid_ids), dtype=np.float32)

    def load_snapshot(self) -> bool:
        """Attempt to directly load the persisted snapshot from disk without reading LanceDB.

        Returns True if successfully loaded, False otherwise.
        """
        with self._lock:
            bin_path = self._data_dir / "expression_index.bin"
            meta_path = self._data_dir / "expression_index.meta.json"
            if not (bin_path.exists() and meta_path.exists()):
                return False
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                idx = hnswlib.Index(space=self._space, dim=self._dim)
                idx.load_index(str(bin_path), max_elements=meta["max_elements"])
                self._index = idx
                self._count = meta["row_count"]
                self._max_elements = meta["max_elements"]
                self._last_max_id = meta["max_expression_id"]
                self._id_checksum = meta["id_checksum"]
                self._indexed_ids = set(idx.get_ids_list())
                self._ready = True
                logger.info(
                    "InCoreHNSWIndex loaded snapshot directly in fast-path: %d elements",
                    self._count,
                )
                return True
            except Exception as e:
                logger.warning("Fast-path snapshot load failed: %s", e)
                return False

    def load_or_build(self, embeddable_rows: list[tuple[int, np.ndarray]]) -> None:
        """Load the snapshot from disk if valid, otherwise build from scratch.

        embeddable_rows: list of (expression_id, 768-dim float vector)
        Guaranteed to drain any early-boot buffered items before marking ready.
        """
        with self._lock:
            # Deduplicate by expression_id (e.g. if table had uncompacted historical versions)
            dedup_dict = {int(eid): v for eid, v in embeddable_rows}
            clean_rows = [(eid, dedup_dict[eid]) for eid in dedup_dict]
            logger.info("Initializing InCoreHNSWIndex with %d unique live rows...", len(clean_rows))

            snapshot = self._load_snapshot_if_valid(clean_rows)

            if snapshot is not None:
                self._index = snapshot
                self._count = len(clean_rows)
                self._max_elements = self._index.get_max_elements()
                self._indexed_ids = set(dedup_dict.keys())
                self._last_max_id = max(self._indexed_ids, default=0)
                self._id_checksum = sum(self._indexed_ids) % (2**64)
                logger.info(
                    "InCoreHNSWIndex loaded from valid snapshot: %d elements (max_id=%d, checksum=%d)",
                    self._count, self._last_max_id, self._id_checksum,
                )
            else:
                self._build_from_scratch_locked(clean_rows)

            # Drain any early boot writes buffered while loading/building
            drained = 0
            while self._pending_queue:
                eid, vec = self._pending_queue.popleft()
                if eid not in self._indexed_ids:
                    self._add_single_locked(eid, vec)
                    drained += 1

            if drained > 0:
                logger.info("InCoreHNSWIndex: drained %d early-boot buffered expressions", drained)

            self._ready = True
            logger.info("InCoreHNSWIndex is READY (%d items indexed).", self._count)

    def _load_snapshot_if_valid(
        self, embeddable_rows: list[tuple[int, np.ndarray]]
    ) -> hnswlib.Index | None:
        """Attempt to load index from disk. Returns None if missing, corrupted, or drifted."""
        bin_path = self._data_dir / "expression_index.bin"
        meta_path = self._data_dir / "expression_index.meta.json"

        if not (bin_path.exists() and meta_path.exists()):
            logger.info("InCoreHNSWIndex: snapshot files not found — performing full build.")
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            live_ids = {int(eid) for eid, _ in embeddable_rows}
            live_count = len(live_ids)
            live_max_id = max(live_ids, default=0)
            live_checksum = sum(live_ids) % (2**64)

            # Strict drift validation
            if meta.get("row_count") != live_count:
                logger.warning(
                    "InCoreHNSWIndex snapshot drifted: meta row_count=%d vs live=%d. Rebuilding.",
                    meta.get("row_count"), live_count,
                )
                return None

            if meta.get("max_expression_id") != live_max_id:
                logger.warning(
                    "InCoreHNSWIndex snapshot drifted: meta max_id=%d vs live=%d. Rebuilding.",
                    meta.get("max_expression_id"), live_max_id,
                )
                return None

            if meta.get("id_checksum") != live_checksum:
                logger.warning(
                    "InCoreHNSWIndex snapshot drifted: meta checksum=%d vs live=%d. Rebuilding.",
                    meta.get("id_checksum"), live_checksum,
                )
                return None

            max_elements = int(meta.get("max_elements", max(live_count * 2, 50_000)))
            index = hnswlib.Index(space=self._space, dim=self._dim)
            index.load_index(str(bin_path), max_elements=max_elements, allow_replace_deleted=True)
            # CRITICAL: load_index resets ef to default (10); restore ef_search
            index.set_ef(self._ef_search)
            return index

        except Exception as exc:
            logger.warning("InCoreHNSWIndex: snapshot validation/loading failed (%s). Rebuilding.", exc)
            return None

    def _build_from_scratch_locked(self, embeddable_rows: list[tuple[int, np.ndarray]]) -> None:
        """Build a new HNSW graph from scratch under self._lock."""
        t0 = time.perf_counter()
        n = len(embeddable_rows)
        self._max_elements = max(n * 2, 50_000)
        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        self._index.init_index(
            max_elements=self._max_elements,
            ef_construction=self._ef_construction,
            M=self._m,
            allow_replace_deleted=True,
        )
        self._index.set_ef(self._ef_search)

        if n > 0:
            ids = [int(eid) for eid, _ in embeddable_rows]
            vectors = np.array([v for _, v in embeddable_rows], dtype=np.float32)
            self._index.add_items(vectors, ids, replace_deleted=True)
            self._indexed_ids = set(ids)
            self._last_max_id = max(ids)
            self._id_checksum = sum(ids) % (2**64)
        else:
            self._indexed_ids = set()
            self._last_max_id = 0
            self._id_checksum = 0

        self._count = n
        logger.info(
            "InCoreHNSWIndex: built from scratch with %d items in %.2fs (max_elements=%d)",
            n, time.perf_counter() - t0, self._max_elements,
        )

    def save(self) -> bool:
        """Persist snapshot to disk atomically using temporary files."""
        with self._lock:
            if not self._ready or self._index is None:
                return False

            try:
                self._data_dir.mkdir(parents=True, exist_ok=True)
                bin_tmp = self._data_dir / "expression_index.bin.tmp"
                bin_path = self._data_dir / "expression_index.bin"
                meta_tmp = self._data_dir / "expression_index.meta.json.tmp"
                meta_path = self._data_dir / "expression_index.meta.json"

                # 1. Save binary index to tmp and atomic replace
                self._index.save_index(str(bin_tmp))
                os.replace(bin_tmp, bin_path)

                # 2. Write metadata to tmp and atomic replace
                meta = {
                    "row_count": self._count,
                    "max_expression_id": self._last_max_id,
                    "id_checksum": self._id_checksum,
                    "max_elements": self._max_elements,
                    "dim": self._dim,
                    "space": self._space,
                    "m": self._m,
                    "ef_construction": self._ef_construction,
                    "ef_search": self._ef_search,
                    "saved_at": time.time(),
                }
                meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                os.replace(meta_tmp, meta_path)

                logger.info(
                    "InCoreHNSWIndex: snapshot saved (%d items, max_id=%d, checksum=%d)",
                    self._count, self._last_max_id, self._id_checksum,
                )
                return True
            except Exception as exc:
                logger.error("InCoreHNSWIndex: failed to save snapshot: %s", exc)
                return False

    # ------------------------------------------------------------------
    # Writes: Add, Batch Add, Remove
    # ------------------------------------------------------------------

    def add(self, expression_id: int, vector: np.ndarray | list[float]) -> None:
        """Add or update a single expression vector. Thread-safe."""
        v = np.array(vector, dtype=np.float32)
        with self._lock:
            if not self._ready:
                self._pending_queue.append((expression_id, v))
                return
            self._add_single_locked(expression_id, v)

    def add_batch(self, rows: list[tuple[int, np.ndarray | list[float]]]) -> None:
        """Add or update multiple expression vectors. Thread-safe."""
        if not rows:
            return
        parsed_rows = [(int(eid), np.array(v, dtype=np.float32)) for eid, v in rows]
        with self._lock:
            if not self._ready:
                self._pending_queue.extend(parsed_rows)
                return
            self._add_batch_locked(parsed_rows)

    def _ensure_capacity_locked(self, count_to_add: int) -> None:
        """Double index capacity if count_to_add would exceed max_elements."""
        if self._count + count_to_add >= self._max_elements:
            new_max = max(self._max_elements * 2, self._count + count_to_add + 10_000)
            logger.info("InCoreHNSWIndex: resizing from %d to %d elements", self._max_elements, new_max)
            self._index.resize_index(new_max)
            self._max_elements = new_max

    def _add_single_locked(self, expression_id: int, vector: np.ndarray) -> None:
        """Internal add for single item under self._lock."""
        self._ensure_capacity_locked(1)
        self._index.add_items(vector, [expression_id], replace_deleted=True)
        if expression_id not in self._indexed_ids:
            self._indexed_ids.add(expression_id)
            self._count += 1
            self._id_checksum = (self._id_checksum + expression_id) % (2**64)
            if expression_id > self._last_max_id:
                self._last_max_id = expression_id

    def _add_batch_locked(self, rows: list[tuple[int, np.ndarray]]) -> None:
        """Internal add for batch under self._lock."""
        self._ensure_capacity_locked(len(rows))
        ids = [eid for eid, _ in rows]
        vectors = np.array([v for _, v in rows], dtype=np.float32)
        self._index.add_items(vectors, ids, replace_deleted=True)
        for eid in ids:
            if eid not in self._indexed_ids:
                self._indexed_ids.add(eid)
                self._count += 1
                self._id_checksum = (self._id_checksum + eid) % (2**64)
                if eid > self._last_max_id:
                    self._last_max_id = eid

    def mark_deleted(self, expression_id: int) -> None:
        """Mark an expression as deleted so future queries ignore it.

        Its slot is eligible for reuse on future add_items() calls.
        """
        with self._lock:
            if not self._ready or self._index is None:
                return
            if expression_id in self._indexed_ids:
                try:
                    self._index.mark_deleted(expression_id)
                    self._indexed_ids.remove(expression_id)
                    self._count = max(0, self._count - 1)
                    self._id_checksum = (self._id_checksum - expression_id) % (2**64)
                    if expression_id == self._last_max_id:
                        self._last_max_id = max(self._indexed_ids, default=0)
                except Exception as exc:
                    logger.warning("InCoreHNSWIndex: mark_deleted failed for #%d: %s", expression_id, exc)

    # ------------------------------------------------------------------
    # Reads: Single Query & Chunked Batch Query
    # ------------------------------------------------------------------

    def query(self, vector: np.ndarray | list[float], k: int = 5) -> list[tuple[int, float]]:
        """Return up to k nearest neighbors as [(expression_id, cosine_distance), ...].

        Cosine distance d in [0, 2]; cosine similarity = 1.0 - d.
        Raises RuntimeError("INDEX_NOT_READY") if called before initialization.
        """
        with self._lock:
            if not self._ready or self._index is None:
                raise RuntimeError("INDEX_NOT_READY")
            actual_k = min(k, self._count)
            if actual_k == 0:
                return []
            v = np.array(vector, dtype=np.float32)
            labels, distances = self._index.knn_query(v, k=actual_k)

        return list(zip(labels[0].tolist(), distances[0].tolist()))

    def query_batch(
        self,
        vectors: np.ndarray | list[list[float]],
        k: int = 5,
        chunk_size: int = _CHUNK_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbors for a batch of vectors, chunking under the lock.

        Releases the lock between chunks so concurrent point queries (autocomplete)
        and incremental writes interleave with <2ms latency.

        Returns:
            (labels, distances) where both are 2D NumPy arrays of shape (len(vectors), k).

        Raises:
            RuntimeError("INDEX_NOT_READY") if called before initialization.
        """
        with self._lock:
            if not self._ready or self._index is None:
                raise RuntimeError("INDEX_NOT_READY")
            actual_k = min(k, self._count)

        n_queries = len(vectors)
        if n_queries == 0 or actual_k == 0:
            return np.empty((n_queries, 0), dtype=np.int64), np.empty((n_queries, 0), dtype=np.float32)

        vec_matrix = np.array(vectors, dtype=np.float32) if not isinstance(vectors, np.ndarray) else vectors

        all_labels: list[np.ndarray] = []
        all_dists: list[np.ndarray] = []

        for i in range(0, n_queries, chunk_size):
            chunk = vec_matrix[i : i + chunk_size]
            with self._lock:
                labels, dists = self._index.knn_query(chunk, k=actual_k)
            all_labels.append(labels)
            all_dists.append(dists)

        merged_labels = np.vstack(all_labels)
        merged_dists = np.vstack(all_dists)

        return merged_labels, merged_dists

    def query_similar(
        self, expression_id: int, k: int = 5, max_distance: float = 0.22
    ) -> list[tuple[int, float]]:
        """Find expressions semantically similar to an existing indexed expression.

        Self-matches (where candidate_id == expression_id) are omitted.
        max_distance=0.22 corresponds to cosine similarity >= 0.78.
        """
        with self._lock:
            if not self._ready or self._index is None:
                raise RuntimeError("INDEX_NOT_READY")
            if expression_id not in self._indexed_ids:
                return []
            vec = self._index.get_items([expression_id])[0]

        # Query k + 1 to account for self-match
        raw_hits = self.query(vec, k=k + 1)
        results = []
        for eid, dist in raw_hits:
            if eid != expression_id and dist <= max_distance:
                results.append((eid, dist))
                if len(results) == k:
                    break
        return results

    def find_duplicate(
        self, vector: np.ndarray | list[float], max_distance: float = 0.05
    ) -> int | None:
        """Find an existing near-duplicate expression vector within max_distance.

        max_distance=0.05 corresponds to cosine similarity >= 0.95.
        Returns the expression_id of the closest existing item, or None if no match.
        """
        with self._lock:
            if not self._ready or self._index is None or self._count == 0:
                return None
            v = np.array(vector, dtype=np.float32)
            labels, distances = self._index.knn_query(v, k=1)
            if distances[0][0] <= max_distance:
                return int(labels[0][0])
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_shared_index: InCoreHNSWIndex | None = None
_init_lock = threading.Lock()


def get_shared_hnsw_index() -> InCoreHNSWIndex:
    """Return the global InCoreHNSWIndex singleton instance."""
    global _shared_index
    if _shared_index is None:
        with _init_lock:
            if _shared_index is None:
                _shared_index = InCoreHNSWIndex()
    return _shared_index
