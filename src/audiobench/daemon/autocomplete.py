"""AutocompleteIndex — HNSW semantic autocomplete fast-path.

Built at daemon startup from all expression vectors in LanceDB.
Refreshed incrementally when new expressions are embedded by the RAG sweep.

Design:
  - Uses hnswlib (cosine space) for O(log n) approximate nearest-neighbour
    lookup at inference time. No embedder touched during lookup — the query
    vector is provided by the caller (already embedded by the handler).
  - _load_vectors() is a separate method so tests can patch it with fake data
    without touching LanceDB or the embedder.
  - Thread-safety: the index is rebuilt atomically. The module-level singleton
    is replaced under a lock.
  - HNSW parameters (M=32, ef_construction=200, ef=100) are tuned for a corpus
    in the 10k–500k range with 768-dimensional Nomic vectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any

import hnswlib
import numpy as np

logger = logging.getLogger("audiobench.daemon.autocomplete")

# HNSW hyperparameters
_DIM: int = 768
_M: int = 32               # Number of bidirectional links per element
_EF_CONSTRUCTION: int = 200  # Controls build quality (higher = better recall)
_EF_SEARCH: int = 100      # Controls query recall (can be set after build)
_MAX_ELEMENTS: int = 1_000_000  # Upper bound on corpus size


class AutocompleteIndex:
    """In-memory HNSW index (via InCoreHNSWIndex) over expression summary vectors.

    Built at daemon startup from LanceDB. Refreshed incrementally as the
    RAG sweep writes new expressions.

    Takes an already-embedded query vector and instantly returns the closest N
    expressions from the corpus.
    """

    def __init__(self, shared_index: Any | None = None) -> None:
        self._shared_index = shared_index
        self._text_cache: dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        if self._shared_index is not None:
            return self._shared_index.is_ready()
        return False

    @ready.setter
    def ready(self, value: bool) -> None:
        pass  # compatibility

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the in-memory HNSW index from LanceDB.

        Safe to call from a thread-pool executor — acquires the instance lock
        for the swap.
        """
        logger.info("Building AutocompleteIndex from LanceDB...")
        rows = self._load_vectors()  # list of (expression_id, vector, content)

        if self._shared_index is None:
            from audiobench.daemon.index import InCoreHNSWIndex
            import tempfile
            self._shared_index = InCoreHNSWIndex(data_dir=Path(tempfile.mkdtemp()))

        if not rows:
            logger.warning("AutocompleteIndex: no expression vectors found — index empty.")
            self._shared_index.load_or_build([])
            return

        with self._lock:
            self._text_cache = {int(r[0]): str(r[2]) for r in rows}

        embeddable_tuples = [(int(r[0]), np.array(r[1], dtype=np.float32)) for r in rows]
        self._shared_index.load_or_build(embeddable_tuples)
        # Persist snapshot
        self._shared_index.save()

        logger.info("AutocompleteIndex ready — %d expressions indexed.", self._shared_index.item_count())

    def lookup(self, query_vector: list[float] | np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Return up to k nearest expressions for the given query vector.

        Args:
            query_vector: 768-dimensional float vector (already embedded by the
                handler; not re-encoded here).
            k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``expression_id`` and ``text``, ordered
            from closest to furthest.

        Raises:
            RuntimeError: if called before :meth:`build`.
        """
        if self._shared_index is None or not self._shared_index.is_ready():
            raise RuntimeError("INDEX_NOT_READY")

        hits = self._shared_index.query(query_vector, k=k)
        results = []
        with self._lock:
            missing_ids = [eid for eid, _ in hits if eid not in self._text_cache]
            if missing_ids:
                try:
                    from audiobench.storage.expression_repository import ExpressionRepository
                    repo = ExpressionRepository()
                    records = repo.get_by_ids(missing_ids)
                    for mid, rec in records.items():
                        self._text_cache[mid] = rec.content
                except Exception:
                    pass

            for eid, dist in hits:
                text = self._text_cache.get(eid, "")
                results.append({"expression_id": eid, "text": text})

        return results

    def item_count(self) -> int:
        """Return the number of items currently in the index."""
        if self._shared_index is None:
            return 0
        return self._shared_index.item_count()

    # ------------------------------------------------------------------
    # Internal — isolated for test patching
    # ------------------------------------------------------------------

    def _load_vectors(self) -> list[tuple[int, list[float], str]]:
        """Load all (expression_id, vector, content) rows from LanceDB.

        Returns an empty list if the table is empty or unavailable.
        Override / patch in tests to inject fake data without touching LanceDB.
        """
        try:
            from audiobench.memory.memory_store import MemoryStore, EMBEDDABLE_SOURCE_TYPES
            store = MemoryStore()
            rows = store.table.search().select(["expression_id", "vector", "content", "source_type"]).to_list()
            return [
                (int(r["expression_id"]), r["vector"], r.get("content", ""))
                for r in rows
                if r.get("source_type") in EMBEDDABLE_SOURCE_TYPES
            ]
        except Exception as exc:
            logger.warning("AutocompleteIndex._load_vectors failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_index: AutocompleteIndex | None = None
_index_lock = threading.Lock()


def get_autocomplete_index() -> AutocompleteIndex:
    """Return the module-level AutocompleteIndex singleton (create if absent)."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                from audiobench.daemon.index import get_shared_hnsw_index
                _index = AutocompleteIndex(shared_index=get_shared_hnsw_index())
    return _index
