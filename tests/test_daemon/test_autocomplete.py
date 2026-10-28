"""
Tests for 2C — AutocompleteIndex HNSW fast-path.

Three invariants (per roadmap):
1. test_build_populates_index — after build() the index reports ready=True and
   item_count > 0.
2. test_lookup_returns_top_k — lookup() returns exactly k results ordered by
   cosine distance when the index has enough entries.
3. test_lookup_before_build_raises — lookup() before build() raises RuntimeError
   with the sentinel 'INDEX_NOT_READY' so the handler can return a clean error.

All tests operate against hnswlib directly — no real LanceDB, no real embedder.
MemoryStore is patched so _build_from_store uses injected vectors.
"""

from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768  # must match production constant in autocomplete.py


def _make_fake_vectors(n: int) -> list[tuple[int, list[float], str]]:
    """Return n (expression_id, unit vector, content) tuples."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        v = rng.standard_normal(_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rows.append((i + 1, v.tolist(), f"expression text {i + 1}"))
    return rows


def _patched_index(n_items: int = 20):
    """Return an AutocompleteIndex whose _load_vectors is patched with fake data."""
    from audiobench.daemon.autocomplete import AutocompleteIndex

    fake_rows = _make_fake_vectors(n_items)

    idx = AutocompleteIndex()
    with patch.object(idx, "_load_vectors", return_value=fake_rows):
        idx.build()
    return idx, fake_rows


# ---------------------------------------------------------------------------
# 1. build() populates the index
# ---------------------------------------------------------------------------

def test_build_populates_index():
    """
    After calling build() with at least one item, ready must be True and the
    internal item count must equal the number of injected vectors.
    """
    n = 15
    idx, fake_rows = _patched_index(n)

    assert idx.ready is True
    assert idx.item_count() == n


# ---------------------------------------------------------------------------
# 2. lookup() returns top-k ordered results
# ---------------------------------------------------------------------------

def test_lookup_returns_top_k():
    """
    lookup() must return exactly k results. Each result must have keys
    'expression_id' and 'text'. Results are ordered closest-first (i.e. the
    nearest neighbour of the query vector is at index 0).
    """
    idx, fake_rows = _patched_index(20)

    # Use the first stored vector as the query — it should be its own nearest
    # neighbour.
    query_vector = fake_rows[0][1]
    k = 5
    results = idx.lookup(query_vector, k=k)

    assert len(results) == k, f"Expected {k} results, got {len(results)}"
    for r in results:
        assert "expression_id" in r
        assert "text" in r
    # Nearest neighbour of self is self
    assert results[0]["expression_id"] == fake_rows[0][0]


# ---------------------------------------------------------------------------
# 3. lookup() before build() raises RuntimeError
# ---------------------------------------------------------------------------

def test_lookup_before_build_raises():
    """
    Calling lookup() on an un-built index must raise RuntimeError with the
    sentinel string 'INDEX_NOT_READY' so _handle_autocomplete returns a clean
    error response instead of crashing.
    """
    from audiobench.daemon.autocomplete import AutocompleteIndex

    idx = AutocompleteIndex()
    assert idx.ready is False

    with pytest.raises(RuntimeError, match="INDEX_NOT_READY"):
        idx.lookup([0.0] * _DIM, k=3)


# ---------------------------------------------------------------------------
# 4. _handle_autocomplete enriches with work_title and work_author
# ---------------------------------------------------------------------------

def test_handle_autocomplete_enriches_work_and_author():
    """_handle_autocomplete must enrich vector hits with work_title and work_author."""
    from unittest.mock import MagicMock
    from audiobench.daemon.server import _handle_autocomplete

    mock_index = MagicMock()
    mock_index.ready = True
    mock_index.lookup.return_value = [
        {"expression_id": 101, "text": "Academic psychology discussion"},
        {"expression_id": 102, "text": "Prior synthesis findings"},
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(_DIM, dtype=np.float32)

    mock_expr1 = MagicMock(id=101, work_id=13, speaker=None, source_type="audio_transcript")
    mock_expr2 = MagicMock(id=102, work_id=None, speaker=None, source_type="search_synthesis")
    mock_repo = MagicMock()
    mock_repo.get_by_ids.return_value = {101: mock_expr1, 102: mock_expr2}

    mock_work = MagicMock(id=13, title="Poor Charlie's Almanack", author="Charles T. Munger")
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_work]
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.autocomplete.get_autocomplete_index", return_value=mock_index), \
         patch("audiobench.memory.singletons.get_primary_embedder", return_value=mock_embedder), \
         patch("audiobench.storage.expression_repository.ExpressionRepository", return_value=mock_repo), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx):
        res = _handle_autocomplete({"prefix": "psychology", "top_k": 2})

    results = res.get("results", [])
    assert len(results) == 2

    # Audio transcript hit with work metadata
    r1 = results[0]
    assert r1["expression_id"] == 101
    assert r1["work_title"] == "Poor Charlie's Almanack"
    assert r1["work_author"] == "Charles T. Munger"
    assert r1["source_type"] == "audio_transcript"

    # Prior synthesis hit with badge author
    r2 = results[1]
    assert r2["expression_id"] == 102
    assert r2["work_author"] == "[Prior Synthesis]"
    assert r2["source_type"] == "search_synthesis"

