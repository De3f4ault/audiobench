"""Tests for P3-5: Silent mid-session BM25 memory hints in ChatREPL."""

from __future__ import annotations

import time


def _make_repl_with_hints(monkeypatch=None):
    """Make a minimal ChatREPL instance with only the hints-relevant attributes."""
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL.__new__(ChatREPL)
    repl._last_hint_at = 0.0
    return repl


# ── Tests ────────────────────────────────────────────────────────────────────

def test_hints_only_fire_on_questions(monkeypatch):
    """Statements must NOT trigger BM25; questions must."""
    fts5_calls = []

    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            fts5_calls.append(rq)
            return []

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        MockFTS5Stream,
    )
    monkeypatch.setattr(
        "audiobench.chat.chat_repl.FTS5Stream",
        MockFTS5Stream,
        raising=False,
    )

    repl = _make_repl_with_hints()

    # Statement — no hint
    result = repl._fetch_memory_hints("This is a statement.")
    assert fts5_calls == []
    assert result == []

    # Question — hint fires
    result = repl._fetch_memory_hints("What did he say about suffering?")
    assert len(fts5_calls) == 1


def test_hints_debounced_within_30_seconds(monkeypatch):
    """If a hint was recently fetched, the next call must be suppressed."""
    fts5_calls = []

    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            fts5_calls.append(rq)
            return []

    monkeypatch.setattr(
        "audiobench.chat.chat_repl.FTS5Stream",
        MockFTS5Stream,
        raising=False,
    )

    repl = _make_repl_with_hints()
    repl._last_hint_at = time.time()  # simulate a recent hint

    result = repl._fetch_memory_hints("Is this a question?")
    assert fts5_calls == []  # debounced
    assert result == []


def test_hints_fire_after_debounce_period(monkeypatch):
    """After the debounce period, hints must fire again."""
    fts5_calls = []

    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            fts5_calls.append(rq)
            return []

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        MockFTS5Stream,
    )

    repl = _make_repl_with_hints()
    repl._last_hint_at = time.time() - 60.0  # 60 seconds ago → past debounce

    repl._fetch_memory_hints("What does this mean?")
    assert len(fts5_calls) == 1


def test_hints_use_only_fts5_not_dense(monkeypatch):
    """Memory hints must ONLY use FTS5, never DenseStream or ColBERTStream."""
    dense_calls = []

    class MockDenseStream:
        def retrieve(self, rq, top_k=3):
            dense_calls.append(1)
            return []

    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            return []

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.DenseStream",
        MockDenseStream,
    )
    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        MockFTS5Stream,
    )

    repl = _make_repl_with_hints()
    repl._fetch_memory_hints("What is the meaning of resilience?")
    assert dense_calls == [], "DenseStream must not be called for memory hints"


def test_hints_not_visible_to_user(monkeypatch, capsys):
    """_fetch_memory_hints must produce zero stdout output."""
    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            return []

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        MockFTS5Stream,
    )

    repl = _make_repl_with_hints()
    repl._fetch_memory_hints("What about resilience?")

    captured = capsys.readouterr()
    assert captured.out == "", "Memory hints must produce no stdout output"


def test_hints_never_raise_on_fts5_failure(monkeypatch):
    """If FTS5 throws, _fetch_memory_hints must return [] without propagating."""
    class BrokenFTS5Stream:
        def retrieve(self, rq, top_k=3):
            raise RuntimeError("FTS5 broken!")

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        BrokenFTS5Stream,
    )

    repl = _make_repl_with_hints()
    # Must not raise
    result = repl._fetch_memory_hints("What went wrong?")
    assert result == []


def test_hints_return_texts_from_fts5(monkeypatch):
    """When FTS5 returns hits, _fetch_memory_hints must return their text."""
    from audiobench.memory.retrieval_streams import SegmentHit

    fake_hits = [
        SegmentHit(segment_id=1, start_time=0.0, end_time=5.0, text="He spoke about resilience deeply."),
        SegmentHit(segment_id=2, start_time=5.0, end_time=10.0, text="She emphasized mental fortitude."),
    ]

    class MockFTS5Stream:
        def retrieve(self, rq, top_k=3):
            return fake_hits

    monkeypatch.setattr(
        "audiobench.memory.retrieval_streams.FTS5Stream",
        MockFTS5Stream,
    )

    repl = _make_repl_with_hints()
    result = repl._fetch_memory_hints("What did he say about resilience?")

    assert len(result) == 2
    assert "resilience" in result[0].lower()

