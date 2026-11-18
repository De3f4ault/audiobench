"""Tests for retrieval streams."""

import pytest

from audiobench.memory.query_reformulator import ReformulatedQuery
from audiobench.memory.retrieval_streams import (
    ColBERTStream,
    DenseStream,
    FTS5Stream,
    RetrievalStream,
)


@pytest.fixture
def any_rq():
    return ReformulatedQuery(
        original="test",
        bm25_keywords="test",
        semantic_query="test",
        hyde_anchor="test",
    )


@pytest.fixture
def db_with_fts5(test_db):
    """Seed test_db with some segment data."""
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord

    with get_session() as session:
        audio = AudioFileRecord(file_path="stream.mp3", file_name="stream.mp3", duration_seconds=100)
        session.add(audio)
        session.commit()

        tx = TranscriptionRecord(audio_file_id=audio.id, file_name="stream.mp3", model_name="x", language="en")
        session.add(tx)
        session.commit()

        # Add segments
        s1 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=1,
            start_time=0.0,
            end_time=10.0,
            text="The quick brown fox jumps over the lazy dog"
        )
        s2 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=2,
            start_time=10.0,
            end_time=20.0,
            text="A completely unrelated sentence about spaceships"
        )
        session.add_all([s1, s2])
        session.commit()

        # We need to manually sync FTS5 in tests if triggers are not active, or just insert it
        # Actually our migrations have triggers that auto-update FTS5 on insert to segments!
        # Let's hope the trigger works in sqlite test.db.


@pytest.fixture
def broken_db_connection(monkeypatch):
    """Mocks get_session to raise an exception."""
    def mock_get_session():
        raise RuntimeError("Database connection lost")
    # Actually get_session is a context manager, so we need a magic mock or just a class
    class BrokenSession:
        def __enter__(self):
            raise RuntimeError("Database connection lost")
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    monkeypatch.setattr("audiobench.memory.retrieval_streams.get_session", lambda: BrokenSession())


@pytest.fixture
def query_counter():
    from sqlalchemy import event

    from audiobench.core.db_engine import _engine

    class QueryCounter:
        def __init__(self):
            self.count = 0

        def __enter__(self):
            self.count = 0
            event.listen(_engine, "before_cursor_execute", self.callback)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            event.remove(_engine, "before_cursor_execute", self.callback)

        def callback(self, conn, cursor, statement, parameters, context, executemany):
            self.count += 1

    return QueryCounter()


def test_fts5_stream_implements_protocol():
    assert isinstance(FTS5Stream(), RetrievalStream)


def test_dense_stream_implements_protocol():
    assert isinstance(DenseStream(), RetrievalStream)


def test_colbert_stream_implements_protocol():
    assert isinstance(ColBERTStream(), RetrievalStream)


def test_fts5_stream_returns_segment_hits_with_timestamps(db_with_fts5):
    rq = ReformulatedQuery(
        original="fox jumps",
        bm25_keywords="fox jumps",
        semantic_query="fox jumps",
        hyde_anchor="fox jumps",
    )
    hits = FTS5Stream().retrieve(rq, top_k=5)
    assert len(hits) > 0
    assert len(hits) <= 5
    for hit in hits:
        assert hit.segment_id > 0
        assert hit.start_time >= 0.0
        assert hit.end_time > hit.start_time
        assert hit.bm25_score < 0  # FTS5 negative convention
        assert len(hit.text) > 0


def test_fts5_stream_returns_empty_on_no_match(db_with_fts5):
    rq = ReformulatedQuery(
        original="xyzzyquux_impossible_word",
        bm25_keywords="xyzzyquux_impossible_word",
        semantic_query="x",
        hyde_anchor="x"
    )
    hits = FTS5Stream().retrieve(rq, top_k=5)
    assert hits == []


def test_stream_retrieve_never_raises(broken_db_connection, any_rq, monkeypatch):
    """Every stream must catch its own errors and return empty list. Never propagate."""
    # Ensure DenseStream and ColBERTStream do not block on a live daemon socket
    class BrokenDaemon:
        def embed_query(self, *args, **kwargs):
            raise RuntimeError("Daemon inference engine unavailable")

        def rerank(self, *args, **kwargs):
            raise RuntimeError("Daemon reranker unavailable")

    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: BrokenDaemon())

    assert FTS5Stream().retrieve(any_rq, top_k=5) == []
    assert DenseStream().retrieve(any_rq, top_k=5) == []
    assert ColBERTStream().retrieve(any_rq, top_k=5) == []


def test_fts5_stream_uses_single_sql_query(db_with_fts5, query_counter):
    rq = ReformulatedQuery(
        original="fox",
        bm25_keywords="fox",
        semantic_query="fox",
        hyde_anchor="fox",
    )
    with query_counter as qc:
        FTS5Stream().retrieve(rq, top_k=10)
    assert qc.count == 1
