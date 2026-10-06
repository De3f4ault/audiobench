"""Tests for ChapterEmbedder daemon intelligence task."""

from unittest.mock import MagicMock
import pytest
from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.chapter_embedder import ChapterEmbedder
from audiobench.storage.models import AudioFileRecord, ChapterRecord, SegmentRecord, TranscriptionRecord, WorkRecord


def test_chapter_embedder_lifecycle():
    mock_store = MagicMock()
    embedder = ChapterEmbedder(memory_store=mock_store)

    with get_session() as session:
        # Create a work and file
        work = WorkRecord(title="Meditations", author="Marcus Aurelius")
        session.add(work)
        session.flush()

        af = AudioFileRecord(
            file_path="/tmp/meditations.mp3",
            file_name="meditations.mp3",
            format="mp3",
            work_id=work.id,
        )
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=af.id,
            source="test",
            full_text="Meditations transcript",
        )
        session.add(tx)
        session.flush()

        # Chapter needing embedding
        chap = ChapterRecord(
            audio_file_id=af.id,
            chapter_index=0,
            title="Book I",
            start_time=0.0,
            end_time=100.0,
            last_embedded_at=None,
        )
        session.add(chap)
        session.flush()

        # Segment with text
        seg = SegmentRecord(
            transcription_id=tx.id,
            segment_index=0,
            text="From my grandfather Verus I learned good morals and the government of my temper.",
            start_time=0.0,
            end_time=10.0,
            chapter_id=chap.id,
        )
        session.add(seg)
        session.commit()

        tx_id = tx.id
        chap_id = chap.id
        work_id = work.id
        af_id = af.id

    # Run one sync cycle
    embedder._run_sync()

    # Verify write_node was called on mock_store
    assert mock_store.write_node.called
    kwargs = mock_store.write_node.call_args.kwargs
    assert kwargs["source_type"] == "chapter_summary"
    assert kwargs["chapter_id"] == chap_id
    assert kwargs["author"] == "Marcus Aurelius"
    assert "Author: Marcus Aurelius" in kwargs["content"]
    assert "Book I" in kwargs["content"]
    assert "grandfather Verus" in kwargs["content"]

    # Verify chapter dirty flag was cleared (last_embedded_at is not None)
    with get_session() as session:
        updated_chap = session.query(ChapterRecord).filter_by(id=chap_id).first()
        assert updated_chap.last_embedded_at is not None

        # Cleanup
        session.delete(seg)
        session.delete(updated_chap)
        tx_row = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
        if tx_row:
            session.delete(tx_row)
        session.delete(session.query(AudioFileRecord).filter_by(id=af_id).first())
        session.delete(session.query(WorkRecord).filter_by(id=work_id).first())
        session.commit()


def test_get_memory_store_singleton(monkeypatch):
    import audiobench.memory.singletons as singletons
    from audiobench.memory.singletons import get_memory_store, get_segment_store

    mock_ms = MagicMock()
    monkeypatch.setattr(singletons, "_memory_store", mock_ms)
    assert get_memory_store() is mock_ms

    mock_ss = MagicMock()
    monkeypatch.setattr(singletons, "_segment_store", mock_ss)
    assert get_segment_store() is mock_ss


def test_chapter_embedder_fallback_to_singleton(monkeypatch):
    import audiobench.memory.singletons as singletons
    mock_ms = MagicMock()
    monkeypatch.setattr(singletons, "_memory_store", mock_ms)

    embedder = ChapterEmbedder()
    store = embedder._get_store()
    assert store is mock_ms

