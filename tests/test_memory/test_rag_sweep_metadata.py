
import lancedb
import pytest

import audiobench.daemon.sweep_state as sweep_state_mod
from audiobench.core.db_session import get_session
from audiobench.daemon import server
from audiobench.daemon.server import _do_sweep_once
from audiobench.daemon.sweep_state import get_sweep_state, init_sweep_state
from audiobench.memory.memory_store import MemoryStore
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord, WorkRecord


@pytest.fixture(autouse=True)
def setup_sweep_state():
    """Reset the sweep state singleton before each test."""
    sweep_state_mod._state = None
    state = init_sweep_state()
    # Mock known hashes to avoid real embedding if possible, but actually we need real embedding to test LanceDB.
    # So we let it run normally.
    yield state
    sweep_state_mod._state = None

def test_work_id_present_in_lancedb_metadata(test_db, tmp_path, monkeypatch):
    from audiobench.core import settings
    # Override settings so LanceDB writes to tmp_path
    monkeypatch.setattr(settings.get_settings(), "data_dir", str(tmp_path))

    # Initialize MemoryStore so it creates the LanceDB table
    server._memory_store = MemoryStore()

    with get_session() as session:
        wr = WorkRecord(id=42, title="Test Work", author="Author")
        session.add(wr)
        session.flush()

        af = AudioFileRecord(file_path="test.mp3", file_name="test.mp3", work_id=42)
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(audio_file_id=af.id, full_text="Hello world test expression", language="en", language_probability=0.99)
        session.add(tx)
        session.flush()

        tx_id = tx.id
        session.commit()

    state = get_sweep_state()
    state.push_transcript(tx_id)

    # Run sweep
    _do_sweep_once()

    # Verify LanceDB
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = db.open_table("expressions")
    results = table.search().to_list()

    assert len(results) >= 1

    # We should have one expression from the transcript.
    # Check that work_id is 42
    expr_node = results[0]
    assert expr_node.get("work_id") == 42
    assert expr_node.get("audio_file_id") == af.id
    assert expr_node.get("confidence") == 0.99
    assert expr_node.get("original_language") == "en"

def test_null_work_id_stored_as_null_not_string(test_db, tmp_path, monkeypatch):
    from audiobench.core import settings
    monkeypatch.setattr(settings.get_settings(), "data_dir", str(tmp_path))

    # Initialize MemoryStore so it creates the LanceDB table
    server._memory_store = MemoryStore()

    with get_session() as session:
        af = AudioFileRecord(file_path="test2.mp3", file_name="test2.mp3", work_id=None)
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(audio_file_id=af.id, full_text="Hello world test expression null work", language="en", language_probability=0.95)
        session.add(tx)
        session.flush()

        tx_id = tx.id
        session.commit()

    state = get_sweep_state()
    state.push_transcript(tx_id)

    # Run sweep
    _do_sweep_once()

    # Verify LanceDB
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = db.open_table("expressions")

    results = table.search().where(f"audio_file_id = {af.id}").to_list()
    assert len(results) >= 1

    expr_node = results[0]
    assert expr_node.get("work_id") is None
    assert expr_node.get("audio_file_id") == af.id
    assert expr_node.get("confidence") == 0.95
    assert expr_node.get("original_language") == "en"


def test_sweep_populates_expression_segment_map_and_chapter_id(test_db, tmp_path, monkeypatch):
    """Test that _do_sweep_once links physical segments into expression_segment_map and assigns chapter_id."""
    from unittest.mock import MagicMock
    from audiobench.core import settings
    from audiobench.storage.models import ChapterRecord, ExpressionRecord, ExpressionSegmentMap, SegmentRecord
    monkeypatch.setattr(settings.get_settings(), "data_dir", str(tmp_path))

    server._memory_store = MemoryStore()
    server._segment_store = MagicMock()

    with get_session() as session:
        af = AudioFileRecord(file_path="test_audio.mp3", file_name="test_audio.mp3")
        session.add(af)
        session.flush()

        chap1 = ChapterRecord(audio_file_id=af.id, chapter_index=0, title="Intro Chapter", start_time=0.0, end_time=30.0)
        chap2 = ChapterRecord(audio_file_id=af.id, chapter_index=1, title="Second Chapter", start_time=30.0, end_time=60.0)
        session.add_all([chap1, chap2])
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=af.id,
            full_text="First chapter speech text here. Second chapter content continues here.",
            language="en",
            language_probability=0.98,
        )
        session.add(tx)
        session.flush()

        seg1 = SegmentRecord(transcription_id=tx.id, segment_index=0, text="First chapter speech text here.", start_time=5.0, end_time=15.0)
        seg2 = SegmentRecord(transcription_id=tx.id, segment_index=1, text="Second chapter content continues here.", start_time=35.0, end_time=45.0)
        session.add_all([seg1, seg2])
        session.commit()

        tx_id = tx.id
        chap1_id = chap1.id
        seg1_id = seg1.id

    state = get_sweep_state()
    state.push_transcript(tx_id)

    _do_sweep_once()

    with get_session() as session:
        t3_exprs = session.query(ExpressionRecord).filter_by(source_id=tx_id, graph_role="sweep_chunk").all()
        assert len(t3_exprs) >= 1

        # Check expression_segment_map
        esm_links = session.query(ExpressionSegmentMap).all()
        assert len(esm_links) >= 1

        # Verify chapter_id is assigned on SQLite expression
        for e in t3_exprs:
            assert e.chapter_id is not None

    # Verify LanceDB has chapter_id and chapter_title
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = db.open_table("expressions")
    results = table.search().where(f"audio_file_id = {af.id}").to_list()
    assert len(results) >= 1
    for r in results:
        assert r.get("chapter_id") is not None
        assert r.get("chapter_title") is not None

