
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
