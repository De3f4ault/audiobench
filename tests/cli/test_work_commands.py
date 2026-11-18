import pytest
from click.testing import CliRunner

from audiobench.cli.commands.work_cmd import work_group
from audiobench.core.db_session import get_session
from audiobench.events import get_bus
from audiobench.storage.models import (
    AudioFileRecord,
    ExpressionRecord,
    TranscriptionRecord,
    WorkRecord,
)


@pytest.fixture
def runner():
    return CliRunner()

def test_work_create_inserts_row(test_db, runner):
    result = runner.invoke(work_group, ["create", "--title", "Being and Time", "--author", "Martin Heidegger"])
    assert result.exit_code == 0
    assert "Created work" in result.output

    with get_session() as session:
        works = session.query(WorkRecord).all()
        assert len(works) == 1
        assert works[0].title == "Being and Time"
        assert works[0].author == "Martin Heidegger"

def test_work_list_shows_all_works(test_db, runner):
    with get_session() as session:
        session.add(WorkRecord(title="Work 1"))
        session.add(WorkRecord(title="Work 2"))
        session.add(WorkRecord(title="Work 3"))
        session.commit()

    result = runner.invoke(work_group, ["list"])
    assert result.exit_code == 0
    assert "Work 1" in result.output
    assert "Work 2" in result.output
    assert "Work 3" in result.output
    # 3 works shown
    assert result.output.count("Work ") >= 3

def test_work_assign_updates_audio_file(test_db, runner):
    with get_session() as session:
        wr = WorkRecord(id=42, title="Test Work")
        af = AudioFileRecord(file_path="test.mp3", file_name="test.mp3")
        session.add(wr)
        session.add(af)
        session.flush()
        af_id = af.id
        session.commit()

    result = runner.invoke(work_group, ["assign", str(af_id), "42"])
    assert result.exit_code == 0
    assert "Assigned 1 audio file(s)" in result.output

    with get_session() as session:
        af = session.query(AudioFileRecord).filter_by(id=af_id).first()
        assert af.work_id == 42

def test_work_assign_propagates_to_expressions(test_db, runner):
    with get_session() as session:
        wr = WorkRecord(id=1, title="Test Work")
        af = AudioFileRecord(file_path="test.mp3", file_name="test.mp3")
        session.add(wr)
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(audio_file_id=af.id)
        session.add(tx)
        session.flush()

        ex = ExpressionRecord(content="test", source_type="audio_transcript", source_id=tx.id)
        session.add(ex)
        session.flush()
        af_id = af.id
        ex_id = ex.id
        session.commit()

    result = runner.invoke(work_group, ["assign", str(af_id), "1"])
    assert result.exit_code == 0

    with get_session() as session:
        ex = session.query(ExpressionRecord).filter_by(id=ex_id).first()
        assert ex.work_id == 1

def test_work_assign_emits_work_assigned_event(test_db, runner):
    with get_session() as session:
        wr = WorkRecord(id=1, title="Test Work")
        af = AudioFileRecord(file_path="test.mp3", file_name="test.mp3")
        session.add(wr)
        session.add(af)
        session.flush()
        af_id = af.id
        session.commit()

    events_caught = []
    def handler(**evt):
        events_caught.append(evt)

    get_bus().on("work_assigned", handler)

    result = runner.invoke(work_group, ["assign", str(af_id), "1"])
    assert result.exit_code == 0

    assert len(events_caught) == 1
    assert events_caught[0]["audio_file_id"] == af_id
    assert events_caught[0]["work_id"] == 1

def test_work_unassigned_lists_only_null_work_id(test_db, runner):
    with get_session() as session:
        wr = WorkRecord(id=1, title="Test Work")
        af1 = AudioFileRecord(file_path="test1.mp3", file_name="test1.mp3", work_id=1)
        af2 = AudioFileRecord(file_path="test2.mp3", file_name="test2.mp3", work_id=1)
        af3 = AudioFileRecord(file_path="test3.mp3", file_name="test3.mp3")
        session.add(wr)
        session.add(af1)
        session.add(af2)
        session.add(af3)
        session.commit()

    result = runner.invoke(work_group, ["unassigned"])
    assert result.exit_code == 0
    assert "test3.mp3" in result.output
    assert "test1.mp3" not in result.output
    assert "test2.mp3" not in result.output
