"""Unit tests for WorkLinker intelligence task and author refinement."""

from unittest.mock import patch
import pytest
from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.work_linker import WorkLinker, _get_first_transcript_text
from audiobench.storage.models import AudioFileRecord, ExpressionRecord, TranscriptionRecord, WorkRecord


def test_get_first_transcript_text_reads_full_text(test_db):
    with get_session() as session:
        af = AudioFileRecord(file_path="/dummy/test.mp3", file_name="test.mp3")
        session.add(af)
        session.flush()

        tr = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="test.mp3",
            full_text="The world's greatest salesman. Author: Og Mandino. Narrated by Og Mandino.",
        )
        session.add(tr)
        session.commit()

        words = _get_first_transcript_text(af.id, session)
        assert words is not None
        assert "Author: Og Mandino" in words


def test_refine_missing_authors_from_transcript(test_db):
    with get_session() as session:
        work = WorkRecord(title="The Greatest Salesman in The World", author=None)
        session.add(work)
        session.flush()

        af = AudioFileRecord(
            file_path="/audio/salesman.mp3",
            file_name="salesman.mp3",
            work_id=work.id,
        )
        session.add(af)
        session.flush()

        tr = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="salesman.mp3",
            full_text="The world's greatest salesman. Author: Og Mandino. Hello world.",
        )
        session.add(tr)
        session.commit()

        linker = WorkLinker()
        refined = linker._refine_missing_authors(session)
        assert refined >= 1

        updated_work = session.get(WorkRecord, work.id)
        assert updated_work.author == "Og Mandino"


def test_refine_missing_authors_from_filename(test_db):
    with get_session() as session:
        work = WorkRecord(title="we as organisms", author=None)
        session.add(work)
        session.flush()

        af = AudioFileRecord(
            file_path="/audio/alan watts we as organisms.mp3",
            file_name="alan watts we as organisms.mp3",
            work_id=work.id,
        )
        session.add(af)
        session.flush()

        linker = WorkLinker()
        refined = linker._refine_missing_authors(session)
        assert refined >= 1

        updated_work = session.get(WorkRecord, work.id)
        assert updated_work.author == "Alan Watts"


def test_refine_missing_authors_scripture_and_prayers(test_db):
    with get_session() as session:
        w_bible = WorkRecord(title="Reading of the First Ten Psalms", author=None)
        w_prayer = WorkRecord(title="Christian Intercessory Prayers for Youth and Restoration", author=None)
        w_shipley = WorkRecord(title="The Unbreakable Mindset DJ Shipley Motiv", author=None)
        w_personal = WorkRecord(title="November 5th Evening Log", author=None)
        session.add_all([w_bible, w_prayer, w_shipley, w_personal])
        session.flush()

        af_personal = AudioFileRecord(
            file_path="/audio/November 5th Evening Log.m4a",
            file_name="November 5th Evening Log.m4a",
            work_id=w_personal.id,
        )
        session.add(af_personal)
        session.commit()

        linker = WorkLinker()
        refined = linker._refine_missing_authors(session)
        assert refined >= 4

        assert session.get(WorkRecord, w_bible.id).author == "The Holy Bible"
        assert session.get(WorkRecord, w_prayer.id).author == "Christian Prayers"
        assert session.get(WorkRecord, w_shipley.id).author == "DJ Shipley"
        assert session.get(WorkRecord, w_personal.id).author == "Personal Recording"


def test_work_linker_enqueues_chunks_for_graph_linking(test_db):
    with get_session() as session:
        af = AudioFileRecord(
            file_path="/audio/seneca_letters.mp3",
            file_name="Seneca - Letters from a Stoic.mp3",
            work_id=None,
        )
        session.add(af)
        session.flush()

        tr = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="Seneca - Letters from a Stoic.mp3",
            full_text="Letters from a stoic by Seneca. Philosophy of life and virtue.",
        )
        session.add(tr)
        session.flush()

        expr = ExpressionRecord(
            content="True happiness is to enjoy the present, without anxious dependence upon the future.",
            source_type="audio_transcript",
            source_id=tr.id,
            graph_role="sweep_chunk",
            work_id=None,
        )
        session.add(expr)
        session.commit()
        expr_id = expr.id

    with patch("audiobench.memory.graph_linker.enqueue_for_linking") as mock_enqueue:
        mock_enqueue.return_value = 1
        linker = WorkLinker()
        linker.run()

        # Check work_id was assigned
        with get_session() as session:
            updated_af = session.get(AudioFileRecord, af.id)
            assert updated_af.work_id is not None
            updated_expr = session.get(ExpressionRecord, expr_id)
            assert updated_expr.work_id == updated_af.work_id

        # Verify enqueue_for_linking was called with the expression ID
        assert mock_enqueue.call_count == 1
        enqueued_ids = mock_enqueue.call_args[0][0]
        assert expr_id in enqueued_ids


def test_reconcile_unlinked_expressions_when_audio_file_already_linked(test_db):
    """Test that expressions created after audio file is linked are reconciled and enqueued."""
    with get_session() as session:
        work = WorkRecord(title="Clean Code", author="Robert C. Martin")
        session.add(work)
        session.flush()

        af = AudioFileRecord(
            file_path="/audio/clean_code.m4b",
            file_name="clean_code.m4b",
            work_id=work.id,  # Audio file already has work_id
        )
        session.add(af)
        session.flush()

        tr = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="clean_code.m4b",
            full_text="Even bad code can function. But if code isn't clean, it can bring a development organization to its knees.",
        )
        session.add(tr)
        session.flush()

        expr = ExpressionRecord(
            content="Even bad code can function. But if code isn't clean, it can bring a development organization to its knees.",
            source_type="audio_transcript",
            source_id=tr.id,
            graph_role="sweep_chunk",
            work_id=None,  # Orphaned expression with NULL work_id
        )
        session.add(expr)
        session.commit()
        expr_id = expr.id

    with patch("audiobench.memory.graph_linker.enqueue_for_linking") as mock_enqueue:
        mock_enqueue.return_value = 1
        linker = WorkLinker()
        linker.run()

        with get_session() as session:
            updated_expr = session.get(ExpressionRecord, expr_id)
            assert updated_expr.work_id == work.id

        assert mock_enqueue.call_count == 1
        enqueued_ids = mock_enqueue.call_args[0][0]
        assert expr_id in enqueued_ids


def test_expression_repository_register_heals_null_work_id(test_db):
    """Test that ExpressionRepository.register updates work_id when existing expression has None."""
    from audiobench.storage.expression_repository import ExpressionRepository

    repo = ExpressionRepository()
    expr1 = repo.register(
        content="Any fool can write code that a computer can understand. Good programmers write code that humans can understand.",
        source_type="audio_transcript",
        source_id=101,
        graph_role="sweep_chunk",
        work_id=None,
    )
    assert expr1.work_id is None

    # Register again with a valid work_id
    expr2 = repo.register(
        content="Any fool can write code that a computer can understand. Good programmers write code that humans can understand.",
        source_type="audio_transcript",
        source_id=101,
        graph_role="sweep_chunk",
        work_id=42,
    )
    assert expr2.id == expr1.id
    assert expr2.work_id == 42



