"""Tests for ChapterRepository and get_ordered_chapters ordering logic."""

import pytest
from audiobench.core.db_session import get_session
from audiobench.storage.chapter_repository import get_chapter_repo
from audiobench.storage.models import AudioFileRecord, ChapterRecord, WorkRecord


def test_get_ordered_chapters():
    repo = get_chapter_repo()

    with get_session() as session:
        # Create a test work
        work = WorkRecord(title="Test Work For Chapters", author="Test Author")
        session.add(work)
        session.flush()

        # File A: sequence_key = "002"
        file_a = AudioFileRecord(
            file_path="/tmp/test_a.mp3",
            file_name="b_file.mp3",
            format="mp3",
            sequence_key="002",
            work_id=work.id,
        )
        # File B: sequence_key = "001"
        file_b = AudioFileRecord(
            file_path="/tmp/test_b.mp3",
            file_name="z_file.mp3",
            format="mp3",
            sequence_key="001",
            work_id=work.id,
        )
        # File C: sequence_key = None, file_name = "alpha.mp3"
        file_c = AudioFileRecord(
            file_path="/tmp/test_c.mp3",
            file_name="alpha.mp3",
            format="mp3",
            sequence_key=None,
            work_id=work.id,
        )
        session.add_all([file_a, file_b, file_c])
        session.flush()

        # Chapters for File A (idx 0, 1)
        ch_a0 = ChapterRecord(
            audio_file_id=file_a.id,
            chapter_index=0,
            title="Track 2 Part 1",
            start_time=0.0,
            end_time=10.0,
        )
        ch_a1 = ChapterRecord(
            audio_file_id=file_a.id,
            chapter_index=1,
            title="Track 2 Part 2",
            start_time=10.0,
            end_time=20.0,
        )

        # Chapter for File B (idx 0)
        ch_b0 = ChapterRecord(
            audio_file_id=file_b.id,
            chapter_index=0,
            title="Track 1 Part 1",
            start_time=0.0,
            end_time=15.0,
        )

        # Chapter for File C (idx 0, unordered)
        ch_c0 = ChapterRecord(
            audio_file_id=file_c.id,
            chapter_index=0,
            title="Unordered Chapter",
            start_time=0.0,
            end_time=30.0,
        )

        session.add_all([ch_a0, ch_a1, ch_b0, ch_c0])
        session.commit()

        # Query ordered chapters for this work
        ordered = repo.get_ordered_chapters(work_id=work.id)

        titles = [c.title for c in ordered]
        # Expected order:
        # 1. sequence_key="001" (File B) -> Track 1 Part 1
        # 2. sequence_key="002" (File A, ch 0) -> Track 2 Part 1
        # 3. sequence_key="002" (File A, ch 1) -> Track 2 Part 2
        # 4. sequence_key=None (File C) -> Unordered Chapter
        assert titles == [
            "Track 1 Part 1",
            "Track 2 Part 1",
            "Track 2 Part 2",
            "Unordered Chapter",
        ]

        # Query scoped to file_a only
        scoped_a = repo.get_ordered_chapters(audio_file_id=file_a.id)
        assert [c.title for c in scoped_a] == ["Track 2 Part 1", "Track 2 Part 2"]

        # Cleanup
        session.delete(ch_a0)
        session.delete(ch_a1)
        session.delete(ch_b0)
        session.delete(ch_c0)
        session.delete(file_a)
        session.delete(file_b)
        session.delete(file_c)
        session.delete(work)
        session.commit()


def test_dirty_flag_trigger_on_summary_and_title_update():
    from datetime import UTC, datetime
    repo = get_chapter_repo()

    with get_session() as session:
        af = AudioFileRecord(
            file_path="/tmp/dirty_test.mp3",
            file_name="dirty_test.mp3",
            format="mp3",
        )
        session.add(af)
        session.flush()

        ch = ChapterRecord(
            audio_file_id=af.id,
            chapter_index=0,
            title="Initial Title",
            start_time=0.0,
            end_time=50.0,
            last_embedded_at=datetime.now(UTC),  # Clean state
        )
        session.add(ch)
        session.commit()
        ch_id = ch.id
        af_id = af.id

    # 1. Update summary -> triggers dirty flag (last_embedded_at = None)
    assert repo.update_chapter_summary(ch_id, "New summary text") is True
    with get_session() as session:
        rec = session.query(ChapterRecord).filter_by(id=ch_id).first()
        assert rec.summary == "New summary text"
        assert rec.last_embedded_at is None

        # Reset to clean state
        rec.last_embedded_at = datetime.now(UTC)
        session.commit()

    # 2. Update title -> triggers dirty flag (last_embedded_at = None)
    assert repo.update_chapter_title(ch_id, "Updated Title") is True
    with get_session() as session:
        rec = session.query(ChapterRecord).filter_by(id=ch_id).first()
        assert rec.title == "Updated Title"
        assert rec.last_embedded_at is None

        # Cleanup
        session.delete(rec)
        session.delete(session.query(AudioFileRecord).filter_by(id=af_id).first())
        session.commit()
