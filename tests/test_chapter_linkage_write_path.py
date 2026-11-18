import pytest
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.transcription_result import Segment, Transcript
from audiobench.storage.models import AudioFileRecord, ChapterRecord, TranscriptionRecord, SegmentRecord
from audiobench.core.db_session import get_session

def test_resolve_segment_chapters_single_chapter():
    repo = TranscriptionRepository()
    with get_session() as session:
        # Create dummy audio file
        af = AudioFileRecord(
            file_path="/tmp/dummy_single.mp3",
            file_name="dummy_single.mp3",
            file_size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            file_hash="dummy_hash_single_001",
        )
        session.add(af)
        session.flush()

        ch = ChapterRecord(
            audio_file_id=af.id,
            chapter_index=0,
            title="Single Chapter",
            start_time=0.0,
            end_time=60.0,
            is_ghost=0,
            transcription_status="pending",
        )
        session.add(ch)
        session.flush()

        segs = [
            Segment(id=0, text="First segment", start=0.0, end=10.0),
            Segment(id=1, text="Second segment", start=10.0, end=30.0),
            Segment(id=2, text="Third segment", start=30.0, end=60.0),
        ]

        resolved = repo._resolve_segment_chapters(segs, af.id, session)
        assert len(resolved) == 3
        assert all(cid == ch.id for cid in resolved)

        # Cleanup
        session.query(ChapterRecord).filter_by(audio_file_id=af.id).delete()
        session.query(AudioFileRecord).filter_by(id=af.id).delete()
        session.commit()

def test_resolve_segment_chapters_multi_chapter():
    repo = TranscriptionRepository()
    with get_session() as session:
        af = AudioFileRecord(
            file_path="/tmp/dummy_multi.mp3",
            file_name="dummy_multi.mp3",
            file_size_bytes=2000,
            format="mp3",
            duration_seconds=120.0,
            sample_rate=16000,
            channels=1,
            file_hash="dummy_hash_multi_002",
        )
        session.add(af)
        session.flush()

        ch1 = ChapterRecord(
            audio_file_id=af.id,
            chapter_index=0,
            title="Chapter 1",
            start_time=0.0,
            end_time=50.0,
            is_ghost=0,
            transcription_status="pending",
        )
        ch2 = ChapterRecord(
            audio_file_id=af.id,
            chapter_index=1,
            title="Chapter 2",
            start_time=50.0,
            end_time=120.0,
            is_ghost=0,
            transcription_status="pending",
        )
        session.add_all([ch1, ch2])
        session.flush()

        segs = [
            Segment(id=0, text="Intro", start=0.0, end=20.0),       # Midpoint 10.0 -> ch1
            Segment(id=1, text="Mid chapter 1", start=25.0, end=45.0), # Midpoint 35.0 -> ch1
            Segment(id=2, text="Start chapter 2", start=52.0, end=80.0), # Midpoint 66.0 -> ch2
            Segment(id=3, text="Tail chapter 2", start=110.0, end=125.0), # Tail bound -> ch2
        ]

        resolved = repo._resolve_segment_chapters(segs, af.id, session)
        assert len(resolved) == 4
        assert resolved[0] == ch1.id
        assert resolved[1] == ch1.id
        assert resolved[2] == ch2.id
        assert resolved[3] == ch2.id

        # Cleanup
        session.query(ChapterRecord).filter_by(audio_file_id=af.id).delete()
        session.query(AudioFileRecord).filter_by(id=af.id).delete()
        session.commit()
