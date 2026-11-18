import json
import subprocess
import tempfile
from pathlib import Path
import pytest

from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.transcription_result import AudioMetadata, Segment, Transcript
from audiobench.storage.models import AudioFileRecord, ChapterRecord, SegmentRecord, TranscriptionRecord
from audiobench.core.db_session import get_session


def test_real_audio_import_and_chapter_linkage_e2e(tmp_path: Path):
    """End-to-end integration test: Real audio file with embedded chapters processed through save_transcription()."""
    audio_path = tmp_path / "test_audiobook.m4b"
    meta_path = tmp_path / "ffmetadata.txt"

    # Create real FFMETADATA with 2 chapters
    meta_path.write_text(""";FFMETADATA1
title=Test Audiobook E2E

[CHAPTER]
TIMEBASE=1/1000
START=0
END=5000
title=Chapter 1: The Beginning

[CHAPTER]
TIMEBASE=1/1000
START=5000
END=10000
title=Chapter 2: The Journey
""")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-i", str(meta_path),
        "-map_metadata", "1",
        "-t", "10",
        "-c:a", "aac",
        str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Build AudioMetadata
    file_hash = AudioMetadata.compute_file_hash(audio_path)
    audio_meta = AudioMetadata(
        file_path=str(audio_path),
        file_name=audio_path.name,
        file_size_bytes=audio_path.stat().st_size,
        format="m4b",
        duration_seconds=10.0,
        sample_rate=16000,
        channels=1,
        file_hash=file_hash,
    )

    # Build Transcript with segments spanning both chapters
    segs = [
        Segment(id=0, text="Welcome to the beginning of this book.", start=0.5, end=3.0),   # mid 1.75 -> Ch 1
        Segment(id=1, text="Here is the rest of chapter one.", start=3.2, end=4.8),        # mid 4.00 -> Ch 1
        Segment(id=2, text="Now we enter chapter two of the story.", start=5.2, end=8.0),  # mid 6.60 -> Ch 2
        Segment(id=3, text="Final words of the second chapter.", start=8.2, end=9.8),      # mid 9.00 -> Ch 2
    ]
    transcript = Transcript(
        segments=segs,
        language="en",
        language_probability=1.0,
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3-turbo",
    )

    repo = TranscriptionRepository()

    # Call save_transcription without chapter_id
    tx_id = repo.save_transcription(transcript, audio_metadata=audio_meta, chapter_id=None)

    with get_session() as session:
        tx_rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
        assert tx_rec is not None
        af_id = tx_rec.audio_file_id
        assert af_id is not None

        # 1. Verify chapters were auto-detected and saved
        chaps = (
            session.query(ChapterRecord)
            .filter_by(audio_file_id=af_id)
            .order_by(ChapterRecord.chapter_index.asc())
            .all()
        )
        assert len(chaps) == 2
        assert "Chapter 1" in chaps[0].title
        assert "Chapter 2" in chaps[1].title
        assert chaps[0].start_time == 0.0
        assert chaps[0].end_time == 5.0
        assert chaps[1].start_time == 5.0
        assert chaps[1].end_time == 10.0

        # 2. Verify all segments are linked to the correct chapters
        saved_segs = (
            session.query(SegmentRecord)
            .filter_by(transcription_id=tx_id)
            .order_by(SegmentRecord.segment_index.asc())
            .all()
        )
        assert len(saved_segs) == 4

        # Zero unlinked segments
        assert all(s.chapter_id is not None for s in saved_segs)

        # Precise chapter mapping
        assert saved_segs[0].chapter_id == chaps[0].id
        assert saved_segs[1].chapter_id == chaps[0].id
        assert saved_segs[2].chapter_id == chaps[1].id
        assert saved_segs[3].chapter_id == chaps[1].id

        print("Real audio import & chapter linkage E2E test passed successfully!")
