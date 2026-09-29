"""Tests for StagingCartItem model."""

from __future__ import annotations

from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, StagingCartItem


def test_staging_cart_item_lifecycle(test_db):
    """Test creating, updating, and deleting StagingCartItem records."""
    with get_session() as session:
        # Create a parent AudioFileRecord
        audio = AudioFileRecord(
            file_path="/fake/path/audio.mp3",
            file_name="audio.mp3",
            file_size_bytes=1024,
            format="mp3",
        )
        session.add(audio)
        session.commit()

        audio_id = audio.id

        # Add to cart
        cart_item = StagingCartItem(
            audio_file_id=audio_id, engine="gemini", model_name="pro", speed_preset="fast"
        )
        session.add(cart_item)
        session.commit()

        # Query and verify
        fetched = session.query(StagingCartItem).filter_by(audio_file_id=audio_id).first()
        assert fetched is not None
        assert fetched.engine == "gemini"
        assert fetched.model_name == "pro"
        assert fetched.speed_preset == "fast"
        assert fetched.audio_file.file_name == "audio.mp3"

        # Test cascade delete
        session.delete(fetched.audio_file)
        session.commit()

        # Cart item should be deleted
        assert session.query(StagingCartItem).count() == 0

# test_job_queue_item_lifecycle removed 2026-09-29: JobQueueItem (System 2)
# retired — job_queue table had zero writes since 2026-08-08.
