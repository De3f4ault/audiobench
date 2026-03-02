"""Repository — CRUD operations for transcription data.

Provides a clean interface over SQLAlchemy for:
- Saving transcriptions with deduplication (by file hash)
- Querying transcription history
- Searching past transcriptions by text
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from src.audiobench.config.logging_config import get_logger
from src.audiobench.core.models import AudioMetadata, Transcript
from src.audiobench.storage.database import get_session
from src.audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord

logger = get_logger("storage.repository")


class TranscriptionRepository:
    """CRUD operations for transcription persistence."""

    def save_transcription(
        self,
        transcript: Transcript,
        audio_metadata: Optional[AudioMetadata] = None,
    ) -> int:
        """Save a transcription result to the database.

        If the same audio file (by hash) was already transcribed, the audio
        record is reused and a new transcription is linked to it.

        Args:
            transcript: The transcription result.
            audio_metadata: Source audio metadata (for dedup by hash).

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            # Find or create audio file record
            audio_record = None
            if audio_metadata and audio_metadata.file_hash:
                audio_record = (
                    session.query(AudioFileRecord)
                    .filter_by(file_hash=audio_metadata.file_hash)
                    .first()
                )

            if audio_record is None and audio_metadata:
                audio_record = AudioFileRecord(
                    file_path=audio_metadata.file_path,
                    file_name=audio_metadata.file_name,
                    file_size_bytes=audio_metadata.file_size_bytes,
                    format=audio_metadata.format,
                    duration_seconds=audio_metadata.duration_seconds,
                    sample_rate=audio_metadata.sample_rate,
                    channels=audio_metadata.channels,
                    file_hash=audio_metadata.file_hash,
                )
                session.add(audio_record)
                session.flush()  # Get the ID

            # Create transcription record
            tx_record = TranscriptionRecord(
                audio_file_id=audio_record.id if audio_record else None,
                source="file",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine=transcript.engine,
                model_name=transcript.model_name,
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                status="completed",
            )
            session.add(tx_record)
            session.flush()

            # Save segments
            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                )
                session.add(seg_record)

            session.commit()
            logger.info(
                "Saved transcription #%d (%d segments)", tx_record.id, len(transcript.segments)
            )
            return tx_record.id

    def save_live_session(self, transcript: Transcript) -> int:
        """Save a live transcription session to the database.

        Live sessions have no source audio file.

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            tx_record = TranscriptionRecord(
                audio_file_id=None,
                source="live",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine="faster-whisper",
                model_name=transcript.model_name if transcript.model_name else "base",
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                status="completed",
            )
            session.add(tx_record)
            session.flush()

            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                )
                session.add(seg_record)

            session.commit()
            logger.info(
                "Saved live session #%d (%d segments)", tx_record.id, len(transcript.segments)
            )
            return tx_record.id

    def find_by_hash(self, file_hash: str) -> Optional[TranscriptionRecord]:
        """Find an existing transcription by audio file hash (deduplication).

        Returns the most recent transcription for the given file hash, or None.
        """
        with get_session() as session:
            audio = session.query(AudioFileRecord).filter_by(file_hash=file_hash).first()
            if audio is None:
                return None

            return (
                session.query(TranscriptionRecord)
                .filter_by(audio_file_id=audio.id)
                .order_by(desc(TranscriptionRecord.created_at))
                .first()
            )

    def get_history(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Get recent transcription history.

        Returns:
            List of dicts with transcription + audio metadata.
        """
        with get_session() as session:
            records = (
                session.query(TranscriptionRecord)
                .order_by(desc(TranscriptionRecord.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )

            results = []
            for rec in records:
                audio = rec.audio_file
                if rec.source == "live":
                    label = "🎤 Live session"
                else:
                    label = audio.file_name if audio else "unknown"
                results.append(
                    {
                        "id": rec.id,
                        "file_name": label,
                        "source": rec.source,
                        "language": rec.language,
                        "model": rec.model_name,
                        "word_count": rec.word_count,
                        "duration": rec.duration_seconds,
                        "status": rec.status,
                        "created_at": rec.created_at.isoformat() if rec.created_at else "",
                        "text_preview": rec.full_text[:100] + "..."
                        if len(rec.full_text) > 100
                        else rec.full_text,
                    }
                )

            return results

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search transcriptions by text content.

        Args:
            query: Search string (case-insensitive LIKE).
            limit: Maximum number of results.

        Returns:
            List of matching transcription dicts.
        """
        with get_session() as session:
            pattern = f"%{query}%"
            records = (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.full_text.ilike(pattern))
                .order_by(desc(TranscriptionRecord.created_at))
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": rec.id,
                    "file_name": rec.audio_file.file_name if rec.audio_file else "unknown",
                    "language": rec.language,
                    "text_preview": rec.full_text[:200],
                    "created_at": rec.created_at.isoformat() if rec.created_at else "",
                }
                for rec in records
            ]

    def get_by_id(self, transcription_id: int) -> Optional[dict]:
        """Get full transcription by ID including all segments."""
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return None

            return {
                "id": rec.id,
                "file_name": (
                    "🎤 Live session"
                    if rec.source == "live"
                    else (rec.audio_file.file_name if rec.audio_file else "unknown")
                ),
                "file_path": (rec.audio_file.file_path if rec.audio_file else None),
                "source": rec.source,
                "full_text": rec.full_text,
                "language": rec.language,
                "language_probability": rec.language_probability,
                "engine": rec.engine,
                "model": rec.model_name,
                "duration": rec.duration_seconds,
                "word_count": rec.word_count,
                "segment_count": rec.segment_count,
                "status": rec.status,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
                "segments": [
                    {
                        "index": seg.segment_index,
                        "text": seg.text,
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "speaker": seg.speaker,
                    }
                    for seg in rec.segments
                ],
            }

    def update_text(self, transcription_id: int, new_text: str) -> bool:
        """Update the full text of a transcription (used by REPL .edit).

        Returns True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            rec.full_text = new_text
            rec.word_count = len(new_text.split())
            session.commit()
            logger.info("Updated text for transcription #%d", transcription_id)
            return True

    def delete_by_id(self, transcription_id: int) -> bool:
        """Delete a transcription by ID.

        Returns True if found and deleted, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            session.delete(rec)
            session.commit()
            logger.info("Deleted transcription #%d", transcription_id)
            return True

    def delete_all(self) -> int:
        """Delete all transcriptions. Returns number deleted."""
        with get_session() as session:
            count = session.query(TranscriptionRecord).count()
            session.query(SegmentRecord).delete()
            session.query(TranscriptionRecord).delete()
            session.commit()
            logger.info("Deleted %d transcription(s)", count)
            return count
