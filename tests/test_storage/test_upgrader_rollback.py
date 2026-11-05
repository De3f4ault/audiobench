"""Unit and integration tests for TranscriptUpgrader and rollback mechanics.

Verifies:
  - In-database archival and atomic replacement with midpoint chapter linking
  - Single-record rollback restoring exact segment timestamps, texts, and original IDs
  - Adjacent records in the same batch untouched by single rollback
  - Batch rollback reversing all changes cleanly
  - Referential integrity & tested invariant: chat_messages and note_captures
    segment_id NULLed on upgrade, body preserved, and still NULL on rollback (Option 1)
  - Stratified post-hoc audit sampling across library duration strata
"""

from __future__ import annotations

from datetime import UTC, datetime
import pytest

from audiobench.core.db_session import get_session
from audiobench.storage.models import (
    ArchivedSegmentRecord,
    ArchivedTranscriptionRecord,
    AudioFileRecord,
    BookmarkRecord,
    ChapterRecord,
    ChatConversation,
    ChatMessage,
    NoteCapture,
    NoteCollection,
    SegmentRecord,
    TranscriptionRecord,
    WorkRecord,
)
from audiobench.transcribe.upgrader import TranscriptUpgrader


@pytest.fixture
def upgrader_env(test_db):
    """Set up two audio files with chapters, transcriptions, segments, and attached annotations."""
    with get_session() as session:
        # File 1: Has chapters, segments, note capture, and chat message attached
        af1 = AudioFileRecord(
            file_name="Meditations - Marcus Aurelius.mp3",
            file_path="/audio/meditations.mp3",
            file_hash="hash_aurelius_12345",
            duration_seconds=3600.0,
        )
        # File 2: Second audio file in same batch
        af2 = AudioFileRecord(
            file_name="Letters from a Stoic - Seneca.mp3",
            file_path="/audio/seneca.mp3",
            file_hash="hash_seneca_67890",
            duration_seconds=1800.0,
        )
        session.add_all([af1, af2])
        session.flush()

        # Chapters for af1
        c1 = ChapterRecord(
            audio_file_id=af1.id,
            chapter_index=0,
            title="Book 1",
            start_time=0.0,
            end_time=1800.0,
            is_ghost=0,
        )
        c2 = ChapterRecord(
            audio_file_id=af1.id,
            chapter_index=1,
            title="Book 2",
            start_time=1800.0,
            end_time=3600.0,
            is_ghost=0,
        )
        session.add_all([c1, c2])
        session.flush()

        # Existing transcription 1
        tx1 = TranscriptionRecord(
            audio_file_id=af1.id,
            file_name=af1.file_name,
            full_text="Old coarse text for Marcus Aurelius.",
            engine="faster-whisper",
            model_name="small",
            duration_seconds=3600.0,
            word_count=6,
            segment_count=2,
        )
        session.add(tx1)
        session.flush()

        # Segments for tx1
        seg1 = SegmentRecord(
            transcription_id=tx1.id,
            segment_index=0,
            text="Old coarse text",
            start_time=0.0,
            end_time=1200.0,
            chapter_id=c1.id,
        )
        seg2 = SegmentRecord(
            transcription_id=tx1.id,
            segment_index=1,
            text="for Marcus Aurelius.",
            start_time=1200.0,
            end_time=2400.0,
            chapter_id=c1.id,
        )
        session.add_all([seg1, seg2])
        session.flush()

        # Existing transcription 2
        tx2 = TranscriptionRecord(
            audio_file_id=af2.id,
            file_name=af2.file_name,
            full_text="Old Seneca text.",
            engine="faster-whisper",
            model_name="small",
            duration_seconds=1800.0,
            word_count=3,
            segment_count=1,
        )
        session.add(tx2)
        session.flush()

        seg3 = SegmentRecord(
            transcription_id=tx2.id,
            segment_index=0,
            text="Old Seneca text.",
            start_time=0.0,
            end_time=1800.0,
        )
        session.add(seg3)
        session.flush()

        # Set chapter transcription_id
        c1.transcription_id = tx1.id
        c2.transcription_id = tx1.id

        # Attach NoteCapture and ChatMessage to seg1
        coll = NoteCollection(title="Philosophy Notes", audio_file_id=af1.id)
        session.add(coll)
        session.flush()

        note = NoteCapture(
            collection_id=coll.id,
            segment_id=seg1.id,
            body="Crucial passage on duty and virtue.",
        )
        conv = ChatConversation(title="Study chat")
        session.add(conv)
        session.flush()

        msg = ChatMessage(
            conversation_id=conv.id,
            role="user",
            content="Can you explain this segment?",
            segment_id=seg1.id,
        )

        bm = BookmarkRecord(
            audio_file_id=af1.id,
            transcription_id=tx1.id,
            timestamp=120.0,
            name="Resume Position",
        )
        session.add_all([note, msg, bm])
        session.commit()

    return test_db


def test_upgrade_and_archive(upgrader_env):
    """Test that upgrade_record archives the old transcription and creates new segments with chapter linkage."""
    batch_id = "test_batch_001"
    incoming_data = {
        "text": "From my grandfather Verus I learned good morals and the government of my temper. From my mother piety.",
        "engine": "whisper",
        "model_name": "large-v3-turbo",
        "duration_seconds": 3600.0,
        "segments": [
            {
                "start": 0.0,
                "end": 30.0,
                "text": "From my grandfather Verus I learned good morals and the government of my temper.",
            },
            {
                "start": 1900.0,
                "end": 1930.0,
                "text": "From my mother piety.",
            },
        ],
    }

    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        # Find af1
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()
        old_tx = session.query(TranscriptionRecord).filter_by(audio_file_id=af1.id).one()
        old_tx_id = old_tx.id
        old_seg_ids = [s.id for s in old_tx.segments]

        res = upgrader.upgrade_record(
            audio_file_id=af1.id,
            incoming_data=incoming_data,
            upgrade_batch_id=batch_id,
            keep_chapters=True,
        )
        session.commit()

        assert res.old_transcription_id == old_tx_id
        assert res.new_transcription_id != old_tx_id
        assert res.new_segments_count == 2
        assert res.archived_segments_count == 2
        assert res.chapters_linked_count == 2

        # Verify archive table
        arch = session.query(ArchivedTranscriptionRecord).filter_by(upgrade_batch_id=batch_id).one()
        assert arch.original_transcription_id == old_tx_id
        assert arch.model_name == "small"
        assert len(arch.segments) == 2
        assert set(s.original_segment_id for s in arch.segments) == set(old_seg_ids)

        # Verify live new transcription
        new_tx = session.query(TranscriptionRecord).filter_by(id=res.new_transcription_id).one()
        assert new_tx.model_name == "large-v3-turbo"
        assert new_tx.is_indexed == 0
        assert len(new_tx.segments) == 2

        # Verify midpoint chapter linkage
        # Segment 0: 0-30s, midpoint 15s -> chapter 0 (0-1800s)
        # Segment 1: 1900-1930s, midpoint 1915s -> chapter 1 (1800-3600s)
        chap0 = session.query(ChapterRecord).filter_by(audio_file_id=af1.id, chapter_index=0).one()
        chap1 = session.query(ChapterRecord).filter_by(audio_file_id=af1.id, chapter_index=1).one()
        assert new_tx.segments[0].chapter_id == chap0.id
        assert new_tx.segments[1].chapter_id == chap1.id

        # Verify bookmark and chapters re-linked to new transcription
        bm = session.query(BookmarkRecord).filter_by(audio_file_id=af1.id).one()
        assert bm.transcription_id == res.new_transcription_id
        assert chap0.transcription_id == res.new_transcription_id
        assert chap1.transcription_id == res.new_transcription_id


def test_single_record_rollback(upgrader_env):
    """Test that single-record rollback restores original transcription and segment IDs."""
    batch_id = "test_batch_002"
    incoming_data = {
        "text": "Upgraded text.",
        "segments": [{"start": 0.0, "end": 10.0, "text": "Upgraded text."}],
    }

    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()
        orig_tx = session.query(TranscriptionRecord).filter_by(audio_file_id=af1.id).one()
        orig_tx_id = orig_tx.id
        orig_full_text = orig_tx.full_text
        orig_seg_ids = [s.id for s in orig_tx.segments]

        # Upgrade
        upgrader.upgrade_record(af1.id, incoming_data, batch_id)
        session.commit()

        # Rollback by audio_file_id
        roll_res = upgrader.rollback_record(af1.id)
        session.commit()

        assert roll_res.restored_transcription_id == orig_tx_id
        assert roll_res.segments_restored_count == len(orig_seg_ids)

        # Confirm restored live record matches original
        restored_tx = session.query(TranscriptionRecord).filter_by(id=orig_tx_id).one()
        assert restored_tx.full_text == orig_full_text
        assert len(restored_tx.segments) == 2
        assert [s.id for s in restored_tx.segments] == orig_seg_ids

        # Confirm bookmark and chapters restored to orig_tx_id
        bm = session.query(BookmarkRecord).filter_by(audio_file_id=af1.id).one()
        assert bm.transcription_id == orig_tx_id
        chap0 = session.query(ChapterRecord).filter_by(audio_file_id=af1.id, chapter_index=0).one()
        assert chap0.transcription_id == orig_tx_id

        # Confirm archive record was consumed
        arch_count = session.query(ArchivedTranscriptionRecord).filter_by(audio_file_id=af1.id).count()
        assert arch_count == 0


def test_adjacent_records_untouched(upgrader_env):
    """Upgrading two files under one batch, then rolling back only one, leaves the other upgraded."""
    batch_id = "test_batch_003"
    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()
        af2 = session.query(AudioFileRecord).filter_by(file_name="Letters from a Stoic - Seneca.mp3").one()

        upgrader.upgrade_record(af1.id, {"text": "New Aurelius", "segments": [{"start": 0, "end": 5, "text": "New Aurelius"}]}, batch_id)
        upgrader.upgrade_record(af2.id, {"text": "New Seneca", "segments": [{"start": 0, "end": 5, "text": "New Seneca"}]}, batch_id)
        session.commit()

        # Roll back only af1
        upgrader.rollback_record(af1.id)
        session.commit()

        # af1 is rolled back to "Old coarse text"
        tx1 = session.query(TranscriptionRecord).filter_by(audio_file_id=af1.id).one()
        assert "Old coarse text" in tx1.full_text

        # af2 remains upgraded to "New Seneca"
        tx2 = session.query(TranscriptionRecord).filter_by(audio_file_id=af2.id).one()
        assert "New Seneca" in tx2.full_text


def test_batch_rollback(upgrader_env):
    """Batch rollback reverses all records belonging to a batch ID."""
    batch_id = "test_batch_004"
    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()
        af2 = session.query(AudioFileRecord).filter_by(file_name="Letters from a Stoic - Seneca.mp3").one()

        upgrader.upgrade_record(af1.id, {"text": "New Aurelius", "segments": []}, batch_id)
        upgrader.upgrade_record(af2.id, {"text": "New Seneca", "segments": []}, batch_id)
        session.commit()

        # Verify both archived
        assert session.query(ArchivedTranscriptionRecord).filter_by(upgrade_batch_id=batch_id).count() == 2

        # Roll back batch
        results = upgrader.rollback_batch(batch_id)
        session.commit()

        assert len(results) == 2
        # Both restored
        tx1 = session.query(TranscriptionRecord).filter_by(audio_file_id=af1.id).one()
        tx2 = session.query(TranscriptionRecord).filter_by(audio_file_id=af2.id).one()
        assert "Old coarse text" in tx1.full_text
        assert "Old Seneca text" in tx2.full_text


def test_relink_behavior_assertion_tested_invariant(upgrader_env):
    """Assert the explicit Option 1 design decision:

    - When an upgrade runs, segments are deleted and chat_messages/note_captures
      FKs are SET NULL by the database engine.
    - When rollback restores the original transcript and segment IDs, the body
      text on notes and chat messages remains completely preserved.
    - However, segment_id remains permanently NULL (it is NOT re-linked).

    This pins the documented Option 1 limitation as a tested invariant.
    If re-linking is implemented in the future (Option 2), update this assertion.
    """
    batch_id = "test_batch_005"
    incoming_data = {
        "text": "New upgraded text",
        "segments": [{"start": 0.0, "end": 20.0, "text": "New upgraded text"}],
    }

    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()

        # Pre-check: note and chat have segment_id pointing to a segment
        note = session.query(NoteCapture).first()
        msg = session.query(ChatMessage).first()
        assert note.segment_id is not None
        assert msg.segment_id is not None
        orig_seg_id = note.segment_id

        # 1. Run upgrade
        upgrader.upgrade_record(af1.id, incoming_data, batch_id)
        session.commit()
        session.expire_all()

        # Assert SET NULL took effect in DB on live segments delete
        note_after_up = session.query(NoteCapture).first()
        msg_after_up = session.query(ChatMessage).first()
        assert note_after_up.segment_id is None
        assert msg_after_up.segment_id is None
        assert note_after_up.body == "Crucial passage on duty and virtue."
        assert msg_after_up.content == "Can you explain this segment?"

        # 2. Run rollback
        upgrader.rollback_record(af1.id)
        session.commit()
        session.expire_all()

        # 3. Assert post-rollback state:
        note_after_rb = session.query(NoteCapture).first()
        msg_after_rb = session.query(ChatMessage).first()

        # Body text is preserved intact
        assert note_after_rb.body == "Crucial passage on duty and virtue."
        assert msg_after_rb.content == "Can you explain this segment?"

        # TESTED INVARIANT: segment_id is still NULL after rollback (Option 1 behavior)
        # This pins the documented Option 1 limitation as a tested invariant.
        # If re-linking is implemented (Option 2), update this assertion.
        assert note_after_rb.segment_id is None
        assert msg_after_rb.segment_id is None


def test_stratified_post_hoc_sampling(upgrader_env):
    """Test that select_post_hoc_audit_samples returns stratified representation across

    durations while enforcing author, work, and genre diversity.
    """
    with get_session() as session:
        upgrader = TranscriptUpgrader(session)

        # Create distinct works and authors
        w_aurelius = WorkRecord(title="Meditations", author="Marcus Aurelius")
        w_seneca = WorkRecord(title="Letters from a Stoic", author="Seneca")
        w_nietzsche = WorkRecord(title="Beyond Good and Evil", author="Friedrich Nietzsche")
        w_kant = WorkRecord(title="Critique of Pure Reason", author="Immanuel Kant")
        w_plato = WorkRecord(title="The Republic", author="Plato")
        session.add_all([w_aurelius, w_seneca, w_nietzsche, w_kant, w_plato])
        session.flush()

        # Create multiple tracks across strata and works
        af_aurelius_short = AudioFileRecord(
            file_name="Meditations_Ch01.mp3",
            file_path="/audio/med_01.mp3",
            duration_seconds=300.0,
            work_id=w_aurelius.id,
            genre="Philosophy",
        )
        af_aurelius_med = AudioFileRecord(
            file_name="Meditations_Ch02.mp3",
            file_path="/audio/med_02.mp3",
            duration_seconds=1500.0,
            work_id=w_aurelius.id,
            genre="Philosophy",
        )
        af_seneca_short = AudioFileRecord(
            file_name="Seneca_Letter01.mp3",
            file_path="/audio/sen_01.mp3",
            duration_seconds=450.0,
            work_id=w_seneca.id,
            genre="Philosophy",
        )
        af_nietzsche_med = AudioFileRecord(
            file_name="Nietzsche_Part01.mp3",
            file_path="/audio/nietz_01.mp3",
            duration_seconds=2200.0,
            work_id=w_nietzsche.id,
            genre="Philosophy",
        )
        af_kant_long = AudioFileRecord(
            file_name="Kant_PureReason_Ch01.mp3",
            file_path="/audio/kant_01.mp3",
            duration_seconds=5000.0,
            work_id=w_kant.id,
            genre="Philosophy",
        )
        af_plato_long = AudioFileRecord(
            file_name="Plato_Republic_Bk01.mp3",
            file_path="/audio/plato_01.mp3",
            duration_seconds=7200.0,
            work_id=w_plato.id,
            genre="Philosophy",
        )

        session.add_all([
            af_aurelius_short,
            af_aurelius_med,
            af_seneca_short,
            af_nietzsche_med,
            af_kant_long,
            af_plato_long,
        ])
        session.commit()

        all_ids = [
            af_aurelius_short.id,
            af_aurelius_med.id,
            af_seneca_short.id,
            af_nietzsche_med.id,
            af_kant_long.id,
            af_plato_long.id,
        ]

        # Request sample size of 5
        samples = upgrader.select_post_hoc_audit_samples(all_ids, sample_size=5)

        assert len(samples) == 5

        # Strata represented
        strata = {s["stratum"] for s in samples}
        assert "short (<15m)" in strata
        assert "medium (15-60m)" in strata
        assert "long (>1h)" in strata

        # Author diversity: with 5 distinct authors available, each sample should have a distinct author
        authors = [s["author"] for s in samples]
        assert len(set(authors)) == 5
        assert set(authors) == {"Marcus Aurelius", "Seneca", "Friedrich Nietzsche", "Immanuel Kant", "Plato"}

        # Field integrity
        for s in samples:
            assert s["audio_file_id"] in all_ids
            assert s["genre"] == "Philosophy"
            assert s["work_title"] in {"Meditations", "Letters from a Stoic", "Beyond Good and Evil", "Critique of Pure Reason", "The Republic"}
            assert s["duration_seconds"] > 0


def test_segment_vector_purge_on_upgrade_and_rollback(upgrader_env, monkeypatch):
    """Test that segment vectors are purged from LanceDB during upgrade and rollback."""
    deleted_calls = []

    class DummyStore:
        def delete_segments(self, seg_ids):
            deleted_calls.append(list(seg_ids))
            return len(seg_ids)

    monkeypatch.setattr("audiobench.memory.singletons.get_segment_store", lambda: DummyStore())

    batch_id = "test_seg_purge_batch"
    incoming_data = {
        "text": "New upgraded text",
        "engine": "whisper",
        "model_name": "large-v3-turbo",
        "duration_seconds": 3600.0,
        "segments": [{"start": 0.0, "end": 20.0, "text": "New segment"}],
    }

    with get_session() as session:
        upgrader = TranscriptUpgrader(session)
        af1 = session.query(AudioFileRecord).filter_by(file_name="Meditations - Marcus Aurelius.mp3").one()
        old_tx = session.query(TranscriptionRecord).filter_by(audio_file_id=af1.id).one()
        old_seg_ids = [s.id for s in old_tx.segments]

        # Upgrade
        upgrader.upgrade_record(
            audio_file_id=af1.id,
            incoming_data=incoming_data,
            upgrade_batch_id=batch_id,
        )
        session.commit()

        assert len(deleted_calls) == 1
        assert deleted_calls[0] == old_seg_ids

        # Rollback
        deleted_calls.clear()
        upgrader.rollback_batch(batch_id)
        session.commit()

        assert len(deleted_calls) == 1
        # The segment IDs purged during rollback should be the new segment created in upgrade
        assert len(deleted_calls[0]) == 1

