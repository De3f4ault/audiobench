"""Tests for Stage 2 Expression Privacy Tier Enforcement."""

import pytest

from audiobench.memory.enums import RelationType, SourceType
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import SegmentRecord, TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository


class MockReplSession:
    """Mock session providing a configurable effective_tier."""

    def __init__(self, tier: int = 0):
        self._tier = tier

    def effective_tier(self) -> int:
        return self._tier


def test_transcription_repository_helpers(test_db):
    """Test get_max_privacy_tier_for_transcriptions and get_max_privacy_tier_for_segments."""
    from audiobench.core.db_session import get_session

    tx_repo = TranscriptionRepository()

    # Empty inputs return 0
    assert tx_repo.get_max_privacy_tier_for_transcriptions([]) == 0
    assert tx_repo.get_max_privacy_tier_for_segments([]) == 0

    with get_session() as s:
        tx1 = TranscriptionRecord(full_text="Tx 1", pipeline_phase="complete")
        tx2 = TranscriptionRecord(full_text="Tx 2", pipeline_phase="complete")
        s.add_all([tx1, tx2])
        s.commit()
        s.refresh(tx1)
        s.refresh(tx2)

        # Segments for tx1: tier 0 and tier 2
        seg1 = SegmentRecord(transcription_id=tx1.id, segment_index=0, text="Public", privacy_tier=0)
        seg2 = SegmentRecord(transcription_id=tx1.id, segment_index=1, text="Intimate", privacy_tier=2)
        # Segments for tx2: only tier 0
        seg3 = SegmentRecord(transcription_id=tx2.id, segment_index=0, text="Public only", privacy_tier=0)
        s.add_all([seg1, seg2, seg3])
        s.commit()
        s.refresh(seg1)
        s.refresh(seg2)
        s.refresh(seg3)

        tx1_id = tx1.id
        tx2_id = tx2.id
        seg1_id = seg1.id
        seg2_id = seg2.id
        seg3_id = seg3.id

    # Test get_max_privacy_tier_for_transcriptions
    assert tx_repo.get_max_privacy_tier_for_transcriptions([tx1_id]) == 2
    assert tx_repo.get_max_privacy_tier_for_transcriptions([tx2_id]) == 0
    assert tx_repo.get_max_privacy_tier_for_transcriptions([tx1_id, tx2_id]) == 2

    # Test get_max_privacy_tier_for_segments
    assert tx_repo.get_max_privacy_tier_for_segments([seg1_id]) == 0
    assert tx_repo.get_max_privacy_tier_for_segments([seg2_id]) == 2
    assert tx_repo.get_max_privacy_tier_for_segments([seg1_id, seg3_id]) == 0
    assert tx_repo.get_max_privacy_tier_for_segments([seg1_id, seg2_id, seg3_id]) == 2


def test_expression_register_and_redaction(test_db):
    """Test register() with privacy_tier, and get_by_id redaction behaviour."""
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import ExpressionRecord

    repo = ExpressionRepository()

    # Public expression (tier 0)
    pub_expr = repo.register(
        content="Public content",
        source_type=SourceType.AUDIO_TRANSCRIPT.value,
        privacy_tier=0,
    )
    # Intimate expression (tier 2)
    int_expr = repo.register(
        content="Sensitive personal voice recording",
        source_type=SourceType.AUDIO_TRANSCRIPT.value,
        privacy_tier=2,
    )

    # 1. Read with session=None (locked, tier 0)
    pub_read_locked = repo.get_by_id(pub_expr.id, session=None)
    assert pub_read_locked is not None
    assert pub_read_locked.content == "Public content"
    assert getattr(pub_read_locked, "redacted", False) is False

    int_read_locked = repo.get_by_id(int_expr.id, session=None)
    assert int_read_locked is not None
    assert int_read_locked.content == "[REDACTED]"
    assert getattr(int_read_locked, "redacted", False) is True

    # 2. Read with unlocked session (tier 2)
    unlocked_session = MockReplSession(tier=2)
    int_read_unlocked = repo.get_by_id(int_expr.id, session=unlocked_session)
    assert int_read_unlocked is not None
    assert int_read_unlocked.content == "Sensitive personal voice recording"
    assert getattr(int_read_unlocked, "redacted", True) is False

    # 3. Read with insufficient tier session (tier 1 < tier 2)
    tier1_session = MockReplSession(tier=1)
    int_read_tier1 = repo.get_by_id(int_expr.id, session=tier1_session)
    assert int_read_tier1 is not None
    assert int_read_tier1.content == "[REDACTED]"

    # 4. Verify SQLite storage was NEVER mutated by the redaction
    with get_session() as s:
        raw_rec = s.query(ExpressionRecord).filter_by(id=int_expr.id).first()
        assert raw_rec is not None
        assert raw_rec.content == "Sensitive personal voice recording"
        assert raw_rec.privacy_tier == 2


def test_expression_batch_redaction(test_db):
    """Test get_by_ids and get_parents_batch session redaction."""
    repo = ExpressionRepository()

    parent = repo.register(
        content="Parent sensitive text",
        source_type=SourceType.AUDIO_TRANSCRIPT.value,
        privacy_tier=2,
    )
    child = repo.register(
        content="Child chunk",
        source_type=SourceType.TRANSCRIPT_SEGMENT.value,
        privacy_tier=2,
    )
    repo.link(from_id=child.id, to_id=parent.id, relation_type=RelationType.SOURCE.value)

    # 1. get_by_ids with locked session
    batch_locked = repo.get_by_ids([parent.id, child.id], session=None)
    assert batch_locked[parent.id].content == "[REDACTED]"
    assert batch_locked[child.id].content == "[REDACTED]"

    # 2. get_by_ids with unlocked session
    batch_unlocked = repo.get_by_ids([parent.id, child.id], session=MockReplSession(tier=2))
    assert batch_unlocked[parent.id].content == "Parent sensitive text"
    assert batch_unlocked[child.id].content == "Child chunk"

    # 3. get_parents_batch with locked session
    parents_locked = repo.get_parents_batch([child.id], session=None)
    assert parents_locked[child.id].content == "[REDACTED]"

    # 4. get_parents_batch with unlocked session
    parents_unlocked = repo.get_parents_batch([child.id], session=MockReplSession(tier=2))
    assert parents_unlocked[child.id].content == "Parent sensitive text"

    # 5. walk_to_parent with locked vs unlocked session
    walk_locked = repo.walk_to_parent(child.id, session=None)
    assert walk_locked.content == "[REDACTED]"
    walk_unlocked = repo.walk_to_parent(child.id, session=MockReplSession(tier=2))
    assert walk_unlocked.content == "Parent sensitive text"


def test_register_batch_with_privacy_tier(test_db):
    """Test register_batch honours privacy_tier and upgrades existing tier."""
    repo = ExpressionRepository()

    items = [
        {"content": "Chunk 1", "source_type": "audio_transcript", "source_id": 1, "privacy_tier": 0},
        {"content": "Chunk 2", "source_type": "audio_transcript", "source_id": 1, "privacy_tier": 2},
    ]

    records = repo.register_batch(items)
    assert len(records) == 2

    rec1 = repo.get_by_id(records[0].id, session=None)
    assert rec1.content == "Chunk 1"

    rec2 = repo.get_by_id(records[1].id, session=None)
    assert rec2.content == "[REDACTED]"

    rec2_unlocked = repo.get_by_id(records[1].id, session=MockReplSession(tier=2))
    assert rec2_unlocked.content == "Chunk 2"


def test_register_with_chapter_id_and_link_segments_batch(test_db):
    """Test register() persists chapter_id and link_segments_batch() populates junction table."""
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, ChapterRecord, ExpressionSegmentMap, SegmentRecord, TranscriptionRecord

    repo = ExpressionRepository()

    with get_session() as s:
        af = AudioFileRecord(file_path="/tmp/test_audio.mp3", file_name="test_audio.mp3", duration_seconds=120.0, format="mp3", file_size_bytes=1024)
        s.add(af)
        s.commit()
        s.refresh(af)

        chap = ChapterRecord(audio_file_id=af.id, chapter_index=0, title="Chapter 1", start_time=0.0, end_time=60.0)
        s.add(chap)
        s.commit()
        s.refresh(chap)

        tx = TranscriptionRecord(audio_file_id=af.id, full_text="Test full text", pipeline_phase="complete")
        s.add(tx)
        s.commit()
        s.refresh(tx)
        seg1 = SegmentRecord(transcription_id=tx.id, segment_index=0, text="Test", start_time=0.0, end_time=1.0)
        seg2 = SegmentRecord(transcription_id=tx.id, segment_index=1, text="full", start_time=1.0, end_time=2.0)
        s.add_all([seg1, seg2])
        s.commit()
        s.refresh(seg1)
        s.refresh(seg2)
        tx_id = tx.id
        chap_id = chap.id
        seg1_id = seg1.id
        seg2_id = seg2.id

    rec = repo.register(
        content="Test chunk with chapter",
        source_type="audio_transcript",
        source_id=tx_id,
        chapter_id=chap_id,
    )
    assert rec.chapter_id == chap_id

    repo.link_segments_batch([(rec.id, seg1_id), (rec.id, seg2_id)])

    with get_session() as s:
        links = s.query(ExpressionSegmentMap).filter_by(expression_id=rec.id).all()
        assert len(links) == 2
        assert {l.segment_id for l in links} == {seg1_id, seg2_id}

