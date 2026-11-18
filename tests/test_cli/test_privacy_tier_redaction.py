"""Tests for privacy_tier enforcement at the repository layer.

Verifies:
1. Fake owner voiceprint enrollment.
2. Segments tagged with privacy_tier >= 2 are redacted with session=None.
3. Segments tagged with privacy_tier=0 remain visible with session=None.
4. An unlocked session (effective_tier >= 2) receives the original segment content.
5. full_text / raw_text are fully replaced when protected segments exist and caller is locked.
6. search() text_preview is [REDACTED] when locked, and original when unlocked.
7. Synchronous idle check in effective_tier() auto-locks expired sessions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import click
import numpy as np
import pytest

from audiobench.cli.repl.session import ReplSession
from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.security import voiceprint
from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository


@pytest.fixture
def fake_voiceprint_enrolled(tmp_data_dir):
    """Enroll a fake 192-dim voiceprint embedding."""
    settings = get_settings()
    sec_dir = settings.security_dir
    sec_dir.mkdir(parents=True, exist_ok=True)

    fake_embedding = np.random.randn(192).astype(np.float32)
    fake_embedding /= np.linalg.norm(fake_embedding)
    np.save(sec_dir / "voiceprint.npy", fake_embedding)

    meta = {"name": "Test Owner", "duration_seconds": 25.0}
    (sec_dir / "enrollment.json").write_text(json.dumps(meta), encoding="utf-8")

    assert voiceprint.is_enrolled()
    return fake_embedding


@pytest.fixture
def protected_transcription(test_db):
    """Seed DB with a transcription containing tier-0 and tier-2 segments."""
    with get_session() as session:
        af = AudioFileRecord(
            file_name="meeting_with_owner.mp3",
            file_path="/tmp/audio/meeting_with_owner.mp3",
            file_hash="hash_protected_tx_123",
            duration_seconds=60.0,
        )
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="Meeting with Owner",
            full_text="Public agenda topic. The owner confidential secret discussion.",
            raw_text="Public agenda topic. The owner confidential secret discussion.",
            language="en",
            engine="faster-whisper",
            model_name="turbo",
            duration_seconds=60.0,
            word_count=9,
            segment_count=2,
            pipeline_phase="complete",
        )
        session.add(tx)
        session.flush()

        seg0 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=0,
            start_time=0.0,
            end_time=5.0,
            text="Public agenda topic.",
            speaker="SPEAKER_00",
            privacy_tier=0,
        )
        seg1 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=1,
            start_time=5.0,
            end_time=12.0,
            text="The owner confidential secret discussion.",
            speaker="SPEAKER_01",
            privacy_tier=2,
        )
        session.add_all([seg0, seg1])
        session.commit()

        tx_id = tx.id

    return tx_id


class TestPrivacyTierRedaction:
    """Test get_by_id and search redaction behavior."""

    def test_voiceprint_is_enrolled(self, fake_voiceprint_enrolled):
        """Confirm our fixture enrolled a fake voiceprint."""
        assert voiceprint.is_enrolled()
        summary = voiceprint.get_enrollment_summary()
        assert summary is not None
        assert "Test Owner" in summary

    def test_get_by_id_locked_session_none(self, protected_transcription):
        """Calling get_by_id(session=None) must redact tier-2 segment and full_text."""
        repo = TranscriptionRepository()
        data = repo.get_by_id(protected_transcription, session=None)

        assert data is not None
        segments = data["segments"]
        assert len(segments) == 2

        # Tier 0 segment: visible
        assert segments[0]["privacy_tier"] == 0
        assert segments[0]["redacted"] is False
        assert segments[0]["text"] == "Public agenda topic."

        # Tier 2 segment: redacted
        assert segments[1]["privacy_tier"] == 2
        assert segments[1]["redacted"] is True
        assert segments[1]["text"] == "[REDACTED]"
        assert "secret discussion" not in segments[1]["text"]

        # full_text / raw_text: fully replaced when protected content exists
        assert "[REDACTED:" in data["full_text"]
        assert r"\unlock 2" in data["full_text"]
        assert "secret discussion" not in data["full_text"]
        assert data["raw_text"] == ""

    def test_get_by_id_unlocked_session(self, protected_transcription):
        """Calling get_by_id(session=unlocked) returns the real text."""
        repo = TranscriptionRepository()

        # Create session unlocked to Tier 2
        session = ReplSession(click.Group("test"))
        session.session_unlock_tier = 2
        assert session.effective_tier() == 2

        data = repo.get_by_id(protected_transcription, session=session)

        assert data is not None
        segments = data["segments"]

        # Both segments visible
        assert segments[0]["text"] == "Public agenda topic."
        assert segments[0]["redacted"] is False

        assert segments[1]["text"] == "The owner confidential secret discussion."
        assert segments[1]["redacted"] is False
        assert segments[1]["privacy_tier"] == 2

        # full_text preserved
        assert data["full_text"] == "Public agenda topic. The owner confidential secret discussion."
        assert data["raw_text"] == "Public agenda topic. The owner confidential secret discussion."

    def test_search_locked_vs_unlocked(self, protected_transcription):
        """search() should redact text_preview when locked and show it when unlocked."""
        repo = TranscriptionRepository()

        # Locked (session=None)
        results_locked = repo.search("secret", limit=5, session=None)
        assert len(results_locked) == 1
        assert results_locked[0]["id"] == protected_transcription
        assert results_locked[0]["text_preview"] == "[REDACTED]"

        # Unlocked (session with tier 2)
        session = ReplSession(click.Group("test"))
        session.session_unlock_tier = 2
        results_unlocked = repo.search("secret", limit=5, session=session)
        assert len(results_unlocked) == 1
        assert results_unlocked[0]["id"] == protected_transcription
        assert results_unlocked[0]["text_preview"] != "[REDACTED]"
        assert "secret" in results_unlocked[0]["text_preview"].lower()

    def test_synchronous_idle_lock(self):
        """effective_tier() synchronously locks if idle time exceeds settings."""
        session = ReplSession(click.Group("test"))
        session.session_unlock_tier = 2
        assert session.effective_tier() == 2

        # Simulate last keystroke 10 minutes ago
        session.last_keystroke_at = datetime.now(UTC) - timedelta(seconds=600)

        mock_settings = MagicMock()
        mock_settings.idle_lock_seconds = 300
        with patch("audiobench.core.settings.get_settings", return_value=mock_settings):
            tier = session.effective_tier()
            assert tier == 0
            assert session.session_unlock_tier == 0
            assert session._idle_lock_fired is True

