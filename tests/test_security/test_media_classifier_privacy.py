"""Security and Privacy Test Suite for Media Classifier & Audiobook Firewall.

Tests asymmetric loss invariants:
- Negative controls: personal voice memos, interviews with authors, long recordings,
  bare path recordings MUST NEVER be autonomously bypassed (never Class A).
- Positive controls: verified legal spoken preambles, .m4b containers,
  and directory provenance corroborated by sibling book artifacts / multi-track sequence
  ARE safely classified as Class A (hard proof).
- Invariant tests: Class B soft heuristics NEVER auto-promote to Tier 0 without
  explicit positive human confirmation.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from audiobench.media.classifier import (
    ClassificationResult,
    classify_media,
    check_transcript_boilerplate,
    check_container,
    check_provenance_and_artifacts,
    check_soft_heuristics,
)


# ── Negative Controls (Privacy Protection) ───────────────────────────────────

def test_short_voice_memo_rejected(tmp_path: Path) -> None:
    """A 45-second voice memo in .m4a format must NEVER be autonomously bypassed."""
    memo = tmp_path / "voice_memo_20261011.m4a"
    memo.touch()

    result = classify_media(file_path=memo, duration_seconds=45.0)

    assert result.category == "class_c_generic_audio"
    assert result.is_autonomous_bypass is False
    assert result.confidence <= 0.50


def test_long_personal_monologue_rejected(tmp_path: Path) -> None:
    """A 45-minute personal monologue / brainstorm in .wav format must NEVER be Class A."""
    audio = tmp_path / "brainstorming_session.wav"
    audio.touch()

    result = classify_media(file_path=audio, duration_seconds=2700.0)

    assert result.category != "class_a_hard_proof"
    assert result.is_autonomous_bypass is False


def test_author_interview_memo_not_autonomous(tmp_path: Path) -> None:
    """An interview where user speaks with an author (duration 40m, artist='Robert C. Martin')

    MUST NOT be autonomously bypassed (Gate 3 flaw). It must be Class B, NEVER Class A!
    """
    interview = tmp_path / "interview_with_uncle_bob.mp3"
    interview.touch()

    meta = {
        "artist": "Robert C. Martin",
        "album": "Clean Architecture Discussion",
        "title": "Episode 14",
    }

    result = classify_media(file_path=interview, metadata=meta, duration_seconds=2400.0)

    assert result.category == "class_b_soft_heuristic"
    assert result.is_autonomous_bypass is False
    assert "album_and_artist_metadata" in result.evidence[0]


def test_bare_audiobook_folder_without_artifacts_not_class_a(tmp_path: Path) -> None:
    """A personal memo accidentally saved in an 'Audiobooks' folder WITHOUT companion book artifacts

    MUST NOT be treated as Class A. It drops to Class B and requires human confirmation.
    """
    ab_dir = tmp_path / "Audiobooks"
    ab_dir.mkdir()
    personal_file = ab_dir / "quick_voice_notes.mp3"
    personal_file.touch()

    # No sibling .epub / .pdf / .mobi exists in ab_dir
    result = classify_media(file_path=personal_file, duration_seconds=120.0)

    assert result.category == "class_b_soft_heuristic"
    assert result.is_autonomous_bypass is False
    assert "bare_audiobook_folder_without_artifacts" in result.evidence[0]


# ── Positive Controls (Legitimate Audiobooks) ────────────────────────────────

def test_audible_preamble_triggers_class_a(tmp_path: Path) -> None:
    """Audio transcript opening with Audible disclaimer triggers Class A autonomous bypass."""
    f = tmp_path / "chapter_01.mp3"
    f.touch()

    transcript_text = "This is Audible. Audible Studios presents The Pragmatic Programmer, written by Andy Hunt."
    result = classify_media(file_path=f, transcript_text=transcript_text, duration_seconds=3600.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert any("audible" in ev for ev in result.evidence)


def test_librivox_preamble_triggers_class_a(tmp_path: Path) -> None:
    """Audio transcript opening with LibriVox disclaimer triggers Class A autonomous bypass."""
    f = tmp_path / "moby_dick_01.mp3"
    f.touch()

    transcript_text = "This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information, visit LibriVox.org."
    result = classify_media(file_path=f, transcript_text=transcript_text, duration_seconds=1800.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert any("librivox" in ev for ev in result.evidence)


def test_tgc_preamble_triggers_class_a(tmp_path: Path) -> None:
    """Audio transcript opening with The Great Courses / The Teaching Company triggers Class A."""
    f = tmp_path / "L01 - Introduction.mp3"
    f.touch()

    transcript_text = "The Great Courses. Philosophy of Religion, Lecture 1."
    result = classify_media(file_path=f, transcript_text=transcript_text, duration_seconds=1800.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert any("the_great_courses" in ev for ev in result.evidence)


def test_m4b_container_triggers_class_a(tmp_path: Path) -> None:
    """Dedicated .m4b container with duration > 10m triggers Class A autonomous bypass."""
    book = tmp_path / "clean_code.m4b"
    book.touch()

    result = classify_media(file_path=book, duration_seconds=36000.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert "container:m4b_audiobook" in result.evidence


def test_path_with_sibling_epub_triggers_class_a(tmp_path: Path) -> None:
    """Directory in Audiobooks path with companion .epub artifact triggers Class A."""
    ab_dir = tmp_path / "Audiobooks" / "Pragmatic Programmer"
    ab_dir.mkdir(parents=True)
    audio = ab_dir / "audio.mp3"
    audio.touch()
    epub = ab_dir / "pragmatic_programmer.epub"
    epub.touch()

    result = classify_media(file_path=audio, duration_seconds=3600.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert any("companion_artifact(.epub)" in ev for ev in result.evidence)


def test_path_with_multitrack_sequence_triggers_class_a(tmp_path: Path) -> None:
    """Directory in Audiobooks path with sequenced chapter files (>= 5 tracks) triggers Class A."""
    ab_dir = tmp_path / "Audiobooks" / "Design Patterns"
    ab_dir.mkdir(parents=True)

    tracks = []
    for i in range(1, 8):
        t = ab_dir / f"{i:02d} - Chapter {i}.mp3"
        t.touch()
        tracks.append(t)

    result = classify_media(file_path=tracks[0], duration_seconds=1200.0)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert any("multi_track_chapter_album" in ev for ev in result.evidence)


def test_explicit_flag_triggers_class_a(tmp_path: Path) -> None:
    """Passing explicit_flag=True guarantees Class A regardless of filename."""
    audio = tmp_path / "custom_recording.wav"
    audio.touch()

    result = classify_media(file_path=audio, explicit_flag=True)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert "explicit_operator_flag" in result.evidence


def test_persisted_confirmation_skips_reprompt(tmp_path: Path) -> None:
    """A file with tags {'audiobook': True, 'user_confirmed': True} immediately triggers Class A."""
    audio = tmp_path / "ambiguous_file.mp3"
    audio.touch()

    meta = {"tags": {"audiobook": True, "user_confirmed": True}}
    result = classify_media(file_path=audio, metadata=meta)

    assert result.category == "class_a_hard_proof"
    assert result.is_autonomous_bypass is True
    assert "persisted_user_confirmation" in result.evidence


# ── Strict Invariant Tests ───────────────────────────────────────────────────

def test_unconfirmed_class_b_never_auto_promotes_to_tier_0(tmp_path: Path) -> None:
    """STRICT INVARIANT: An unconfirmed Class B file must NEVER auto-promote to Tier 0.

    Even across multiple rescans, time advancement, or lack of confirmation,
    is_autonomous_bypass must remain False.
    """
    file_path = tmp_path / "SomeAuthor - Discussion.mp3"
    file_path.touch()

    meta = {"artist": "Some Author", "title": "Discussion"}
    dur = 1200.0  # 20 min

    # Run classification multiple times (simulating re-scans / daemon cycles)
    for _ in range(5):
        result = classify_media(file_path=file_path, metadata=meta, duration_seconds=dur)
        assert result.category == "class_b_soft_heuristic"
        assert result.is_autonomous_bypass is False
        assert result.review_reason is not None


def test_headless_class_b_defers_without_biometric_flood(tmp_path: Path) -> None:
    """Class B files must tag pending_privacy_review and NOT call _schedule_biometric_pass."""
    from unittest.mock import MagicMock
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.transcription_result import Transcript, Segment, AudioMetadata

    repo = TranscriptionRepository()
    repo._schedule_biometric_pass = MagicMock()
    repo._import_to_library = MagicMock(return_value=str(tmp_path / "imported.mp3"))

    audio_file = tmp_path / "SomeAuthor - Clean Architecture Notes.mp3"
    audio_file.touch()

    meta = AudioMetadata(
        file_path=str(audio_file),
        file_name=audio_file.name,
        file_size_bytes=1000,
        format="mp3",
        duration_seconds=1800.0,
        sample_rate=16000,
        channels=1,
        file_hash="mock_hash_class_b_test",
    )
    transcript = Transcript(
        segments=[Segment(id=0, start=0.0, end=5.0, text="Discussing the architecture.")],
        duration_seconds=1800.0,
    )

    tx_id = repo.save_transcription(transcript, audio_metadata=meta, overwrite=True)
    assert tx_id > 0

    # Assert biometric pass was NOT called (no CPU flood)
    repo._schedule_biometric_pass.assert_not_called()

    # Assert file is in pending_privacy_reviews
    pending = repo.get_pending_privacy_reviews()
    matching = [p for p in pending if p["file_name"] == audio_file.name]
    assert len(matching) == 1
    assert "stem_author_title_split" in matching[0]["reason"]


def test_review_resolution_lifecycle(tmp_path: Path) -> None:
    """Test resolving a pending review as audiobook (Tier 0) vs generic (biometric pass)."""
    from unittest.mock import MagicMock
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.transcription_result import Transcript, Segment, AudioMetadata

    repo = TranscriptionRepository()
    repo._schedule_biometric_pass = MagicMock()
    repo._import_to_library = MagicMock(return_value=str(tmp_path / "imported_res.mp3"))

    audio_file = tmp_path / "Author - ResolveTest.mp3"
    audio_file.touch()

    meta = AudioMetadata(
        file_path=str(audio_file),
        file_name=audio_file.name,
        file_size_bytes=1000,
        format="mp3",
        duration_seconds=1800.0,
        sample_rate=16000,
        channels=1,
        file_hash="mock_hash_resolve_test",
    )
    transcript = Transcript(
        segments=[Segment(id=0, start=0.0, end=5.0, text="Test segment.", privacy_tier=1)],
        duration_seconds=1800.0,
    )

    repo.save_transcription(transcript, audio_metadata=meta, overwrite=True)
    pending = repo.get_pending_privacy_reviews()
    item = next(p for p in pending if p["file_name"] == audio_file.name)

    # 1. Resolve as Audiobook -> sets Tier 0, tags audiobook, clears pending
    success = repo.resolve_privacy_review(item["id"], as_audiobook=True)
    assert success is True

    # Pending list should now be clear of this item
    pending_after = repo.get_pending_privacy_reviews()
    assert not any(p["id"] == item["id"] for p in pending_after)


def test_direct_schedule_biometric_pass_intercepts_audiobook(tmp_path: Path) -> None:
    """Verifies that ANY path calling _schedule_biometric_pass directly (e.g. alignment_worker)

    is intercepted by the firewall, bypassing scheduler enqueueing for verified audiobooks.
    """
    from unittest.mock import patch
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.transcription_result import Transcript, Segment, AudioMetadata

    repo = TranscriptionRepository()
    repo._import_to_library = lambda p: str(p)

    book = tmp_path / "pragmatic_programmer.m4b"
    book.touch()

    meta = AudioMetadata(
        file_path=str(book),
        file_name=book.name,
        file_size_bytes=5000000,
        format="m4b",
        duration_seconds=36000.0,
        sample_rate=44100,
        channels=2,
        file_hash="mock_hash_direct_schedule_test",
    )
    transcript = Transcript(
        segments=[Segment(id=0, start=0.0, end=5.0, text="Intro to pragmatic programmer.")],
        duration_seconds=36000.0,
    )

    tx_id = repo.save_transcription(transcript, audio_metadata=meta, overwrite=True)

    with patch("audiobench.jobs.scheduler.enqueue") as mock_enqueue:
        # Simulate alignment_worker or complete_transcription calling _schedule_biometric_pass directly:
        repo._schedule_biometric_pass([1, 2, 3], str(book), tx_id, run_inline=False)
        mock_enqueue.assert_not_called()


def test_tui_b_toggle_persistence_and_invariants(tmp_path: Path) -> None:
    """Verifies that the TUI [B] toggle persistence path through save_transcription(is_audiobook=True):
    1. Sets user_confirmed=True and audiobook=True in audio_files.tags
    2. Clears pending_privacy_review from the review queue
    3. Guarantees privacy_tier=0 on saved segments
    4. Bypasses biometric pass
    5. Survives subsequent re-scans without re-prompting.
    """
    import json
    from unittest.mock import MagicMock
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.transcribe.transcription_result import Transcript, Segment, AudioMetadata
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, SegmentRecord

    repo = TranscriptionRepository()
    repo._schedule_biometric_pass = MagicMock()
    repo._import_to_library = lambda p: str(p)

    audio_file = tmp_path / "Ambiguous Speaker - Deep Work.mp3"
    audio_file.touch()

    meta = AudioMetadata(
        file_path=str(audio_file),
        file_name=audio_file.name,
        file_size_bytes=2000,
        format="mp3",
        duration_seconds=1200.0,
        sample_rate=16000,
        channels=1,
        file_hash="mock_hash_tui_b_toggle",
    )
    transcript = Transcript(
        segments=[Segment(id=0, start=0.0, end=5.0, text="Deep work chapter notes.")],
        duration_seconds=1200.0,
    )

    # 1. First save without confirmation (Class B soft heuristic: stem split)
    tx_id_1 = repo.save_transcription(transcript, audio_metadata=meta, overwrite=True, is_audiobook=False)
    repo._schedule_biometric_pass.assert_not_called()
    pending = repo.get_pending_privacy_reviews()
    assert any(p["file_name"] == audio_file.name for p in pending)

    # 2. Operator toggles [B] in TUI and imports with is_audiobook=True
    tx_id_2 = repo.save_transcription(transcript, audio_metadata=meta, overwrite=True, is_audiobook=True)
    repo._schedule_biometric_pass.assert_not_called()

    # Verify pending review is CLEARED
    pending_after = repo.get_pending_privacy_reviews()
    assert not any(p["file_name"] == audio_file.name for p in pending_after)

    # Verify tags persistence in DB: user_confirmed=True, audiobook=True
    with get_session() as session:
        audio_rec = session.query(AudioFileRecord).filter_by(file_name=audio_file.name).first()
        assert audio_rec is not None
        tags = json.loads(audio_rec.tags)
        assert tags.get("user_confirmed") is True
        assert tags.get("audiobook") is True
        assert "pending_privacy_review" not in tags

        # Verify segments have privacy_tier == 0
        segs = session.query(SegmentRecord).filter_by(transcription_id=tx_id_2).all()
        assert all(s.privacy_tier == 0 for s in segs)

    # 3. Subsequent scan without is_audiobook flag recognizes persisted confirmation
    from audiobench.media.classifier import classify_media
    cls_result = classify_media(file_path=audio_file, metadata={"tags": tags})
    assert cls_result.is_autonomous_bypass is True
    assert "persisted_user_confirmation" in cls_result.evidence



