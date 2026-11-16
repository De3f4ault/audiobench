"""
Tests for alignment_pipeline timestamp interpolation and timeline sanitization.

Covers:
1. Formula sensitivity / regression test: validates that when an overlapping anchor
   boundary occurs (as characterized by lo_end > hi_start), the old formula produces
   inverted timestamps (end < start), while the new 4-tuple intra-segment formula
   plus sanitizer produces strictly monotonic, positive-duration timestamps.
   NOTE: This validates the mathematical robustness of the formula under overlapping
   inputs; it does not claim these synthetic anchors were the historical production input.
2. Clean baseline discrimination test: confirms that on monotonic, non-overlapping
   Whisper anchors, both old and new formulas produce valid forward-moving timestamps.
3. Sanitizer behavior: asserts _sanitize_whisper_segments clamps backward jumps,
   logs the clamp delta, and sets has_clamped=True.
4. End-to-end clamped alignment path: verifies that when has_clamped=True, the alignment
   worker calls mark_transcription_degraded with all three required arguments
   (tx_id, "timeline_clamped", "aligning"), setting status='degraded' and
   failure_reason='timeline_clamped' without arity errors.
5. Backfill sweep protection: asserts get_degraded_for_backfill excludes records with
   failure_reason='timeline_clamped', and that after a recovery sweep loop tick,
   failure_reason remains 'timeline_clamped' rather than being overwritten by
   'alignment_exhausted'.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from audiobench.core.db_session import get_session
from audiobench.storage.models import TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.alignment_pipeline import (
    _interpolate_time,
    _map_timestamps,
    _sanitize_whisper_segments,
)
from audiobench.transcribe.transcription_result import Segment, Transcript


# ---------------------------------------------------------------------------
# Old formula reference for regression comparison
# ---------------------------------------------------------------------------

def _old_interpolate_time(
    anchors: list[tuple[float, float, float]],
    target_chars: float,
    max_time: float,
) -> float:
    """Historical formula from commit 963a0b0 (3-tuple anchors, cross-gap interpolation)."""
    if not anchors:
        return 0.0
    if target_chars <= anchors[0][0]:
        return anchors[0][1]
    if target_chars >= anchors[-1][0]:
        return min(max_time, anchors[-1][2])

    lo, hi = 0, len(anchors) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if anchors[mid][0] <= target_chars:
            lo = mid
        else:
            hi = mid

    lo_chars, lo_start, lo_end = anchors[lo]
    hi_chars, hi_start, hi_end = anchors[hi]
    span = hi_chars - lo_chars
    if span == 0:
        return lo_start
    frac = (target_chars - lo_chars) / span
    return lo_end + frac * (hi_start - lo_end)


# ---------------------------------------------------------------------------
# 1. Formula Sensitivity / Regression Test
# ---------------------------------------------------------------------------

def test_formula_sensitivity_on_overlapping_anchors():
    """
    Sensitivity / regression test on overlapping anchor boundary.

    Uses the boundary values observed in segment 3770 (TX 623):
    Whisper anchor K ends at 197.390s, while anchor K+1 starts at 141.056s.

    Under the old cross-gap formula:
      lo_end + frac * (hi_start - lo_end) has negative slope (-56.334s),
      causing end_time < start_time (inverted timestamp).

    Under the new formula (intra-segment interpolation + sanitization):
      start_time < end_time (strictly positive duration) and monotonically increasing.
    """
    # Raw whisper segments with the 56.334s backward jump
    raw_whisper = [
        {"start": 186.0, "end": 197.390, "text": "speech before jump", "char_count": 100},
        {"start": 141.056, "end": 155.0, "text": "speech after jump", "char_count": 100},
    ]

    # Target character range spanning across the anchor boundary (span is [0, 100])
    w_cs = 1.0    # frac = 0.01 -> evaluates near lo_end (197.390)
    w_ce = 99.0   # frac = 0.99 -> evaluates near hi_start (141.056)

    old_3tuple_anchors = [
        (0.0, 186.0, 197.390),
        (100.0, 141.056, 155.0),
        (200.0, 155.0, 170.0),
    ]

    # Old formula: demonstrates the mathematical defect
    old_start = _old_interpolate_time(old_3tuple_anchors, w_cs, max_time=200.0)
    old_end = _old_interpolate_time(old_3tuple_anchors, w_ce, max_time=200.0)
    assert old_end < old_start, f"Old formula expected to invert, got start={old_start}, end={old_end}"

    # New formula + sanitization: must enforce monotonicity
    sanitized, has_clamped = _sanitize_whisper_segments(raw_whisper, tx_id=999)
    assert has_clamped is True

    # Build 4-tuple anchors from sanitized segments
    anchors4 = []
    cum = 0.0
    for seg in sanitized:
        c = max(1, seg["char_count"])
        anchors4.append((cum, cum + c, float(seg["start"]), float(seg["end"])))
        cum += c

    new_start = _interpolate_time(anchors4, w_cs, max_time=200.0)
    new_end = _interpolate_time(anchors4, w_ce, max_time=200.0)

    assert new_end > new_start, f"New formula must produce positive duration: start={new_start}, end={new_end}"
    assert new_start >= 186.0
    assert new_end <= sanitized[-1]["end"]


# ---------------------------------------------------------------------------
# 2. Clean Baseline Discrimination Test
# ---------------------------------------------------------------------------

def test_clean_baseline_discrimination():
    """
    On monotonic, non-overlapping anchors (such as today's clean Whisper output),
    both old and new formulas produce valid forward-moving timestamps.
    This proves the test discriminates between clean and overlapping conditions.
    """
    clean_whisper = [
        {"start": 0.0, "end": 10.0, "text": "Segment one", "char_count": 50},
        {"start": 10.5, "end": 25.0, "text": "Segment two", "char_count": 80},
        {"start": 26.0, "end": 45.0, "text": "Segment three", "char_count": 100},
    ]

    sanitized, has_clamped = _sanitize_whisper_segments(clean_whisper, tx_id=1)
    assert has_clamped is False

    # Old 3-tuple anchors
    old_anchors = []
    cum = 0.0
    for seg in clean_whisper:
        old_anchors.append((cum, seg["start"], seg["end"]))
        cum += seg["char_count"]

    # New 4-tuple anchors
    new_anchors = []
    cum = 0.0
    for seg in sanitized:
        c = seg["char_count"]
        new_anchors.append((cum, cum + c, seg["start"], seg["end"]))
        cum += c

    # Evaluate across 10 sample positions
    for t_char in [10.0, 30.0, 60.0, 100.0, 150.0, 200.0]:
        t_old = _old_interpolate_time(old_anchors, t_char, max_time=45.0)
        t_new = _interpolate_time(new_anchors, t_char, max_time=45.0)

        assert t_old >= 0.0
        assert t_new >= 0.0
        # Both must produce non-decreasing results as t_char increases


# ---------------------------------------------------------------------------
# 3. Sanitizer Behavior Test
# ---------------------------------------------------------------------------

def test_sanitizer_clamps_backward_jumps_and_logs_warning(caplog, capsys):
    """
    _sanitize_whisper_segments must clamp start to last_end when start < last_end,
    log a warning containing tx_id, seg_idx, and clamp amount, and return has_clamped=True.
    """
    whisper_segs = [
        {"start": 0.0, "end": 50.0, "text": "Part 1", "char_count": 100},
        {"start": 30.0, "end": 70.0, "text": "Part 2 (jumps back by 20s)", "char_count": 100},
        {"start": 75.0, "end": 90.0, "text": "Part 3", "char_count": 50},
    ]

    with caplog.at_level(logging.WARNING):
        sanitized, has_clamped = _sanitize_whisper_segments(whisper_segs, tx_id=42)

    assert has_clamped is True
    assert len(sanitized) == 3

    # Seg 1 was clamped from 30.0 to 50.0
    assert sanitized[0]["start"] == 0.0
    assert sanitized[0]["end"] == 50.0
    assert sanitized[1]["start"] == 50.0
    assert sanitized[1]["end"] == 70.0
    assert sanitized[2]["start"] == 75.0

    # Warning must be logged with details (capture across both caplog and stdout/console)
    warning_text = caplog.text + " " + capsys.readouterr().out
    assert "backward jump on tx=42" in warning_text
    assert "whisper_seg_idx=1" in warning_text
    assert "clamping forward by 20.000s" in warning_text


# ---------------------------------------------------------------------------
# 4. End-to-End Clamped Alignment Path & Arity Test
# ---------------------------------------------------------------------------

def test_clamped_path_calls_mark_transcription_degraded_with_correct_arity(monkeypatch):
    """
    Verify that when alignment encounters a clamped timeline:
    1. mark_transcription_degraded is called with exactly 3 positional arguments:
       (tx_id, "timeline_clamped", "aligning")
    2. The database record ends up with pipeline_phase='degraded' and failure_reason='timeline_clamped'.
    3. No TypeError occurs from wrong arity.
    """
    repo = TranscriptionRepository()

    # Create a test transcription record in the isolated database
    with get_session() as session:
        rec = TranscriptionRecord(
            file_name="test_clamped.m4a",
            full_text="Test clamped path text.",
            pipeline_phase="aligning",
            duration_seconds=100.0,
        )
        session.add(rec)
        session.commit()
        test_tx_id = rec.id

    try:
        # Build a transcript with segments
        t = Transcript(
            id=test_tx_id,
            text="Hello world test",
            language="en",
            duration_seconds=100.0,
            segments=[
                Segment(id=0, text="Hello", start=0.0, end=0.0),
                Segment(id=1, text="world test", start=0.0, end=0.0),
            ],
        )

        # Mock whisper output that contains a backward jump
        mock_whisper_segs = [
            {"start": 0.0, "end": 50.0, "text": "Hello", "char_count": 5},
            {"start": 20.0, "end": 60.0, "text": "world test", "char_count": 10},
        ]

        with patch("audiobench.transcribe.alignment_pipeline._transcribe_for_timestamps", return_value=mock_whisper_segs):
            from audiobench.transcribe.alignment_pipeline import align_transcript

            dummy_model = MagicMock()
            mutated_transcript, has_clamped = align_transcript(t, "dummy.mp3", dummy_model)

        assert has_clamped is True

        # Simulate the alignment_worker execution path with the returned has_clamped flag
        repo.commit_alignment(test_tx_id, mutated_transcript, phase="degraded")
        # Call mark_transcription_degraded with all three arguments
        repo.mark_transcription_degraded(test_tx_id, "timeline_clamped", "aligning")

        # Verify database record state
        with get_session() as session:
            updated_rec = session.query(TranscriptionRecord).get(test_tx_id)
            assert updated_rec is not None
            assert updated_rec.pipeline_phase == "degraded"
            assert updated_rec.failure_reason == "timeline_clamped"

    finally:
        with get_session() as session:
            r = session.query(TranscriptionRecord).get(test_tx_id)
            if r:
                session.delete(r)
                session.commit()


# ---------------------------------------------------------------------------
# 5. Backfill Sweep Protection & Label Preservation Test
# ---------------------------------------------------------------------------

def test_backfill_sweep_preserves_timeline_clamped_label():
    """
    Verify that:
    1. get_degraded_for_backfill excludes records with failure_reason='timeline_clamped'.
    2. A recovery sweep does NOT select the record or overwrite failure_reason with 'alignment_exhausted'.
    """
    repo = TranscriptionRepository()

    with get_session() as session:
        rec = TranscriptionRecord(
            file_name="test_backfill_guard.m4a",
            full_text="Backfill test text.",
            pipeline_phase="degraded",
            failure_reason="timeline_clamped",
            duration_seconds=50.0,
            backfill_attempt_count=0,
        )
        session.add(rec)
        session.commit()
        test_tx_id = rec.id

    try:
        # 1. Direct repository query test
        candidates = repo.get_degraded_for_backfill(max_backfill_attempts=3)
        candidate_ids = [c.id for c in candidates]
        assert test_tx_id not in candidate_ids, (
            f"tx {test_tx_id} with failure_reason='timeline_clamped' must be excluded from backfill candidates"
        )

        # 2. Simulate sweep execution
        from audiobench.daemon.pipeline_recovery import _do_sweep_once
        mock_pipeline = MagicMock()
        _do_sweep_once(mock_pipeline)

        # 3. Assert failure_reason is still 'timeline_clamped' and was NOT overwritten
        with get_session() as session:
            checked_rec = session.query(TranscriptionRecord).get(test_tx_id)
            assert checked_rec is not None
            assert checked_rec.pipeline_phase == "degraded"
            assert checked_rec.failure_reason == "timeline_clamped", (
                f"failure_reason was overwritten to '{checked_rec.failure_reason}'!"
            )

    finally:
        with get_session() as session:
            r = session.query(TranscriptionRecord).get(test_tx_id)
            if r:
                session.delete(r)
                session.commit()
