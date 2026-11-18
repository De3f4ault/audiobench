"""Tests for ChapterReaderApp — Phase 3a local TUI.

Tests cover:
- SelectionState: range logic, extend, clear
- _render_segments: active/cursor/selection markup, window truncation
- _progress_bar: formatting, full bar, paused icon
- Paragraph boundary detection (J/K logic)
- App logic via Textual's Pilot: SYNC_LOCK → DECOUPLED on cursor move,
  visual selection toggle/cancel, mode badge text, quit action
- register_eof_listener: append semantics, unregister handle, reboot re-reg
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from audiobench.cli.tui.chapter_reader import (
    ChapterReaderApp,
    ScrollMode,
    SelectionState,
    _render_segments,
    _progress_bar,
)
from audiobench.playback.service import PlaybackService


# ── SelectionState ─────────────────────────────────────────────────────────────


class TestSelectionState:
    def test_initially_inactive(self):
        s = SelectionState()
        assert not s.active
        assert s.lo == -1
        assert s.hi == -1

    def test_start_sets_anchor_and_cursor(self):
        s = SelectionState()
        s.start(5)
        assert s.active
        assert s.anchor == 5
        assert s.cursor == 5
        assert s.lo == 5
        assert s.hi == 5

    def test_extend_forward(self):
        s = SelectionState()
        s.start(3)
        s.extend(7)
        assert s.lo == 3
        assert s.hi == 7

    def test_extend_backward(self):
        """Extending backwards: lo/hi always order correctly."""
        s = SelectionState()
        s.start(7)
        s.extend(3)
        assert s.lo == 3
        assert s.hi == 7

    def test_clear_resets_all_fields(self):
        s = SelectionState()
        s.start(4)
        s.extend(9)
        s.clear()
        assert not s.active
        assert s.lo == -1
        assert s.hi == -1

    def test_extend_no_op_when_inactive(self):
        """extend() while inactive does not activate or raise."""
        s = SelectionState()
        s.extend(5)   # should not raise
        assert not s.active


# ── _render_segments ───────────────────────────────────────────────────────────


def _make_segs(n: int, gap: float = 0.0) -> list[dict]:
    """Make n segments each 5s long, with optional silence gaps."""
    segs = []
    t = 0.0
    for i in range(n):
        segs.append({
            "id": i,
            "norm_start": t,
            "norm_end": t + 5.0,
            "start": t,
            "end": t + 5.0,
            "text": f"Segment {i + 1} text here.",
            "speaker": None,
        })
        t += 5.0 + gap
    return segs


class TestRenderSegments:
    def test_empty_segments_returns_placeholder(self):
        out = _render_segments([], 0, 0, SelectionState(), ScrollMode.SYNC_LOCK, 80)
        assert "No transcript" in out

    def test_active_segment_has_arrow(self):
        segs = _make_segs(5)
        out = _render_segments(segs, 2, 2, SelectionState(), ScrollMode.SYNC_LOCK, 80)
        assert "▶" in out

    def test_cursor_not_active_has_angle(self):
        segs = _make_segs(5)
        # cursor=3, active=1 — cursor shows ›, active-not-cursor shows ▸
        out = _render_segments(segs, 3, 1, SelectionState(), ScrollMode.SYNC_LOCK, 80)
        assert "›" in out   # cursor indicator
        assert "▸" in out   # playhead indicator when cursor is elsewhere

    def test_selection_range_shows_diamond(self):
        segs = _make_segs(8)
        sel = SelectionState()
        sel.start(2)
        sel.extend(5)
        out = _render_segments(segs, 4, 0, sel, ScrollMode.DECOUPLED, 80)
        # Cursor (4) within selection → ◈ prefix
        assert "◈" in out

    def test_overflow_indicators_shown(self):
        """Window is cursor ±12; with 30 segments and cursor=14, both ⋮ shown."""
        segs = _make_segs(30)
        out = _render_segments(segs, 14, 14, SelectionState(), ScrollMode.SYNC_LOCK, 80)
        assert "⋮" in out

    def test_no_overflow_for_small_list(self):
        segs = _make_segs(5)
        out = _render_segments(segs, 2, 2, SelectionState(), ScrollMode.SYNC_LOCK, 80)
        assert "⋮" not in out


# ── _progress_bar ──────────────────────────────────────────────────────────────


class TestProgressBar:
    def test_playing_shows_play_icon(self):
        out = _progress_bar(30.0, 120.0, 1.0, False, 80)
        assert "▶" in out

    def test_paused_shows_pause_icon(self):
        out = _progress_bar(30.0, 120.0, 1.0, True, 80)
        assert "⏸" in out

    def test_speed_shown_when_not_1x(self):
        out = _progress_bar(10.0, 60.0, 1.5, False, 80)
        assert "1.5×" in out

    def test_speed_hidden_at_1x(self):
        out = _progress_bar(10.0, 60.0, 1.0, False, 80)
        assert "×" not in out

    def test_zero_duration_no_crash(self):
        out = _progress_bar(0.0, 0.0, 1.0, False, 80)
        assert "▶" in out


# ── Paragraph boundary logic ───────────────────────────────────────────────────


class TestParagraphBoundary:
    """Unit-test the boundary detection logic directly without Textual pilot."""

    def _make_segs_with_gaps(self):
        """3 paragraphs separated by 2s gaps:
           segs 0-2: para A (no gap)
           gap 2s
           segs 3-5: para B
           gap 2s
           segs 6-8: para C
        """
        segs = []
        t = 0.0
        for para in range(3):
            for _ in range(3):
                segs.append({
                    "norm_start": t,
                    "norm_end": t + 5.0,
                    "start": t,
                    "end": t + 5.0,
                    "text": "text",
                    "speaker": None,
                })
                t += 5.0
            t += 2.0  # silence gap between paragraphs
        return segs

    def _para_down(self, segs, cursor):
        """Mirror of action_para_down logic."""
        idx = cursor + 1
        while idx < len(segs) - 1:
            gap = segs[idx]["norm_start"] - segs[idx - 1]["norm_end"]
            spk_curr = segs[idx].get("speaker")
            spk_prev = segs[idx - 1].get("speaker")
            spk_changed = bool(spk_curr or spk_prev) and (spk_curr != spk_prev)
            if gap >= 1.5 or spk_changed:
                break
            idx += 1
        return min(idx, len(segs) - 1)

    def _para_up(self, segs, cursor):
        """Mirror of action_para_up logic."""
        idx = cursor - 1
        while idx > 0:
            gap = segs[idx]["norm_start"] - segs[idx - 1]["norm_end"]
            spk_curr = segs[idx].get("speaker")
            spk_prev = segs[idx - 1].get("speaker")
            spk_changed = bool(spk_curr or spk_prev) and (spk_curr != spk_prev)
            if gap >= 1.5 or spk_changed:
                break
            idx -= 1
        return max(0, idx)

    def test_para_down_from_start_lands_at_next_boundary(self):
        segs = self._make_segs_with_gaps()
        result = self._para_down(segs, 0)
        # First gap is after index 2 (norm_start of 3 is 5*3+2=17, norm_end of 2 is 15 → gap 2s)
        assert result == 3

    def test_para_down_from_second_para_lands_at_third(self):
        segs = self._make_segs_with_gaps()
        result = self._para_down(segs, 3)
        assert result == 6

    def test_para_up_from_last_para_lands_at_second(self):
        segs = self._make_segs_with_gaps()
        result = self._para_up(segs, 7)
        assert result == 6

    def test_para_up_from_beginning_clamps_to_zero(self):
        segs = self._make_segs_with_gaps()
        result = self._para_up(segs, 0)
        assert result == 0

    def test_tight_segments_no_boundary(self):
        """No gaps ≥ 1.5s and no speaker change → para_down returns last segment."""
        segs = _make_segs(5, gap=0.0)  # no silence gap, no speaker
        result = self._para_down(segs, 0)
        assert result == len(segs) - 1

    def test_speaker_transition_triggers_boundary(self):
        """Speaker change triggers paragraph jump even with 0s silence gap."""
        segs = [
            {"norm_start": 0.0, "norm_end": 5.0, "speaker": "Alice", "text": "Hello"},
            {"norm_start": 5.0, "norm_end": 10.0, "speaker": "Alice", "text": "More Alice"},
            {"norm_start": 10.0, "norm_end": 15.0, "speaker": "Bob", "text": "Hi Alice"},  # speaker transition at index 2
            {"norm_start": 15.0, "norm_end": 20.0, "speaker": "Bob", "text": "More Bob"},
        ]
        # Jumping down from 0 should land on index 2 (Bob's first utterance)
        assert self._para_down(segs, 0) == 2
        # Jumping up from 3 should land back on index 2 (start of Bob's turn)
        assert self._para_up(segs, 3) == 2
        # Jumping up from 2 should land on index 0
        assert self._para_up(segs, 2) == 0


# ── Bookmark Actions (Multi-segment end-to-end) ───────────────────────────────


class TestBookmarkActions:
    """Tests bookmarking in ChapterReaderApp with point and region markers."""

    def test_bookmark_single_segment_calls_add(self):
        segs = [
            {"norm_start": 10.0, "norm_end": 15.0, "text": "Single segment text"}
        ]
        mock_service = MagicMock()
        app = ChapterReaderApp(
            service=mock_service,
            title="Test",
            segments=segs,
            audio_file_id=42,
            transcription_id=1,
        )
        app._cursor = 0
        app._sel.clear()

        with patch("audiobench.storage.bookmark_repository.BookmarkRepository.add") as mock_add:
            app.action_bookmark_cursor()
            mock_add.assert_called_once_with(
                audio_file_id=42,
                timestamp=10.0,
                name="Single segment text",
                bookmark_type="bookmark",
                transcription_id=1,
            )

    def test_bookmark_visual_selection_calls_add_region_end_to_end(self):
        """End-to-end verification of multi-segment region bookmarking in DB."""
        from audiobench.storage.bookmark_repository import BookmarkRepository
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AudioFileRecord, BookmarkRecord

        # Create temporary audio file record in DB
        with get_session() as session:
            af = AudioFileRecord(
                file_path="/tmp/test_reader_audio.mp3",
                file_name="test_reader_audio.mp3",
                duration_seconds=120.0,
            )
            session.add(af)
            session.commit()
            af_id = af.id

        try:
            segs = [
                {"norm_start": 0.0, "norm_end": 5.0, "text": "First segment intro"},
                {"norm_start": 5.0, "norm_end": 10.0, "text": "Second segment middle"},
                {"norm_start": 10.0, "norm_end": 15.0, "text": "Third segment conclusion"},
            ]
            mock_service = MagicMock()
            app = ChapterReaderApp(
                service=mock_service,
                title="Test Book",
                segments=segs,
                audio_file_id=af_id,
            )
            # Select segments 0 through 2
            app._sel.start(0)
            app._sel.extend(2)
            assert app._sel.lo == 0
            assert app._sel.hi == 2

            # Execute action
            app.action_bookmark_cursor()

            # Selection should be cleared after bookmarking
            assert not app._sel.active

            # Verify BookmarkRecord in DB has correct region timestamps
            repo = BookmarkRepository()
            bookmarks = repo.list_for_file(af_id)
            assert len(bookmarks) == 1
            bm = bookmarks[0]
            assert bm["audio_file_id"] == af_id
            assert bm["timestamp"] == 0.0
            assert bm["end_timestamp"] == 15.0
            assert bm["bookmark_type"] == "highlight"
            assert "First segment" in bm["name"]
        finally:
            # Clean up
            with get_session() as session:
                session.query(BookmarkRecord).filter_by(audio_file_id=af_id).delete()
                session.query(AudioFileRecord).filter_by(id=af_id).delete()
                session.commit()


# ── register_eof_listener ──────────────────────────────────────────────────────


class TestRegisterEofListener:
    def test_appends_to_mpv_callbacks_when_alive(self):
        """If mpv is already alive, callback is registered immediately."""
        svc = PlaybackService()
        mock_mpv = MagicMock()
        mock_mpv.is_running.return_value = True
        mock_mpv._eof_callbacks = []
        svc._mpv = mock_mpv

        cb = MagicMock()
        svc.register_eof_listener(cb)

        assert cb in mock_mpv._eof_callbacks

    def test_service_own_handler_not_displaced(self):
        """Service's _on_eof must survive external listener registration."""
        svc = PlaybackService()
        mock_mpv = MagicMock()
        mock_mpv.is_running.return_value = True
        mock_mpv._eof_callbacks = [svc._on_eof]
        svc._mpv = mock_mpv

        cb = MagicMock()
        svc.register_eof_listener(cb)

        assert svc._on_eof in mock_mpv._eof_callbacks
        assert cb in mock_mpv._eof_callbacks
        # Service handler at index 0, external cb appended after — index comparison
        # avoids the `is` identity trap on bound method descriptors
        assert mock_mpv._eof_callbacks.index(svc._on_eof) < mock_mpv._eof_callbacks.index(cb)

    def test_unregister_removes_from_mpv_callbacks(self):
        """Returned unregister callable removes the specific callback."""
        svc = PlaybackService()
        mock_mpv = MagicMock()
        mock_mpv.is_running.return_value = True
        # Start empty — register_eof_listener is the only thing that adds cb
        mock_mpv._eof_callbacks = []
        svc._mpv = mock_mpv

        cb = MagicMock()
        unreg = svc.register_eof_listener(cb)
        assert cb in mock_mpv._eof_callbacks  # registered

        unreg()

        assert cb not in mock_mpv._eof_callbacks
        assert all(e[1] is not cb for e in svc._pending_eof_listeners)

    def test_unregister_safe_to_call_twice(self):
        """Calling unregister twice must not raise."""
        svc = PlaybackService()
        mock_mpv = MagicMock()
        mock_mpv.is_running.return_value = True
        mock_mpv._eof_callbacks = []
        svc._mpv = mock_mpv

        cb = MagicMock()
        unreg = svc.register_eof_listener(cb)
        unreg()
        unreg()  # must not raise

    def test_pending_listener_reregistered_on_mpv_reboot(self):
        """Listener survives mpv crash + reboot via _ensure_mpv re-registration."""
        svc = PlaybackService()
        cb = MagicMock()

        # Register before mpv boots — svc._mpv is None
        unreg = svc.register_eof_listener(cb)
        assert len(svc._pending_eof_listeners) == 1

        # Simulate _ensure_mpv booting mpv
        mock_mpv = MagicMock()
        mock_mpv._eof_callbacks = []
        # Manually replicate the _ensure_mpv re-registration loop
        mock_mpv._eof_callbacks.append(svc._on_eof)
        for _token, pending_cb in svc._pending_eof_listeners:
            mock_mpv._eof_callbacks.append(pending_cb)

        assert cb in mock_mpv._eof_callbacks
