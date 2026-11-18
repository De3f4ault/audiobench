"""Tests for Phase 2B (chapter auto-seeding) and 2D (chapter telemetry)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from audiobench.chapters.cue_parser import ChapterInfo
from audiobench.playback.service import PlaybackService


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mpv(chapters=None, duration=100.0):
    mpv = MagicMock()
    mpv.get_chapters.return_value = chapters if chapters is not None else []
    mpv.get_duration.return_value = duration
    return mpv


def _make_chapter_info(index, title="Chapter", start=0.0, end=10.0):
    return ChapterInfo(index=index, title=title, start_time=start, end_time=end, is_ghost=False, id=index + 1)


# Import-path helper: _maybe_seed_chapters uses `from audiobench.storage.chapter_repository import get_chapter_repo`
_REPO_PATH = "audiobench.storage.chapter_repository.get_chapter_repo"
# get_chapter_telemetry uses `from audiobench.core.db_session import get_session`
_SESSION_PATH = "audiobench.core.db_session.get_session"


# ── 2B: _maybe_seed_chapters ──────────────────────────────────────────────────


class TestMaybeSeedChapters:
    """Tests for _maybe_seed_chapters guard and seeding logic."""

    def test_guard_skips_when_real_chapters_exist(self):
        """Guard: any chapter_index > 0 in DB → skip seeding; mpv not queried."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[{"time": 0.0, "title": "A"}, {"time": 50.0, "title": "B"}])

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = [
            _make_chapter_info(0, start=0.0, end=50.0),
            _make_chapter_info(1, start=50.0, end=100.0),  # index 1 > 0 → guard fires
        ]

        with patch(_REPO_PATH, return_value=mock_repo):
            svc._maybe_seed_chapters(mpv, audio_file_id=1)

        mock_repo.save_chapters.assert_not_called()
        mpv.get_chapters.assert_not_called()  # skipped before querying mpv

    def test_guard_allows_seeding_when_only_index_zero_exists(self):
        """Guard: sole index-0 row (ffprobe placeholder) is replaceable."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[
            {"time": 0.0, "title": "Part 1"},
            {"time": 45.0, "title": "Part 2"},
        ], duration=90.0)

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = [
            _make_chapter_info(0, start=0.0, end=90.0),  # only index 0 — placeholder
        ]
        mock_repo.save_chapters.return_value = []

        with patch(_REPO_PATH, return_value=mock_repo):
            svc._maybe_seed_chapters(mpv, audio_file_id=2)

        mock_repo.save_chapters.assert_called_once()
        saved = mock_repo.save_chapters.call_args[0][1]
        assert len(saved) == 2
        assert saved[0].index == 0
        assert saved[1].index == 1
        assert saved[0].start_time == 0.0
        assert saved[1].start_time == 45.0
        assert saved[1].end_time == 90.0  # last chapter uses duration as end

    def test_guard_allows_seeding_when_no_existing_chapters(self):
        """No existing rows at all → seed from mpv."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[{"time": 0.0, "title": "Only Chapter"}], duration=60.0)

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = []
        mock_repo.save_chapters.return_value = []

        with patch(_REPO_PATH, return_value=mock_repo):
            svc._maybe_seed_chapters(mpv, audio_file_id=3)

        mock_repo.save_chapters.assert_called_once()
        saved = mock_repo.save_chapters.call_args[0][1]
        assert len(saved) == 1
        assert saved[0].index == 0
        assert saved[0].title == "Only Chapter"
        assert saved[0].end_time == 60.0

    def test_mpv_returns_no_chapters_skips_save(self):
        """If mpv returns empty chapter list (both attempts), no save is issued."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[])

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = []

        with patch(_REPO_PATH, return_value=mock_repo), \
             patch("time.sleep"):  # suppress 500ms retry sleep in test
            svc._maybe_seed_chapters(mpv, audio_file_id=4)

        mock_repo.save_chapters.assert_not_called()

    def test_chapter_end_times_computed_from_next_chapter_start(self):
        """end_time for each chapter = next chapter's start_time (not duration, except last)."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[
            {"time": 0.0, "title": "Ch1"},
            {"time": 30.0, "title": "Ch2"},
            {"time": 75.0, "title": "Ch3"},
        ], duration=120.0)

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = []
        mock_repo.save_chapters.return_value = []

        with patch(_REPO_PATH, return_value=mock_repo):
            svc._maybe_seed_chapters(mpv, audio_file_id=5)

        saved = mock_repo.save_chapters.call_args[0][1]
        assert saved[0].end_time == 30.0    # next chapter start
        assert saved[1].end_time == 75.0    # next chapter start
        assert saved[2].end_time == 120.0   # file duration (last chapter)

    def test_fallback_title_generated_when_mpv_title_empty(self):
        """Chapters with no title from mpv get 'Chapter N' generated title."""
        svc = PlaybackService()
        mpv = _make_mpv(chapters=[
            {"time": 0.0, "title": ""},
            {"time": 60.0, "title": None},
        ], duration=120.0)

        mock_repo = MagicMock()
        mock_repo.get_chapters.return_value = []
        mock_repo.save_chapters.return_value = []

        with patch(_REPO_PATH, return_value=mock_repo):
            svc._maybe_seed_chapters(mpv, audio_file_id=6)

        saved = mock_repo.save_chapters.call_args[0][1]
        assert saved[0].title == "Chapter 1"
        assert saved[1].title == "Chapter 2"


# ── 2D: get_chapter_telemetry ─────────────────────────────────────────────────


def _mock_session_ctx(rows):
    """Return a context manager mock whose query chain returns rows."""
    mock_session = MagicMock()
    mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = rows
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_session)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, mock_session


class TestGetChapterTelemetry:
    """Tests for the segment_id → chapter metadata mapping."""

    def test_empty_input_returns_empty_dict(self):
        result = PlaybackService.get_chapter_telemetry([])
        assert result == {}

    def test_maps_segment_ids_to_chapter_metadata(self):
        """Returns correct chapter fields, with str keys, for each segment_id with a chapter."""
        rows = [
            (10, 1, 0, "Introduction", 0.0, 30.0),
            (11, 2, 1, "Main Argument", 30.0, 75.0),
        ]
        ctx, _ = _mock_session_ctx(rows)

        with patch(_SESSION_PATH, return_value=ctx):
            result = PlaybackService.get_chapter_telemetry([10, 11])

        # Keys are always strings — consistent across local and daemon paths
        assert result == {
            "10": {"chapter_id": 1, "chapter_index": 0, "title": "Introduction",
                   "start_time": 0.0, "end_time": 30.0},
            "11": {"chapter_id": 2, "chapter_index": 1, "title": "Main Argument",
                   "start_time": 30.0, "end_time": 75.0},
        }

    def test_segments_without_chapter_absent_from_result(self):
        """Segments with NULL chapter_id are INNER JOINed out and absent from result."""
        # Only segment 10 has a chapter; 99 has no chapter_id (JOIN excludes it)
        rows = [(10, 1, 0, "Intro", 0.0, 30.0)]
        ctx, _ = _mock_session_ctx(rows)

        with patch(_SESSION_PATH, return_value=ctx):
            result = PlaybackService.get_chapter_telemetry([10, 99])

        assert "10" in result     # present — has chapter
        assert "99" not in result  # absent — NULL chapter_id excluded by INNER JOIN
        assert 10 not in result    # no int keys in result
        assert 99 not in result

    def test_db_error_returns_empty_dict(self):
        """DB failures return {} rather than raising — callers degrade gracefully."""
        with patch(_SESSION_PATH, side_effect=RuntimeError("DB down")):
            result = PlaybackService.get_chapter_telemetry([1, 2, 3])

        assert result == {}

    def test_server_handler_returns_str_keyed_result_unchanged(self):
        """Server handler passes through str-keyed result from get_chapter_telemetry.

        get_chapter_telemetry() returns str keys canonically — the handler must
        not modify them (no redundant conversion, no accidental int coercion).
        """
        from audiobench.daemon.server import _handle_playback_chapter_telemetry

        fake_result = {"10": {
            "chapter_id": 1, "chapter_index": 0,
            "title": "Ch", "start_time": 0.0, "end_time": 30.0,
        }}
        with patch.object(PlaybackService, "get_chapter_telemetry", return_value=fake_result):
            result = _handle_playback_chapter_telemetry({"segment_ids": [10]})

        assert "10" in result
        assert 10 not in result
        assert result["10"]["title"] == "Ch"
