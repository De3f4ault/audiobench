"""Tests for Phase 4 changes in play.py.

Covers:
- _build_atempo_filter: boundary cases for the chained-atempo filter builder
- _play_with_lyrics: routing smoke tests (daemon path, service fallback, stats)
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── _build_atempo_filter ───────────────────────────────────────────────────────

from audiobench.cli.commands.play import _build_atempo_filter


class TestBuildAtempoFilter:
    """Unit tests for the ffplay atempo chain builder."""

    def test_normal_in_range_speed(self):
        """Speeds in [0.5, 2.0] produce a single atempo filter."""
        result = _build_atempo_filter(1.5)
        assert result == "atempo=1.500000"
        assert "," not in result  # no chaining needed

    def test_speed_below_min_clamped_to_half(self):
        """Speeds below 0.5 are clamped to 0.5 (ffplay lower bound)."""
        result = _build_atempo_filter(0.2)
        assert result == "atempo=0.500000"

    def test_exact_lower_bound(self):
        """0.5x is on the boundary — single filter, no chain."""
        result = _build_atempo_filter(0.5)
        assert result == "atempo=0.500000"
        assert "," not in result

    def test_exact_upper_in_range(self):
        """2.0x is the upper boundary of a single filter — no chain needed."""
        result = _build_atempo_filter(2.0)
        assert result == "atempo=2.000000"
        assert "," not in result

    def test_speed_3x_chains_two_filters(self):
        """3.0x → atempo=2.0,atempo=1.500000."""
        result = _build_atempo_filter(3.0)
        parts = result.split(",")
        assert len(parts) == 2
        assert parts[0] == "atempo=2.0"
        assert parts[1] == "atempo=1.500000"

    def test_speed_4x_chains_two_filters_at_boundary(self):
        """4.0x → atempo=2.0,atempo=2.000000 (both at upper bound)."""
        result = _build_atempo_filter(4.0)
        parts = result.split(",")
        assert len(parts) == 2
        assert parts[0] == "atempo=2.0"
        assert parts[1] == "atempo=2.000000"

    def test_speed_5x_chains_three_filters(self):
        """5.0x → three filters: 2.0 * 2.0 * 1.25 = 5.0."""
        result = _build_atempo_filter(5.0)
        parts = result.split(",")
        assert len(parts) == 3
        assert parts[0] == "atempo=2.0"
        assert parts[1] == "atempo=2.0"
        assert parts[2] == "atempo=1.250000"

    def test_per_filter_value_never_exceeds_2(self):
        """No individual filter value should exceed 2.0 (ffmpeg hard limit)."""
        for speed in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
            result = _build_atempo_filter(speed)
            for part in result.split(","):
                val = float(part.split("=")[1])
                assert val <= 2.0 + 1e-9, (
                    f"Filter value {val} exceeds 2.0 at speed={speed}: {result}"
                )
                assert val >= 0.5 - 1e-9, (
                    f"Filter value {val} below 0.5 at speed={speed}: {result}"
                )

    def test_product_of_chain_equals_requested_speed(self):
        """Product of all chained atempo values should equal the (clamped) requested speed."""
        for speed in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
            result = _build_atempo_filter(speed)
            product = 1.0
            for part in result.split(","):
                product *= float(part.split("=")[1])
            expected = max(0.5, speed)
            assert math.isclose(product, expected, rel_tol=1e-5), (
                f"Chain product {product:.6f} != expected {expected:.6f} for speed={speed}: {result}"
            )


# ── _play_with_lyrics routing ──────────────────────────────────────────────────


class TestPlayWithLyricsRouting:
    """Smoke tests for the new _play_with_lyrics routing logic.

    Mocks both daemon and service/reader layers so nothing actually plays audio.
    Verifies routing decisions and post-exit stats behaviour.
    """

    _SEGS = [{"norm_start": 0.0, "norm_end": 5.0, "text": "Hello"}]

    def test_service_fallback_when_daemon_unavailable(self):
        """When daemon raises, PlaybackService is used as fallback."""
        from audiobench.cli.commands.play import _play_with_lyrics

        mock_svc = MagicMock()
        mock_svc.status.return_value = {"position": 30.0, "duration": 120.0}

        with (
            patch("audiobench.daemon.factory.get_daemon_client", side_effect=Exception("no daemon")),
            patch("audiobench.playback.service.PlaybackService", return_value=mock_svc) as MockSvc,
            patch("audiobench.cli.tui.chapter_reader.open_reader") as mock_reader,
        ):
            _play_with_lyrics(
                Path("/tmp/test.mp3"),
                self._SEGS,
                "Test File",
                0.0,
                1.5,
                120.0,
                audio_file_id=1,
                transcription_id=2,
            )

            MockSvc.assert_called_once()
            mock_svc.play.assert_called_once_with(
                "/tmp/test.mp3",
                start_pos=0.0,
                speed=1.5,
                audio_file_id=1,
                transcription_id=2,
            )
            # open_reader called with the service as its first positional arg
            mock_reader.assert_called_once()
            assert mock_reader.call_args[0][0] is mock_svc

    def test_post_exit_stats_read_service_status(self):
        """Post-exit stats call service.status() for final position."""
        from audiobench.cli.commands.play import _play_with_lyrics

        mock_svc = MagicMock()
        mock_svc.status.return_value = {"position": 45.0, "duration": 90.0}

        with (
            patch("audiobench.daemon.factory.get_daemon_client", side_effect=Exception("no daemon")),
            patch("audiobench.playback.service.PlaybackService", return_value=mock_svc),
            patch("audiobench.cli.tui.chapter_reader.open_reader"),
            patch("audiobench.cli.commands.play.console") as mock_console,
        ):
            _play_with_lyrics(
                Path("/tmp/test.mp3"),
                self._SEGS,
                "Test",
                0.0,
                1.0,
                90.0,
            )

        mock_svc.status.assert_called()
        # Verify a stat line references the duration percentage
        all_printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "50%" in all_printed  # 45/90 = 50%

    def test_stats_fall_back_gracefully_when_status_raises(self):
        """If service.status() raises, stats fall back to start_seconds without crashing."""
        from audiobench.cli.commands.play import _play_with_lyrics

        mock_svc = MagicMock()
        mock_svc.status.side_effect = RuntimeError("mpv is gone")

        with (
            patch("audiobench.daemon.factory.get_daemon_client", side_effect=Exception("no daemon")),
            patch("audiobench.playback.service.PlaybackService", return_value=mock_svc),
            patch("audiobench.cli.tui.chapter_reader.open_reader"),
            patch("audiobench.cli.commands.play.console"),
        ):
            # Must not raise
            _play_with_lyrics(
                Path("/tmp/test.mp3"),
                self._SEGS,
                "Test",
                10.0,
                1.0,
                60.0,
            )
