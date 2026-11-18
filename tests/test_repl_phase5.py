"""Tests for Phase 5 REPL backslash commands.

Covers:
- DaemonClient: four new playback methods send correct IPC payloads
- \\loop: timestamp parsing, set/clear/show routing, end<=start guard
- \\filter: set/clear/show routing
- \\chapters: nothing-playing guard, jump routing, list rendering
- \\read: daemon-path guard (isinstance DaemonClient → honest message),
         local-path segment resolution, nothing-playing guard
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest


# ── DaemonClient method wrappers ──────────────────────────────────────────────


class TestDaemonClientNewMethods:
    """Verify the four new DaemonClient methods send the right IPC payloads."""

    def _make_client(self):
        from audiobench.daemon.client import DaemonClient
        client = DaemonClient.__new__(DaemonClient)
        client._send = MagicMock(return_value={"ok": True})
        return client

    def test_playback_loop_segment(self):
        c = self._make_client()
        c.playback_loop_segment(90.0, 165.5)
        c._send.assert_called_once_with(
            "playback_loop_segment", {"start": 90.0, "end": 165.5}
        )

    def test_playback_clear_loop(self):
        c = self._make_client()
        c.playback_clear_loop()
        c._send.assert_called_once_with("playback_clear_loop", {})

    def test_playback_set_filter(self):
        c = self._make_client()
        c.playback_set_filter("bass=g=5")
        c._send.assert_called_once_with("playback_set_filter", {"filter_str": "bass=g=5"})

    def test_playback_set_filter_clear(self):
        """Empty string clears the filter."""
        c = self._make_client()
        c.playback_set_filter("")
        c._send.assert_called_once_with("playback_set_filter", {"filter_str": ""})

    def test_playback_chapter_telemetry(self):
        c = self._make_client()
        c.playback_chapter_telemetry([1, 2, 3])
        c._send.assert_called_once_with(
            "playback_chapter_telemetry", {"segment_ids": [1, 2, 3]}
        )


# ── \loop ──────────────────────────────────────────────────────────────────────


def _make_status_from_service(**overrides) -> dict:
    """Generate a mock status dictionary derived directly from PlaybackService schema."""
    from audiobench.playback.service import PlaybackService
    base = PlaybackService().status()
    base.update(overrides)
    return base


class TestCmdLoop:
    def _make_session(self):
        return MagicMock()

    def _mock_client(self, status=None):
        client = MagicMock()
        client.playback_status.return_value = status or _make_status_from_service(playing=True, state="playing")
        client.playback_loop_segment.return_value = _make_status_from_service(
            playing=True, state="playing", position=100.0, loop_a=90.0, loop_b=165.0
        )
        client.playback_clear_loop.return_value = _make_status_from_service(
            playing=True, state="playing", position=100.0, loop_a=None, loop_b=None
        )
        return client

    def test_loop_set_seconds(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "90 165")
        client.playback_loop_segment.assert_called_once_with(90.0, 165.0)
        assert any("Loop requested" in p for p in printed)

    def test_loop_set_mmss(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "1:30 2:45")
        client.playback_loop_segment.assert_called_once_with(90.0, 165.0)
        assert any("Loop requested" in p for p in printed)

    def test_loop_clear(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "clear")
        client.playback_clear_loop.assert_called_once()
        client.playback_loop_segment.assert_not_called()
        assert any("Loop cleared" in p for p in printed)

    def test_loop_set_error_handled(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        client.playback_loop_segment.return_value = {"error": "mpv socket closed"}
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "90 165")
        assert any("Could not set loop" in p and "mpv socket closed" in p for p in printed)

    def test_loop_end_before_start_rejected(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "2:00 1:30")
        client.playback_loop_segment.assert_not_called()
        assert any("must be after" in p for p in printed)

    def test_loop_bare_shows_status(self):
        """Bare \\loop with no args shows current loop bounds."""
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client(status={"playing": True, "loop_a": 90.0, "loop_b": 165.0})
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "")
        client.playback_loop_segment.assert_not_called()
        assert any("A-B loop:" in p and "01:30" in p and "02:45" in p for p in printed)

    def test_loop_wrong_arg_count_shows_usage(self):
        from audiobench.cli.repl.backslash_commands import cmd_loop
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_loop(self._make_session(), "1:30")  # only one timestamp
        client.playback_loop_segment.assert_not_called()
        assert any("Usage" in p or "usage" in p for p in printed)


# ── \filter ────────────────────────────────────────────────────────────────────


class TestCmdFilter:
    def _make_session(self):
        return MagicMock()

    def _mock_client(self, filter_val=None):
        client = MagicMock()
        client.playback_status.return_value = _make_status_from_service(
            playing=True, state="playing", filter=filter_val or ""
        )
        client.playback_set_filter.return_value = _make_status_from_service(
            playing=True, state="playing", filter=filter_val or ""
        )
        return client

    def test_filter_set(self):
        from audiobench.cli.repl.backslash_commands import cmd_filter
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_filter(self._make_session(), "bass=g=5")
        client.playback_set_filter.assert_called_once_with("bass=g=5")
        assert any("Filter applied" in p and "bass=g=5" in p for p in printed)

    def test_filter_clear(self):
        from audiobench.cli.repl.backslash_commands import cmd_filter
        client = self._mock_client()
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_filter(self._make_session(), "clear")
        client.playback_set_filter.assert_called_once_with("")
        assert any("Filter cleared" in p for p in printed)

    def test_filter_set_error_handled(self):
        from audiobench.cli.repl.backslash_commands import cmd_filter
        client = self._mock_client()
        client.playback_set_filter.return_value = {"error": "bad filter string"}
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_filter(self._make_session(), "invalid=???")
        assert any("Could not apply filter" in p and "bad filter string" in p for p in printed)

    def test_filter_bare_shows_active_filter(self):
        from audiobench.cli.repl.backslash_commands import cmd_filter
        client = self._mock_client(filter_val="treble=g=-3")
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_filter(self._make_session(), "")
        client.playback_set_filter.assert_not_called()
        assert any("Active filter" in p and "treble=g=-3" in p for p in printed)

    def test_mock_status_schema_matches_real_service(self):
        """In-test contract assertion: mock status keys must match PlaybackService.status() exactly."""
        from audiobench.playback.service import PlaybackService
        real_keys = set(PlaybackService().status().keys())
        mock_keys = set(_make_status_from_service().keys())
        assert mock_keys == real_keys, f"Schema mismatch: {mock_keys ^ real_keys}"


# ── \chapters ──────────────────────────────────────────────────────────────────


class TestCmdChapters:
    def _make_session(self):
        return MagicMock()

    def test_nothing_playing_returns_early(self):
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        client = MagicMock()
        client.playback_status.return_value = {"playing": False}
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_chapters(self._make_session(), "")
        assert any("Nothing" in p or "nothing" in p for p in printed)

    def test_no_audio_file_id_returns_early(self):
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        client = MagicMock()
        client.playback_status.return_value = {"playing": True, "audio_file_id": None}
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_chapters(self._make_session(), "")
        assert any("audio file ID" in p for p in printed)

    def test_jump_to_chapter_calls_seek(self):
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        from audiobench.chapters.cue_parser import ChapterInfo

        chapter = ChapterInfo(
            index=2,
            title="The Middle",
            start_time=180.0,
            end_time=360.0,
            is_ghost=False,
            id=10,
        )

        client = MagicMock()
        client.playback_status.return_value = {"playing": True, "audio_file_id": 7, "position": 50.0}

        mock_repo = MagicMock()
        mock_repo.get_chapters_for_file.return_value = [chapter]

        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.storage.chapter_repository.get_chapter_repo", return_value=mock_repo),
            patch("audiobench.cli.repl.backslash_commands.console"),
        ):
            cmd_chapters(self._make_session(), "2")

        client.playback_seek.assert_called_once_with(180.0)

    def test_list_chapters_renders_properly(self):
        """Verify bare \\chapters renders real ChapterInfo objects without AttributeError."""
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        from audiobench.chapters.cue_parser import ChapterInfo

        chapters = [
            ChapterInfo(index=0, title="Intro", start_time=0.0, end_time=100.0, is_ghost=False, id=1),
            ChapterInfo(index=1, title="Core Chapter", start_time=100.0, end_time=200.0, is_ghost=False, id=2),
            ChapterInfo(index=2, title="Conclusion", start_time=200.0, end_time=300.0, is_ghost=False, id=3),
        ]

        client = MagicMock()
        # Playing at position 150.0 (inside Chapter 1)
        client.playback_status.return_value = {
            "playing": True,
            "audio_file_id": 998,
            "position": 150.0,
        }

        mock_repo = MagicMock()
        mock_repo.get_chapters_for_file.return_value = chapters

        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.storage.chapter_repository.get_chapter_repo", return_value=mock_repo),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_chapters(self._make_session(), "")

        all_text = " ".join(printed)
        assert "Book Progress:" in all_text
        assert "Intro" in all_text
        assert "Core Chapter" in all_text
        assert "Conclusion" in all_text
        assert "▶" in all_text  # Active chapter indicator
        assert "ACTIVE" in all_text

    def test_list_chapters_all_renders_table(self):
        """Verify \\chapters all renders the full table format."""
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        from audiobench.chapters.cue_parser import ChapterInfo

        chapters = [
            ChapterInfo(index=0, title="Intro", start_time=0.0, end_time=100.0, is_ghost=False, id=1),
            ChapterInfo(index=1, title="Core Chapter", start_time=100.0, end_time=200.0, is_ghost=False, id=2),
            ChapterInfo(index=2, title="Conclusion", start_time=200.0, end_time=300.0, is_ghost=False, id=3),
        ]

        client = MagicMock()
        client.playback_status.return_value = {
            "playing": True,
            "audio_file_id": 998,
            "position": 150.0,
        }

        mock_repo = MagicMock()
        mock_repo.get_chapters_for_file.return_value = chapters

        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.storage.chapter_repository.get_chapter_repo", return_value=mock_repo),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_chapters(self._make_session(), "all")

        all_text = " ".join(printed)
        assert "Chapters (3)" in all_text
        assert "Intro" in all_text
        assert "Core Chapter" in all_text
        assert "Conclusion" in all_text
        assert "▶" in all_text

    def test_chapters_hud_renders_transcript_and_timeline(self):
        """Verify HUD renders timeline bar, runway, and 3-tone transcript lines."""
        from audiobench.cli.repl.backslash_commands import cmd_chapters
        from audiobench.chapters.cue_parser import ChapterInfo

        chapters = [
            ChapterInfo(index=0, title="Intro", start_time=0.0, end_time=100.0, is_ghost=False, id=1),
            ChapterInfo(index=1, title="Core Chapter", start_time=100.0, end_time=200.0, is_ghost=False, id=2),
            ChapterInfo(index=2, title="Conclusion", start_time=200.0, end_time=300.0, is_ghost=False, id=3),
        ]

        client = MagicMock()
        client.playback_status.return_value = {
            "playing": True,
            "audio_file_id": 998,
            "position": 150.0,
            "duration": 300.0,
        }
        client.playback_context.return_value = {
            "segments": [
                {"norm_start": 120.0, "text": "First thought of core chapter"},
                {"norm_start": 150.0, "text": "Now speaking this active line"},
                {"norm_start": 180.0, "text": "Next upcoming thought"},
            ],
            "current_idx": 1,
            "position": 150.0,
        }

        mock_repo = MagicMock()
        mock_repo.get_chapters_for_file.return_value = chapters

        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=client),
            patch("audiobench.storage.chapter_repository.get_chapter_repo", return_value=mock_repo),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.size.width = 80
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_chapters(self._make_session(), "")

        all_text = " ".join(printed)
        assert "Book Progress:" in all_text
        assert "Chapter 1 of 3" in all_text
        assert "▲" in all_text
        assert "First thought of core chapter" in all_text
        assert "Now speaking this active line" in all_text
        assert "Next upcoming thought" in all_text
        assert "Tip: \\chapters <n> to jump" in all_text


# ── \read ───────────────────────────────────────────────────────────────────────


class TestCmdRead:
    def _make_session(self):
        return MagicMock()

    def test_daemon_path_opens_reader_via_proxy(self):
        """When client is DaemonClient, wrap in DaemonPlaybackProxy and launch open_reader."""
        from audiobench.cli.repl.backslash_commands import cmd_read
        from audiobench.daemon.client import DaemonClient
        from audiobench.playback.daemon_proxy import DaemonPlaybackProxy

        daemon_client = MagicMock(spec=DaemonClient)
        daemon_client.playback_status.return_value = {
            "playing": True,
            "transcription_id": 42,
            "audio_file_id": 7,
            "last_eof_reason": None,
        }

        mock_record = {
            "id": 42,
            "file_name": "test.mp3",
            "audio_file_id": 7,
            "segments": [{"norm_start": 0.0, "norm_end": 5.0, "text": "Hello"}],
        }
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = mock_record

        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=daemon_client),
            patch("audiobench.storage.repository.TranscriptionRepository", return_value=mock_repo),
            patch("audiobench.cli.tui.chapter_reader.open_reader") as mock_reader,
            patch("audiobench.cli.repl.backslash_commands.console"),
        ):
            cmd_read(self._make_session(), "")

        mock_reader.assert_called_once()
        service_arg = mock_reader.call_args[0][0]
        assert isinstance(service_arg, DaemonPlaybackProxy)

    def test_local_path_nothing_playing_no_arg_returns_warning(self):
        """Local client, nothing playing, no transcript arg → warning message."""
        from audiobench.cli.repl.backslash_commands import cmd_read
        from audiobench.daemon.local_client import LocalRetrievalClient

        local_client = MagicMock(spec=LocalRetrievalClient)
        mock_service = MagicMock()
        mock_service.status.return_value = {
            "playing": False, "transcription_id": None, "audio_file_id": None
        }
        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=local_client),
            patch("audiobench.playback.service.get_playback_service", return_value=mock_service),
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_read(self._make_session(), "")

        assert any("Nothing" in p or "nothing" in p for p in printed)

    def test_local_path_opens_reader_when_playing(self):
        """Local client, track playing with transcription_id → open_reader called."""
        from audiobench.cli.repl.backslash_commands import cmd_read
        from audiobench.daemon.local_client import LocalRetrievalClient

        local_client = MagicMock(spec=LocalRetrievalClient)
        mock_service = MagicMock()
        mock_service.status.return_value = {
            "playing": True,
            "transcription_id": 42,
            "audio_file_id": 7,
        }

        mock_record = {
            "id": 42,
            "file_name": "test.mp3",
            "audio_file_id": 7,
            "segments": [{"norm_start": 0.0, "norm_end": 5.0, "text": "Hello"}],
        }
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = mock_record

        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=local_client),
            patch("audiobench.playback.service.get_playback_service", return_value=mock_service),
            patch("audiobench.storage.repository.TranscriptionRepository", return_value=mock_repo),
            patch("audiobench.cli.tui.chapter_reader.open_reader") as mock_reader,
            patch("audiobench.cli.repl.backslash_commands.console"),
        ):
            cmd_read(self._make_session(), "")

        mock_reader.assert_called_once()
        call_kwargs = mock_reader.call_args
        assert call_kwargs[0][0] is mock_service  # service is first positional arg

    def test_local_path_no_segments_prints_warning(self):
        """Local client, transcript found but no segments → warning, no reader open."""
        from audiobench.cli.repl.backslash_commands import cmd_read
        from audiobench.daemon.local_client import LocalRetrievalClient

        local_client = MagicMock(spec=LocalRetrievalClient)
        mock_service = MagicMock()
        mock_service.status.return_value = {
            "playing": True,
            "transcription_id": 42,
            "audio_file_id": 7,
        }

        mock_record = {"id": 42, "file_name": "test.mp3", "audio_file_id": 7, "segments": []}
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = mock_record

        printed = []
        with (
            patch("audiobench.daemon.factory.get_daemon_client", return_value=local_client),
            patch("audiobench.playback.service.get_playback_service", return_value=mock_service),
            patch("audiobench.storage.repository.TranscriptionRepository", return_value=mock_repo),
            patch("audiobench.cli.tui.chapter_reader.open_reader") as mock_reader,
            patch("audiobench.cli.repl.backslash_commands.console") as mock_con,
        ):
            mock_con.print.side_effect = lambda m="": printed.append(str(m))
            cmd_read(self._make_session(), "")

        mock_reader.assert_not_called()
        assert any("No transcript" in p or "segment" in p.lower() for p in printed)
