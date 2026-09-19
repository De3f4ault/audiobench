"""Unit tests for playback controls, timestamp parser, and command dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from rich.console import Console

from audiobench.playback.controls import (
    PLAYBACK_COMMANDS,
    extract_timestamps,
    find_next_timestamp,
    find_prev_timestamp,
    fmt_timestamp,
    fmt_timestamp_slug,
    handle_playback_command,
    parse_timestamp,
    render_playback_status_line,
    render_progress_bar,
)


def test_playback_commands_registry():
    assert "/play" in PLAYBACK_COMMANDS
    assert "/seek" in PLAYBACK_COMMANDS
    assert "/pause" in PLAYBACK_COMMANDS
    assert "/resume" in PLAYBACK_COMMANDS
    assert "/explain" in PLAYBACK_COMMANDS
    assert "/clip" in PLAYBACK_COMMANDS


def test_parse_timestamp():
    assert parse_timestamp("0:00") == 0.0
    assert parse_timestamp("6:30") == 390.0
    assert parse_timestamp("06:30") == 390.0
    assert parse_timestamp("1:02:30") == 3750.0
    assert parse_timestamp("10:30.5") == 630.5
    assert parse_timestamp("45") == 45.0
    assert parse_timestamp("") is None
    assert parse_timestamp("abc") is None
    assert parse_timestamp("invalid:time") is None


def test_fmt_timestamp():
    assert fmt_timestamp(0.0) == "00:00"
    assert fmt_timestamp(390.0) == "06:30"
    assert fmt_timestamp(3750.0) == "01:02:30"
    assert fmt_timestamp(45.0) == "00:45"


def test_fmt_timestamp_slug():
    assert fmt_timestamp_slug(390.0) == "06m30s"
    assert fmt_timestamp_slug(3750.0) == "01h02m30s"
    assert fmt_timestamp_slug(0.0) == "00m00s"


def test_extract_timestamps_sorted_and_clustered():
    # Out of chronological order in the text
    text = (
        "He concluded at (1:05:00) after speaking.\n"
        "Earlier at (6:30), he discussed anxiety.\n"
        "And right at (6:32), he repeated that same thought.\n"
        "The introduction started at (0:00)."
    )
    extracted = extract_timestamps(text)
    # Must be sorted chronologically and clustered (6:32 dropped because it's within 3s of 6:30)
    expected = [
        ("0:00", 0.0),
        ("6:30", 390.0),
        ("1:05:00", 3900.0),
    ]
    assert extracted == expected


def test_render_playback_status_line():
    client = MagicMock()

    # When not playing
    client.playback_status.return_value = {"playing": False}
    assert render_playback_status_line(client) is None

    # When playing
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "position": 390.0,
        "duration": 862.0,
        "speed": 1.0,
    }
    assert render_playback_status_line(client) == "[PLAY] 06:30/14:22"

    # When paused with non-1.0 speed
    client.playback_status.return_value = {
        "playing": True,
        "paused": True,
        "position": 390.0,
        "duration": 862.0,
        "speed": 1.5,
    }
    assert render_playback_status_line(client) == "[PAUSE] 06:30/14:22 [1.5x]"


def test_handle_playback_command():
    client = MagicMock()
    console = Console(quiet=True)

    # /seek
    client.playback_seek.return_value = {"position": 390.0}
    handled = handle_playback_command("/seek", "6:30", client, console)
    assert handled is True
    client.playback_seek.assert_called_with(390.0)

    # /seek relative
    client.playback_seek_relative.return_value = {"position": 405.0}
    handled = handle_playback_command("/seek", "+15", client, console)
    assert handled is True
    client.playback_seek_relative.assert_called_with(15.0)

    # /pause
    client.playback_pause.return_value = {"paused": True}
    handled = handle_playback_command("/pause", "", client, console)
    assert handled is True
    client.playback_pause.assert_called_once()

    # /resume
    client.playback_resume.return_value = {"paused": False}
    handled = handle_playback_command("/resume", "", client, console)
    assert handled is True
    client.playback_resume.assert_called_once()

    # /speed
    client.playback_speed.return_value = {"speed": 1.5}
    handled = handle_playback_command("/speed", "1.5", client, console)
    assert handled is True
    client.playback_speed.assert_called_with(1.5)

    # /stop
    client.playback_stop.return_value = {"playing": False}
    handled = handle_playback_command("/stop", "", client, console)
    assert handled is True
    client.playback_stop.assert_called_once()


def test_handle_playback_command_catches_exception():
    client = MagicMock()
    client.playback_seek.side_effect = RuntimeError("Daemon error [playback_seek]: mpv IPC socket not created")
    console = Console(quiet=True)

    handled = handle_playback_command("/seek", "1:00", client, console)
    assert handled is True


def test_mpv_controller_cleans_up_on_premature_exit(tmp_path):
    from audiobench.playback import MpvController
    from unittest.mock import patch

    sock_path = str(tmp_path / "test_mpv.sock")
    ctrl = MpvController(socket_path=sock_path)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1

    with patch("subprocess.Popen", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="mpv process exited prematurely"):
            ctrl.start("/dummy/audio.mp3")


def test_mpv_controller_kills_proc_on_socket_timeout(tmp_path):
    from audiobench.playback import MpvController
    from unittest.mock import patch

    sock_path = str(tmp_path / "test_mpv.sock")
    ctrl = MpvController(socket_path=sock_path)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="mpv IPC socket not created"):
            ctrl.start("/dummy/audio.mp3")

    mock_proc.kill.assert_called()


def test_find_next_timestamp():
    citations = [("01:20", 80.0), ("03:40", 220.0), ("08:00", 480.0)]
    # Before all citations
    assert find_next_timestamp(citations, 0.0) == (1, "01:20", 80.0)
    # Inside first citation section
    assert find_next_timestamp(citations, 100.0) == (2, "03:40", 220.0)
    # Approaching 2nd citation (before tolerance)
    assert find_next_timestamp(citations, 218.0) == (2, "03:40", 220.0)
    # Within tolerance of 2nd citation (219.5s vs 220.0s) -> moves to 3rd
    assert find_next_timestamp(citations, 219.5) == (3, "08:00", 480.0)
    # Between 2nd and 3rd
    assert find_next_timestamp(citations, 250.0) == (3, "08:00", 480.0)
    # At/past last citation
    assert find_next_timestamp(citations, 480.0) is None
    assert find_next_timestamp(citations, 500.0) is None


def test_find_prev_timestamp():
    citations = [("01:20", 80.0), ("03:40", 220.0), ("08:00", 480.0)]
    # Before first citation
    assert find_prev_timestamp(citations, 10.0) == (1, "01:20", 80.0)
    # Just after 2nd citation starts (<= 3s) -> step back to 1st
    assert find_prev_timestamp(citations, 221.0) == (1, "01:20", 80.0)
    # Well into 2nd citation (> 3s) -> rewind to start of 2nd
    assert find_prev_timestamp(citations, 240.0) == (2, "03:40", 220.0)
    # Well into 3rd citation (> 3s) -> rewind to start of 3rd
    assert find_prev_timestamp(citations, 500.0) == (3, "08:00", 480.0)


def test_render_progress_bar():
    # 50% progress
    bar = render_progress_bar(50.0, 100.0, width=10)
    assert "━" in bar
    assert "░" in bar
    # 0% or negative duration
    assert "─" in render_progress_bar(0.0, 0.0, width=10)


def test_render_playback_status_line_with_citations():
    client = MagicMock()
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "position": 240.0,
        "duration": 600.0,
        "speed": 1.0,
    }
    citations = [("01:20", 80.0), ("03:40", 220.0), ("08:00", 480.0)]
    line = render_playback_status_line(client, citations)
    assert "[PLAY] 04:00/10:00" in line
    assert "[2/3]" in line
    assert "n:next" in line or "jump" in line


def test_next_prev_slash_commands():
    client = MagicMock()
    client.playback_status.return_value = {"playing": True, "position": 100.0}
    console = Console(quiet=True)
    citations = [("01:20", 80.0), ("03:40", 220.0), ("08:00", 480.0)]

    # /next
    handled = handle_playback_command("/next", "", client, console, active_citations=citations)
    assert handled is True
    client.playback_seek.assert_called_with(220.0)

    # /prev
    handled = handle_playback_command("/prev", "", client, console, active_citations=citations)
    assert handled is True
    client.playback_seek.assert_called_with(80.0)

    # /3 direct index
    handled = handle_playback_command("/3", "", client, console, active_citations=citations)
    assert handled is True
    client.playback_seek.assert_called_with(480.0)


def test_lyrics_slash_command(monkeypatch):
    client = MagicMock()
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "position": 50.0,
        "duration": 300.0,
        "speed": 1.0,
        "file": "/tmp/sample.mp3",
    }
    console = Console(record=True)
    mock_show = MagicMock()
    monkeypatch.setattr("audiobench.playback.lyrics_hud.show_daemon_lyrics", mock_show)

    for alias in ("/lyrics", "/karaoke", "/follow", "/lyric"):
        handled = handle_playback_command(alias, "", client, console)
        assert handled is True
    assert mock_show.call_count == 4


def test_show_daemon_lyrics_non_interactive():
    from audiobench.playback.lyrics_hud import show_daemon_lyrics

    client = MagicMock()
    client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "position": 10.0,
        "duration": 60.0,
        "speed": 1.0,
        "file": "/tmp/test.mp3",
    }
    console = Console(record=True)
    show_daemon_lyrics(client, console)
    output = console.export_text()
    assert "Live Lyrics Follow" in output
    assert "test.mp3" in output


def test_show_daemon_lyrics_not_playing():
    from audiobench.playback.lyrics_hud import show_daemon_lyrics

    client = MagicMock()
    client.playback_status.return_value = {"playing": False}
    console = Console(record=True)
    show_daemon_lyrics(client, console)
    output = console.export_text()
    assert "No audio currently playing" in output


def test_normalize_segments_duplicate_starts():
    from audiobench.playback.lyrics_hud import _normalize_segments

    raw = [
        {"id": 1, "start": 33.37, "end": 43.84, "text": "Sentence A."},
        {"id": 2, "start": 43.84, "end": 44.34, "text": "Sentence B short."},
        {"id": 3, "start": 43.84, "end": 44.34, "text": "Sentence C medium length here."},
        {"id": 4, "start": 43.84, "end": 49.04, "text": "Sentence D which is the longest sentence in this cluster."},
        {"id": 5, "start": 49.04, "end": 59.52, "text": "Sentence E next."},
    ]
    norm = _normalize_segments(raw)
    assert len(norm) == 5

    # Segments 2, 3, 4 should now have distinct, monotonic start and end times
    assert norm[1]["norm_start"] < norm[2]["norm_start"] < norm[3]["norm_start"]
    assert norm[1]["norm_end"] <= norm[2]["norm_start"]
    assert norm[2]["norm_end"] <= norm[3]["norm_start"]
    assert norm[3]["norm_end"] <= norm[4]["norm_start"]

    # Each segment has non-empty text and valid norm_start < norm_end
    for s in norm:
        assert s["norm_start"] < s["norm_end"]


def test_find_active_segment_intervals_and_gaps():
    from audiobench.playback.lyrics_hud import _find_active_segment

    segments = [
        {"norm_start": 10.0, "norm_end": 15.0, "text": "First"},
        {"norm_start": 20.0, "norm_end": 25.0, "text": "Second"},
        {"norm_start": 25.0, "norm_end": 30.0, "text": "Third"},
    ]

    # Exactly inside intervals
    assert _find_active_segment(segments, 12.0) == 0
    assert _find_active_segment(segments, 22.0) == 1
    assert _find_active_segment(segments, 27.0) == 2

    # In silence gap (15.0 to 20.0):
    # At 16.0s (near First): stays on First
    assert _find_active_segment(segments, 16.0) == 0
    # At 19.5s (within 0.8s of Second): transitions to Second
    assert _find_active_segment(segments, 19.5) == 1

    # Before start and after end
    assert _find_active_segment(segments, 5.0) == 0
    assert _find_active_segment(segments, 50.0) == 2


def test_lyrics_command_standardization_in_registry():
    from audiobench.chat.command_meta import COMMAND_REGISTRY, get_all_command_names, get_command_meta

    # Primary command is /lyrics
    lyrics_meta = get_command_meta("/lyrics")
    assert lyrics_meta is not None
    assert lyrics_meta.name == "/lyrics"

    # /lyric is not an advertised alias (standardized to /lyrics)
    assert "/lyric" not in lyrics_meta.aliases
    assert "/lyric" not in get_all_command_names()

    # Searching for /lyr matches ONLY /lyrics
    matching = [name for name in get_all_command_names() if name.startswith("/lyr")]
    assert matching == ["/lyrics"]


def test_extract_timestamps_synthesis_formats():
    """Verify timestamp extraction handles ranges, tables, prefixes, and markdown from AI synthesis."""
    text = (
        "Explains modern culture (0:00-0:33).\n"
        "Timestamp 03:30 - The tension\n"
        '"Does being happy make you less successful?" (03:19-03:28)\n'
        "(03:31-03:46)\n"
        '(Earlier context) "If the end goal is happiness" (03:13-03:17)\n'
        "(as earlier noted at 0:33)\n"
        "burnout cycle described later (7:15-7:30).\n"
        "state at 10:46\n"
        "2026-09-19 19:37:04 Some ISO log\n"
        "Aspect ratio 16:9 and port localhost:8080\n"
    )
    extracted = extract_timestamps(text)
    # Start of ranges and standalone timestamps preserved, trailing range ends excluded
    # Deduplication clusters 03:30 & 03:31 (diff <= 3s)
    expected_times = [
        ("0:00", 0.0),
        ("0:33", 33.0),
        ("03:13", 193.0),
        ("03:19", 199.0),
        ("03:30", 210.0),
        ("7:15", 435.0),
        ("10:46", 646.0),
    ]
    assert extracted == expected_times


