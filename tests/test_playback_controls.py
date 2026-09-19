"""Unit tests for playback controls, timestamp parser, and command dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from rich.console import Console

from audiobench.playback.controls import (
    PLAYBACK_COMMANDS,
    extract_timestamps,
    fmt_timestamp,
    fmt_timestamp_slug,
    handle_playback_command,
    parse_timestamp,
    render_playback_status_line,
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
