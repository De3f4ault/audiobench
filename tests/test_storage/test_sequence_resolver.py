"""Tests for sequence_resolver module."""

import pytest
from audiobench.storage.sequence_resolver import resolve_sequence_key, _parse_int_prefix


def test_parse_int_prefix():
    assert _parse_int_prefix("9") == 9
    assert _parse_int_prefix("09") == 9
    assert _parse_int_prefix("9/12") == 9
    assert _parse_int_prefix("  14 / 20  ") == 14
    assert _parse_int_prefix("invalid") is None
    assert _parse_int_prefix("") is None
    assert _parse_int_prefix(None) is None


def test_branch_1_disc_and_track():
    tags = {"disc": "2", "track": "5"}
    assert resolve_sequence_key("some_file.mp3", tags) == "02-005"

    tags_total = {"disc": "1/2", "track": "9/10"}
    assert resolve_sequence_key("file.mp3", tags_total) == "01-009"


def test_branch_2_album_disc_and_track():
    tags = {
        "album": "Out of Your Mind - Disc 1 - The Nature of Consciousness",
        "track": "3",
    }
    assert resolve_sequence_key("watts.mp3", tags) == "01-003"

    tags_cd = {
        "album": "Out Of Your Mind - CD4 - The Web Of Life",
        "track": "8",
    }
    assert resolve_sequence_key("watts_cd.mp3", tags_cd) == "04-008"

    tags_disc10 = {
        "album": "Out of Your Mind - Disc 10 - The World As Self",
        "track": "12",
    }
    assert resolve_sequence_key("watts_10.mp3", tags_disc10) == "10-012"


def test_branch_3_plain_track():
    tags = {"track": "7", "album": "Frankenstein"}
    assert resolve_sequence_key("ch7.mp3", tags) == "007"


def test_branch_4a_leading_number():
    assert resolve_sequence_key("01 - Part 1 Chapter 1 — Fyodor Dostoyevsky.mp3", {}) == "001"
    assert resolve_sequence_key("02_chapter_two.mp3", {}) == "002"
    # Year guard: 1984 should not be treated as a chapter track 1984
    assert resolve_sequence_key("1984 George Orwell.mp3", {}) is None


def test_branch_4b_librivox_pattern():
    assert resolve_sequence_key("tale_of_two_cities_07_dickens_64kb.mp3", {}) == "007"
    assert resolve_sequence_key("crime_and_punishment_04_64kb.mp3", {}) == "004"
    assert resolve_sequence_key("ulysses_12b_joyce_64kb.mp3", {}) == "012b"


def test_branch_4c_part_pattern():
    assert resolve_sequence_key("The Power of Now-Part07.mp3", {}) == "007"
    assert resolve_sequence_key("Genius Makers-Disk04.mp3", {}) == "004"
    assert resolve_sequence_key("Book_Chapter_03.mp3", {}) == "003"


def test_branch_5_none():
    assert resolve_sequence_key("Cameron Hanes - Endure.m4b", {}) is None
    assert resolve_sequence_key("Sunday Service.m4a", {}) is None
    assert resolve_sequence_key("Random Speech.mp3", {"genre": "Speech"}) is None
