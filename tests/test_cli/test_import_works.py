from pathlib import Path

import pytest

from audiobench.cli.commands.import_cmd import extract_metadata


def test_extract_metadata_from_filename(tmp_path: Path):
    f = tmp_path / "Martin Heidegger - Being and Time.mp3"
    f.touch()
    title, author = extract_metadata(f)
    assert title == "Being and Time"
    assert author == "Martin Heidegger"

def test_extract_metadata_from_filename_no_author(tmp_path: Path):
    f = tmp_path / "Chapter 1.mp3"
    f.touch()
    title, author = extract_metadata(f)
    assert title is None
    assert author is None

def test_extract_metadata_with_mutagen(tmp_path: Path):
    pytest.importorskip("mutagen")
    from unittest.mock import MagicMock, patch

    mock_m = MagicMock()
    mock_m.get.side_effect = lambda k, default: {
        "title": ["The Concept of Time"],
        "artist": ["Heidegger"],
    }.get(k, default)

    with patch("mutagen.File", return_value=mock_m):
        title, author = extract_metadata(tmp_path / "test.mp3")
        assert title == "The Concept of Time"
        assert author == "Heidegger"
