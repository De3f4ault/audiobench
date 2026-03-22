"""Shared fixtures for the AudioBench test suite.

Provides:
    - runner: Click CliRunner for CLI tests
    - tmp_data_dir: Temporary data directory (isolates DB, presets, etc.)
    - test_settings: Patched settings pointing to tmp_data_dir
    - test_db: Initialized temp database with session factory
    - sample_audio_dir: Directory with fake audio files for collection tests
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "plugins").mkdir()
    (data_dir / "presets").mkdir()
    (data_dir / "logs").mkdir()
    return data_dir


@pytest.fixture
def test_settings(tmp_data_dir):
    """Patched settings that use a temp directory for all data.

    This ensures tests never touch the real database or user data.
    """
    from audiobench.core.settings import AudioBenchSettings, get_settings

    db_path = tmp_data_dir / "test.db"
    settings = AudioBenchSettings(
        database_url=f"sqlite:///{db_path}",
        data_dir=tmp_data_dir,
        models_dir=tmp_data_dir / "models",
    )

    # Clear the lru_cache and patch
    get_settings.cache_clear()
    with patch("audiobench.core.settings.get_settings", return_value=settings):
        yield settings
    get_settings.cache_clear()


@pytest.fixture
def test_db(test_settings):
    """Initialize a test database and return the settings.

    The DB is created fresh in a temp dir — destroyed after the test.
    """
    # Reset the module-level engine
    import audiobench.core.db_engine as db_mod

    old_engine = db_mod._engine
    db_mod._engine = None

    from audiobench.core.db_engine import init_db

    init_db()
    yield test_settings

    # Cleanup
    db_mod._engine = old_engine


@pytest.fixture
def sample_audio_dir(tmp_path):
    """Create a directory tree with fake audio files for testing file_collector."""
    root = tmp_path / "audio"
    root.mkdir()

    # Flat files
    (root / "meeting.mp3").write_text("fake")
    (root / "podcast.m4a").write_text("fake")
    (root / "notes.txt").write_text("not audio")
    (root / "draft_take.mp3").write_text("fake")

    # Subdirectory
    sub = root / "sub"
    sub.mkdir()
    (sub / "interview.wav").write_text("fake")
    (sub / "backup.mp3").write_text("fake")

    return root
