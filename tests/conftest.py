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
import pytest
from click.testing import CliRunner


@pytest.fixture(scope="session", autouse=True)
def _isolate_daemon_socket_for_tests():
    """Redirect daemon socket to an isolated path for all tests.

    Ensures no test ever connects to /tmp/audiobench-daemon.sock or touches
    the running production daemon.
    """
    old_sock = os.environ.get("AUDIOBENCH_DAEMON_SOCKET_PATH")
    os.environ["AUDIOBENCH_DAEMON_SOCKET_PATH"] = "/tmp/audiobench-test-isolated.sock"
    from audiobench.core.settings import get_settings
    get_settings.cache_clear()
    yield
    if old_sock is None:
        os.environ.pop("AUDIOBENCH_DAEMON_SOCKET_PATH", None)
    else:
        os.environ["AUDIOBENCH_DAEMON_SOCKET_PATH"] = old_sock
    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _isolate_database_for_tests(tmp_path_factory):
    """Session-wide database isolation fallback.

    Redirects AUDIOBENCH_DATABASE_URL to an isolated temporary SQLite database
    so that any test that does not explicitly request the `test_db` fixture
    is guaranteed NEVER to connect to or pollute data/transcriptions.db.
    """
    from audiobench.core.settings import get_settings

    tmp_dir = tmp_path_factory.mktemp("session_db")
    tmp_db_file = tmp_dir / "session_fallback.db"

    old_db_url = os.environ.get("AUDIOBENCH_DATABASE_URL")
    old_data_dir = os.environ.get("AUDIOBENCH_DATA_DIR")

    os.environ["AUDIOBENCH_DATABASE_URL"] = f"sqlite:///{tmp_db_file}"
    os.environ["AUDIOBENCH_DATA_DIR"] = str(tmp_dir)
    get_settings.cache_clear()

    # Pre-create tables in the session fallback DB
    import audiobench.core.db_engine as db_mod
    import audiobench.core.db_session as dbs_mod
    from audiobench.storage.models import Base

    db_mod._engine = None
    dbs_mod._SessionLocal = None

    engine = db_mod.get_engine()
    Base.metadata.create_all(bind=engine)
    engine.dispose()
    db_mod._engine = None
    dbs_mod._SessionLocal = None

    yield

    if old_db_url is None:
        os.environ.pop("AUDIOBENCH_DATABASE_URL", None)
    else:
        os.environ["AUDIOBENCH_DATABASE_URL"] = old_db_url

    if old_data_dir is None:
        os.environ.pop("AUDIOBENCH_DATA_DIR", None)
    else:
        os.environ["AUDIOBENCH_DATA_DIR"] = old_data_dir

    get_settings.cache_clear()
    db_mod._engine = None
    dbs_mod._SessionLocal = None


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
def test_settings(tmp_data_dir, monkeypatch):
    """Patched settings that use a temp directory for all data.

    This ensures tests never touch the real database or user data.
    """
    from audiobench.core.settings import get_settings

    db_path = tmp_data_dir / "test.db"

    # Inject test paths into the environment so pydantic-settings picks them up.
    # This avoids aliasing bugs where modules import `get_settings` directly
    # before we can mock.patch it.
    monkeypatch.setenv("AUDIOBENCH_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUDIOBENCH_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("AUDIOBENCH_MODELS_DIR", str(tmp_data_dir / "models"))

    # Clear the lru_cache so the next call rebuilds from the patched environment
    get_settings.cache_clear()
    settings = get_settings()

    yield settings

    get_settings.cache_clear()


@pytest.fixture
def test_db(test_settings):
    """Initialize a test database and return the settings.

    The DB is created fresh in a temp dir — destroyed after the test.

    Both module-level singletons must be reset: _engine in db_engine and
    _SessionLocal in db_session. Resetting only _engine leaves db_session
    bound to the previous test's SQLite file, causing cross-test state leakage.
    """
    import audiobench.core.db_engine as db_mod
    import audiobench.core.db_session as sess_mod

    old_engine = db_mod._engine
    old_session_factory = sess_mod._SessionLocal

    # Tear down both singletons so init_db builds fresh against the new tmp path
    db_mod._engine = None
    sess_mod._SessionLocal = None

    from audiobench.core.db_engine import init_db

    init_db()
    yield test_settings

    # Dispose the test engine so SQLite file handles are released before restoring
    if db_mod._engine is not None:
        db_mod._engine.dispose()

    # Restore originals (supports nested/parallel fixture usage)
    db_mod._engine = old_engine
    sess_mod._SessionLocal = old_session_factory


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
