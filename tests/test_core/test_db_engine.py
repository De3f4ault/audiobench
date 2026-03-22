"""Tests for DB engine — init, connection, table creation."""

from __future__ import annotations

from unittest.mock import patch

from audiobench.core.settings import AudioBenchSettings


class TestDatabaseEngine:
    """Test database engine initialization."""

    def test_init_db_creates_tables(self, tmp_path):
        """init_db() should create all tables without error."""
        import audiobench.core.db_engine as db_mod

        db_path = tmp_path / "test.db"
        settings = AudioBenchSettings(
            database_url=f"sqlite:///{db_path}",
        )
        old_engine = db_mod._engine
        db_mod._engine = None

        with patch("audiobench.core.db_engine.get_settings", return_value=settings):
            from audiobench.core.db_engine import init_db

            init_db()
            assert db_path.exists()

        db_mod._engine = old_engine

    def test_get_engine_returns_same_instance(self, tmp_path):
        """get_engine() should return the same engine on repeat calls."""
        import audiobench.core.db_engine as db_mod

        db_path = tmp_path / "test2.db"
        settings = AudioBenchSettings(
            database_url=f"sqlite:///{db_path}",
        )
        old_engine = db_mod._engine
        db_mod._engine = None

        with patch("audiobench.core.db_engine.get_settings", return_value=settings):
            from audiobench.core.db_engine import get_engine

            e1 = get_engine()
            e2 = get_engine()
            assert e1 is e2

        db_mod._engine = old_engine

    def test_sqlite_creates_parent_dir(self, tmp_path):
        """For SQLite URLs, get_engine() should create parent directories."""
        import audiobench.core.db_engine as db_mod

        nested = tmp_path / "deep" / "nested" / "test.db"
        settings = AudioBenchSettings(
            database_url=f"sqlite:///{nested}",
        )
        old_engine = db_mod._engine
        db_mod._engine = None

        with patch("audiobench.core.db_engine.get_settings", return_value=settings):
            from audiobench.core.db_engine import get_engine

            get_engine()
            assert nested.parent.exists()

        db_mod._engine = old_engine
