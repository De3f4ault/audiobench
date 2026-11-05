"""Tests for the database migration runner."""

import sqlite3
from pathlib import Path

from audiobench.storage import migrate


def test_migration_runner_idempotency(tmp_path: Path, monkeypatch):
    """Test that migrations can be run multiple times safely."""
    # 1. Setup a dummy migrations directory
    dummy_migrations_dir = tmp_path / "dummy_migrations"
    dummy_migrations_dir.mkdir()

    # Create two dummy migration files
    m1 = dummy_migrations_dir / "001_initial.sql"
    m1.write_text("CREATE TABLE test_table (id INTEGER PRIMARY KEY);")

    m2 = dummy_migrations_dir / "002_add_column.sql"
    m2.write_text("ALTER TABLE test_table ADD COLUMN name TEXT;")

    # Patch the MIGRATIONS_DIR in migrate module
    monkeypatch.setattr(migrate, "MIGRATIONS_DIR", dummy_migrations_dir)

    db_path = tmp_path / "test_migrations.db"

    # 2. Run migrations first time
    migrate.run_migrations(db_path, allow_path_mismatch=True)

    # Verify WAL mode and schema_version table
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].lower() == "wal"

        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version")
        versions = [row[0] for row in cursor.fetchall()]
        assert versions == [1, 2]

        # Verify dummy migrations actually applied
        cursor = conn.execute("PRAGMA table_info(test_table)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "id" in columns
        assert "name" in columns

    # 3. Run migrations second time (idempotency check)
    migrate.run_migrations(db_path, allow_path_mismatch=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version")
        versions = [row[0] for row in cursor.fetchall()]
        assert versions == [1, 2]  # Still just 1 and 2, no duplicates or errors


def test_real_migrations(tmp_path: Path):
    """Test that the actual SQL migrations in the project apply cleanly to an empty database."""
    db_path = tmp_path / "real_migrations.db"

    # Create the original schema via SQLAlchemy first, as the app does
    from sqlalchemy import create_engine

    from audiobench.storage.models import Base

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)

    # Run migrations using the real MIGRATIONS_DIR
    migrate.run_migrations(db_path, allow_path_mismatch=True)

    with sqlite3.connect(db_path) as conn:
        # Check that schema_version has records (at least 6 since we have 6 migrations)
        cursor = conn.execute("SELECT COUNT(*) FROM schema_version")
        count = cursor.fetchone()[0]
        assert count >= 6

        # Check some of the new tables exist
        cursor = conn.execute("PRAGMA table_info(expressions)")
        assert len(cursor.fetchall()) > 0

        # We shouldn't fail if we run it again (idempotent)
    migrate.run_migrations(db_path, allow_path_mismatch=True)

