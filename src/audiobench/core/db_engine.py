"""Database engine — SQLAlchemy engine creation and table initialization.

Handles engine creation from the configured database URL and
provides the init_db() function to create all ORM tables.

Usage:
    from audiobench.core.db_engine import get_engine, init_db

    init_db()  # Create all tables
    engine = get_engine()  # Get the engine instance
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine as _create_engine

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("core.db_engine")

# Module-level engine (lazy init)
_engine = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url

        # For SQLite, ensure the parent directory exists
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "")
            if db_path and not db_path.startswith(":"):
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = _create_engine(
            url,
            echo=False,
            pool_pre_ping=True,
        )

        # Enable WAL mode for SQLite — allows concurrent reads and writes,
        # which is needed because background title generation writes while
        # the main thread reads.
        if url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, _connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info("Database engine created: %s", url.split("@")[-1] if "@" in url else url)

    return _engine


def init_db() -> None:
    """Create all database tables and run pending migrations."""
    from audiobench.storage.models import Base  # noqa: F811

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    # Run idempotent migrations for existing databases
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        if db_path and not db_path.startswith(":"):
            try:
                from audiobench.storage.migrations.m002_db_hardening import migrate

                migrate(db_path)
            except Exception as e:
                logger.warning("Migration m002 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m003_fix_segments_fk import (
                    migrate as migrate_003,
                )

                migrate_003(db_path)
            except Exception as e:
                logger.warning("Migration m003 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m004_ai_features import (
                    migrate as migrate_004,
                )

                migrate_004(db_path)
            except Exception as e:
                logger.warning("Migration m004 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m005_bookmarks import (
                    migrate as migrate_005,
                )

                migrate_005(db_path)
            except Exception as e:
                logger.warning("Migration m005 failed (non-fatal): %s", e)

    logger.info("Database tables created")
