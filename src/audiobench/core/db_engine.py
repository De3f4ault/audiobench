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
from typing import Any

from sqlalchemy import Engine
from sqlalchemy import create_engine as _create_engine

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("core.db_engine")

# Module-level engine (lazy init)
_engine = None


def get_engine() -> Engine:
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
            def _set_sqlite_pragmas(dbapi_conn: Any, _connection_record: Any) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=15000")   # wait up to 15s on lock, not 0ms
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA mmap_size=268435456")
                cursor.execute("PRAGMA cache_size=-32000")
                cursor.close()

        # ── Announce the resolved DB path loudly ─────────────────────────────
        # This intentionally goes to stdout (not just the log file) so that
        # any invocation — CLI, daemon, test script — shows exactly which file
        # the engine is pointed at before any query runs. If this path looks
        # wrong, stop and fix it rather than debugging silent data divergence.
        resolved_db = url.split("///")[-1] if "sqlite:///" in url else url.split("@")[-1]
        logger.debug(f"[db] Canonical database: {Path(resolved_db).resolve()}")
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

            try:
                from audiobench.storage.migrations.m006_refined_at import (
                    migrate as migrate_006,
                )

                migrate_006(db_path)
            except Exception as e:
                logger.warning("Migration m006 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m007_background_jobs import (
                    migrate as migrate_007,
                )

                migrate_007(db_path)
            except Exception as e:
                logger.warning("Migration m007 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m008_chapters import (
                    migrate as migrate_008,
                )

                migrate_008(db_path)
            except Exception as e:
                logger.warning("Migration m008 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m012_backfill_columns import (
                    migrate as migrate_012,
                )

                migrate_012(db_path)
            except Exception as e:
                logger.warning("Migration m012 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m009_rag_indexing import (
                    migrate as migrate_009,
                )

                migrate_009(db_path)
            except Exception as e:
                logger.warning("Migration m009 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m010_command_graph import (
                    migrate as migrate_010,
                )

                migrate_010(db_path)
            except Exception as e:
                logger.warning("Migration m010 failed (non-fatal): %s", e)

            try:
                from audiobench.storage.migrations.m011_strategy_column import (
                    migrate as migrate_011,
                )

                migrate_011(db_path)
            except Exception as e:
                logger.warning("Migration m011 failed (non-fatal): %s", e)

            # Run the new idempotent SQL migrations (Phase 0/1 and security layer)
            # Includes: 001–024 (core schema) and 025 (privacy_tier — security layer)
            try:
                from pathlib import Path

                from audiobench.storage.migrate import run_migrations

                run_migrations(Path(db_path))
            except Exception as e:
                logger.error("SQL migrations failed: %s", e)
                raise

    logger.info("Database tables created")
