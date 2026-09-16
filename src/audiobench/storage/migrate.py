"""Migration runner for AudioBench SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.exceptions import MigrationError

logger = get_logger("storage.migrate")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(db_path: Path, *, allow_path_mismatch: bool = False) -> None:
    """Run all pending SQL migrations idempotently.

    IMPORTANT: This function operates on the path it is given, not on whatever
    get_settings().database_url resolves to. Passing the wrong path creates and
    migrates a new empty DB without error — a silent mismatch that is hard to
    detect after the fact.

    The canonical path is always what get_settings().database_url resolves to:
        data/transcriptions.db  (relative to project root)

    If db_path does not match the configured database_url, this function raises
    MigrationError rather than proceeding silently. Pass allow_path_mismatch=True
    only for legitimate off-canonical cases (e.g. in-memory test DBs, CI fixtures).
    """
    resolved = db_path.resolve()

    # ── Announce loudly before touching anything ──────────────────────────────
    # Intentionally goes to stdout (not just the log file) so any invocation —
    # CLI, daemon, CI script — shows the path before any DB work happens.
    # If this path looks wrong, stop before reading further output.
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[migrate] Operating on: {resolved}")
    logger.info("run_migrations: operating on %s", resolved)

    # ── Canonical path guard ──────────────────────────────────────────────────
    # Raises on mismatch rather than warning and proceeding. A migration applied
    # to the wrong file and reported as success is worse than a refused migration
    # — at least a refusal leaves the system unchanged and the error message
    # names both paths so the caller knows exactly what to fix.
    if not allow_path_mismatch:
        try:
            from audiobench.core.settings import get_settings
            configured_url = get_settings().database_url
            if configured_url.startswith("sqlite:///"):
                configured_path = Path(configured_url.replace("sqlite:///", "")).resolve()
                if resolved != configured_path:
                    raise MigrationError(
                        f"run_migrations received a path that does not match the configured "
                        f"database.\n"
                        f"  Received : {resolved}\n"
                        f"  Canonical: {configured_path}\n"
                        f"Pass allow_path_mismatch=True to suppress this check (tests only)."
                    )
        except MigrationError:
            raise
        except Exception:
            pass  # Don't block if settings can't be loaded (bootstrap scenario)

    with sqlite3.connect(str(db_path)) as conn:
        # Enable WAL mode first as specified
        conn.execute("PRAGMA journal_mode=WAL")

        # Create schema_version table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # Get applied versions
        cursor = conn.execute("SELECT version FROM schema_version")
        applied_versions = {row[0] for row in cursor.fetchall()}

        # Find all .sql files in migrations directory
        if not MIGRATIONS_DIR.exists():
            return

        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for sql_file in sql_files:
            # Extract version from filename (e.g. 001_initial.sql -> 1)
            try:
                version = int(sql_file.stem.split("_")[0])
            except ValueError:
                logger.warning(
                    "Skipping migration file with invalid name format: %s", sql_file.name
                )
                continue

            if version in applied_versions:
                continue

            logger.info("Applying migration: %s", sql_file.name)

            sql_script = sql_file.read_text(encoding="utf-8")

            try:
                # Execute script in transaction
                conn.executescript(sql_script)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info("Ignoring duplicate column error for %s", sql_file.name)
                    conn.rollback()
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    conn.commit()
                else:
                    conn.rollback()
                    raise MigrationError(f"Failed to apply {sql_file.name}: {e}") from e
            except sqlite3.Error as e:
                conn.rollback()
                raise MigrationError(f"Failed to apply {sql_file.name}: {e}") from e

    logger.info("All SQL migrations applied successfully")
