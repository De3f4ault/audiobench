"""Migration: Fix stale foreign key in segments table.

SQLite's ALTER TABLE RENAME leaves behind stale FK references in other
tables.  If the transcriptions table was ever recreated via rename →
create → copy, the segments FK still points at the old temp name
(e.g. "_transcriptions_old").  This migration detects the problem and
rebuilds the segments table with the correct FK.

Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.003")


def _segments_fk_is_broken(cursor: sqlite3.Cursor) -> bool:
    """Return True if segments has a FK that does NOT point to 'transcriptions'."""
    cursor.execute("PRAGMA foreign_key_list(segments)")
    rows = cursor.fetchall()
    # foreign_key_list columns: (id, seq, table, from, to, on_update, on_delete, match)
    for row in rows:
        referenced_table = row[2]
        if referenced_table != "transcriptions":
            logger.warning(
                "segments FK points to '%s' instead of 'transcriptions'",
                referenced_table,
            )
            return True
    return False


def migrate(db_path: str) -> None:
    """Rebuild segments table with the correct FK if needed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _segments_fk_is_broken(cursor):
            logger.info("Migration 003 skipped — segments FK is correct")
            return

        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.execute("BEGIN TRANSACTION")

        cursor.execute("""
            CREATE TABLE segments_fixed (
                id       INTEGER NOT NULL PRIMARY KEY,
                transcription_id INTEGER NOT NULL,
                segment_index    INTEGER NOT NULL,
                text             TEXT    NOT NULL,
                start_time       FLOAT  NOT NULL,
                end_time         FLOAT  NOT NULL,
                speaker          VARCHAR(64),
                FOREIGN KEY(transcription_id) REFERENCES transcriptions (id)
            )
        """)

        cursor.execute("INSERT INTO segments_fixed SELECT * FROM segments")
        cursor.execute("DROP TABLE segments")
        cursor.execute("ALTER TABLE segments_fixed RENAME TO segments")

        # Recreate the index that was on the old table
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_segments_transcription_id "
            "ON segments (transcription_id)"
        )

        conn.commit()
        logger.info("Migration 003 completed — segments FK fixed")

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.execute("PRAGMA foreign_keys=ON")
        conn.close()
