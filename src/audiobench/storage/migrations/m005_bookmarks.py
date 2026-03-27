"""Migration: Create bookmarks table.

Adds the ``bookmarks`` table for persisting point bookmarks and region
markers linked to audio files.  Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.005")


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER NOT NULL
                                  REFERENCES audio_files(id) ON DELETE CASCADE,
                transcription_id INTEGER
                                  REFERENCES transcriptions(id) ON DELETE SET NULL,
                timestamp     REAL    NOT NULL,
                end_timestamp REAL,
                name          VARCHAR(512) DEFAULT 'Untitled',
                notes         TEXT,
                bookmark_type VARCHAR(16)  DEFAULT 'bookmark',
                color         VARCHAR(16),
                created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_bookmarks_audio_file_id "
            "ON bookmarks(audio_file_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_bookmarks_transcription_id "
            "ON bookmarks(transcription_id)"
        )
        conn.commit()
        logger.info("Migration 005 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
