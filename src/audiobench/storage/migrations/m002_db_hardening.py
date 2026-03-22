"""Migration: Add new columns and indexes for database hardening.

Adds:
- transcriptions.file_name (denormalized from audio_files)
- chat_conversations.engine
- Indexes on frequently queried columns

This migration is safe to run multiple times (idempotent).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.002")


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _index_exists(cursor: sqlite3.Cursor, index_name: str) -> bool:
    """Check if an index exists."""
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    )
    return cursor.fetchone() is not None


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # ── New columns ──
        if not _column_exists(cursor, "transcriptions", "file_name"):
            cursor.execute(
                "ALTER TABLE transcriptions ADD COLUMN file_name VARCHAR(256) DEFAULT '' NOT NULL"
            )
            # Backfill from audio_files
            cursor.execute("""
                UPDATE transcriptions
                SET file_name = (
                    SELECT audio_files.file_name
                    FROM audio_files
                    WHERE audio_files.id = transcriptions.audio_file_id
                )
                WHERE audio_file_id IS NOT NULL AND file_name = ''
            """)
            # Backfill live sessions
            cursor.execute("""
                UPDATE transcriptions
                SET file_name = '🎤 Live session'
                WHERE source = 'live' AND file_name = ''
            """)
            logger.info("Added file_name column to transcriptions")

        if not _column_exists(cursor, "chat_conversations", "engine"):
            cursor.execute(
                "ALTER TABLE chat_conversations ADD COLUMN engine VARCHAR(64) DEFAULT 'ollama'"
            )
            logger.info("Added engine column to chat_conversations")

        # ── Indexes ──
        indexes = [
            ("ix_transcriptions_created_at", "transcriptions", "created_at"),
            ("ix_transcriptions_language", "transcriptions", "language"),
            ("ix_transcriptions_audio_file_id", "transcriptions", "audio_file_id"),
            ("ix_segments_transcription_id", "segments", "transcription_id"),
            ("ix_chat_conversations_created_at", "chat_conversations", "created_at"),
            ("ix_chat_messages_conversation_id", "chat_messages", "conversation_id"),
        ]

        for idx_name, table, column in indexes:
            if not _index_exists(cursor, idx_name):
                cursor.execute(f"CREATE INDEX {idx_name} ON {table} ({column})")
                logger.info("Created index %s", idx_name)

        conn.commit()
        logger.info("Migration 002 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
