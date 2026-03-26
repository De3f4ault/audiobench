"""Migration: Add columns for AI features.

Adds:
- chat_messages.model_name — tracks which model generated each response
  (for split-screen comparison and resume)
- transcriptions.raw_text — preserves original Whisper output before
  LLM refinement

This migration is safe to run multiple times (idempotent).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.004")


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # ── chat_messages.model_name ──
        if not _column_exists(cursor, "chat_messages", "model_name"):
            cursor.execute(
                "ALTER TABLE chat_messages ADD COLUMN model_name VARCHAR(128)"
            )
            logger.info("Added model_name column to chat_messages")

        # ── transcriptions.raw_text ──
        if not _column_exists(cursor, "transcriptions", "raw_text"):
            cursor.execute(
                "ALTER TABLE transcriptions ADD COLUMN raw_text TEXT"
            )
            logger.info("Added raw_text column to transcriptions")

        conn.commit()
        logger.info("Migration 004 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
