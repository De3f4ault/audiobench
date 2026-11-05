"""Add failure_reason and attempt_count to transcriptions"""

import sqlite3


def upgrade(db: sqlite3.Connection) -> None:
    db.execute("ALTER TABLE transcriptions ADD COLUMN failure_reason VARCHAR(256)")
    db.execute("ALTER TABLE transcriptions ADD COLUMN attempt_count INTEGER DEFAULT 0")
    db.execute("ALTER TABLE transcriptions ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")
    db.execute("CREATE INDEX IF NOT EXISTS ix_transcriptions_pipeline_phase ON transcriptions(status)")
    # Backfill existing completed transcriptions to 'complete' instead of 'completed'
    db.execute("UPDATE transcriptions SET status = 'complete' WHERE status = 'completed'")

def downgrade(db: sqlite3.Connection) -> None:
    # SQLite does not easily support DROP COLUMN without recreating the table,
    # so we just drop the index and leave the columns.
    db.execute("DROP INDEX IF EXISTS ix_transcriptions_pipeline_phase")
    db.execute("UPDATE transcriptions SET status = 'completed' WHERE status = 'complete'")
