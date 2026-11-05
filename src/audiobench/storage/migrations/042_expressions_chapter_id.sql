-- Migration 042: Add chapter_id foreign key to expressions table.
--
-- Enables chapter-scoped semantic queries and direct chapter linking for expressions.
-- Single-chapter expressions are backfilled; multi-chapter expressions remain NULL
-- until expression_segment_map is populated (Decision 1 backlog).

ALTER TABLE expressions ADD COLUMN chapter_id INTEGER REFERENCES chapters(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_expressions_chapter_id ON expressions (chapter_id);
