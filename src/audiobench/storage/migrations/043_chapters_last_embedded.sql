-- Migration 043: Add last_embedded_at dirty flag to chapters table.
--
-- Enables ChapterEmbedder background task to discover mutated or newly ingested
-- chapters that require vector embedding (Tier 2 Passage / Chapter level).

ALTER TABLE chapters ADD COLUMN last_embedded_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_chapters_last_embedded_at ON chapters (last_embedded_at);
