-- Migration 053: Create partial index on expression_relations.prompt_version
--
-- SQLite's executescript() silently drops CREATE INDEX statements that appear
-- alongside an ALTER TABLE in the same script (migration 052). This migration
-- contains only the index creation so it runs as a clean single statement.
--
-- The partial index covers only rows where prompt_version IS NOT NULL (i.e.,
-- rows written by step5_triadic_extraction). This avoids index bloat on the
-- ~116k existing semantic/thematic rows that have no prompt_version.

CREATE INDEX IF NOT EXISTS ix_expression_relations_prompt_version
    ON expression_relations (prompt_version)
    WHERE prompt_version IS NOT NULL;
