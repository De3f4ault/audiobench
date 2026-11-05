-- Migration 040: Composite index on expression_relations for fast SEMANTIC edge queries.
--
-- Before this migration, queries like:
--   SELECT * FROM expression_relations WHERE relation_type = 'semantic' ORDER BY weight DESC
-- did a full table scan over 24,000+ rows.
--
-- This index makes the Observatory InferencesFeed panel and `\related` REPL
-- command fast even when the knowledge graph grows to 100,000+ edges.
--
-- The index covers:
--   relation_type  — filter to SEMANTIC edges only
--   weight         — sort by similarity descending
--   from_expression_id, to_expression_id  — already covered by the unique index
--                                            but included here for covering scans
--
-- Estimated size: ~1.5 MB per 100,000 edges.

CREATE INDEX IF NOT EXISTS idx_expr_relations_semantic_weight
ON expression_relations (relation_type, weight DESC, from_expression_id, to_expression_id)
WHERE relation_type = 'semantic';
