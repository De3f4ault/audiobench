-- Migration 051: Directional indices on expression_relations for partial SEMANTIC edge lookups.
-- Fixes SQLite query planner regression where idx_expr_relations_semantic_weight
-- (relation_type, weight DESC, from_expression_id, to_expression_id) was erroneously
-- selected for point lookups on from_expression_id or to_expression_id, causing full
-- index scans across 27,000+ entries (166s stalls in candidate sampling and island queries).

CREATE INDEX IF NOT EXISTS idx_expr_relations_semantic_from
ON expression_relations (from_expression_id)
WHERE relation_type = 'semantic';

CREATE INDEX IF NOT EXISTS idx_expr_relations_semantic_to
ON expression_relations (to_expression_id)
WHERE relation_type = 'semantic';
