-- Migration 039: Purge pure boilerplate sweep_chunk expressions.
--
-- Deletes standalone publisher/platform disclaimer expressions from the
-- sweep_chunk tier.  These are short chunks containing ONLY production
-- credits (e.g. LibriVox public-domain stamps, Audible intros) with no
-- substantive content.  They were becoming degree-172 super-attractor hubs
-- in the knowledge graph, connecting unrelated audiobooks via identical
-- boilerplate text.
--
-- Safety guarantees:
--   - Deletes only graph_role='sweep_chunk' rows (never segments, which
--     preserve audio playback and timestamp alignment).
--   - Length < 300 chars guards against accidentally deleting hybrid chunks
--     that contain boilerplate followed by real literary content.
--   - The UNIQUE constraint on expression_relations(from_id, to_id, relation_type)
--     means cascade deletion removes only the specific bogus edges attached to
--     these nodes, leaving all real SEMANTIC and SOURCE edges intact.
--
-- Verified count before running: 26 rows affected (all confirmed pure stamps).

DELETE FROM expressions
WHERE graph_role = 'sweep_chunk'
  AND LENGTH(content) < 300
  AND (
    LOWER(SUBSTR(content, 1, 200)) LIKE '%librivox%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%liebervox%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%librefox%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%libravox%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%library vox%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%this is audible%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%sounds true presents%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%tantor audio%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%brilliance audio%'
    OR LOWER(SUBSTR(content, 1, 200)) LIKE '%blackstone audio%'
  );
-- expression_relations rows cascade-delete automatically via FK constraint.
