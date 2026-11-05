-- Migration 046: Purge erroneous same-work semantic edges created by audio_file_id bug.
--
-- Background:
-- ConnectionSurfer and PatternDetector previously conflated audio_file_id with work_id,
-- causing 8,561 intra-book chapter edges across multi-file audiobooks (e.g. Brothers
-- Karamazov, Moby Dick, Great Expectations, Crime and Punishment, Ulysses) to masquerade
-- as cross-work semantic bridges and exhaust author-level edge caps.
--
-- This migration permanently purges any semantic edge where from_expression_id and
-- to_expression_id belong to the same work_id. True cross-work edges are strictly preserved.

DELETE FROM expression_relations
WHERE id IN (
    SELECT r.id
    FROM expression_relations r
    JOIN expressions e1 ON r.from_expression_id = e1.id
    JOIN expressions e2 ON r.to_expression_id = e2.id
    WHERE r.relation_type = 'semantic'
      AND e1.work_id IS NOT NULL 
      AND e2.work_id IS NOT NULL 
      AND e1.work_id = e2.work_id
);
