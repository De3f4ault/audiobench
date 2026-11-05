-- Migration 033: Persist prior synthesis hits per search query
--
-- prior_synthesis_json stores a JSON array of the synthesis hit objects
-- that were surfaced at search time (the [synthesis] blocks). These are the LLM
-- answers from earlier searches in the same session that were retrieved
-- by SynthesisStream and displayed above the audio fragments.
--
-- Without this column, synthesis hits exist only in memory at search time
-- and are silently dropped when the session closes. On resume+replay, the
-- displayed results are missing the synthesis context that was visible live.
--
-- Stored as JSON so no schema change is needed to the hit structure and
-- the column is nullable for backwards compatibility with old rows.

ALTER TABLE search_queries
    ADD COLUMN prior_synthesis_json TEXT DEFAULT NULL;
