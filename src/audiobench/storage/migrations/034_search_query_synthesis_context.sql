-- Migration 034: Synthesis Context Relational Storage
--
-- Replaces the JSON-based prior_synthesis_json column introduced in 033
-- with a dedicated relational table. This stores the [synthesis] hits
-- that were surfaced during a search.
--
-- Advantages:
-- 1. No JSON parsing required on resume.
-- 2. source_query_id preserves the chain of thought (e.g., S12's context came from S8 and S10).
-- 3. Enables querying semantic evolution (e.g., how often was S5's synthesis cited).

CREATE TABLE IF NOT EXISTS search_query_synthesis_context (
    query_id        INTEGER NOT NULL REFERENCES search_queries(id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,          -- display order (1, 2, etc.)
    source_type     TEXT    NOT NULL,          -- 'search_synthesis' or 'search_session_summary'
    source_query_id INTEGER,                   -- the query this synthesis originally came from (if applicable)
    content         TEXT    NOT NULL,          -- denormalized synthesis text as displayed
    PRIMARY KEY (query_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_sqsc_query ON search_query_synthesis_context(query_id);
