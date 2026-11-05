-- Migration 048: Link chat conversations to specific search queries within a search session
--
-- Enables chat threads spawned from a specific search query step to be linked back to
-- their exact search query (sequence_num, query_text), preserving provenance and
-- enabling step-tagged titles ([S#] <Topic>).

ALTER TABLE chat_conversations ADD COLUMN search_query_id INTEGER REFERENCES search_queries(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_chat_conversations_search_query_id ON chat_conversations (search_query_id);
