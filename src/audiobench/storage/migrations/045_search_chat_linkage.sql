-- Migration 045: Link chat conversations to search sessions.
--
-- Enables chat threads spawned from search sessions to be linked back to their
-- parent search session, allowing multi-thread exploration and thread navigation.

ALTER TABLE chat_conversations ADD COLUMN search_session_id INTEGER REFERENCES search_sessions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_chat_conversations_search_session_id ON chat_conversations (search_session_id);
