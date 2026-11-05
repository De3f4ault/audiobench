-- Migration 044: Couple chat messages and conversations to audio and segments.
--
-- Enables voice mode in chat, linking chat messages to time-coded segments,
-- audio assets on disk, duration, and Gemini Live voice metadata.

ALTER TABLE chat_messages ADD COLUMN modality VARCHAR(16) DEFAULT 'text';
ALTER TABLE chat_messages ADD COLUMN duration_seconds FLOAT DEFAULT NULL;
ALTER TABLE chat_messages ADD COLUMN audio_path VARCHAR(512) DEFAULT NULL;
ALTER TABLE chat_messages ADD COLUMN interrupted BOOLEAN DEFAULT 0;
ALTER TABLE chat_messages ADD COLUMN voice_name VARCHAR(64) DEFAULT NULL;
ALTER TABLE chat_messages ADD COLUMN segment_id INTEGER REFERENCES segments(id) ON DELETE SET NULL;

ALTER TABLE chat_conversations ADD COLUMN audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE SET NULL;
ALTER TABLE chat_conversations ADD COLUMN transcription_id INTEGER REFERENCES transcriptions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_chat_messages_segment_id ON chat_messages (segment_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_modality ON chat_messages (modality);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_audio_file_id ON chat_conversations (audio_file_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_transcription_id ON chat_conversations (transcription_id);
