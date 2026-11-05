-- Migration 041: Enrich audio_files table with format and sequence metadata.
--
-- Adds normalized columns for:
--   track_title   - Track/chapter name from ffprobe tags['title'] (81% present)
--   sequence_key  - Deterministic sort key computed by sequence_resolver (e.g. '01-002', '007', '012b')
--   codec         - Stream audio codec name (100% present, e.g. 'mp3', 'aac', 'flac')
--   bit_rate      - Format/stream bitrate in bps (99.6% present, NULL when unknown)
--   genre         - Genre tag from ffprobe tags['genre'] (29% present, NULL when absent)

ALTER TABLE audio_files ADD COLUMN track_title TEXT;
ALTER TABLE audio_files ADD COLUMN sequence_key TEXT;
ALTER TABLE audio_files ADD COLUMN codec TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE audio_files ADD COLUMN bit_rate INTEGER;
ALTER TABLE audio_files ADD COLUMN genre TEXT;

CREATE INDEX IF NOT EXISTS idx_audio_files_sequence_key ON audio_files (sequence_key);
