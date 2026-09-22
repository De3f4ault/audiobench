-- Migration 035: YouTube Channel Workspace
-- Adds YouTubeChannelNode (standing relationship per channel),
-- YouTubePlaylistCache (uploads playlist cache, filled by Phase 2),
-- and youtube_channel_id to audio_files for per-channel library counting.
--
-- YouTubeChannelNode.whiteboard_text is NEVER sent to LanceDB — enforced in channel_store.py.
-- engagement_weight is NOT stored — computed on read from last_visited_at.
--
-- YouTubePlaylistCache.videos_json entries have the shape:
--   {video_id, title, published_at, duration_pt, availability}
--   availability: "public" | "unlisted" | "private_or_removed"
-- video_count_available is the denominator for the "89/312" display.
-- video_count_total is the raw API count (includes private placeholders).

-- Add channel ID to audio files for per-channel library counting
ALTER TABLE audio_files ADD COLUMN youtube_channel_id TEXT;
CREATE INDEX IF NOT EXISTS idx_audio_files_yt_channel ON audio_files(youtube_channel_id);

-- First-class channel node
CREATE TABLE IF NOT EXISTS youtube_channel_nodes (
    id                    INTEGER  PRIMARY KEY AUTOINCREMENT,
    channel_id            TEXT     NOT NULL UNIQUE,  -- YouTube UC... ID
    title                 TEXT     NOT NULL,
    thumbnail_url         TEXT,
    whiteboard_text       TEXT,                      -- operational note, never embedded
    whiteboard_updated_at DATETIME,
    last_visited_at       DATETIME,                  -- only stored engagement signal
    created_at            DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ycn_last_visited
    ON youtube_channel_nodes(last_visited_at DESC);

-- Uploads playlist cache (filled by Phase 2, schema locked here)
CREATE TABLE IF NOT EXISTS youtube_playlist_cache (
    id                    INTEGER  PRIMARY KEY AUTOINCREMENT,
    channel_id            TEXT     NOT NULL UNIQUE
                              REFERENCES youtube_channel_nodes(channel_id)
                              ON DELETE CASCADE,
    fetched_at            DATETIME NOT NULL,
    video_count_total     INTEGER  NOT NULL DEFAULT 0,  -- raw API totalResults
    video_count_available INTEGER  NOT NULL DEFAULT 0,  -- public + unlisted only (UI denominator)
    videos_json           TEXT     NOT NULL DEFAULT '[]',
    -- Shape of each entry in videos_json:
    --   {"video_id": "...", "title": "...", "published_at": "YYYY-MM-DD",
    --    "duration_pt": "PT14M20S", "availability": "public|unlisted|private_or_removed"}
    -- availability is captured from the API at fetch time — never derived from title.
    next_page_token       TEXT                          -- NULL = fully fetched; token = partial
);
