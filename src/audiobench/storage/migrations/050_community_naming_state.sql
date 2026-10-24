-- Migration 050: Community Naming State and Retry Backoff
-- Adds tracking columns to expression_communities for the asynchronous LLM naming sweep.

ALTER TABLE expression_communities ADD COLUMN naming_attempted_at DATETIME;
ALTER TABLE expression_communities ADD COLUMN naming_failure_count INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_expression_communities_sweep
ON expression_communities (is_active, name_source, naming_attempted_at);
