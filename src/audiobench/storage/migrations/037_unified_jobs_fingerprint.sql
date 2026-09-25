-- 037_unified_jobs_fingerprint.sql
-- Adds fingerprint column to unified_jobs for deterministic duplicate detection
-- across concurrent terminals and CLI invocations.

ALTER TABLE unified_jobs ADD COLUMN fingerprint VARCHAR(64);
CREATE INDEX IF NOT EXISTS ix_unified_jobs_fingerprint ON unified_jobs (fingerprint);
