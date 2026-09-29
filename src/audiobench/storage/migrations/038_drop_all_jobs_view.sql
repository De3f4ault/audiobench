-- 022_drop_all_jobs_view.sql
-- Drops the `all_jobs` SQL view introduced in 021_unified_jobs.sql.
--
-- Context (2026-09-29):
-- The `all_jobs` view was a UNION ALL of `jobs` and `job_queue` tables created
-- to give the TUI OperationsPanel a unified feed. It never included `unified_jobs`
-- (System 3, introduced 2026-09-17), so the panel has never shown a live job.
-- The last write to `jobs` was 2026-08-31; the last write to `job_queue` was
-- 2026-08-08. Both source tables are being retired in the Phase 2 cleanup.
-- The `OperationsPanel` is being repointed to query `unified_jobs` directly.
--
-- Using DROP VIEW IF EXISTS so this is safe to run on any DB that missed 021.

DROP VIEW IF EXISTS all_jobs;
