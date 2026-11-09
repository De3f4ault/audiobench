-- Migration 054: Create missing index on notes.status
--
-- Migration 012 declared this index but it was silently dropped by
-- executescript() because the CREATE INDEX statement followed a CREATE TABLE
-- in the same script. The notes table currently has 0 rows so there is no
-- query-performance impact from the absence, but the index is part of the
-- declared schema and should exist for consistency.
--
-- Pattern: ALTER TABLE ADD COLUMN or CREATE TABLE followed by CREATE INDEX
-- in the same migration file is unsafe with executescript(). Each DDL that
-- depends on a preceding schema change must be in its own numbered file.

CREATE INDEX IF NOT EXISTS ix_notes_status ON notes (status);
