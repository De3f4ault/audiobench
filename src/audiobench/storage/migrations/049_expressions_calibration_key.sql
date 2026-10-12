-- Migration 049: Add calibration_key column to expressions table
--
-- Disambiguates state payloads stored under source_type = 'daemon_calibration'
-- so subsystems (PatternDetector, TimingModel, CalibrationTracker, ConnectionSurfer,
-- BlindSpotDetector) do not collide or overwrite each other's state.

ALTER TABLE expressions ADD COLUMN calibration_key VARCHAR(64);

CREATE INDEX IF NOT EXISTS ix_expressions_calibration_key ON expressions (calibration_key);
