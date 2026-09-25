-- Migration 036: Backfill model_name = 'default'
--
-- Root cause: transcriber.py used getattr(engine, "model_name", ...) but engines
-- store the model as `_model_name` (private). The double-miss fell through to
-- getattr(settings, "model", "default") where "model" also doesn't exist, so the
-- literal string "default" was written for every new transcription.
--
-- Fix: restore the correct model name based on engine type.
--   faster-whisper rows → large-v3-turbo  (configured model_name default)
--   gemini rows         → gemini-2.5-flash (configured gemini_model default)
--
-- Safe to re-run (WHERE clause is idempotent).

UPDATE transcriptions
SET model_name = 'large-v3-turbo'
WHERE model_name = 'default'
  AND engine = 'faster-whisper';

UPDATE transcriptions
SET model_name = 'gemini-2.5-flash'
WHERE model_name = 'default'
  AND engine = 'gemini';
