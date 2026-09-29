"""
Tests for 1B — _emit_event writes to log_event (journal DB) instead of a flat file.

Three invariants being tested:
1. _emit_event with a valid job_id calls log_event() with the correct subsystem,
   event_type, entity_type/id, and metadata — not a flat file.
2. _emit_event with job_id=None is a no-op (no log_event call, no file write).
3. Phase events emitted during the pipeline have subsystem="transcribe" and the
   correct event_type.

runner.py invariant (separate from transcriber):
4. submit_job no longer creates a job_{id}.events file on disk.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. _emit_event writes to log_event, not a flat file
# ---------------------------------------------------------------------------

def test_emit_event_writes_to_log_event():
    """
    _emit_event(job_id=5, phase='done', words=120) must call log_event() with:
      - subsystem = "transcribe"
      - event_type = "done"
      - entity_type = "job"
      - entity_id = 5
    and must NOT access self._settings.data_dir (no flat-file write).
    """
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    t = TranscriptionPipeline.__new__(TranscriptionPipeline)
    t._settings = MagicMock()

    with patch("audiobench.transcribe.transcriber.log_event") as mock_log:
        t._emit_event(5, phase="done", words=120, speakers=2, duration=300)

    mock_log.assert_called_once()
    pos_args = mock_log.call_args[0]
    kw = mock_log.call_args[1]

    assert pos_args[0] == "transcribe", "subsystem must be 'transcribe'"
    assert pos_args[1] == "done",       "event_type must match the phase kwarg"
    assert kw.get("entity_type") == "job"
    assert kw.get("entity_id") == 5

    # Must not touch the filesystem via settings.data_dir
    t._settings.data_dir.__truediv__.assert_not_called()


def test_emit_event_without_job_id_is_noop():
    """
    _emit_event(None, phase='transcribing') must not call log_event and
    must not access the filesystem.
    """
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    t = TranscriptionPipeline.__new__(TranscriptionPipeline)
    t._settings = MagicMock()

    with patch("audiobench.transcribe.transcriber.log_event") as mock_log:
        t._emit_event(None, phase="transcribing", progress=50)

    mock_log.assert_not_called()
    t._settings.data_dir.__truediv__.assert_not_called()


def test_pipeline_phase_events_have_correct_subsystem():
    """
    Any phase update — transcribing, saving, done — must use
    subsystem='transcribe' and the phase name as event_type.
    """
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    t = TranscriptionPipeline.__new__(TranscriptionPipeline)
    t._settings = MagicMock()

    with patch("audiobench.transcribe.transcriber.log_event") as mock_log:
        t._emit_event(7, phase="transcribing", progress=42)

    assert mock_log.called
    pos_args = mock_log.call_args[0]
    assert pos_args[0] == "transcribe"    # subsystem
    assert pos_args[1] == "transcribing"  # event_type

    kw = mock_log.call_args[1]
    assert kw.get("entity_id") == 7
    assert kw.get("entity_type") == "job"


# ---------------------------------------------------------------------------
# 4. runner.py: submit_job does NOT create a flat .events file
# ---------------------------------------------------------------------------

# test_submit_job_does_not_create_events_file removed 2026-09-29.
# It tested that runner.submit_job() no longer creates .events files.
# runner.py (System 1) was deleted in the job system consolidation.

