import json
from pathlib import Path

from audiobench.cli.display.job_stream import render_job_event, tail_job_events
from audiobench.cli.display.phase_tracker import PhaseTracker
from audiobench.core.db_session import get_session
from audiobench.jobs.scheduler import enqueue
from audiobench.storage.models import UnifiedJob


def test_tracker_abort_emits_error_event(tmp_path):
    events_file = tmp_path / "job_test.events"
    tracker = PhaseTracker(
        events_file=events_file,
        quiet=True,
    )

    tracker.update("transcribing", "Transcribing chunk 1/2...", 0.5)
    tracker.abort("Simulated failure")

    assert events_file.exists()
    lines = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]

    types = [evt.get("t") for evt in lines]
    assert "error" in types
    assert "done" not in types

    error_evt = [e for e in lines if e.get("t") == "error"][0]
    assert error_evt["message"] == "Simulated failure"


def test_tracker_finalize_emits_done_event(tmp_path):
    events_file = tmp_path / "job_test.events"
    tracker = PhaseTracker(
        events_file=events_file,
        quiet=True,
    )

    tracker.update("transcribing", "Complete", 1.0)
    tracker.finalize()

    assert events_file.exists()
    lines = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]

    types = [evt.get("t") for evt in lines]
    assert "done" in types
    assert "error" not in types


def test_job_stream_tail_stops_on_error_event(test_db, tmp_path):
    events_file = tmp_path / "job_test.events"
    events_file.write_text(
        json.dumps({"t": "phase", "phase": "transcribing", "message": "Chunk 1/2"}) + "\n"
        + json.dumps({"t": "error", "message": "Protobuf depth exceeded"}) + "\n"
    )

    id_cloud = enqueue(job_type="transcribe", args=["test.mp3"], slot="transcription:cloud")

    ok = tail_job_events(id_cloud, events_file)
    assert ok is False
