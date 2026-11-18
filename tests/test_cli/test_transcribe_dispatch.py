"""Tests for the intelligent transcribe command dispatcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from audiobench.cli.commands.transcribe import transcribe
from audiobench.core.db_session import get_session
from audiobench.jobs.scheduler import enqueue
from audiobench.storage.models import UnifiedJob


def test_dispatch_explicit_files_priority(runner, test_db, tmp_path):
    f1 = tmp_path / "song1.mp3"
    f2 = tmp_path / "song2.mp3"
    f1.write_bytes(b"dummy audio 1")
    f2.write_bytes(b"dummy audio 2")

    with patch("audiobench.cli.display.job_stream.stream_jobs_sequence") as mock_stream, \
         patch("audiobench.jobs.scheduler.ensure_worker") as mock_worker:
        mock_stream.return_value = True

        result = runner.invoke(transcribe, [str(f1), str(f2)])
        assert result.exit_code == 0
        assert mock_worker.called
        assert mock_stream.called

        # Extract queued job IDs passed to stream_jobs_sequence
        queued_ids = mock_stream.call_args[0][0]
        assert len(queued_ids) == 2

        with get_session() as session:
            jobs = session.query(UnifiedJob).filter(UnifiedJob.id.in_(queued_ids)).all()
            assert len(jobs) == 2
            for j in jobs:
                assert j.priority == 10
                assert j.slot == "transcription:local"
                assert j.batch_total == 2


def test_dispatch_folder_input_routes_to_normal_queue(runner, test_db, tmp_path):
    sub = tmp_path / "audio_dir"
    sub.mkdir()
    f1 = sub / "track1.mp3"
    f2 = sub / "track2.mp3"
    f1.write_bytes(b"audio 1")
    f2.write_bytes(b"audio 2")

    with patch("audiobench.cli.display.job_stream.stream_jobs_sequence") as mock_stream, \
         patch("audiobench.jobs.scheduler.ensure_worker") as mock_worker:

        result = runner.invoke(transcribe, [str(sub)])
        assert result.exit_code == 0
        assert mock_worker.called
        # Folder input should NOT attach/stream in foreground
        assert not mock_stream.called
        assert "background queue" in result.output

        with get_session() as session:
            jobs = session.query(UnifiedJob).filter_by(slot="transcription:local").all()
            assert len(jobs) == 2
            for j in jobs:
                assert j.priority == 100


def test_dispatch_many_files_routes_to_normal_queue(runner, test_db, tmp_path):
    # 6 files (> 5 priority threshold)
    files = []
    for i in range(6):
        f = tmp_path / f"part_{i}.mp3"
        f.write_bytes(b"audio")
        files.append(str(f))

    with patch("audiobench.cli.display.job_stream.stream_jobs_sequence") as mock_stream, \
         patch("audiobench.jobs.scheduler.ensure_worker") as mock_worker:

        result = runner.invoke(transcribe, files)
        assert result.exit_code == 0
        assert mock_worker.called
        assert not mock_stream.called
        assert "background queue" in result.output

        with get_session() as session:
            jobs = session.query(UnifiedJob).filter_by(slot="transcription:local").all()
            assert len(jobs) == 6
            for j in jobs:
                assert j.priority == 100


def test_dispatch_gemini_routes_to_cloud_slot(runner, test_db, tmp_path):
    f = tmp_path / "sample.mp3"
    f.write_bytes(b"sample audio")

    with patch("audiobench.cli.display.job_stream.stream_jobs_sequence") as mock_stream, \
         patch("audiobench.jobs.scheduler.ensure_worker") as mock_worker:

        result = runner.invoke(transcribe, [str(f), "--engine", "gemini"])
        assert result.exit_code == 0
        assert mock_worker.called
        assert mock_stream.called
        assert "cloud lane" in result.output

        queued_ids = mock_stream.call_args[0][0]
        with get_session() as session:
            job = session.get(UnifiedJob, queued_ids[0])
            assert job.slot == "transcription:cloud"


def test_dispatch_deduplication_skips_active(runner, test_db, tmp_path):
    f1 = tmp_path / "active.mp3"
    f1.write_bytes(b"audio content")

    # Already queued
    enqueue(job_type="transcribe", args=["transcribe", str(f1.resolve())], file_label=f1.name)

    with patch("audiobench.cli.display.job_stream.stream_jobs_sequence") as mock_stream:
        result = runner.invoke(transcribe, [str(f1)])
        assert result.exit_code == 0
        assert "already completed or queued" in result.output
        assert not mock_stream.called


def test_dispatch_deduplication_on_unsupported_platform(runner, test_db, tmp_path):
    f1 = tmp_path / "win_active.mp3"
    f1.write_bytes(b"audio content")

    # Already queued
    enqueue(job_type="transcribe", args=["transcribe", str(f1.resolve())], file_label=f1.name)

    with patch("audiobench.cli.commands.transcribe.SUPPORTS_BACKGROUND_JOBS", False):
        result = runner.invoke(transcribe, [str(f1)])
        assert result.exit_code == 0
        assert "already completed or queued" in result.output
