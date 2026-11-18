"""Tests for the unified jobs CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from audiobench.cli.commands.jobs_cmd import jobs
from audiobench.core.db_session import get_session
from audiobench.jobs.scheduler import enqueue
from audiobench.storage.models import UnifiedJob


def test_jobs_list_empty(runner, test_db):
    result = runner.invoke(jobs, [])
    assert result.exit_code == 0
    assert "No jobs found" in result.output


def test_jobs_list_with_active_and_recent(runner, test_db):
    id1 = enqueue(
        job_type="transcribe",
        args=["transcribe", "meeting.m4a"],
        file_label="meeting.m4a",
    )
    id2 = enqueue(
        job_type="youtube_fetch",
        slot="network",
        args=["youtube", "_fetch_internal", "abc"],
        file_label="Interview",
    )

    with get_session() as session:
        j2 = session.get(UnifiedJob, id2)
        j2.status = "done"
        session.commit()

    result = runner.invoke(jobs, [])
    assert result.exit_code == 0
    assert "meeting.m4a" in result.output
    assert "Interview" in result.output
    assert "Summary:" in result.output
    assert "1 pending" in result.output
    assert "1 done" in result.output


def test_jobs_batch_progress_display(runner, test_db):
    for i in range(1, 6):
        enqueue(
            job_type="transcribe",
            args=["transcribe", f"f{i}.mp3"],
            file_label=f"f{i}.mp3",
            batch_id="b-test",
            batch_label="Test Recordings",
            batch_index=i,
            batch_total=5,
        )

    import os

    with get_session() as session:
        j1 = session.query(UnifiedJob).filter_by(file_label="f1.mp3").first()
        j1.status = "done"
        j2 = session.query(UnifiedJob).filter_by(file_label="f2.mp3").first()
        j2.status = "running"
        j2.pid = os.getpid()
        session.commit()

    result = runner.invoke(jobs, [])
    assert result.exit_code == 0
    assert "Test Recordings" in result.output
    assert "1/5" in result.output  # 1 done out of 5


def test_jobs_cancel_command(runner, test_db):
    jid = enqueue(job_type="transcribe", args=["cancel_me.mp3"])
    result = runner.invoke(jobs, ["cancel", str(jid)])
    assert result.exit_code == 0
    assert f"Job #{jid} cancelled" in result.output

    with get_session() as session:
        job = session.get(UnifiedJob, jid)
        assert job.status == "cancelled"


def test_jobs_cancel_batch_command(runner, test_db):
    enqueue(job_type="transcribe", args=["1.mp3"], batch_id="b-cancel")
    enqueue(job_type="transcribe", args=["2.mp3"], batch_id="b-cancel")

    result = runner.invoke(jobs, ["cancel", "--batch", "b-cancel"])
    assert result.exit_code == 0
    assert "Cancelled 2 job(s) in batch b-cancel" in result.output


def test_jobs_cancel_all_command(runner, test_db):
    enqueue(job_type="transcribe", args=["1.mp3"])
    enqueue(job_type="transcribe", args=["2.mp3"])

    result = runner.invoke(jobs, ["cancel", "--all"])
    assert result.exit_code == 0
    assert "Cancelled 2 job(s)" in result.output


def test_jobs_retry_command(runner, test_db):
    jid = enqueue(job_type="transcribe", args=["retry_me.mp3"])
    with get_session() as session:
        j = session.get(UnifiedJob, jid)
        j.status = "failed"
        session.commit()

    result = runner.invoke(jobs, ["retry", str(jid)])
    assert result.exit_code == 0
    assert f"Job #{jid} re-enqueued as pending" in result.output

    with get_session() as session:
        j = session.get(UnifiedJob, jid)
        assert j.status == "pending"


def test_jobs_prune_command(runner, test_db, tmp_path):
    jid = enqueue(job_type="transcribe", args=["prune_me.mp3"])
    with get_session() as session:
        j = session.get(UnifiedJob, jid)
        j.status = "done"
        session.commit()

    result = runner.invoke(jobs, ["prune"])
    assert result.exit_code == 0
    assert "Pruned 1 job(s)" in result.output

    with get_session() as session:
        assert session.get(UnifiedJob, jid) is None


def test_jobs_tail_command(runner, test_db, tmp_path):
    events_file = tmp_path / "test.events"
    with open(events_file, "w") as f:
        f.write(json.dumps({"t": "phase", "phase": "converting"}) + "\n")
        f.write(
            json.dumps({"t": "segment", "start": 0.0, "end": 5.0, "text": "Hello world"}) + "\n"
        )
        f.write(json.dumps({"t": "done"}) + "\n")

    jid = enqueue(job_type="transcribe", args=["tail_me.mp3"])
    with get_session() as session:
        j = session.get(UnifiedJob, jid)
        j.events_path = str(events_file)
        j.status = "done"
        session.commit()

    result = runner.invoke(jobs, ["tail", str(jid)])
    assert result.exit_code == 0
    assert "converting..." in result.output
    assert "Hello world" in result.output
    assert "Completed successfully" in result.output


def test_jobs_tail_falls_back_to_log(runner, test_db, tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("2026-09-17 [INFO] Biometric worker started\n2026-09-17 [INFO] Processed 10 segments\n")

    # Job has no events file, only log file
    jid = enqueue(job_type="biometric_pass", args=["_biometric_worker", "1", "a.mp3", "[]"])
    with get_session() as session:
        j = session.get(UnifiedJob, jid)
        j.log_path = str(log_file)
        j.events_path = None
        j.status = "done"
        session.commit()

    result = runner.invoke(jobs, ["tail", str(jid)])
    assert result.exit_code == 0
    assert "streaming execution log instead" in result.output
    assert "Biometric worker started" in result.output
    assert "Processed 10 segments" in result.output


def test_jobs_diff_command(runner, test_db):
    id1 = enqueue(
        job_type="transcribe",
        args=["transcribe", "meeting.mp3", "--model", "base"],
        file_label="meeting.mp3",
    )
    id2 = enqueue(
        job_type="transcribe",
        args=["transcribe", "meeting.mp3", "--model", "large-v3-turbo"],
        file_label="meeting.mp3",
        allow_duplicate=True,
    )

    result = runner.invoke(jobs, ["diff", str(id1), str(id2)])
    assert result.exit_code == 0
    assert "Comparison: Job #" in result.output
    assert "meeting.mp3" in result.output
    assert "base" in result.output
    assert "large-v3-turbo" in result.output
    assert "SIMILAR" in result.output


def test_jobs_interactive_retry_all_excludes_cancelled_by_default(runner, test_db, monkeypatch):
    id_failed = enqueue(job_type="transcribe", args=["failed.mp3"])
    id_cancelled = enqueue(job_type="transcribe", args=["cancelled.mp3"])

    with get_session() as session:
        jf = session.get(UnifiedJob, id_failed)
        jf.status = "failed"
        jc = session.get(UnifiedJob, id_cancelled)
        jc.status = "cancelled"
        session.commit()

    monkeypatch.setattr("audiobench.cli.commands.jobs_cmd.ensure_worker", lambda: None)

    # First pass: /retry --all should only retry the failed job, preserving cancelled
    prompts1 = iter(["/retry --all", "/exit"])
    class MockPtSession1:
        def prompt(self, _):
            return next(prompts1)

    monkeypatch.setattr(
        "audiobench.cli.shared.repl_shell.make_prompt_session",
        lambda **kwargs: MockPtSession1(),
    )

    result = runner.invoke(jobs, ["-i"])
    assert result.exit_code == 0
    assert "Re-enqueued 1 job(s) as pending." in result.output

    with get_session() as session:
        jf = session.get(UnifiedJob, id_failed)
        jc = session.get(UnifiedJob, id_cancelled)
        assert jf.status == "pending"
        assert jc.status == "cancelled"

    # Second pass: /retry --all --include-cancelled should re-enqueue cancelled jobs as well
    prompts2 = iter(["/retry --all --include-cancelled", "/exit"])
    class MockPtSession2:
        def prompt(self, _):
            return next(prompts2)

    monkeypatch.setattr(
        "audiobench.cli.shared.repl_shell.make_prompt_session",
        lambda **kwargs: MockPtSession2(),
    )

    result2 = runner.invoke(jobs, ["-i"])
    assert result2.exit_code == 0
    assert "Re-enqueued 1 job(s) as pending." in result2.output

    with get_session() as session:
        jc = session.get(UnifiedJob, id_cancelled)
        assert jc.status == "pending"




