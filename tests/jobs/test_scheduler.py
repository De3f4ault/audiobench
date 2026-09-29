"""Tests for unified scheduler and worker daemon."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiobench.core.db_session import get_session
from audiobench.jobs.scheduler import (
    cancel_batch,
    cancel_job,
    claim_next_pending,
    deduplicate_files,
    drain_slot,
    enqueue,
    get_slot_capacity,
    prune_jobs,
    retry_job,
    startup_recovery,
)
from audiobench.storage.models import UnifiedJob


def test_enqueue_returns_id(test_db):
    job_id = enqueue(
        job_type="transcribe",
        args=["transcribe", "test.mp3"],
        slot="transcription",
        file_label="test.mp3",
        batch_id="b-123",
        batch_total=1,
    )
    assert isinstance(job_id, int)
    assert job_id > 0

    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        assert job is not None
        assert job.status == "pending"
        assert job.slot == "transcription"
        assert job.file_label == "test.mp3"
        assert job.batch_id == "b-123"


def test_slot_capacity_whisper_and_gemini(test_db, monkeypatch):
    from audiobench.core.settings import get_settings

    # Default whisper -> 1
    assert get_slot_capacity("transcription") == 1
    assert get_slot_capacity("network") == 3
    assert get_slot_capacity("indexing") == 1

    # Gemini default -> 1
    monkeypatch.setenv("AUDIOBENCH_ENGINE", "gemini")
    get_settings.cache_clear()
    assert get_slot_capacity("transcription") == 1

    # Gemini configured slots -> 3
    monkeypatch.setenv("AUDIOBENCH_TRANSCRIPTION_CLOUD_SLOTS", "3")
    get_settings.cache_clear()
    assert get_slot_capacity("transcription") == 3
    get_settings.cache_clear()


def test_enqueue_priority(test_db):
    from audiobench.jobs.scheduler import enqueue_priority

    id_normal = enqueue(job_type="transcribe", args=["normal.mp3"], priority=100)
    id_prio = enqueue_priority(job_type="transcribe", args=["prio.mp3"])

    with get_session() as session:
        j_prio = session.get(UnifiedJob, id_prio)
        assert j_prio is not None
        assert j_prio.priority == 10

    # Claim should yield priority first
    job = claim_next_pending("transcription")
    assert job is not None
    assert job.id == id_prio


def test_claim_next_pending(test_db):
    id1 = enqueue(job_type="transcribe", args=["1.mp3"], priority=100)
    id2 = enqueue(job_type="transcribe", args=["2.mp3"], priority=50)

    # id2 has lower priority number (higher urgency) -> claimed first
    job = claim_next_pending("transcription")
    assert job is not None
    assert job.id == id2
    assert job.status == "running"

    # id1 claimed next
    job2 = claim_next_pending("transcription")
    assert job2 is not None
    assert job2.id == id1
    assert job2.status == "running"

    # None left
    assert claim_next_pending("transcription") is None


def test_startup_recovery_dead_pid(test_db):
    job_id = enqueue(job_type="transcribe", args=["test.mp3"], max_attempts=1)
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        job.status = "running"
        job.pid = 9999999  # dead PID
        session.commit()

    recovered = startup_recovery()
    assert recovered == 1

    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        assert job.status == "failed"
        assert job.exit_code == -1
        assert "Process died" in (job.error_summary or "")


def test_startup_recovery_retries_if_attempts_remain(test_db):
    job_id = enqueue(job_type="transcribe", args=["test.mp3"], max_attempts=2)
    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        job.status = "running"
        job.pid = 9999999
        session.commit()

    recovered = startup_recovery()
    assert recovered == 1

    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        assert job.status == "pending"
        assert job.attempt == 2
        assert job.pid is None


def test_deduplicate_files(test_db):
    enqueue(job_type="transcribe", args=["a.mp3"], file_label="a.mp3")
    enqueue(job_type="transcribe", args=["b.mp3"], file_label="b.mp3")

    with get_session() as session:
        j = session.query(UnifiedJob).filter_by(file_label="b.mp3").first()
        j.status = "done"
        session.commit()

    files = [Path("a.mp3"), Path("b.mp3"), Path("c.mp3")]
    deduped = deduplicate_files(files)
    assert [f.name for f in deduped] == ["c.mp3"]


def test_cancel_job(test_db):
    job_id = enqueue(job_type="transcribe", args=["test.mp3"])
    success = cancel_job(job_id)
    assert success is True

    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        assert job.status == "cancelled"


def test_cancel_batch(test_db):
    id1 = enqueue(job_type="transcribe", args=["1.mp3"], batch_id="b1")
    id2 = enqueue(job_type="transcribe", args=["2.mp3"], batch_id="b1")
    id3 = enqueue(job_type="transcribe", args=["3.mp3"], batch_id="b2")

    cancelled = cancel_batch("b1")
    assert cancelled == 2

    with get_session() as session:
        assert session.get(UnifiedJob, id1).status == "cancelled"
        assert session.get(UnifiedJob, id2).status == "cancelled"
        assert session.get(UnifiedJob, id3).status == "pending"


def test_retry_job(test_db):
    job_id = enqueue(job_type="transcribe", args=["test.mp3"])
    cancel_job(job_id)

    retried_id = retry_job(job_id)
    assert retried_id == job_id

    with get_session() as session:
        job = session.get(UnifiedJob, job_id)
        assert job.status == "pending"
        assert job.attempt == 1


def test_prune_jobs(test_db, tmp_path):
    job_id = enqueue(job_type="transcribe", args=["test.mp3"])
    log_file = tmp_path / "test.log"
    log_file.write_text("sample log")

    with get_session() as session:
        j = session.get(UnifiedJob, job_id)
        j.status = "done"
        j.log_path = str(log_file)
        session.commit()

    assert log_file.exists()
    pruned = prune_jobs()
    assert pruned == 1
    assert not log_file.exists()

    with get_session() as session:
        assert session.get(UnifiedJob, job_id) is None


def test_drain_slot_empty_queue(test_db):
    processed = drain_slot("transcription")
    assert processed == 0


def test_biometric_worker_command_resolution(test_db):
    import json
    import sys

    raw_args = ["_biometric_worker", "42", "audio.mp3", "[]"]
    if raw_args and raw_args[0] == "_biometric_worker":
        cmd = [sys.executable, "-m", "audiobench._biometric_worker", *raw_args[1:]]
    else:
        cmd = [sys.executable, "-m", "audiobench", *raw_args]

    assert cmd == [sys.executable, "-m", "audiobench._biometric_worker", "42", "audio.mp3", "[]"]


def test_transcription_precedence_over_indexing(test_db):
    from audiobench.jobs.scheduler import has_active_transcription

    # 1. Enqueue indexing job first
    bio_id = enqueue(job_type="biometric_pass", args=["_biometric_worker"], slot="indexing")
    # 2. Enqueue transcription job
    tx_id = enqueue(job_type="transcribe", args=["sample.mp3"], slot="transcription")

    assert has_active_transcription() is True
    assert get_slot_capacity("indexing") == 0

    # Indexing cannot be claimed while transcription is pending
    assert claim_next_pending("indexing") is None

    # Transcription can be claimed
    tx_job = claim_next_pending("transcription")
    assert tx_job is not None
    assert tx_job.id == tx_id

    # Mark transcription done
    with get_session() as session:
        j = session.get(UnifiedJob, tx_id)
        j.status = "done"
        session.commit()

    # Now indexing can proceed
    assert has_active_transcription() is False
    assert get_slot_capacity("indexing") == 1

    bio_job = claim_next_pending("indexing")
    assert bio_job is not None
    assert bio_job.id == bio_id


def test_preempt_indexing_jobs(test_db):
    from audiobench.jobs.scheduler import preempt_indexing_jobs

    bio_id = enqueue(job_type="biometric_pass", args=["_biometric_worker"], slot="indexing")
    with get_session() as session:
        j = session.get(UnifiedJob, bio_id)
        j.status = "running"
        j.pid = 9999999  # Non-existent PID
        session.commit()

    preempted = preempt_indexing_jobs("Yielding for transcription")
    assert preempted == 1

    with get_session() as session:
        j = session.get(UnifiedJob, bio_id)
        assert j.status == "pending"
        assert j.pid is None
        assert "Yielding for transcription" in (j.error_summary or "")


def test_compute_job_fingerprint_normalization():
    from audiobench.jobs.scheduler import compute_job_fingerprint

    # Flag order independence
    fp1 = compute_job_fingerprint(
        "transcribe",
        ["transcribe", "test.mp3", "--model", "large-v3", "--language", "en"],
        file_label="test.mp3",
    )
    fp2 = compute_job_fingerprint(
        "transcribe",
        ["transcribe", "test.mp3", "--language", "en", "--model", "large-v3"],
        file_label="test.mp3",
    )
    assert fp1 == fp2

    # Parameter differentiation
    fp3 = compute_job_fingerprint(
        "transcribe",
        ["transcribe", "test.mp3", "--model", "base", "--language", "en"],
        file_label="test.mp3",
    )
    assert fp1 != fp3


def test_enqueue_duplicate_interception(test_db):
    # First invocation in Terminal 1
    id1 = enqueue(
        job_type="transcribe",
        args=["transcribe", "duplicate_test.mp3", "--model", "large-v3"],
        file_label="duplicate_test.mp3",
    )

    # Second invocation in Terminal 2 (identical file and settings)
    id2 = enqueue(
        job_type="transcribe",
        args=["transcribe", "duplicate_test.mp3", "--model", "large-v3"],
        file_label="duplicate_test.mp3",
    )

    # Must return identical ID without creating duplicate row
    assert id1 == id2

    with get_session() as session:
        count = (
            session.query(UnifiedJob)
            .filter_by(file_label="duplicate_test.mp3")
            .count()
        )
        assert count == 1


def test_cloud_slot_does_not_gate_or_preempt_indexing(test_db):
    from audiobench.jobs.scheduler import claim_next_pending, has_active_local_compute

    # 1. Enqueue cloud transcribe job
    id_cloud = enqueue(job_type="transcribe", args=["cloud.mp3"], slot="transcription:cloud")
    # Cloud job does not count as local compute
    assert not has_active_local_compute()

    # 2. Enqueue indexing job - can be claimed because only cloud is pending
    id_index = enqueue(job_type="biometric_pass", args=["idx"], slot="indexing")
    claimed_index = claim_next_pending("indexing")
    assert claimed_index is not None
    assert claimed_index.id == id_index

    # 3. Claim cloud job - does NOT preempt running indexing job
    claimed_cloud = claim_next_pending("transcription:cloud")
    assert claimed_cloud is not None
    assert claimed_cloud.id == id_cloud

    with get_session() as session:
        idx_job = session.get(UnifiedJob, id_index)
        assert idx_job.status == "running"  # Still running!


def test_local_slot_preempts_indexing(test_db):
    from audiobench.jobs.scheduler import claim_next_pending, has_active_local_compute

    # 1. Start indexing job
    id_index = enqueue(job_type="biometric_pass", args=["idx"], slot="indexing")
    claimed_index = claim_next_pending("indexing")
    assert claimed_index is not None
    assert claimed_index.id == id_index

    # 2. Enqueue local transcribe job
    id_local = enqueue(job_type="transcribe", args=["local.mp3"], slot="transcription:local")
    assert has_active_local_compute()

    # 3. Claiming local job preempts running indexing job
    claimed_local = claim_next_pending("transcription:local")
    assert claimed_local is not None

    with get_session() as session:
        idx_job = session.get(UnifiedJob, id_index)
        assert idx_job.status == "pending"  # Preempted back to pending!


def test_alignment_slot_gates_indexing_and_waits_for_running(test_db):
    from audiobench.jobs.scheduler import claim_next_pending, enqueue_alignment, has_active_local_compute

    # 1. Start indexing job
    id_index = enqueue(job_type="biometric_pass", args=["idx"], slot="indexing")
    claimed_index = claim_next_pending("indexing")
    assert claimed_index is not None
    assert claimed_index.status == "running"

    # 2. Enqueue alignment job
    id_align = enqueue_alignment(tx_id=99, audio_path="/tmp/test.mp3", parent_job_id=42)
    assert has_active_local_compute()

    # 3. Align job waits for running indexing job to finish (does not brutally SIGTERM mid-chunk)
    assert claim_next_pending("transcription:align") is None

    # 4. Once indexing finishes, align job can be claimed
    with get_session() as session:
        j = session.get(UnifiedJob, id_index)
        j.status = "done"
        session.commit()

    claimed_align = claim_next_pending("transcription:align")
    assert claimed_align is not None
    assert claimed_align.id == id_align
    assert claimed_align.slot == "transcription:align"


def test_preempted_biometric_pass_survives_startup_recovery(test_db):
    """A biometric_pass job preempted once (max_attempts=3) must survive startup_recovery
    as pending, not be permanently marked failed.

    Context (Bug #263): preempt_indexing_jobs() resets job status to 'pending' and
    nulls the PID. If max_attempts=1, startup_recovery found attempt(1) >= max_attempts(1)
    and marked it failed. With max_attempts=3, attempt(1) < max_attempts(3) => stays pending.
    """
    from datetime import UTC, datetime

    # Enqueue a biometric_pass job with max_attempts=3 (now the default)
    job_id = enqueue(
        job_type="biometric_pass",
        args=["_biometric_worker", "42", "/audio/test.mp3", "[]"],
        slot="indexing",
        file_label="test.mp3",
        max_attempts=3,
    )

    # Simulate what startup_recovery sees: a running job with a dead PID (preempted mid-flight)
    with get_session() as session:
        j = session.get(UnifiedJob, job_id)
        j.status = "running"
        j.pid = 9999999  # dead PID
        j.attempt = 1
        session.commit()

    recovered = startup_recovery()
    assert recovered == 1, "startup_recovery must report 1 recovered job"

    with get_session() as session:
        j = session.get(UnifiedJob, job_id)
        assert j.status == "pending", (
            f"Expected pending after one preemption with max_attempts=3, got {j.status!r}"
        )
        assert j.attempt == 2, f"Expected attempt=2 after recovery, got {j.attempt}"


def test_preempted_align_job_survives_startup_recovery(test_db):
    """An align job preempted once must survive startup_recovery as pending (max_attempts=3)."""
    from audiobench.jobs.scheduler import enqueue_alignment

    job_id = enqueue_alignment(tx_id=123, audio_path="/audio/test.mp3")

    with get_session() as session:
        j = session.get(UnifiedJob, job_id)
        assert j.max_attempts == 3, (
            f"enqueue_alignment must set max_attempts=3, got {j.max_attempts}"
        )
        j.status = "running"
        j.pid = 9999999  # dead PID
        session.commit()

    startup_recovery()

    with get_session() as session:
        j = session.get(UnifiedJob, job_id)
        assert j.status == "pending", (
            f"Align job should be pending after one preemption, got {j.status!r}"
        )
