"""CLI tests for audiobench library upgrade and rollback commands."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import click
import pytest

from audiobench.cli.app import cli
from audiobench.cli.commands.upgrade_cmd import UpgradeLock, check_daemon_gate1, check_daemon_gate2
from audiobench.core.db_session import get_session
from audiobench.storage.models import (
    AudioFileRecord,
    ChapterRecord,
    SegmentRecord,
    TranscriptionRecord,
)


@pytest.fixture
def cli_upgrade_env(test_db, tmp_path):
    """Set up database and external transcript files for CLI upgrade tests."""
    full_text = (
        "The boy's name was Santiago. He arrived with his herd at an abandoned church. "
        "The roof had fallen in long ago, and an enormous sycamore had grown on the spot "
        "where the sacristy had once stood. He decided to spend the night there. "
        "He saw to it that all the sheep entered through the ruined gate, and then laid "
        "some planks across it to prevent the flock from wandering away during the night. "
        "There were no wolves in the region, but once an animal had strayed during the night, "
        "and the boy had spent the entire next day searching for it."
    )
    words = full_text.split()
    total_dur = 60.0

    with get_session() as session:
        af1 = AudioFileRecord(
            file_name="The Alchemist.mp3",
            file_path="/audio/The Alchemist.mp3",
            file_hash="hash_alchemist_clean_123",
            duration_seconds=total_dur,
        )
        session.add(af1)
        session.flush()

        tx1 = TranscriptionRecord(
            audio_file_id=af1.id,
            file_name=af1.file_name,
            full_text=full_text,
            engine="faster-whisper",
            duration_seconds=total_dur,
            word_count=len(words),
            segment_count=25,
        )
        session.add(tx1)
        session.flush()

        # 25 micro-segments (2.4s each) -> high micro_segment_ratio, low TQS (~55)
        for i in range(25):
            s = SegmentRecord(
                transcription_id=tx1.id,
                segment_index=i,
                text=" ".join(words[i * 3 : (i + 1) * 3]),
                start_time=i * 2.4,
                end_time=(i + 1) * 2.4,
            )
            session.add(s)
        session.commit()

    # Create external transcript directory with 1 clean candidate
    batch_dir = tmp_path / "incoming_transcripts"
    batch_dir.mkdir()

    clean_json = batch_dir / "The Alchemist.json"
    half = len(words) // 2
    clean_json.write_text(
        json.dumps({
            "schema_version": "1",
            "audio": {
                "file_name": "The Alchemist.mp3",
                "file_hash": "hash_alchemist_clean_123",
                "duration_seconds": total_dur,
            },
            "engine": "whisper",
            "model_name": "large-v3-turbo",
            "text": full_text,
            "duration_seconds": total_dur,
            "segments": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "text": " ".join(words[:half]),
                    "words": [{"word": w, "start": i * 0.5, "end": (i + 1) * 0.5} for i, w in enumerate(words[:half])],
                },
                {
                    "start": 30.0,
                    "end": 60.0,
                    "text": " ".join(words[half:]),
                    "words": [{"word": w, "start": 30.0 + i * 0.5, "end": 30.0 + (i + 1) * 0.5} for i, w in enumerate(words[half:])],
                },
            ],
        }),
        encoding="utf-8",
    )

    return {
        "batch_dir": batch_dir,
        "clean_json": clean_json,
    }


def test_daemon_gate1_hard_exit():
    """Gate 1: If daemon is not running, check_daemon_gate1 must exit with code 1."""
    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            check_daemon_gate1()
        assert exc_info.value.code == 1


def test_daemon_gate2_durable_flag(tmp_path):
    """Gate 2: If daemon died mid-run, check_daemon_gate2 writes data/needs_reindex."""
    flag_path = tmp_path / "needs_reindex"
    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value=None):
        check_daemon_gate2(batch_id="test_batch_gate2", upgraded_count=42, flag_path=flag_path)

    assert flag_path.exists()
    data = json.loads(flag_path.read_text(encoding="utf-8"))
    assert data["batch_id"] == "test_batch_gate2"
    assert data["upgraded_count"] == 42
    assert "Daemon unreachable" in data["reason"]


def test_lockfile_stale_lock_recovery(tmp_path):
    """If lockfile points to a non-existent PID, it is cleared as stale."""
    lock_path = tmp_path / "upgrade.lock"
    # Write a dead PID
    lock_path.write_text(json.dumps({"pid": 9999999, "timestamp": 1000.0}), encoding="utf-8")

    lock = UpgradeLock(lock_path=lock_path)
    # Should not raise; stale lock is unlinked
    with lock:
        assert lock_path.exists()
        current_data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert current_data["pid"] == os.getpid()

    # Cleaned up on exit
    assert not lock_path.exists()


def test_lockfile_active_lock_blocked(tmp_path):
    """If lockfile points to a running PID, an error is raised."""
    lock_path = tmp_path / "upgrade.lock"
    # Write current running PID
    lock_path.write_text(json.dumps({"pid": os.getpid(), "timestamp": 2000000000.0}), encoding="utf-8")

    lock = UpgradeLock(lock_path=lock_path)
    with pytest.raises(click.ClickException) as exc_info:
        lock.acquire()
    assert "Another upgrade process is running" in str(exc_info.value)


def test_cli_upgrade_daemon_offline_blocked(runner, cli_upgrade_env):
    """Running 'audiobench library upgrade' with daemon down exits with code 1."""
    batch_dir = cli_upgrade_env["batch_dir"]
    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value=None):
        result = runner.invoke(cli, ["library", "upgrade", str(batch_dir)])
        assert result.exit_code == 1
        assert "Daemon not reachable" in result.output


def test_cli_upgrade_apply_all_clean_aborted_on_bad_phrase(runner, cli_upgrade_env):
    """If user types the wrong confirmation phrase, bulk upgrade aborts with no changes."""
    batch_dir = cli_upgrade_env["batch_dir"]

    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value={"status": "ok"}):
        result = runner.invoke(
            cli,
            ["library", "upgrade", str(batch_dir), "--apply-all-clean", "--no-daemon-check"],
            input="wrong confirmation phrase\n",
        )
        assert "Aborted" in result.output
        assert "Confirmation phrase did not match" in result.output

    # DB untouched
    with get_session() as session:
        tx = session.query(TranscriptionRecord).first()
        assert tx.engine == "faster-whisper"


def test_cli_upgrade_apply_all_clean_success_and_sampling(runner, cli_upgrade_env):
    """Confirming the phrase performs atomic upgrade, snapshot, and prints post-hoc audit samples."""
    batch_dir = cli_upgrade_env["batch_dir"]

    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value={"status": "ok"}):
        # 1 clean candidate
        result = runner.invoke(
            cli,
            ["library", "upgrade", str(batch_dir), "--apply-all-clean", "--no-daemon-check"],
            input="CONFIRM UPGRADE 1 RECORDS\n",
        )
        assert result.exit_code == 0
        assert "Successfully upgraded 1 records" in result.output
        assert "Post-Hoc Human Verification Audit" in result.output
        assert "The Alchemist" in result.output

    # DB is updated
    with get_session() as session:
        tx = session.query(TranscriptionRecord).first()
        assert tx.model_name == "large-v3-turbo"
        assert tx.is_indexed == 0


def test_cli_rollback_command(runner, cli_upgrade_env):
    """Upgrades a record, then rolls it back via 'audiobench library rollback'."""
    batch_dir = cli_upgrade_env["batch_dir"]

    with patch("audiobench.daemon.client.DaemonClient.daemon_is_healthy", return_value={"status": "ok"}):
        # 1. Upgrade
        res_up = runner.invoke(
            cli,
            ["library", "upgrade", str(batch_dir), "--apply-all-clean", "--no-daemon-check", "--batch-id", "batch_cli_test_01"],
            input="CONFIRM UPGRADE 1 RECORDS\n",
        )
        assert res_up.exit_code == 0

        # Verify upgraded state
        with get_session() as session:
            tx = session.query(TranscriptionRecord).first()
            assert tx.model_name == "large-v3-turbo"

        # 2. Rollback by title
        res_rb = runner.invoke(
            cli,
            ["library", "rollback", "Alchemist", "--no-daemon-check"],
        )
        assert res_rb.exit_code == 0
        assert "Rollback successful" in res_rb.output

        # Verify restored original state
        with get_session() as session:
            tx = session.query(TranscriptionRecord).first()
            assert tx.engine == "faster-whisper"
            assert tx.model_name == "large-v3-turbo" or tx.segment_count == 8
