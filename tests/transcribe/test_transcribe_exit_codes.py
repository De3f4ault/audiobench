"""Tests for Phase 3 bug fix: unhandled exceptions during transcription
must cause a non-zero exit code so worker_daemon marks the job as failed.

Background (TX #1504 audit):
  - Gemini hit RecursionError inside json_repair.repair_json().
  - The exception propagated to cli/commands/transcribe.py:1341 (except Exception).
  - When job_id is set, the handler calls sys.exit(1).
  - worker_daemon polls returncode and records the job as failed on != 0.

These tests verify the contract: any unhandled exception in pipeline.transcribe_file()
causes sys.exit(1) when running under worker_daemon (job_id is set).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from audiobench.cli.commands.transcribe import transcribe


@pytest.fixture()
def runner():
    return CliRunner()


def test_engine_exception_with_job_id_exits_nonzero(runner, tmp_path):
    """When TranscriptionPipeline.transcribe_file() raises and --job-id is set,
    the CLI must exit with code 1 so worker_daemon records the job as failed.

    Exercises the exact failure mode from TX #1504: RecursionError propagated
    through the Gemini engine and reached the outer except Exception handler.
    """
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)  # minimal MP3 header

    # Patch at the class method so any call to transcribe_file raises
    with patch(
        "audiobench.transcribe.transcriber.TranscriptionPipeline.transcribe_file",
        side_effect=RecursionError("Input nesting exceeds the supported parser recursion depth"),
    ):
        result = runner.invoke(
            transcribe,
            [str(audio), "--job-id", "999"],
            catch_exceptions=True,
        )

    # The command must have exited with code 1 (sys.exit(1) from the exception handler)
    assert result.exit_code == 1, (
        f"Expected exit code 1 (so worker_daemon marks job failed), "
        f"got {result.exit_code}. Output: {result.output!r}"
    )
