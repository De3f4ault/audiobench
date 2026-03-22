"""Tests for CLI commands — smoke + basic functionality."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from audiobench.cli.app import cli


@pytest.fixture
def runner():
    return CliRunner()


# ── Smoke tests (every command's --help works) ──

COMMANDS = [
    "transcribe",
    "subtitle",
    "listen",
    "speak",
    "download-voice",
    "summarize",
    "ask",
    "chat",
    "history",
    "search",
    "show",
    "export",
    "delete",
    "preset",
    "install-completion",
    "download",
    "info",
    "doctor",
    "status",
    "cleanup",
    "vocab",
    "repl",
    "analyze",
    "convert",
    "merge",
    "inspect",
]


class TestCommandRegistration:
    """Verify all commands are registered and respond to --help."""

    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AudioBench" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("command", COMMANDS)
    def test_command_help(self, runner, command):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0, f"{command} --help failed: {result.output}"

    def test_all_expected_commands_registered(self, runner):
        result = runner.invoke(cli, ["--help"])
        for cmd in COMMANDS:
            assert cmd in result.output, f"Command '{cmd}' missing from --help"


class TestInfoCommand:
    """Test the info command output."""

    def test_info_shows_settings(self, runner):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Model" in result.output
        assert "Device" in result.output


class TestGlobalFlags:
    """Test global CLI flags."""

    def test_json_flag_accepted(self, runner):
        result = runner.invoke(cli, ["--json", "--help"])
        assert result.exit_code == 0

    def test_verbose_flag_accepted(self, runner):
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_debug_flag_accepted(self, runner):
        result = runner.invoke(cli, ["--debug", "--help"])
        assert result.exit_code == 0
