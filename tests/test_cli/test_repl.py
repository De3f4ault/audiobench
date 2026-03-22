"""Tests for REPL session state and dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import click


class TestReplSession:
    """Test ReplSession state management."""

    def _make_session(self):
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        return ReplSession(group)

    def test_initial_state(self):
        session = self._make_session()
        assert session.last_id is None
        assert session.context_record is None
        assert session._command_count == 0

    def test_prompt_no_context(self):
        session = self._make_session()
        prompt = session.prompt
        assert "ab" in prompt
        assert "#" not in prompt

    def test_expand_vars_no_context(self):
        session = self._make_session()
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "$last"]  # No expansion without context

    def test_expand_vars_with_context(self):
        session = self._make_session()
        session.last_id = 42
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "42"]

    def test_auto_inject_id_context_aware(self):
        session = self._make_session()
        session.last_id = 10
        args = ["show"]
        result = session.auto_inject_id(args)
        assert result == ["show", "10"]

    def test_auto_inject_id_skips_explicit(self):
        session = self._make_session()
        session.last_id = 10
        args = ["show", "99"]
        result = session.auto_inject_id(args)
        assert result == ["show", "99"]  # User-specified ID preserved

    def test_auto_inject_id_skips_non_aware(self):
        session = self._make_session()
        session.last_id = 10
        args = ["history"]
        result = session.auto_inject_id(args)
        assert result == ["history"]  # history is not context-aware

    def test_clear_context(self):
        session = self._make_session()
        session.last_id = 42
        session.context_record = {"id": 42}
        session.clear_context()
        assert session.last_id is None
        assert session.context_record is None


class TestSlashCommands:
    """Test slash command routing."""

    def test_exit_returns_true(self):
        from audiobench.cli.repl.slash_commands import handle_slash_command
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/exit", session) is True

    def test_quit_returns_true(self):
        from audiobench.cli.repl.slash_commands import handle_slash_command
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/quit", session) is True

    def test_unknown_returns_false(self):
        from audiobench.cli.repl.slash_commands import handle_slash_command
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/nonexistent", session) is False


class TestDotCommands:
    """Test dot-command registry and typo correction."""

    def test_dot_commands_registry_not_empty(self):
        from audiobench.cli.repl.dot_commands import DOT_COMMANDS

        assert len(DOT_COMMANDS) > 15

    def test_suggest_close_match(self):
        from audiobench.cli.repl.dot_commands import _suggest_dot_command

        result = _suggest_dot_command(".stat")
        assert result == ".stats"

    def test_suggest_no_match(self):
        from audiobench.cli.repl.dot_commands import _suggest_dot_command

        result = _suggest_dot_command(".zzzzz")
        assert result is None

    def test_all_dot_commands_start_with_dot(self):
        from audiobench.cli.repl.dot_commands import DOT_COMMANDS

        for cmd in DOT_COMMANDS:
            assert cmd.startswith("."), f"{cmd} doesn't start with dot"


class TestAliases:
    """Test bare-word alias mapping."""

    def test_help_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["help"] == "/help"

    def test_exit_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["exit"] == "/exit"

    def test_quit_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["quit"] == "/exit"
