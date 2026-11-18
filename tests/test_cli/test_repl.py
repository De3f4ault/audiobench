"""Tests for REPL session state and dispatch."""

from __future__ import annotations

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
        assert session.focus is None
        assert session._command_count == 0

    def test_prompt_no_context(self):
        session = self._make_session()
        prompt = session.prompt
        # prompt may be a prompt_toolkit HTML object, which has a .value string
        prompt_text = getattr(prompt, "value", str(prompt))
        assert "❯" in prompt_text
        assert "#" not in prompt_text

    def test_expand_vars_no_context(self):
        session = self._make_session()
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "$last"]  # No expansion without context

    def test_expand_vars_with_context(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=42, label="Transcript #42")
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "42"]

    def test_auto_inject_id_context_aware(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["show"]
        result = session.auto_inject_id(args)
        assert result == ["show", "10"]

    def test_auto_inject_id_skips_explicit(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["show", "99"]
        result = session.auto_inject_id(args)
        assert result == ["show", "99"]  # User-specified ID preserved

    def test_auto_inject_id_skips_non_aware(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["history"]
        result = session.auto_inject_id(args)
        assert result == ["history"]  # history is not context-aware

    def test_clear_context(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=42, label="Transcript #42")
        session.focus = None
        assert session.last_id is None
        assert session.focus is None


class TestSlashCommands:
    """Test slash command routing."""

    def test_exit_returns_true(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/exit", session) is True

    def test_quit_returns_true(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/quit", session) is True

    def test_unknown_returns_false(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/nonexistent", session) is False


class TestDotCommands:
    """Test dot-command registry."""

    def test_dot_commands_registry_not_empty(self):
        from audiobench.cli.repl.dot_commands import _DOT_HANDLERS

        assert len(_DOT_HANDLERS) >= 1

    def test_all_dot_commands_lowercase(self):
        from audiobench.cli.repl.dot_commands import _DOT_HANDLERS

        for cmd in _DOT_HANDLERS.keys():
            assert cmd == cmd.lower(), f"{cmd} must be registered lowercase"


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


class TestPrefixNormalization:
    """Test transparent / vs \\ command routing and \\context handler."""

    def _make_session(self):
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        return ReplSession(group)

    def test_slash_focus_routes_without_warning(self):
        from audiobench.cli.display.theme import console
        from audiobench.cli.repl import _dispatch_single

        session = self._make_session()
        # Dispatch /focus (without arguments)
        with console.capture() as cap:
            should_exit = _dispatch_single(session, "/focus")
        assert should_exit is False
        out = cap.get()
        # Must execute cmd_focus directly, never reprimand the user
        assert "Did you mean \\focus? (backslash, not slash)" not in out
        assert "No focus set" in out

    def test_backslash_context_handler_registered(self):
        from audiobench.cli.display.theme import console
        from audiobench.cli.repl import _dispatch_single
        from audiobench.core.focused_entity import FocusedEntity

        session = self._make_session()
        # When unfocused
        with console.capture() as cap:
            _dispatch_single(session, "\\context")
        out1 = cap.get()
        assert "No context set" in out1
        assert "Did you mean: \\next?" not in out1

        # When focused
        session.focus = FocusedEntity(type="transcript", id=42, label="Transcript #42")
        with console.capture() as cap:
            _dispatch_single(session, "\\context")
        out2 = cap.get()
        assert "#42" in out2
        assert "transcript" in out2.lower()

    def test_slash_context_transparent_routing(self):
        from audiobench.cli.display.theme import console
        from audiobench.cli.repl import _dispatch_single
        from audiobench.core.focused_entity import FocusedEntity

        session = self._make_session()
        session.focus = FocusedEntity(type="transcript", id=101, label="Transcript #101")
        with console.capture() as cap:
            _dispatch_single(session, "/context")
        out = cap.get()
        assert "#101" in out
        assert "transcript" in out.lower()
        assert "Did you mean" not in out


class TestActionMenuBracketEscaping:
    """Test that Rich does not swallow single-letter menu hotkeys [v], [g], [f]."""

    def test_action_menu_options_display_brackets(self):
        import io
        from rich.console import Console
        from audiobench.cli.display.theme import ACCENT

        out_io = io.StringIO()
        test_console = Console(file=out_io, force_terminal=False)

        options = [
            ("v", "View full content"),
            ("g", "Show graph context"),
            ("f", "Focus source file"),
            ("n", "Capture to active/context note"),
            ("a", "Use as context for .ask synthesis"),
            ("s", "Show surrounding expressions"),
            ("q", "Back to results"),
        ]
        for key, desc in options:
            test_console.print(f"    [{ACCENT}]\\[{key}][/] {desc}")

        rendered = out_io.getvalue()
        for key, _ in options:
            assert f"[{key}]" in rendered, f"Expected visible hotkey [{key}] in rendered output"
