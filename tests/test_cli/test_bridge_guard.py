"""Tests 5–7: Bridge staleness guard verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from audiobench.cli.repl.backslash_commands import _check_bridge_staleness
from audiobench.cli.repl.session import ReplSession


class TestBridgeGuardStaleness:
    """Tests 5–7: _check_bridge_staleness behavior under fresh and stale citation states."""

    def test_stale_bridge_warning(self) -> None:
        """Test 5: citations[for_cite] != bridge_eid emits warning with \\bookmark cite {N} hint."""
        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 2002, 2: 3003}
        session.last_bookmark_bridges_for_cite = 1
        session.last_bookmark_bridges_for_eid = 1001

        bridge = MagicMock()
        bridge.target_expression_id = 1001  # Old cite 1 was 1001, now cite 1 is 2002 -> mismatch

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=1)

        mock_print.assert_called_once()
        msg = mock_print.call_args[0][0]
        assert "⚠ Bridge b1 is stale" in msg
        assert "\\bookmark cite 1" in msg

    def test_fresh_bridge_no_warning(self) -> None:
        """Test 6: citations[for_cite] == bridge_eid is fresh -> no warning emitted."""
        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 1001, 2: 2002}
        session.last_bookmark_bridges_for_cite = 1
        session.last_bookmark_bridges_for_eid = 1001

        bridge = MagicMock()
        bridge.target_expression_id = 1001  # Matches citations[1]

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=1)

        mock_print.assert_not_called()

    def test_stale_bridge_fallback_hint(self) -> None:
        """Test 7: for_cite=None, bridge not in citations -> stale with \\bookmark {eid} hint."""
        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {}  # Empty citations
        session.last_bookmark_bridges_for_cite = None
        session.last_bookmark_bridges_for_eid = 1001

        bridge = MagicMock()
        bridge.target_expression_id = 5050

        # When citations is non-empty but doesn't contain bridge_eid and for_cite is None
        session.last_synthesis_citations = {1: 1001, 2: 2002}
        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=2)

        mock_print.assert_called_once()
        msg = mock_print.call_args[0][0]
        assert "\\bookmark cite 2" in msg

    def test_stale_bridge_warning_on_focus(self) -> None:
        """Test cmd_focus('b1') triggers staleness check and emits warning when bridge is stale."""
        from audiobench.cli.repl.backslash_commands import cmd_focus

        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 9999}
        session.last_bookmark_bridges_for_cite = 1
        session.last_bookmark_bridges_for_eid = 1001

        bridge = MagicMock()
        bridge.target_expression_id = 1001  # Different from citations[1]=9999 -> stale
        session.last_bookmark_bridges = {1: bridge}

        with patch("audiobench.cli.repl.backslash_commands._focus_bridge") as mock_focus_bridge, \
             patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print, \
             patch("audiobench.core.db_engine.init_db"):
            cmd_focus(session, "b1")

        mock_focus_bridge.assert_called_once_with(session, bridge, 1)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "⚠ Bridge b1 is stale" in printed
        assert "\\bookmark cite 1" in printed

    def test_fresh_bridge_in_bookmark_sources_no_warning(self) -> None:
        """When bridge_eid is in last_bookmark_bridge_sources, no warning is emitted even if citations diverge."""
        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 1001, 2: 2002}
        session.last_bookmark_bridges_for_cite = 2
        session.last_bookmark_bridges_for_eid = 2002
        session.last_bookmark_bridge_sources = {110327, 3003}

        bridge = MagicMock()
        bridge.target_expression_id = 110327

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=3)

        mock_print.assert_not_called()

    def test_stale_bridge_after_synthesis_cleared_sources(self) -> None:
        """When a new synthesis clears last_bookmark_bridge_sources, the old bridge is detected as stale."""
        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 9001, 2: 9002}
        session.last_bookmark_bridges_for_cite = 2
        session.last_bookmark_bridges_for_eid = 2002
        session.last_bookmark_bridge_sources = set()  # Cleared by new synthesis

        bridge = MagicMock()
        bridge.target_expression_id = 110327

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=3)

        mock_print.assert_called_once()
        msg = mock_print.call_args[0][0]
        assert "⚠ Bridge b3 is stale" in msg
        assert "\\bookmark cite 2" in msg


class TestBookmarkBridgeNavigation:
    """Tests for \\bookmark b<n> argument resolution."""

    def test_bookmark_bridge_syntax_resolves_and_renders(self) -> None:
        """\\bookmark b3 resolves target_expression_id from session.last_bookmark_bridges."""
        from audiobench.cli.repl.backslash_commands import cmd_bookmark

        session = ReplSession(cli_group=MagicMock())
        mock_bridge = MagicMock()
        mock_bridge.target_expression_id = 110327
        session.last_bookmark_bridges = {3: mock_bridge}

        with patch("audiobench.memory.bookmark_service.generate_smart_bookmark_card") as mock_gen, \
             patch("audiobench.cli.repl.backslash_commands._render_smart_bookmark_card") as mock_render:
            mock_card = MagicMock()
            mock_card.expression_id = 110327
            mock_gen.return_value = mock_card

            cmd_bookmark(session, "b3")

            mock_gen.assert_called_once_with(110327)
            mock_render.assert_called_once_with(session, mock_card, cite_num=None)

    def test_bookmark_bridge_not_found(self) -> None:
        """\\bookmark b9 prints warning when bridge does not exist."""
        from audiobench.cli.repl.backslash_commands import cmd_bookmark

        session = ReplSession(cli_group=MagicMock())
        session.last_bookmark_bridges = {}

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            cmd_bookmark(session, "b9")

        mock_print.assert_called_once()
        assert "Semantic bridge [B9] not found" in mock_print.call_args[0][0]


