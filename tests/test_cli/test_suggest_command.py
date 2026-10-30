"""Tests for upgraded \\suggest command (Next Listen & Command History)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import click
import pytest

from audiobench.cli.repl.session import ReplSession
from audiobench.cli.repl.backslash_commands import cmd_suggest
from audiobench.core.focused_entity import FocusedEntity
from audiobench.storage.models import WorkRecord


@pytest.fixture
def repl_session():
    group = click.Group("test")
    session = ReplSession(group)
    session.focus = FocusedEntity(type="transcript", id=592, label="Ulysses Chapter 1")
    return session


def test_cmd_suggest_next_listen_with_consensus(repl_session):
    """When a work is focused, \\suggest displays Next Listen recommendations with clique consensus."""
    mock_db = MagicMock()

    # 1. Resolve work_id for transcript 592 -> work_id = 1 (Solomon)
    mock_curr_work = MagicMock(id=1, title="No Excuses", author="Robert C. Solomon")

    # 2. cand_works: Frankl (wid=2) and Dostoyevsky (wid=290)
    mock_cand_works = [
        (2, "Man's Search for Meaning", "Viktor E. Frankl", 33, 0.8548, 0.812),
        (290, "The Brothers Karamazov", "Fyodor Dostoyevsky", 15, 0.8446, 0.798),
    ]

    # 3. Neighbors of curr_wid (1): {2, 290}
    mock_conn_curr = [(2,), (290,)]

    # 4. For candidate 1 (Frankl): neighbors are {1, 290} -> mutual with Solomon is {290} (Dostoyevsky)!
    mock_conn_frankl = [(1, "Robert C. Solomon", "No Excuses"), (290, "Fyodor Dostoyevsky", "The Brothers Karamazov")]

    # Bridge for Frankl
    mock_bridge_frankl = ("Everything can be taken from a man but one thing: the last of human freedoms...", 42)

    # Playable tx for Frankl
    mock_tx_frankl = (102,)

    # 5. For candidate 2 (Dostoyevsky): neighbors are {1, 2} -> mutual with Solomon is {2} (Frankl)!
    mock_conn_dostoyevsky = [(1, "Robert C. Solomon", "No Excuses"), (2, "Viktor E. Frankl", "Man's Search for Meaning")]

    # Bridge for Dostoyevsky
    mock_bridge_dosto = ("The mystery of human existence lies not in just staying alive...", 55)

    # Playable tx for Dostoyevsky
    mock_tx_dosto = (290,)

    mock_db.execute.side_effect = [
        # curr_wid lookup:
        MagicMock(fetchone=lambda: (1,)),
        # cand_works query:
        MagicMock(fetchall=lambda: mock_cand_works),
        # conn_to_curr:
        MagicMock(fetchall=lambda: mock_conn_curr),
        # frankl neighbors:
        MagicMock(fetchall=lambda: mock_conn_frankl),
        # frankl bridge:
        MagicMock(fetchone=lambda: mock_bridge_frankl),
        # frankl tx:
        MagicMock(fetchone=lambda: mock_tx_frankl),
        # dostoyevsky neighbors:
        MagicMock(fetchall=lambda: mock_conn_dostoyevsky),
        # dostoyevsky bridge:
        MagicMock(fetchone=lambda: mock_bridge_dosto),
        # dostoyevsky tx:
        MagicMock(fetchone=lambda: mock_tx_dosto),
    ]

    def query_mock(model):
        m = MagicMock()
        if model is WorkRecord:
            m.filter_by.return_value.first.return_value = mock_curr_work
        return m

    mock_db.query.side_effect = query_mock

    class MockSessionCtx:
        def __enter__(self):
            return mock_db

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("audiobench.core.db_session.get_session", return_value=MockSessionCtx()), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_suggest(repl_session, "")

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Next Listen Recommendations" in printed
        assert "Viktor E. Frankl" in printed
        assert "Man's Search for Meaning" in printed
        assert "Consensus with Fyodor Dostoyevsky" in printed
        assert "85% match" in printed
        assert "\\play 102" in printed
        assert "Fyodor Dostoyevsky" in printed
        assert "The Brothers Karamazov" in printed
        assert "Consensus with Viktor E. Frankl" in printed
        assert "\\play 290" in printed


def test_cmd_suggest_explicit_commands_flag(repl_session):
    """When the user passes \\suggest commands, it shows command history regardless of focus."""
    mock_repo = MagicMock()
    mock_repo.get_recent_commands.return_value = [{"command": "related"}]
    mock_repo.get_next_command_suggestions.return_value = [
        {"command": "play", "count": 7},
        {"command": "synthesize", "count": 4},
    ]
    mock_repo.list_workflows.return_value = []

    with patch("audiobench.storage.command_graph_repository.get_command_graph_repo", return_value=mock_repo), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_suggest(repl_session, "commands")

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Last command:" in printed
        assert "related" in printed
        assert "\\play" in printed
        assert "7×" in printed
        assert "\\synthesize" in printed
        assert "4×" in printed


def test_cmd_suggest_fallback_when_no_focus():
    """When no focus is set, \\suggest falls back to command history."""
    group = click.Group("test")
    session = ReplSession(group)
    session.focus = None

    mock_repo = MagicMock()
    mock_repo.get_recent_commands.return_value = [{"command": "search"}]
    mock_repo.get_next_command_suggestions.return_value = [
        {"command": "focus", "count": 12},
    ]
    mock_repo.list_workflows.return_value = []

    with patch("audiobench.storage.command_graph_repository.get_command_graph_repo", return_value=mock_repo), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_suggest(session, "")

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Last command:" in printed
        assert "search" in printed
        assert "\\focus" in printed
        assert "12×" in printed
