"""Unit and integration tests for audiobench.memory.bookmark_service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from audiobench.memory.bookmark_service import (
    DialecticBridge,
    SmartBookmarkCard,
    _format_timestamp,
    generate_smart_bookmark_card,
    resolve_bookmark_expression,
)


class TestTimestampFormat:
    def test_format_timestamp_edge_cases(self) -> None:
        assert _format_timestamp(None) == "00:00"
        assert _format_timestamp(-5.0) == "00:00"
        assert _format_timestamp(0) == "00:00"
        assert _format_timestamp(45.2) == "00:45"
        assert _format_timestamp(125.0) == "02:05"
        assert _format_timestamp(3665.0) == "1:01:05"


class TestResolveBookmarkExpression:
    def test_direct_expression_id_returns_immediately(self) -> None:
        assert resolve_bookmark_expression(expression_id=12345) == 12345

    def test_none_inputs_return_none(self) -> None:
        assert resolve_bookmark_expression() is None
        assert resolve_bookmark_expression(audio_file_id=1, timestamp_seconds=None) is None


class TestGenerateSmartBookmarkCardMocked:
    def test_generates_two_tiered_card(self) -> None:
        # Mock DB queries:
        # 1. Main expression
        main_expr = (101, 10, "Man must find meaning even in intense suffering.", "Man's Search for Meaning", "Viktor Frankl", 1, 300.0, 310.0)
        # 2. Prior premise
        prior_premise = ("We must examine our underlying presuppositions regarding fate and conditionality.",)
        # 3. Following resolution
        later_resolution = ("Ultimately, meaning is not given by outside circumstance, but chosen internally.",)
        # 4. Dialectic bridges
        bridges_raw = [
            (201, 20, "The Brothers Karamazov", "Fyodor Dostoyevsky", 0.825, "Ivan argues passionately against the unmerited suffering of the innocent in this world."),
            (301, 30, "The Way of Zen", "Alan Watts", 0.795, "Suffering arises when we cling to an illusion of separate control and ego identity."),
        ]
        # 5. Triadic inter-edges: Dostoyevsky connects to Alan Watts!
        inter_edges = (201, 301, "Fyodor Dostoyevsky", "Alan Watts")

        with patch("audiobench.memory.bookmark_service.get_session") as mock_session_ctx:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session

            mock_session.execute.side_effect = [
                MagicMock(fetchone=lambda: main_expr),
                MagicMock(fetchone=lambda: prior_premise),
                MagicMock(fetchone=lambda: later_resolution),
                MagicMock(fetchall=lambda: bridges_raw),
                MagicMock(fetchone=lambda: inter_edges),
            ]

            card = generate_smart_bookmark_card(101)

            assert isinstance(card, SmartBookmarkCard)
            assert card.expression_id == 101
            assert card.work_title == "Man's Search for Meaning"
            assert card.author == "Viktor Frankl"
            assert card.timestamp_display == "05:00"
            assert "meaning even in intense suffering" in card.quote

            # Tier 1 checks
            assert card.preceding_context is not None
            assert "presuppositions" in card.preceding_context
            assert card.following_resolution is not None
            assert "chosen internally" in card.following_resolution

            # Tier 2 checks
            assert len(card.dialectic_bridges) == 2
            assert card.dialectic_bridges[0].author == "Fyodor Dostoyevsky"
            assert card.dialectic_bridges[0].cosine_similarity == 0.825
            assert card.dialectic_bridges[0].is_clique_partner is True
            assert card.dialectic_bridges[1].author == "Alan Watts"
            assert card.dialectic_bridges[1].is_clique_partner is True

            # Triadic clique consensus triangle
            assert card.consensus_triangle == ["Viktor Frankl", "Fyodor Dostoyevsky", "Alan Watts"]

            # Rendered markdown verification
            assert "Tier 1: Local Narrative Arc" in card.rendered_markdown
            assert "Tier 2: Cross-Library Semantic Bridges" in card.rendered_markdown
            assert "Consensus Triangle Badge" in card.rendered_markdown
            assert "Fyodor Dostoyevsky" in card.rendered_markdown
            assert "Alan Watts" in card.rendered_markdown


class TestBookmarkCommand:
    def test_cmd_bookmark_direct_and_cite(self) -> None:
        from audiobench.cli.repl.backslash_commands import cmd_bookmark
        from audiobench.cli.repl.session import ReplSession

        session = ReplSession(cli_group=MagicMock())
        session.last_synthesis_citations = {1: 101, 2: 202}

        mock_card = SmartBookmarkCard(
            expression_id=101,
            work_id=1,
            work_title="Test Work",
            author="Test Author",
            timestamp_display="01:23",
            quote="A meaningful quote",
            rendered_markdown="# Mock Bookmark Card",
        )

        with patch("audiobench.memory.bookmark_service.generate_smart_bookmark_card") as mock_gen, \
             patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            mock_gen.return_value = mock_card

            # Test direct ID
            cmd_bookmark(session, "101")
            mock_gen.assert_called_with(101)

            # Test cite lookup
            cmd_bookmark(session, "cite 1")
            mock_gen.assert_called_with(101)

            # Test invalid cite
            cmd_bookmark(session, "cite 99")
            # Should not call generate for non-existent citation 99
            assert mock_gen.call_count == 2

