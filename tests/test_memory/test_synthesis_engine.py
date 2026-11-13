"""Unit and integration tests for synthesis_retriever and synthesis_engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from audiobench.chat.synthesis_engine import (
    SynthesisResponse,
    format_synthesis_prompt,
    generate_synthesis,
)
from audiobench.memory.query_engine import Ok, Err
from audiobench.memory.synthesis_retriever import (
    SynthesisFragment,
    retrieve_synthesis_context,
)
from conftest import empty_meta


class TestFormatSynthesisPrompt:
    def test_format_prompt_contains_citations(self) -> None:
        frags = [
            SynthesisFragment(
                expression_id=1,
                work_id=10,
                work_title="Book A",
                author="Author A",
                content="Passage from author A on guilt.",
                q_sim=0.85,
                edge_weight=1.0,
                effective_rel=0.85,
                mmr_score=0.85,
                source_mode="ANN",
            ),
            SynthesisFragment(
                expression_id=2,
                work_id=20,
                work_title="Book B",
                author="Author B",
                content="Passage from author B on conscience.",
                q_sim=0.75,
                edge_weight=0.82,
                effective_rel=0.78,
                mmr_score=0.76,
                source_mode="GRAPH_BRIDGE",
                is_clique_partner=True,
            ),
        ]
        prompt = format_synthesis_prompt("How does guilt affect mind?", frags)
        assert "Source [1]: Author A — Book A" in prompt
        assert "Source [2]: Author B — Book B (Consensus Triad Partner)" in prompt
        assert "Passage from author A on guilt." in prompt
        assert "[1], [2], [3]" in prompt


class TestRetrieveSynthesisContextMocked:
    def test_retrieval_and_mmr_selection(self) -> None:
        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(101, 0.15), (102, 0.20)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0]], dtype=np.float32)

        sample_ann_rows = [
            (101, 1, "A very deep contemplation on the psychological roots of suffering.", "Work 1", "Author 1", "audio_transcript", None, None),
            (102, 2, "A philosophical analysis of how suffering can be embraced constructively.", "Work 2", "Author 2", "audio_transcript", None, None),
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            # 1. Fetch ANN metadata
            # 2. Fetch graph neighbors (empty in this test)
            # 3. Fetch 3-cliques (empty)
            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, _meta = retrieve_synthesis_context("suffering and psychology", top_k=2, index=mock_index)

            assert len(results) == 2
            assert results[0].expression_id == 101
            assert results[0].author == "Author 1"
            assert results[1].expression_id == 102
            assert results[1].author == "Author 2"
            assert results[0].source_mode == "ANN"


class TestGenerateSynthesisMocked:
    def test_generate_synthesis_success(self) -> None:
        frags = [
            SynthesisFragment(
                expression_id=501,
                work_id=1,
                work_title="Title A",
                author="Author A",
                content="Passage A about resilience.",
                q_sim=0.88,
                edge_weight=1.0,
                effective_rel=0.88,
                mmr_score=0.88,
                source_mode="ANN",
            )
        ]

        with patch("audiobench.chat.synthesis_engine.retrieve_synthesis_context") as mock_ret, \
             patch("audiobench.chat.synthesis_engine._call_llm") as mock_llm:
            mock_ret.return_value = (frags, empty_meta())
            mock_llm.return_value = Ok("Author A argues that resilience is built internally [1].", model="gemma4:31b-cloud")

            resp = generate_synthesis("resilience")
            assert isinstance(resp, SynthesisResponse)
            assert resp.citations == {1: 501}
            assert "[1]" in resp.synthesis_text
            assert resp.model_used == "gemma4:31b-cloud"

    def test_generate_synthesis_fallback_digest(self) -> None:
        frags = [
            SynthesisFragment(
                expression_id=502,
                work_id=2,
                work_title="Title B",
                author="Author B",
                content="Passage B on existential choice.",
                q_sim=0.82,
                edge_weight=1.0,
                effective_rel=0.82,
                mmr_score=0.82,
                source_mode="ANN",
            )
        ]

        with patch("audiobench.chat.synthesis_engine.retrieve_synthesis_context") as mock_ret, \
             patch("audiobench.chat.synthesis_engine._call_llm") as mock_llm:
            mock_ret.return_value = (frags, empty_meta())
            mock_llm.return_value = Err("Ollama offline")

            resp = generate_synthesis("choice")
            assert resp.model_used == "digest_fallback"
            assert "Dialectic Digest: choice" in resp.synthesis_text
            assert resp.citations == {1: 502}


class TestSynthesizeCommand:
    def test_cmd_synthesize_populates_session_citations(self) -> None:
        from audiobench.cli.repl.backslash_commands import cmd_synthesize
        from audiobench.cli.repl.session import ReplSession

        session = ReplSession(cli_group=MagicMock())

        mock_resp = SynthesisResponse(
            query="guilt and breakdown",
            synthesis_text="Dostoyevsky [1] and Munger [2] examine self-deception.",
            citations={1: 101, 2: 202},
            fragments=[
                SynthesisFragment(
                    expression_id=101,
                    work_id=1,
                    work_title="Crime and Punishment",
                    author="Fyodor Dostoyevsky",
                    content="Raskolnikov's confession...",
                    q_sim=0.85,
                    edge_weight=1.0,
                    effective_rel=0.85,
                    mmr_score=0.85,
                    source_mode="ANN",
                ),
                SynthesisFragment(
                    expression_id=202,
                    work_id=2,
                    work_title="Poor Charlie's Almanack",
                    author="Charles Munger",
                    content="Psychology of human misjudgment...",
                    q_sim=0.80,
                    edge_weight=0.82,
                    effective_rel=0.81,
                    mmr_score=0.79,
                    source_mode="GRAPH_BRIDGE",
                ),
            ],
            elapsed_seconds=0.45,
            model_used="ollama",
        )

        with patch("audiobench.chat.synthesis_engine.generate_synthesis") as mock_gen, \
             patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            mock_gen.return_value = mock_resp

            cmd_synthesize(session, "guilt and breakdown")

            # Check that citations are stored on session for \bookmark cite
            assert session.last_synthesis_citations == {1: 101, 2: 202}


class TestEmbedQueryDaemonDelegation:
    def test_embed_query_delegates_to_daemon(self) -> None:
        from audiobench.memory.synthesis_retriever import embed_query

        mock_daemon = MagicMock()
        mock_daemon.embed_query.return_value = [3.0, 4.0]

        with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_daemon) as mock_get_client, \
             patch("audiobench.memory.singletons.get_primary_embedder") as mock_primary:
            vec = embed_query("test query")

            mock_get_client.assert_called_once()
            mock_daemon.embed_query.assert_called_once_with("test query")
            mock_primary.assert_not_called()
            np.testing.assert_allclose(vec, [0.6, 0.8], atol=1e-5)

# ─── Tests 1–4: Stale Bridge Guard Session Fields ───────────────────────────
# Tests 5–7 for the REPL staleness warning live in
# tests/test_cli/test_bridge_guard.py which exercises _check_bridge_staleness directly.

class TestStaleBridgeGuardSessionFields:
    """Tests 1–4: Session stamping fields are set and cleared correctly."""

    def test_last_bookmark_bridges_for_eid_set_on_render(self) -> None:
        """Test 1: _render_smart_bookmark_card stamps for_eid on the session."""
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.backslash_commands import _render_smart_bookmark_card

        session = ReplSession(cli_group=MagicMock())
        assert session.last_bookmark_bridges_for_eid is None

        card = MagicMock()
        card.expression_id = 999
        card.work_title = "Test Work"
        card.author = "Test Author"
        card.timestamp_display = "2026-01-01"
        card.quote = "Some excerpt text"
        card.community_id = None
        card.preceding_context = None
        card.following_resolution = None
        card.bridges = []
        card.consensus_triangle = None

        with patch("audiobench.cli.repl.backslash_commands.console.print"):
            _render_smart_bookmark_card(session, card, cite_num=3)

        assert session.last_bookmark_bridges_for_eid == 999
        assert session.last_bookmark_bridges_for_cite == 3

    def test_for_eid_defaults_to_none_before_render(self) -> None:
        """Test 2: Fields are None on a fresh session — no stale accidental reads."""
        from audiobench.cli.repl.session import ReplSession
        session = ReplSession(cli_group=MagicMock())
        assert session.last_bookmark_bridges_for_eid is None
        assert session.last_bookmark_bridges_for_cite is None

    def test_for_eid_overwrites_on_second_render(self) -> None:
        """Test 3: Rendering a second bookmark card replaces the eid/cite stamp."""
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.backslash_commands import _render_smart_bookmark_card

        session = ReplSession(cli_group=MagicMock())

        def make_card(eid: int) -> MagicMock:
            card = MagicMock()
            card.expression_id = eid
            card.work_title = "W"
            card.author = "A"
            card.timestamp_display = "2026-01-01"
            card.quote = "text"
            card.community_id = None
            card.preceding_context = None
            card.following_resolution = None
            card.bridges = []
            card.consensus_triangle = None
            return card

        with patch("audiobench.cli.repl.backslash_commands.console.print"):
            _render_smart_bookmark_card(session, make_card(100), cite_num=1)
            _render_smart_bookmark_card(session, make_card(200), cite_num=2)

        assert session.last_bookmark_bridges_for_eid == 200
        assert session.last_bookmark_bridges_for_cite == 2

    def test_check_bridge_staleness_emits_warning_when_stale(self) -> None:
        """Test 4: _check_bridge_staleness emits Rich warning when cite points to different eid."""
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.backslash_commands import _check_bridge_staleness

        session = ReplSession(cli_group=MagicMock())
        # citation 1 → eid 9999; bridge was built from eid 1001 — mismatch → stale
        session.last_synthesis_citations = {1: 9999}
        session.last_bookmark_bridges_for_cite = 1
        session.last_bookmark_bridges_for_eid = 1001

        bridge = MagicMock()
        bridge.target_expression_id = 42

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            _check_bridge_staleness(session, bridge, b_idx=1)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "stale" in printed.lower() or "\u26a0" in printed


# ─── Tests 1–4: Per-Author MMR Cap & Sub-Caps ───────────────────────────────

class TestAuthorCapEnforced:
    """Tests 1–4: Cap enforcement on named authors, pseudo-buckets, and reduced fragment count."""

    def test_author_cap_enforced(self) -> None:
        """Test 1: At most DEFAULT_MAX_PER_AUTHOR (6) fragments from a single named author."""
        from audiobench.memory.synthesis_retriever import (
            DEFAULT_MAX_PER_AUTHOR,
            retrieve_synthesis_context,
        )

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(100 + i, 0.05 * i) for i in range(1, 11)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                100 + i,
                1,
                f"Passage number {i} discussing black swans, antifragility, and extreme events in markets.",
                "The Black Swan",
                "Nassim Taleb",
                "audio_transcript",
                None,
                None,
            )
            for i in range(1, 11)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context("black swan", top_k=10, index=mock_index)

            assert len(results) == DEFAULT_MAX_PER_AUTHOR
            assert all(f.author == "Nassim Taleb" for f in results)
            assert meta.author_counts_pool["Nassim Taleb"] == 10
            assert meta.author_counts_selected["Nassim Taleb"] == DEFAULT_MAX_PER_AUTHOR
            assert meta.capped_authors == {"Nassim Taleb"}

    def test_prior_bucket_sub_cap_enforced(self) -> None:
        """Test 2: At most DEFAULT_PRIOR_CAP (3) fragments from PRIOR_BUCKET."""
        from audiobench.memory.synthesis_retriever import (
            DEFAULT_PRIOR_CAP,
            retrieve_synthesis_context,
        )

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(200 + i, 0.05 * i) for i in range(1, 6)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                200 + i,
                None,
                f"Prior synthesis reflection number {i} examining ancient stoicism and resilience in adversity.",
                None,
                None,
                "search_synthesis",
                f"Query {i}",
                None,
            )
            for i in range(1, 6)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context("stoicism", top_k=5, index=mock_index)

            assert len(results) == DEFAULT_PRIOR_CAP
            assert all(f.author == "[Prior Synthesis]" for f in results)
            assert meta.prior_count == DEFAULT_PRIOR_CAP

    def test_unknown_bucket_sub_cap_enforced(self) -> None:
        """Test 3: At most DEFAULT_UNKNOWN_CAP (5) fragments from UNKNOWN_BUCKET."""
        from audiobench.memory.synthesis_retriever import (
            DEFAULT_UNKNOWN_CAP,
            retrieve_synthesis_context,
        )

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(300 + i, 0.05 * i) for i in range(1, 9)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                300 + i,
                i,
                f"Unattributed lecture fragment number {i} on moral sentiment and historical ethics in society.",
                "Anonymous Lecture",
                None,
                "audio_transcript",
                None,
                None,
            )
            for i in range(1, 9)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context("moral sentiment", top_k=8, index=mock_index)

            assert len(results) == DEFAULT_UNKNOWN_CAP
            assert all(f.author == "Unknown Author" for f in results)
            assert meta.unknown_count == DEFAULT_UNKNOWN_CAP

    def test_synthesis_with_reduced_fragment_count(self) -> None:
        """Test 4: Reduced fragment count (10 frags on top_k=20) succeeds without error."""
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        frags = [
            SynthesisFragment(
                expression_id=500 + i,
                work_id=i,
                work_title=f"Work {i}",
                author=f"Author {i}",
                content=f"Passage content for fragment {i} discussing antifragility.",
                q_sim=0.85,
                edge_weight=1.0,
                effective_rel=0.85,
                mmr_score=0.85,
                source_mode="ANN",
            )
            for i in range(1, 11)
        ]
        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Taleb": 10},
            author_counts_selected={"Taleb": 6},
            capped_authors={"Taleb"},
            prior_count=0,
            unknown_count=0,
        )

        with patch("audiobench.chat.synthesis_engine.retrieve_synthesis_context", return_value=(frags, meta)), \
             patch("audiobench.chat.synthesis_engine._call_llm") as mock_llm:
            mock_llm.return_value = Ok("Synthesized text spanning multiple sources [1] [2].", model="gemma4:31b-cloud")

            resp = generate_synthesis("antifragile", top_k=20)

            assert len(resp.fragments) == 10
            assert len(resp.citations) == 10
            assert resp.citations[1] == 501
            assert resp.citations[10] == 510
            assert resp.retrieval_meta is not None
            assert resp.retrieval_meta.capped_authors == {"Taleb"}


# ─── Tests 8–13: Per-Author MMR Cap, Footer & Fallback ─────────────────────

class TestRetrievalMetaStructure:
    """Tests 8: RetrievalMeta is well-formed and returned from retrieve_synthesis_context."""

    def test_retrieval_meta_fields_present(self) -> None:
        """Test 8a: RetrievalMeta has all required fields with correct types."""
        from audiobench.memory.synthesis_retriever import RetrievalMeta
        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Taleb": 8, "Munger": 3},
            author_counts_selected={"Taleb": 6, "Munger": 3},
            capped_authors={"Taleb"},
            prior_count=0,
            unknown_count=1,
        )
        assert meta.pool_size == 10
        assert meta.k_ann == 40
        assert meta.capped_authors == {"Taleb"}
        assert meta.prior_count == 0
        assert meta.unknown_count == 1

    def test_retrieve_synthesis_context_returns_tuple(self) -> None:
        """Test 8b: Tripwire - retrieve_synthesis_context returns (list[SynthesisFragment], RetrievalMeta)."""
        from audiobench.memory.synthesis_retriever import RetrievalMeta, retrieve_synthesis_context

        mock_index = MagicMock()
        mock_index.query.return_value = []  # no ANN hits → empty pool path

        with patch("audiobench.memory.synthesis_retriever.ensure_hnsw_ready", return_value=mock_index), \
             patch("audiobench.memory.synthesis_retriever.embed_query", return_value=np.zeros(128, dtype=np.float32)):
            result = retrieve_synthesis_context("any query", top_k=4, index=mock_index)

        assert isinstance(result, tuple), "Must return a tuple"
        frags, meta = result
        assert isinstance(frags, list)
        assert isinstance(meta, RetrievalMeta)


class TestFooterDiagnostics:
    """Tests 9–13: Footer rendering diagnostics, cap notification, and fallback signals."""

    def test_footer_cap_fires_when_pool_exceeds_cap(self) -> None:
        """Test 9: Pool exceeds cap (10 pool, 6 selected) -> cap notice is rendered."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 10},
            author_counts_selected={"Nassim Taleb": 6},
            capped_authors={"Nassim Taleb"},
            prior_count=0,
            unknown_count=0,
        )
        resp = SynthesisResponse(
            query="black swan",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(6)],
            elapsed_seconds=1.2,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "6 sources (top_k=20)" in printed
        assert "Per-author cap fired" in printed
        assert "Nassim Taleb (10 in pool, capped at 6)" in printed
        # 2026-11-13: --focused is now implemented. Hint was removed when the
        # flag didn't exist; now it's restored because the flag works.
        assert "\\synthesize --focused for depth" in printed

    def test_footer_focused_mode_renders_diagnostics(self) -> None:
        """Focused mode renders pool/selected metrics, cap bypassed, and k_ann limit."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 10},
            author_counts_selected={"Nassim Taleb": 10},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
            focused=True,
            focused_author="Nassim Taleb",
            author_match_status="resolved",
        )
        resp = SynthesisResponse(
            query="black swan",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(10)],
            elapsed_seconds=1.1,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Focused mode" in printed
        assert "Nassim Taleb (10/10 selected, cap bypassed; k_ann=40)" in printed
        assert "capped at 6" not in printed

    def test_footer_general_focused_mode_with_cap_bypassed(self) -> None:
        """General focused mode surfaces authors that genuinely exceeded the cap."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=20,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 15, "David Grim": 5},
            author_counts_selected={"Nassim Taleb": 10, "David Grim": 3},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
            focused=True,
            focused_author=None,
            author_match_status="resolved",
        )
        resp = SynthesisResponse(
            query="complexity",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(13)],
            elapsed_seconds=1.1,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Focused mode" in printed
        assert "Nassim Taleb (10/15 selected, cap bypassed)" in printed
        # David Grim had 3 selected (<= 6) so Grim is not reported as cap bypassed
        assert "David Grim" not in printed
        assert "k_ann=40" in printed

    def test_footer_general_focused_mode_below_cap(self) -> None:
        """When general focused mode selects <= 6 for all authors, 'cap bypassed' is not falsely claimed."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=17,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 8, "David Grim": 9},
            author_counts_selected={"Nassim Taleb": 2, "David Grim": 1},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
            focused=True,
            focused_author=None,
            author_match_status="resolved",
        )
        resp = SynthesisResponse(
            query="fragility",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(3)],
            elapsed_seconds=0.8,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Focused mode" in printed
        assert "3 selected across pool (k_ann=40)" in printed
        assert "cap bypassed" not in printed

    def test_author_filter_without_focused_cap_hint_shown(self) -> None:
        """When --author filter hits cap without --focused, hint suggests adding --focused."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 10},
            author_counts_selected={"Nassim Taleb": 6},
            capped_authors={"Nassim Taleb"},
            prior_count=0,
            unknown_count=0,
            focused=False,
            focused_author="Nassim Taleb",
            author_match_status="resolved",
        )
        resp = SynthesisResponse(
            query="black swan",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(6)],
            elapsed_seconds=1.0,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Author filter active:" in printed
        assert "Nassim Taleb (10 in pool, capped at 6)" in printed
        assert "add --focused to uncap" in printed

    def test_footer_focused_zero_match_warning(self) -> None:
        """When author filter matches 0 candidates with --focused, footer warns and notes broad focus."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=6,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 6},
            author_counts_selected={"Nassim Taleb": 6},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
            focused=True,
            focused_author="Tolle",
            author_match_status="zero_match",
        )
        resp = SynthesisResponse(
            query="presence",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(6)],
            elapsed_seconds=0.9,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
            top_k=20,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Focused mode (author 'Tolle' not found in pool; applied broadly):" in printed
        assert "6 selected (k_ann=40)" in printed

    def test_footer_cap_not_fired_when_pool_equals_cap(self) -> None:
        """Test 10: Pool equals cap (6 pool, 6 selected) -> no cap notice is rendered."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        # pool == cap -> capped_authors is empty
        meta = RetrievalMeta(
            pool_size=6,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 6},
            author_counts_selected={"Nassim Taleb": 6},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
        )
        resp = SynthesisResponse(
            query="black swan",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(6)],
            elapsed_seconds=0.8,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Per-author cap fired" not in printed

    def test_footer_cap_not_fired_when_top_k_limited(self) -> None:
        """Test 11: top_k limited (10 pool, 5 selected, cap=6) -> no cap notice is rendered."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer
        from audiobench.memory.synthesis_retriever import RetrievalMeta

        meta = RetrievalMeta(
            pool_size=10,
            k_ann=40,
            author_counts_pool={"Nassim Taleb": 10},
            author_counts_selected={"Nassim Taleb": 5},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
        )
        resp = SynthesisResponse(
            query="black swan",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(5)],
            elapsed_seconds=0.7,
            model_used="gemini-2.5-flash",
            retrieval_meta=meta,
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Per-author cap fired" not in printed

    def test_footer_shows_fallback_reason(self) -> None:
        """Test 12: Both providers fail -> footer renders fallback reason with digest notice."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer

        resp = SynthesisResponse(
            query="risk",
            synthesis_text="Digest text",
            fragments=[MagicMock() for _ in range(3)],
            elapsed_seconds=0.5,
            model_used="digest_fallback",
            fallback_from="all_providers",
            fallback_reason="Quota exceeded (429)",
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Both LLM providers failed" in printed
        assert "Quota exceeded (429)" in printed

    def test_fallback_reason_populated_on_secondary_success(self) -> None:
        """Test 13: Secondary success carries fallback_from='ollama' and reason into footer."""
        from audiobench.cli.repl.backslash_commands import render_synthesis_footer

        resp = SynthesisResponse(
            query="risk",
            synthesis_text="Text",
            fragments=[MagicMock() for _ in range(3)],
            elapsed_seconds=2.1,
            model_used="gemini-2.5-flash",
            fallback_from="ollama",
            fallback_reason="ollama: 402 payment required",
        )

        with patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            render_synthesis_footer(resp)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Ollama unavailable" in printed
        assert "402 payment required" in printed
        assert "gemini-2.5-flash" in printed


class TestFocusedAndAuthorSynthesis:
    """Tests for --focused (cap bypass) and --author (filtering) retrieval & CLI parsing."""

    def test_parse_synthesize_args(self) -> None:
        """Verify robust argument extraction without positional author inference."""
        from audiobench.cli.repl.backslash_commands import parse_synthesize_args

        q, f, a = parse_synthesize_args('"what did taleb say" "about risk"')
        assert q == "what did taleb say about risk"
        assert f is False
        assert a is None

        q, f, a = parse_synthesize_args('--focused "risk and fragility"')
        assert q == "risk and fragility"
        assert f is True
        assert a is None

        q, f, a = parse_synthesize_args('"risk and fragility" --focused')
        assert q == "risk and fragility"
        assert f is True
        assert a is None

        q, f, a = parse_synthesize_args('--focused=Taleb "risk and fragility"')
        assert q == "risk and fragility"
        assert f is True
        assert a == "Taleb"

        q, f, a = parse_synthesize_args('--author "Nassim Taleb" "risk and fragility"')
        assert q == "risk and fragility"
        assert f is False
        assert a == "Nassim Taleb"

        q, f, a = parse_synthesize_args('-a Taleb "risk and fragility"')
        assert q == "risk and fragility"
        assert f is False
        assert a == "Taleb"

        q, f, a = parse_synthesize_args('--focused --author Taleb "risk and fragility"')
        assert q == "risk and fragility"
        assert f is True
        assert a == "Taleb"

        # Bare token after --focused is treated as part of query, not author
        q, f, a = parse_synthesize_args('--focused Taleb "risk and fragility"')
        assert q == "Taleb risk and fragility"
        assert f is True
        assert a is None

    def test_focused_bypasses_cap_in_retriever(self) -> None:
        """With focused=True, all 10 candidates from single author are selected (no cap at 6)."""
        from audiobench.memory.synthesis_retriever import retrieve_synthesis_context

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(100 + i, 0.05 * i) for i in range(1, 11)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                100 + i,
                1,
                f"Passage number {i} discussing black swans, antifragility, and tail risk in markets.",
                "The Black Swan",
                "Nassim Taleb",
                "audio_transcript",
                None,
                None,
            )
            for i in range(1, 11)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context(
                "black swan", top_k=10, index=mock_index, focused=True
            )

            assert len(results) == 10
            assert all(f.author == "Nassim Taleb" for f in results)
            assert meta.focused is True
            assert meta.capped_authors == set()

    def test_author_filter_without_focused_retains_cap(self) -> None:
        """With author='Taleb' but focused=False, cap of 6 remains active and registers capped_authors."""
        from audiobench.memory.synthesis_retriever import retrieve_synthesis_context, DEFAULT_MAX_PER_AUTHOR

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(100 + i, 0.05 * i) for i in range(1, 11)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                100 + i,
                1,
                f"Passage number {i} discussing black swans, antifragility, and tail risk in markets.",
                "The Black Swan",
                "Nassim Taleb",
                "audio_transcript",
                None,
                None,
            )
            for i in range(1, 11)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context(
                "black swan", top_k=10, index=mock_index, focused=False, author="Taleb"
            )

            assert len(results) == DEFAULT_MAX_PER_AUTHOR
            assert meta.focused is False
            assert meta.focused_author == "Nassim Taleb"
            assert meta.author_match_status == "resolved"
            assert meta.capped_authors == {"Nassim Taleb"}

    def test_author_filter_with_focused_uncaps(self) -> None:
        """With author='Taleb' and focused=True, all 10 candidates are selected."""
        from audiobench.memory.synthesis_retriever import retrieve_synthesis_context

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(100 + i, 0.05 * i) for i in range(1, 11)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (
                100 + i,
                1,
                f"Passage number {i} discussing black swans, antifragility, and tail risk in markets.",
                "The Black Swan",
                "Nassim Taleb",
                "audio_transcript",
                None,
                None,
            )
            for i in range(1, 11)
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context(
                "black swan", top_k=10, index=mock_index, focused=True, author="Taleb"
            )

            assert len(results) == 10
            assert meta.focused is True
            assert meta.focused_author == "Nassim Taleb"
            assert meta.author_match_status == "resolved"
            assert meta.capped_authors == set()

    def test_author_multi_match_returns_graceful_response(self) -> None:
        """When author matches multiple authors in candidate pool, generate_synthesis returns contained response."""
        from audiobench.chat.synthesis_engine import generate_synthesis
        from audiobench.memory.synthesis_retriever import AmbiguousAuthorError

        with patch("audiobench.chat.synthesis_engine.retrieve_synthesis_context") as mock_ret:
            mock_ret.side_effect = AmbiguousAuthorError("Nick", ["Nick Bostrom", "Nicholas Taleb"])

            resp = generate_synthesis("superintelligence", author="Nick")

            assert "⚠ Author 'Nick' matches multiple in pool: Nicholas Taleb, Nick Bostrom. Be more specific." in resp.synthesis_text
            assert resp.fragments == []
            assert resp.model_used == "none"
            assert resp.retrieval_meta is not None
            assert resp.retrieval_meta.author_match_status == "multi_match"

    def test_author_zero_match_with_focused_falls_back(self) -> None:
        """When author matches 0 candidates under focused mode, falls back to broad focus."""
        from audiobench.memory.synthesis_retriever import retrieve_synthesis_context

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(101, 0.1), (102, 0.2)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (101, 1, "Passage about risk and uncertainty in complex adaptive economic systems.", "The Black Swan", "Nassim Taleb", "audio_transcript", None, None),
            (102, 2, "Passage about bias and heuristics in human decision making under pressure.", "Thinking Fast", "Daniel Kahneman", "audio_transcript", None, None),
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context(
                "presence", top_k=5, index=mock_index, focused=True, author="Tolle"
            )

            assert len(results) == 2
            assert meta.focused is True
            assert meta.focused_author == "Tolle"
            assert meta.author_match_status == "zero_match"

    def test_author_zero_match_without_focused_returns_empty(self) -> None:
        """When author matches 0 candidates without focused mode, returns empty pool."""
        from audiobench.memory.synthesis_retriever import retrieve_synthesis_context

        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.query.return_value = [(101, 0.1)]
        mock_index.get_items.side_effect = lambda ids: np.array([[1.0, 0.0] for _ in ids], dtype=np.float32)

        sample_ann_rows = [
            (101, 1, "Passage about risk and uncertainty in complex adaptive economic systems.", "The Black Swan", "Nassim Taleb", "audio_transcript", None, None),
        ]

        with patch("audiobench.memory.synthesis_retriever.get_session") as mock_session_ctx, \
             patch("audiobench.memory.synthesis_retriever.embed_query") as mock_embed:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_embed.return_value = np.array([1.0, 0.0], dtype=np.float32)

            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_ann_rows),
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: []),
            ]

            results, meta = retrieve_synthesis_context(
                "presence", top_k=5, index=mock_index, focused=False, author="Tolle"
            )

            assert len(results) == 0
            assert meta.author_match_status == "zero_match"

    def test_cmd_synthesize_forwards_focused_and_author_args(self) -> None:
        """cmd_synthesize correctly parses and passes focused and author flags to generate_synthesis."""
        from audiobench.cli.repl.backslash_commands import cmd_synthesize
        from audiobench.cli.repl.session import ReplSession

        session = ReplSession(cli_group=MagicMock())
        mock_resp = SynthesisResponse(
            query="risk",
            synthesis_text="Synthesized text",
            fragments=[],
            elapsed_seconds=0.1,
            model_used="gemini",
        )

        with patch("audiobench.chat.synthesis_engine.generate_synthesis", return_value=mock_resp) as mock_gen, \
             patch("audiobench.cli.repl.backslash_commands.console.print"):
            cmd_synthesize(session, "--focused --author Taleb risk")
            mock_gen.assert_called_once_with(
                "risk",
                community_id=None,
                focused=True,
                author="Taleb",
            )



