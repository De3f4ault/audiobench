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

            results = retrieve_synthesis_context("suffering and psychology", top_k=2, index=mock_index)

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
            mock_ret.return_value = frags
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
            mock_ret.return_value = frags
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

