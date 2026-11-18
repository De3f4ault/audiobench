"""Gate 1: Extractive Digest Parity Test.

Verifies byte-identical parity between the legacy synthesis_engine.py fallback digest
and llm_providers._render_extractive_digest, as well as ProviderRouter extractive fallback.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from audiobench.chat.providers.llm_providers import _render_extractive_digest
from audiobench.chat.providers.provider_outcome import OutcomeType
from audiobench.chat.providers.router import FallbackPolicy, ProviderRouter
from audiobench.core.settings import AudioBenchSettings
from audiobench.memory.synthesis_retriever import SynthesisFragment


def legacy_synthesis_engine_digest(fragments: list[SynthesisFragment], query: str) -> str:
    """Exact verbatim logic from legacy synthesis_engine.py:317-328."""
    digest_lines = [
        f"### Dialectic Digest: {query}",
        "",
        "*(LLM service unavailable — displaying grounded comparative excerpts)*",
        "",
    ]
    for idx, f in enumerate(fragments, 1):
        digest_lines.append(
            f"**[{idx}] {f.author}** — *{f.work_title}* (Relevance: `{f.effective_rel:.3f}`)\n"
            f"> \"{f.content}\"\n"
        )
    return "\n".join(digest_lines)


def test_extractive_digest_byte_identical_parity():
    fragments = [
        SynthesisFragment(
            expression_id=101,
            work_id=1,
            work_title="Meditations",
            author="Marcus Aurelius",
            content="You have power over your mind - not outside events. Realize this, and you will find strength.",
            q_sim=0.92,
            edge_weight=1.0,
            effective_rel=0.92,
            mmr_score=0.92,
            source_mode="ANN",
        ),
        SynthesisFragment(
            expression_id=102,
            work_id=2,
            work_title="Enchiridion",
            author="Epictetus",
            content="Men are disturbed not by things, but by the view which they take of them.",
            q_sim=0.88,
            edge_weight=1.0,
            effective_rel=0.885,
            mmr_score=0.885,
            source_mode="ANN",
        ),
        SynthesisFragment(
            expression_id=103,
            work_id=3,
            work_title="Letters from a Stoic",
            author="Seneca",
            content="We suffer more often in imagination than in reality.",
            q_sim=0.81,
            edge_weight=0.9,
            effective_rel=0.81234,
            mmr_score=0.812,
            source_mode="ANN",
        ),
    ]

    query = "stoic resilience in adversity"

    legacy_output = legacy_synthesis_engine_digest(fragments, query)
    router_helper_output = _render_extractive_digest(fragments, query=query)

    # Byte-identical parity check
    assert router_helper_output == legacy_output
    assert router_helper_output.encode("utf-8") == legacy_output.encode("utf-8")
    assert f"### Dialectic Digest: {query}" in router_helper_output
    assert not router_helper_output.startswith("### Dialectic Digest\n")


def test_router_extractive_digest_outcome_parity():
    """Verify that when router degrades with digest_mode='extractive', output matches legacy."""
    settings = AudioBenchSettings(
        llm_provider_order=["ollama"],
        gemini_api_key=None,
    )
    router = ProviderRouter(settings)

    # Mock ollama to fail so router falls through to degraded
    mock_ollama = MagicMock()
    mock_ollama.generate.side_effect = RuntimeError("Ollama connection refused")
    router._slots[0].provider = mock_ollama

    fragments = [
        SynthesisFragment(
            expression_id=201,
            work_id=10,
            work_title="The Republic",
            author="Plato",
            content="The measure of a man is what he does with power.",
            q_sim=0.95,
            edge_weight=1.0,
            effective_rel=0.950,
            mmr_score=0.950,
            source_mode="ANN",
        )
    ]
    query = "virtue and justice"

    outcome = router.generate(
        messages=[{"role": "user", "content": "synthesize"}],
        context_fragments=fragments,
        query=query,
        digest_mode="extractive",
        # NO_LOCAL_CPU: test is verifying the digest degradation path specifically;
        # we don't want a live Ollama server to intercept with a local model and
        # mask whether the digest path is reached.
        fallback_policy=FallbackPolicy.NO_LOCAL_CPU,
    )

    assert outcome.type == OutcomeType.DEGRADED
    assert outcome.degraded_mode == "extractive"
    expected = legacy_synthesis_engine_digest(fragments, query)
    assert outcome.content == expected
    assert outcome.content.encode("utf-8") == expected.encode("utf-8")
    assert f"### Dialectic Digest: {query}" in outcome.content
    assert not outcome.content.startswith("### Dialectic Digest\n")
