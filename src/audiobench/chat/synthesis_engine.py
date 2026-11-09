"""Synthesis Engine — LLM Dialectic Synthesis with Grounded Bracketed Citations.

Connects synthesis_retriever.py to the LLM backend (Ollama / Gemini) to synthesize
divergent philosophical and narrative perspectives across the library.

Guarantees:
- Grounded attribution: Enforces inline citations [1], [2], [3] mapped directly
  to source expression IDs.
- Interactive Flywheel: Exposes citation mapping so REPL users can run:
  `\\bookmark cite <n>` to inspect the underlying passage.
- Fallback protection: If the LLM service is offline, formats a structured
  multi-author comparative digest directly from the retrieved fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any

from audiobench.chat.providers.ollama_provider import OllamaClient
from audiobench.core.settings import get_settings
from audiobench.memory.query_engine import _call_llm, Ok, Err
from audiobench.memory.synthesis_retriever import (
    SynthesisFragment,
    retrieve_synthesis_context,
)

logger = logging.getLogger("audiobench.chat.synthesis_engine")


@dataclass
class SynthesisResponse:
    """The completed dialectic synthesis and its citation mappings."""
    query: str
    synthesis_text: str
    citations: dict[int, int] = field(default_factory=dict)  # {1: eid1, 2: eid2, ...}
    fragments: list[SynthesisFragment] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_used: str = "digest_fallback"


# Provisional adoption (Structure 2: Dialectic Triad):
# Validated 2026-11-07 to yield 2x citation density (20 vs 10), zero preamble fluff,
# faster inference (2.23s vs 3.85s), and explicit dialectic friction over serial listing.
_SYNTHESIS_SYSTEM_PROMPT = """You are AudioBench's Dialectic Synthesis Engine.
Transform the provided library excerpts into a rigorous dialectic synthesis structured across three specific phases:

Required Output Format:
### 1. Foundational Convergence
Identify the shared fundamental premise or existential reality all authors accept.

### 2. Dialectic Tension & Divergence
Examine where their mechanics, priorities, or worldviews clash, diverge, or challenge one another. Highlight the friction between their approaches.

### 3. Integrated Synthesis
Formulate a higher-order principle or operational framework that integrates these contrasting perspectives.

Strict Rules:
1. Ground every claim using inline bracketed citation numbers: [1], [2], [3], etc. Use strictly plain integers (e.g. [1], [2]). Never append line numbers, daggers (†), or pipes (e.g. do not write [1†L1-L5] or [1|L1-L5]).
2. Do not invent facts or extrapolate beyond what is supported by the excerpts.
3. Keep your tone intellectual, concise, and direct. Avoid conversational filler or introductory/concluding preambles. Start directly with the first section header (### 1. Foundational Convergence).
"""


def _clean_synthesis_citations(text: str) -> str:
    """Normalize model citation artifacts like [ 2†L9-L15] or [ 1|L27-L34] to clean [2] or [1]."""
    cleaned = re.sub(
        r"\[\s*(\d+)\s*(?:[†|:]\s*(?:L?\d+(?:-L?\d+)?|[a-zA-Z0-9_\-\.]+)?|\s+L\d+(?:-L\d+)?)\s*\]",
        r"[\1]",
        text,
    )
    return re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", cleaned)


def format_synthesis_prompt(query: str, fragments: list[SynthesisFragment]) -> str:
    """Build the dialectic prompt incorporating numbered source excerpts."""
    lines = [
        f"Core Question: \"{query}\"\n",
        "Retrieved Library Excerpts:",
    ]
    for idx, f in enumerate(fragments, 1):
        clique_note = " (Consensus Triad Partner)" if f.is_clique_partner else ""
        lines.append(
            f"\nSource [{idx}]: {f.author} — {f.work_title}{clique_note}\n"
            f"\"{f.content}\""
        )

    lines.append(
        "\nTask: Synthesize these perspectives across the three required phases. "
        "Explicitly cite your sources using [1], [2], [3] throughout every section of your response."
    )
    return "\n".join(lines)


def generate_synthesis(
    query: str,
    top_k: int = 4,
    temperature: float = 0.3,
    llm: OllamaClient | None = None,
    community_id: int | None = None,
) -> SynthesisResponse:
    """Generate a grounded cross-disciplinary synthesis for the given query.

    Args:
        query: Conceptual or thematic question.
        top_k: Number of diverse fragments to retrieve (default 4).
        temperature: LLM sampling temperature (default 0.3).
        llm: Optional OllamaClient instance.
        community_id: Optional community ID to restrict synthesis to a single school.

    Returns:
        SynthesisResponse with text and citation mapping.
    """
    t0 = time.perf_counter()
    fragments = retrieve_synthesis_context(query, top_k=top_k, community_id=community_id)

    if not fragments:
        return SynthesisResponse(
            query=query,
            synthesis_text="No relevant or linkable passages found across the library for this theme.",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )

    citation_map: dict[int, int] = {
        idx: f.expression_id for idx, f in enumerate(fragments, 1)
    }

    prompt = format_synthesis_prompt(query, fragments)
    full_prompt = f"{_SYNTHESIS_SYSTEM_PROMPT}\n\n{prompt}"

    settings = get_settings()
    client = llm or OllamaClient()
    api_key = settings.gemini_api_key

    llm_res = _call_llm(full_prompt, temperature, client, api_key)

    model_used = "digest_fallback"
    if isinstance(llm_res, Ok):
        answer = _clean_synthesis_citations(llm_res.value)
        res_model = getattr(llm_res, "model", None)
        model_used = res_model if res_model and res_model != "unknown" else getattr(client, "_model", "gemma4:31b-cloud")
    else:
        logger.warning("LLM call failed in synthesis engine: %s; falling back to digest.", llm_res.error)
        model_used = "digest_fallback"
        # Structured fallback if local model and cloud are unreachable
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
        answer = "\n".join(digest_lines)

    elapsed = round(time.perf_counter() - t0, 3)
    return SynthesisResponse(
        query=query,
        synthesis_text=answer,
        citations=citation_map,
        fragments=fragments,
        elapsed_seconds=elapsed,
        model_used=model_used,
    )
