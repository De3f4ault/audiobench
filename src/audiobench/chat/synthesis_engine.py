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
from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.memory.query_engine import _call_llm, Ok, Err
from audiobench.memory.synthesis_retriever import (
    AmbiguousAuthorError,
    RetrievalMeta,
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
    # Footer fields — populated by generate_synthesis when LLM fallback fires or cap fires.
    fallback_from: str | None = None       # "ollama" if primary failed and secondary succeeded
    fallback_reason: str | None = None     # primary error string on fallback success
    retrieval_meta: RetrievalMeta | None = None
    top_k: int = 20                        # requested top_k for query context in footer


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
4. Epistemic Distinction for Prior Syntheses: Sources whose author is "[Prior Synthesis]" or "[Past Session]" (or whose source_type is search_synthesis / search_session_summary) represent prior user conclusions and syntheses from earlier sessions, NOT external primary authors. When citing such a source, frame it explicitly as a prior user conclusion (e.g. "In previous sessions, you concluded that [1]...", "Your earlier analysis established [1]..."). Distinguish it from primary audio evidence, and when relevant, contrast or integrate how new primary authors support, deepen, or challenge these earlier conclusions.
"""


def _clean_synthesis_citations(text: str) -> str:
    """Normalize model citation artifacts like [ 2†L9-L15] or [ 1|L27-L34] to clean [2] or [1]."""
    cleaned = re.sub(
        r"\[\s*(\d+)\s*(?:[†|:]\s*(?:L?\d+(?:-L?\d+)?|[a-zA-Z0-9_\-\.]+)?|\s+L\d+(?:-L\d+)?)\s*\]",
        r"[\1]",
        text,
    )
    return re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", cleaned)


PRIOR_SOURCE_TYPES = frozenset({"search_synthesis", "search_session_summary"})
PRIOR_AUTHOR_LABELS = frozenset({"[Prior Synthesis]", "[Past Session]"})


def _is_prior_source(f: SynthesisFragment) -> bool:
    """A source is prior-synthesis if it's an LLM-generated artifact from an earlier session."""
    st = getattr(f, "source_type", None) or ""
    author = getattr(f, "author", "") or ""
    return st in PRIOR_SOURCE_TYPES or author in PRIOR_AUTHOR_LABELS


def format_synthesis_prompt(query: str, fragments: list[SynthesisFragment]) -> str:
    """Build the dialectic prompt incorporating numbered source excerpts."""
    lines = [
        f"Core Question: \"{query}\"\n",
        "Retrieved Library Excerpts:",
    ]
    has_prior = False
    for idx, f in enumerate(fragments, 1):
        clique_note = " (Consensus Triad Partner)" if f.is_clique_partner else ""
        type_tag = ""
        if _is_prior_source(f):
            has_prior = True
            st = getattr(f, "source_type", "search_synthesis") or "search_synthesis"
            type_tag = f" [source_type: {st}]"
        lines.append(
            f"\nSource [{idx}]: {f.author} — {f.work_title}{clique_note}{type_tag}\n"
            f"\"{f.content}\""
        )

    task_instructions = [
        "\nTask: Synthesize these perspectives across the three required phases. "
        "Explicitly cite your sources using [1], [2], [3] throughout every section of your response."
    ]
    if has_prior:
        task_instructions.append(
            "\nCRITICAL: Sources tagged with [source_type: search_synthesis] or authored by "
            "[Prior Synthesis] represent conclusions from previous sessions. Frame them as "
            "'In previous sessions you concluded...' rather than treating them as external "
            "evidence or published works."
        )

    lines.append("\n".join(task_instructions))
    return "\n".join(lines)


def fetch_typed_edge_hints(fragments: list[SynthesisFragment]) -> str:
    """Query typed-edge graph for co-retrieved pairs; return pre-registered hint block.

    For each pair (i, j) where both expression_ids appear in the retrieved set
    and a typed edge exists (created_by='step5_triadic_extraction'), emit a hint
    line in the pre-registered format:
        fragments [{i}, {j}] — {RELATION_TYPE} (prompt_v={version}, weight={w:.3f})

    Returns empty string if no co-retrieved typed pairs exist.

    The hint block is prefixed with a verification instruction so the LLM uses
    it as metadata rather than as authoritative fact — important given the 1.4%
    residual error rate in v1.0/v1.1 classifications.
    """
    if not fragments:
        return ""

    # Build expression_id -> fragment-index mapping (1-based, matching prompt)
    eid_to_idx: dict[int, int] = {f.expression_id: i for i, f in enumerate(fragments, 1)}
    retrieved_ids = list(eid_to_idx.keys())
    if len(retrieved_ids) < 2:
        return ""

    id_list = ",".join(str(i) for i in retrieved_ids)
    try:
        from sqlalchemy import text as sa_text
        with get_session() as session:
            rows = session.execute(sa_text(f"""
                SELECT from_expression_id, to_expression_id,
                       relation_type, weight, prompt_version
                FROM expression_relations
                WHERE created_by = 'step5_triadic_extraction'
                  AND relation_type IN ('contradicts', 'elaborates', 'inspired_by')
                  AND from_expression_id IN ({id_list})
                  AND to_expression_id IN ({id_list})
                ORDER BY weight DESC
            """)).fetchall()
    except Exception:
        logger.debug("fetch_typed_edge_hints: DB query failed, skipping hints", exc_info=True)
        return ""

    if not rows:
        return ""

    hint_lines = []
    seen_pairs: set[tuple[int, int]] = set()
    for from_id, to_id, rel_type, weight, prompt_ver in rows:
        pair_key = (min(from_id, to_id), max(from_id, to_id))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        i = eid_to_idx[from_id]
        j = eid_to_idx[to_id]
        hint_lines.append(
            f"  fragments [{i}, {j}] — {rel_type.upper()}"
            f" (prompt_v={prompt_ver or 'unknown'}, weight={weight:.3f})"
        )

    if not hint_lines:
        return ""

    header = (
        "\nLibrary graph hints (verify against passage text before citing — "
        f"{len(hint_lines)} typed relation(s) detected among retrieved sources):"
    )
    return header + "\n" + "\n".join(hint_lines) + "\n"


def generate_synthesis(
    query: str,
    top_k: int = 20,
    temperature: float = 0.3,
    llm: OllamaClient | None = None,
    community_id: int | None = None,
    use_typed_edges: bool = False,
    focused: bool = False,
    author: str | None = None,
) -> SynthesisResponse:
    """Generate a grounded cross-disciplinary synthesis for the given query.

    Args:
        query: Conceptual or thematic question.
        top_k: Number of diverse fragments to retrieve (default 20).
        temperature: LLM sampling temperature (default 0.3).
        llm: Optional OllamaClient instance.
        community_id: Optional community ID to restrict synthesis to a single school.
        use_typed_edges: Whether to inject library graph hints for co-retrieved pairs.
        focused: When True, bypasses the per-author diversity cap across named authors.
        author: Optional author name to filter candidate pool before MMR selection.

    Returns:
        SynthesisResponse with text and citation mapping.
    """
    t0 = time.perf_counter()
    # Unpack happens before _call_llm so digest fallback has access to fragments
    # regardless of LLM outcome (user's watch-point from code review).
    try:
        fragments, retrieval_meta = retrieve_synthesis_context(
            query,
            top_k=top_k,
            community_id=community_id,
            focused=focused,
            author=author,
        )
    except AmbiguousAuthorError as e:
        meta = RetrievalMeta(
            pool_size=0,
            k_ann=40,
            author_counts_pool={},
            author_counts_selected={},
            capped_authors=set(),
            prior_count=0,
            unknown_count=0,
            focused=focused,
            focused_author=e.requested,
            author_match_status="multi_match",
        )
        return SynthesisResponse(
            query=query,
            synthesis_text=f"⚠ Author '{e.requested}' matches multiple in pool: {', '.join(e.candidates)}. Be more specific.",
            citations={},
            fragments=[],
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            model_used="none",
            top_k=top_k,
            retrieval_meta=meta,
        )

    if not fragments:
        if author and retrieval_meta.author_match_status == "zero_match":
            msg = f"No passages by author '{author}' found in candidate pool for this query."
        else:
            msg = "No relevant or linkable passages found across the library for this theme."
        return SynthesisResponse(
            query=query,
            synthesis_text=msg,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            top_k=top_k,
            retrieval_meta=retrieval_meta,
        )

    citation_map: dict[int, int] = {
        idx: f.expression_id for idx, f in enumerate(fragments, 1)
    }

    prompt = format_synthesis_prompt(query, fragments)

    # Option 2: inject typed-edge metadata for co-retrieved pairs.
    # Pre-registered format: "verify against passage text" to avoid hallucinated
    # edge labels. No-op when use_typed_edges=False (baseline A/B condition)
    # or when no co-retrieved typed pairs exist (Q1 abstract queries).
    typed_hints = fetch_typed_edge_hints(fragments) if use_typed_edges else ""
    if typed_hints:
        logger.debug("Injecting %d typed-edge hint(s) into synthesis prompt", typed_hints.count("fragments ["))
        prompt = prompt + typed_hints

    full_prompt = f"{_SYNTHESIS_SYSTEM_PROMPT}\n\n{prompt}"

    settings = get_settings()
    client = llm or OllamaClient()
    api_key = settings.gemini_api_key

    llm_res = _call_llm(full_prompt, temperature, client, api_key)

    fallback_from: str | None = None
    fallback_reason: str | None = None

    if isinstance(llm_res, Ok):
        answer = _clean_synthesis_citations(llm_res.value)
        # Sentinel-aware: Ok.model defaults to "unknown" (truthy) so bare `or` is a no-op.
        # Compare against sentinel explicitly and fall back to client model name.
        model_used = (
            llm_res.model if llm_res.model and llm_res.model != "unknown"
            else getattr(client, "_model", "gemma4:31b-cloud")
        )
        # Carry primary-failure info when secondary (Gemini) succeeded — set by _call_llm.
        fallback_from = getattr(llm_res, "fallback_from", None)
        fallback_reason = getattr(llm_res, "fallback_reason", None)
    else:
        logger.warning("LLM call failed in synthesis engine: %s; falling back to digest.", llm_res.error)
        model_used = "digest_fallback"
        fallback_from = "all_providers"
        fallback_reason = llm_res.error
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
        fallback_from=fallback_from,
        fallback_reason=fallback_reason,
        retrieval_meta=retrieval_meta,
        top_k=top_k,
    )
