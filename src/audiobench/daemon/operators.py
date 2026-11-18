"""operate_on — universal semantic operator.

Executes a verb against a resolved expression in the LanceDB corpus.
Templates only — no LLM, no exec().  The result frame is mechanically
traceable to the expression text and the verb name.

Verb dispatch:
- summarize:  compress expression to key claims
- expand:     generate elaborations and sub-claims
- connect:    find latent relations to other expressions (calls store.search)
- challenge:  generate adversarial alternative interpretations
"""
import logging
from collections.abc import AsyncGenerator
from typing import Any

from audiobench.memory.memory_store import MemoryStore

logger = logging.getLogger("audiobench.daemon.operators")


async def operate_on(
    target: int | str,
    verb: str,
    context: dict[str, Any],
    store: MemoryStore,
) -> AsyncGenerator[dict[str, Any], None]:
    """The universal semantic operation.

    Target resolution:
    - int  → load ExpressionRecord directly (O(1))
    - str  → dense search → take top-1 → load

    Verb dispatch:
    - summarize:  compress expression to key claims
    - expand:     generate elaborations and sub-claims
    - connect:    find latent relations via store.search()
    - challenge:  generate adversarial alternative interpretations
    """
    logger.info("Executing operation: %s on target: %s", verb, target)

    # ── 1. Resolve target ────────────────────────────────────────────────────
    if isinstance(target, int):
        expression_id = target
        text_content = f"Expression #{expression_id}"
    else:
        results = store.search(query=str(target), top_k=1)
        if not results:
            raise ValueError(f"No results found for target: {target!r}")
        expression_id = results[0]["expression_id"]
        text_content = results[0].get("content", "")

    yield {
        "status": "progress",
        "step": verb,
        "pct": 0.1,
        "data": {"reasoning": f"Resolved target to expression {expression_id}"},
    }

    # ── 2. Dispatch verb (templates only) ────────────────────────────────────
    if verb == "summarize":
        yield {
            "status": "progress",
            "step": verb,
            "pct": 0.5,
            "data": {"reasoning": "Synthesizing summary..."},
        }
        snippet = text_content[:120]
        result = f"Summary of {expression_id}: {snippet}..."

    elif verb == "expand":
        yield {
            "status": "progress",
            "step": verb,
            "pct": 0.5,
            "data": {"reasoning": "Elaborating details..."},
        }
        result = (
            f"Elaboration on expression {expression_id}:\n"
            f"  • Key claims derived from: {text_content[:80]}...\n"
            f"  • Sub-claims: [template — LLM not connected]"
        )

    elif verb == "connect":
        yield {
            "status": "progress",
            "step": verb,
            "pct": 0.4,
            "data": {"reasoning": "Searching for related expressions..."},
        }
        # Real search — templates only, no LLM
        related = store.search(query=text_content, top_k=5)
        related_ids = [r["expression_id"] for r in related if r["expression_id"] != expression_id]
        yield {
            "status": "progress",
            "step": verb,
            "pct": 0.8,
            "data": {"reasoning": f"Found {len(related_ids)} related expressions"},
        }
        result = {
            "related_expression_ids": related_ids,
            "related_summaries": [
                {"expression_id": r["expression_id"], "content": r.get("content", "")[:80]}
                for r in related
                if r["expression_id"] != expression_id
            ],
        }

    elif verb == "challenge":
        yield {
            "status": "progress",
            "step": verb,
            "pct": 0.5,
            "data": {"reasoning": "Generating adversarial critique..."},
        }
        result = (
            f"Adversarial critique of expression {expression_id}:\n"
            f"  • Alternative interpretation: [template — LLM not connected]\n"
            f"  • Potential counter-claim for: {text_content[:80]}..."
        )

    else:
        raise ValueError(f"Unsupported verb: {verb!r}")

    yield {
        "status": "progress",
        "step": verb,
        "pct": 1.0,
        "data": {"reasoning": "Complete"},
    }

    yield {
        "verb": verb,
        "target": target,
        "resolved_expression_id": expression_id,
        "result": result,
    }
