"""PipelineExecutor — chains {verb, params} steps against daemon models.

Each step is a dict with keys:
  verb   — one of: search, embed, chunk, synthesize
  params — verb-specific arguments

The executor maintains a shared context dict. Each step may read from it
(implicit inputs from prior steps) and write results back to it.

Streams one progress frame on step start, one on step completion, and a
final ``{"status": "ok", ...}`` frame carrying the accumulated context.

Unknown verbs yield an error frame (status="error") and halt the pipeline.

Streaming verbs (synthesize):
  Instead of a single return dict, these verbs are async generators that
  yield intermediate reasoning frames (status="progress", data.reasoning)
  before yielding a sentinel ``{"_result": {...}}`` frame carrying the
  context payload to merge.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

# ---------------------------------------------------------------------------
# Verb handler registry — module-level so tests can patch individual entries.
# Each entry is a callable (args: dict) -> dict.
# Populated lazily at first build() call to avoid circular import at module load.
# ---------------------------------------------------------------------------
_VERB_HANDLERS: dict[str, Any] = {}


def _get_verb_handlers() -> dict[str, Any]:
    """Return the verb handler table, populating it on first call."""
    if not _VERB_HANDLERS:
        from audiobench.daemon.server import (
            _handle_chunk,
            _handle_embed,
            _handle_search,
        )
        _VERB_HANDLERS["search"] = _handle_search
        _VERB_HANDLERS["chunk"] = _handle_chunk
        _VERB_HANDLERS["embed"] = _handle_embed
    return _VERB_HANDLERS


class PipelineExecutor:
    """Execute a sequence of pipeline steps against the daemon's loaded models.

    Each step is ``{verb, params}``.  Executor maintains a shared ``context``
    dict that passes one step's outputs as the next step's implicit inputs.

    Supported verbs
    ---------------
    search      → queries MemoryStore; writes ``results`` to context
    embed       → embeds one expression; writes ``embedded=True`` to context
    chunk       → chunks a text; writes ``chunks`` to context
    synthesize  → streams reasoning frames, then writes answer+reasoning to context

    Streams progress frames so the client can display incremental feedback.
    """

    async def run(
        self, args: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        from audiobench.daemon.intelligence.timing_model import get_timing_model

        steps = args.get("steps", [])
        timing = get_timing_model()
        context: dict[str, Any] = {}

        for step in steps:
            verb = step.get("verb", "")
            params = step.get("params", {})

            yield {"status": "progress", "step": verb, "event": "start"}

            start_t = time.time()
            try:
                # _iter_dispatch yields 0-N reasoning frames then a _result sentinel.
                result: dict[str, Any] = {}
                async for frame in self._iter_dispatch(verb, params, context):
                    if "_result" in frame:
                        result = frame["_result"]
                    else:
                        # Re-yield intermediate reasoning frames to the caller.
                        yield frame

                context.update(result)
                duration_ms = (time.time() - start_t) * 1000.0
                timing.record_latency(verb, duration_ms)
                yield {"status": "progress", "step": verb, "event": "complete"}
            except _UnknownVerb as exc:
                yield {
                    "status": "error",
                    "step": verb,
                    "verb": verb,
                    "error": str(exc),
                }
                return  # Halt pipeline on unknown verb

        yield {"status": "ok", "success": True, "data": {"context": context}}

    # ------------------------------------------------------------------
    # Internal dispatch — always an async generator
    # ------------------------------------------------------------------

    async def _iter_dispatch(
        self, verb: str, params: dict[str, Any], context: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Route a verb, yielding 0-N reasoning frames then a _result sentinel.

        Lookup order:
        1. _VERB_HANDLERS dict (populated lazily; testable by direct injection).
        2. Built-in streaming verbs (_synthesize).
        3. _UnknownVerb sentinel.
        """
        # Check the handler table first (covers search/chunk/embed + test overrides).
        handlers = _get_verb_handlers()
        if verb in handlers:
            result = handlers[verb]({"verb": verb, **params})
            # Normalise: ensure result is a dict for context.update()
            yield {"_result": result if isinstance(result, dict) else {}}
            return

        # Built-in streaming verbs.
        if verb == "synthesize":
            async for frame in self._synthesize(params, context):
                yield frame
            return

        raise _UnknownVerb(f"Unsupported pipeline verb: {verb!r}")

    async def _synthesize(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming synthesis — emits reasoning frames then a _result sentinel.

        Templates only — no LLM, no exec().  Phase 2D wires the real LLM
        when it becomes available; for now the reasoning trace is mechanically
        derived from the question and available context.
        """
        question = params.get("question") or context.get("question", "")
        ctx_text = params.get("context") or str(context.get("results", ""))

        # ── Reasoning trace (streaming) ──────────────────────────────────────
        yield {
            "status": "progress",
            "step": "synthesize",
            "data": {
                "reasoning": f"Framing question: {question!r}"
            },
        }

        yield {
            "status": "progress",
            "step": "synthesize",
            "data": {
                "reasoning": (
                    f"Drawing on context ({len(ctx_text)} chars): "
                    f"{ctx_text[:80]}..."
                )
            },
        }

        yield {
            "status": "progress",
            "step": "synthesize",
            "data": {"reasoning": "Composing answer from available evidence..."},
        }

        # ── Final result (sentinel) ──────────────────────────────────────────
        answer = (
            f"[synthesize] question={question!r} | "
            f"context_used={len(ctx_text)} chars | "
            f"LLM: not connected (template)"
        )
        reasoning_summary = (
            f"Template reasoning over question={question!r}. "
            f"No LLM connected — answer is mechanically derived."
        )

        yield {
            "_result": {
                "answer": answer,
                "reasoning": reasoning_summary,
                "context_used": ctx_text[:200],
            }
        }

    async def _search(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        handle = _get_verb_handlers()["search"]
        query = params.get("query") or context.get("query", "")
        top_k = int(params.get("top_k", 5))
        result = handle({"query": query, "top_k": top_k, **params})
        return {"results": result.get("results", [])}

    async def _embed(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        handle = _get_verb_handlers()["embed"]
        result = handle(dict(params))
        return {"embedded": True, "embed_result": result}

    async def _chunk(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        handle = _get_verb_handlers()["chunk"]
        text = params.get("text") or context.get("text", "")
        result = handle({"text": text, **params})
        return {"chunks": result.get("chunks", [])}


class _UnknownVerb(ValueError):
    """Raised by _iter_dispatch when a verb is not recognised."""
