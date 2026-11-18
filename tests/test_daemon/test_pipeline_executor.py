"""
Tests for 2A — PipelineExecutor step chaining.

Three invariants (per roadmap):
1. test_each_step_emits_start_and_complete_frames — every step in the steps
   array must produce a 'start' progress frame and a 'complete' progress frame.
2. test_context_threads_through_steps — the output of one step is accessible
   in the context that the next step receives (context threading).
3. test_unknown_verb_yields_error_frame_not_crash — an unknown verb must yield
   an error frame with status='error' and not raise an exception.

All tests call PipelineExecutor.run() directly as an async generator.
Verb handlers are patched so no real model is invoked.
"""

import asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(gen) -> list[dict]:
    """Drain an async generator and return all yielded frames."""
    frames = []
    async for frame in gen:
        frames.append(frame)
    return frames


def _run(coro_or_gen):
    """Run a coroutine or consume an async generator synchronously."""
    async def _inner():
        if hasattr(coro_or_gen, '__anext__'):
            return await _collect(coro_or_gen)
        return await coro_or_gen
    return asyncio.run(_inner())


# ---------------------------------------------------------------------------
# 1. Each step emits start and complete frames
# ---------------------------------------------------------------------------

def test_each_step_emits_start_and_complete_frames():
    """
    For a two-step pipeline, there must be exactly one 'start' and one
    'complete' progress frame per step, plus a final 'ok' frame.
    """
    import audiobench.daemon.pipeline as _pipe
    from audiobench.daemon.pipeline import PipelineExecutor

    steps = [
        {"verb": "search", "params": {"query": "test query", "top_k": 3}},
        {"verb": "chunk",  "params": {"text": "some text"}},
    ]

    # Pre-populate the handler table with fakes before the executor runs.
    orig = dict(_pipe._VERB_HANDLERS)
    _pipe._VERB_HANDLERS["search"] = lambda args: {"results": [{"expression_id": 1, "content": "x"}]}
    _pipe._VERB_HANDLERS["chunk"]  = lambda args: {"chunks": ["chunk A", "chunk B"]}
    try:
        frames = _run(PipelineExecutor().run({"steps": steps}))
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)

    progress = [f for f in frames if f.get("status") == "progress"]
    ok_frames = [f for f in frames if f.get("status") == "ok"]

    starts    = [f for f in progress if f.get("event") == "start"]
    completes = [f for f in progress if f.get("event") == "complete"]

    assert len(starts) == 2, f"Expected 2 start frames, got {len(starts)}"
    assert len(completes) == 2, f"Expected 2 complete frames, got {len(completes)}"
    assert len(ok_frames) == 1, f"Expected 1 ok frame, got {len(ok_frames)}"

    assert starts[0]["step"] == "search"
    assert starts[1]["step"] == "chunk"


# ---------------------------------------------------------------------------
# 2. Context threads through steps
# ---------------------------------------------------------------------------

def test_context_threads_through_steps():
    """
    Output written to context by step N must be visible to step N+1 via the
    context dict carried in the final ok frame.
    """
    import audiobench.daemon.pipeline as _pipe
    from audiobench.daemon.pipeline import PipelineExecutor

    steps = [
        {"verb": "search", "params": {"query": "threading test", "top_k": 2}},
        {"verb": "chunk",  "params": {"text": "some content"}},
    ]

    orig = dict(_pipe._VERB_HANDLERS)
    _pipe._VERB_HANDLERS["search"] = lambda args: {"results": [{"expression_id": 42, "content": "ctx"}]}
    _pipe._VERB_HANDLERS["chunk"]  = lambda args: {"chunks": ["c1"]}
    try:
        frames = _run(PipelineExecutor().run({"steps": steps}))
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)

    ok_frames = [f for f in frames if f.get("status") == "ok"]
    assert ok_frames, "No ok frame emitted"
    ctx = ok_frames[0].get("data", {}).get("context", {})

    # Both step outputs must be in context
    assert "results" in ctx, f"'results' missing from context: {ctx}"
    assert "chunks" in ctx, f"'chunks' missing from context: {ctx}"
    assert ctx["results"][0]["expression_id"] == 42
    assert ctx["chunks"] == ["c1"]


# ---------------------------------------------------------------------------
# 3. Unknown verb yields error frame, does not raise
# ---------------------------------------------------------------------------

def test_unknown_verb_yields_error_frame_not_crash():
    """
    A step with an unrecognised verb must yield a frame with status='error'
    and must NOT raise an exception out of the generator. The executor must
    continue to yield the final ok frame (or an error-summary frame).
    """
    from audiobench.daemon.pipeline import PipelineExecutor

    steps = [{"verb": "totally_unknown_verb", "params": {}}]

    # This must not raise
    frames = _run(PipelineExecutor().run({"steps": steps}))

    error_frames = [f for f in frames if f.get("status") == "error"]
    assert error_frames, \
        f"Expected at least one error frame for unknown verb, got: {frames}"
    assert "verb" in error_frames[0] or "step" in error_frames[0]
