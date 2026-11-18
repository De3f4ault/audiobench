"""
Tests for 2D — Streaming reasoning trace via the synthesize verb.

Three invariants (per roadmap):
1. test_synthesize_emits_reasoning_frames — the synthesize verb in
   PipelineExecutor must yield at least one reasoning progress frame
   (status='progress', data.reasoning present) before the final ok frame.
2. test_pipeline_stream_client_yields_progress_frames — DaemonClient has a
   pipeline_stream() method that yields progress frames from the daemon's
   _stream transport; terminal ok/error frames end the stream.
3. test_synthesize_result_has_answer_and_reasoning_keys — the context dict
   written by the synthesize step must contain both 'answer' and 'reasoning'
   keys so downstream steps and the final ok frame carry structured output.
"""

import asyncio
from unittest.mock import patch

import audiobench.daemon.pipeline as _pipe
from audiobench.daemon.pipeline import PipelineExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(gen) -> list[dict]:
    frames = []
    async for frame in gen:
        frames.append(frame)
    return frames


def _run_async(coro_or_gen):
    async def _inner():
        if hasattr(coro_or_gen, "__anext__"):
            return await _collect(coro_or_gen)
        return await coro_or_gen
    return asyncio.run(_inner())


def _inject_fakes(**overrides):
    orig = dict(_pipe._VERB_HANDLERS)
    _pipe._VERB_HANDLERS.update({
        "search": lambda args: {"results": []},
        "chunk":  lambda args: {"chunks": []},
        "embed":  lambda args: {"embedded": True},
    })
    _pipe._VERB_HANDLERS.update(overrides)
    return orig


# ---------------------------------------------------------------------------
# 1. synthesize emits reasoning frames
# ---------------------------------------------------------------------------

def test_synthesize_emits_reasoning_frames():
    """
    A pipeline with a single 'synthesize' step must produce at least one
    progress frame whose data dict contains a 'reasoning' key — so the TUI
    can display incremental reasoning without waiting for the final result.
    """
    orig = _inject_fakes()
    try:
        steps = [{"verb": "synthesize", "params": {"question": "What is memory?"}}]
        frames = _run_async(PipelineExecutor().run({"steps": steps}))
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)

    progress = [f for f in frames if f.get("status") == "progress"]
    reasoning_frames = [
        f for f in progress
        if isinstance(f.get("data"), dict) and "reasoning" in f["data"]
    ]
    assert reasoning_frames, (
        "Expected at least one progress frame with data.reasoning, "
        f"got progress frames: {progress}"
    )


# ---------------------------------------------------------------------------
# 2. pipeline_stream() on DaemonClient yields progress frames
# ---------------------------------------------------------------------------

def test_pipeline_stream_client_yields_progress_frames():
    """
    DaemonClient.pipeline_stream(steps) must be a generator that yields
    progress frames from the daemon's _stream transport and stops when it
    receives a terminal ok or error frame.
    """
    from audiobench.daemon.client import DaemonClient

    fake_frames = [
        {"status": "progress", "step": "synthesize", "event": "start"},
        {"status": "progress", "step": "synthesize", "event": "complete",
         "data": {"reasoning": "thinking..."}},
        {"status": "ok", "success": True, "data": {"context": {}}},
    ]

    with patch.object(DaemonClient, "_stream", return_value=iter(fake_frames)):
        client = DaemonClient.__new__(DaemonClient)
        results = list(client.pipeline_stream(
            steps=[{"verb": "synthesize", "params": {}}]
        ))

    progress = [f for f in results if f.get("status") == "progress"]
    assert len(progress) == 2, f"Expected 2 progress frames, got {results}"
    # ok frame must NOT be in yielded results (consumed internally)
    assert not any(f.get("status") == "ok" for f in results), \
        "ok frame should not be yielded to caller"


# ---------------------------------------------------------------------------
# 3. synthesize result has answer and reasoning keys
# ---------------------------------------------------------------------------

def test_synthesize_result_has_answer_and_reasoning_keys():
    """
    After a synthesize step, the context dict in the final ok frame must
    contain both 'answer' and 'reasoning' keys — structured output for
    downstream steps and display.
    """
    orig = _inject_fakes()
    try:
        steps = [{"verb": "synthesize", "params": {
            "question": "Summarise the corpus",
            "context": "prior search results here",
        }}]
        frames = _run_async(PipelineExecutor().run({"steps": steps}))
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)

    ok_frames = [f for f in frames if f.get("status") == "ok"]
    assert ok_frames, "No ok frame emitted"

    ctx = ok_frames[0].get("data", {}).get("context", {})
    assert "answer" in ctx, f"'answer' missing from context: {ctx}"
    assert "reasoning" in ctx, f"'reasoning' missing from context: {ctx}"
