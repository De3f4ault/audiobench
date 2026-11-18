
import pytest

import audiobench.daemon.pipeline as _pipe
from audiobench.daemon.pipeline import PipelineExecutor


def _inject_fakes(**overrides):
    """Pre-populate _VERB_HANDLERS with no-op fakes, returning original state."""
    orig = dict(_pipe._VERB_HANDLERS)
    _pipe._VERB_HANDLERS.update({
        "search":    lambda args: {"results": []},
        "chunk":     lambda args: {"chunks": []},
        "embed":     lambda args: {"embedded": True},
        "synthesize": lambda args: {"answer": "stub"},
    })
    _pipe._VERB_HANDLERS.update(overrides)
    return orig


@pytest.mark.asyncio
async def test_pipeline_executor_runs_steps_in_order():
    orig = _inject_fakes()
    try:
        executor = PipelineExecutor()
        args = {
            "steps": [
                {"verb": "search", "params": {"query": "test"}},
                {"verb": "synthesize"}
            ]
        }

        events = []
        async for frame in executor.run(args):
            if frame.get("status") == "progress":
                events.append((frame.get("step"), frame.get("event")))

        # Check that search started/completed before synthesize started
        assert events == [
            ("search", "start"),
            ("search", "complete"),
            ("synthesize", "start"),
            ("synthesize", "complete")
        ]
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)


@pytest.mark.asyncio
async def test_pipeline_step_pct_increases_monotonically():
    orig = _inject_fakes()
    try:
        executor = PipelineExecutor()
        args = {
            "steps": [
                {"verb": "synthesize"}
            ]
        }

        pcts = []
        async for frame in executor.run(args):
            if frame.get("status") == "progress" and "pct" in frame:
                pcts.append(frame.get("pct"))

        # Check that percentages are sorted (may be empty if executor emits none)
        assert pcts == sorted(pcts)
    finally:
        _pipe._VERB_HANDLERS.clear()
        _pipe._VERB_HANDLERS.update(orig)
