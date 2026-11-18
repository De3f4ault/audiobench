"""
Tests for 2B — operate_on universal operator (supplementary tests).

The existing test_operators.py covers int-target and str-target resolution.
These three tests verify the remaining invariants:

1. test_unsupported_verb_raises_value_error — an unrecognised verb must raise
   ValueError with the verb name in the message (no exec, no LLM).
2. test_connect_verb_calls_store_search_for_related — the 'connect' verb must
   call store.search() to find expressions related to the target (not just
   return a static stub string).
3. test_progress_frames_have_pct_in_bounds — every progress frame emitted by
   operate_on must have a 'pct' value in [0.0, 1.0].
"""

from unittest.mock import MagicMock

import pytest

from audiobench.daemon.operators import operate_on
from audiobench.memory.memory_store import MemoryStore

# ---------------------------------------------------------------------------
# 1. Unknown verb raises ValueError with verb name in message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsupported_verb_raises_value_error():
    """
    An unrecognised verb must raise ValueError before yielding any final
    result frame.  The exception message must contain the verb name so callers
    can log meaningful errors.
    """
    store = MagicMock(spec=MemoryStore)

    with pytest.raises(ValueError, match="totally_unknown"):
        async for _ in operate_on(
            target=1, verb="totally_unknown", context={}, store=store
        ):
            pass


# ---------------------------------------------------------------------------
# 2. connect verb calls store.search() for related expressions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_verb_calls_store_search_for_related():
    """
    The 'connect' verb must call store.search() to find expressions related
    to the resolved target — it must not just return a static string.
    The final result frame must contain the related expression IDs.
    """
    store = MagicMock(spec=MemoryStore)
    # Primary resolution: int target — no search needed
    # Related-expression search: returns two candidates
    store.search.return_value = [
        {"expression_id": 10, "content": "related A"},
        {"expression_id": 11, "content": "related B"},
    ]

    frames = []
    async for frame in operate_on(target=5, verb="connect", context={}, store=store):
        frames.append(frame)

    result = frames[-1]
    assert result["verb"] == "connect"
    assert result["resolved_expression_id"] == 5

    # store.search must have been called (for related-expression lookup)
    store.search.assert_called()

    # Related IDs must appear in the result somehow (list or str representation)
    result_str = str(result.get("result", ""))
    assert "10" in result_str or "related" in result_str, \
        f"Expected related expression IDs in result, got: {result}"


# ---------------------------------------------------------------------------
# 3. Progress frames have pct in [0.0, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_frames_have_pct_in_bounds():
    """
    Every progress frame emitted by operate_on must carry a 'pct' float in
    [0.0, 1.0].  This is a contract for progress-bar rendering in the TUI.
    """
    store = MagicMock(spec=MemoryStore)
    store.search.return_value = [{"expression_id": 7, "content": "some content"}]

    for verb in ("summarize", "expand", "connect", "challenge"):
        frames = []
        async for frame in operate_on(target=1, verb=verb, context={}, store=store):
            frames.append(frame)

        progress_frames = [f for f in frames if f.get("status") == "progress"]
        assert progress_frames, f"No progress frames emitted for verb={verb!r}"

        for f in progress_frames:
            pct = f.get("pct")
            assert pct is not None, f"Missing 'pct' in frame: {f}"
            assert 0.0 <= pct <= 1.0, \
                f"pct={pct!r} out of [0,1] for verb={verb!r}, frame={f}"
