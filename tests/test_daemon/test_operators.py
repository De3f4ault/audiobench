from unittest.mock import MagicMock

import pytest

from audiobench.daemon.operators import operate_on
from audiobench.memory.memory_store import MemoryStore


@pytest.mark.asyncio
async def test_operate_on_int_target_loads_expression_by_id():
    store = MagicMock(spec=MemoryStore)

    events = []
    async for frame in operate_on(target=42, verb="summarize", context={}, store=store):
        events.append(frame)

    # the last event is the result dict
    result = events[-1]
    assert result["verb"] == "summarize"
    assert result["target"] == 42
    assert result["resolved_expression_id"] == 42
    assert "Summary of 42:" in result["result"]
    store.search.assert_not_called()

@pytest.mark.asyncio
async def test_operate_on_str_target_dense_searches():
    store = MagicMock(spec=MemoryStore)
    store.search.return_value = [{"expression_id": 99, "content": "mock text content"}]

    events = []
    async for frame in operate_on(target="search query", verb="expand", context={}, store=store):
        events.append(frame)

    result = events[-1]
    assert result["verb"] == "expand"
    assert result["target"] == "search query"
    assert result["resolved_expression_id"] == 99

    # Verify dense search was called with top_k=1
    store.search.assert_called_once_with(query="search query", top_k=1)
