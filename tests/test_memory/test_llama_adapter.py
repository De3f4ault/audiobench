from unittest.mock import patch

import numpy as np
import pytest

from audiobench.memory.llama_adapter import NomicEmbeddingAdapter


@patch("audiobench.memory.llama_adapter.EmbeddingEngine")
def test_llama_adapter_delegates_correctly(mock_engine_cls):
    """Verify LlamaIndex adapter delegates to EmbeddingEngine."""
    mock_engine = mock_engine_cls.return_value
    mock_engine.embed_for_storage.return_value = np.array([0.1, 0.2, 0.3])
    mock_engine.embed_for_query.return_value = np.array([0.4, 0.5, 0.6])

    adapter = NomicEmbeddingAdapter()

    # Test query delegation
    query_res = adapter._get_query_embedding("test query")
    mock_engine.embed_for_query.assert_called_once_with("test query")
    assert query_res == [0.4, 0.5, 0.6]

    # Test text delegation
    text_res = adapter._get_text_embedding("test text")
    mock_engine.embed_for_storage.assert_called_once_with("test text")
    assert text_res == [0.1, 0.2, 0.3]


@pytest.mark.anyio
@patch("audiobench.memory.llama_adapter.EmbeddingEngine")
async def test_llama_adapter_async_delegation(mock_engine_cls):
    """Verify async LlamaIndex adapter methods delegate correctly."""
    mock_engine = mock_engine_cls.return_value
    mock_engine.embed_for_storage.return_value = np.array([0.1, 0.2, 0.3])
    mock_engine.embed_for_query.return_value = np.array([0.4, 0.5, 0.6])

    adapter = NomicEmbeddingAdapter()

    # Test async query
    query_res = await adapter._aget_query_embedding("async query")
    mock_engine.embed_for_query.assert_called_once_with("async query")
    assert query_res == [0.4, 0.5, 0.6]

    # Test async text
    text_res = await adapter._aget_text_embedding("async text")
    mock_engine.embed_for_storage.assert_called_once_with("async text")
    assert text_res == [0.1, 0.2, 0.3]

    # Test batch texts
    batch_res = adapter._get_text_embeddings(["one", "two"])
    assert mock_engine.embed_for_storage.call_count == 3  # 1 earlier + 2 now
    assert batch_res == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
