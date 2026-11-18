from unittest.mock import MagicMock

import pytest


def make_query_result(query="test", sources=1):
    from audiobench.memory.query_engine import ResearchResult
    from audiobench.memory.rrf_fusion import FusedResult
    result = ResearchResult(query=query)
    result.sources = [
        FusedResult(
            segment_id=i,
            start_time=10.0,
            end_time=20.0,
            text=f"fragment {i}",
            rrf_score=0.1,
            stream_contributions=(("fts5", 1),)
        ) for i in range(sources)
    ]
    return result

@pytest.fixture
def mock_llm(monkeypatch):
    from audiobench.chat.providers.ollama_provider import OllamaClient
    mock = MagicMock(spec=OllamaClient)
    mock.stream.return_value = [{"content": "expanded synthesis chunk"}]
    monkeypatch.setattr("audiobench.chat.providers.ollama_provider.OllamaClient", lambda *args, **kwargs: mock)
    return mock

@pytest.fixture
def mock_engine(monkeypatch):
    from audiobench.memory.query_engine import ResearchEngine
    mock = MagicMock(spec=ResearchEngine)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd.ResearchEngine", lambda *args, **kwargs: mock)
    return mock

def test_expand_uses_same_fragments_not_new_search(mock_llm, mock_engine):
    from audiobench.cli.commands.memory_cmd import _stream_expanded_synthesis
    result = make_query_result(sources=5)
    _stream_expanded_synthesis(result)
    assert mock_engine.search.call_count == 0
    # LLM must have been called with a prompt containing fragment content
    call_args = mock_llm.stream.call_args[1]["prompt"]
    for src in result.sources:
        assert src.text[:30] in call_args

def test_expand_prompt_requests_timestamps(mock_llm):
    from audiobench.cli.commands.memory_cmd import _stream_expanded_synthesis
    _stream_expanded_synthesis(make_query_result(sources=3))
    prompt = mock_llm.stream.call_args[1]["prompt"]
    assert "timestamp" in prompt.lower() or "time" in prompt.lower()

def test_expand_prompt_requests_open_question(mock_llm):
    from audiobench.cli.commands.memory_cmd import _stream_expanded_synthesis
    _stream_expanded_synthesis(make_query_result())
    prompt = mock_llm.stream.call_args[1]["prompt"]
    assert "open question" in prompt.lower() or "explore" in prompt.lower()
