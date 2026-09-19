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
def mock_input_returns(monkeypatch):
    def _setter(returns: list[str]):
        it = iter(returns)
        fn = lambda *args, **kwargs: next(it)
        monkeypatch.setattr("builtins.input", fn)

        class MockPromptSession:
            def prompt(self, *args, **kwargs):
                return fn()

        monkeypatch.setattr(
            "audiobench.cli.commands.memory_cmd._make_search_prompt_session",
            lambda *args, **kwargs: MockPromptSession(),
        )
    return _setter

@pytest.fixture
def mock_engine(monkeypatch):
    from audiobench.memory.query_engine import ResearchEngine
    mock = MagicMock(spec=ResearchEngine)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd.ResearchEngine", lambda *args, **kwargs: mock)
    return mock

@pytest.fixture
def mock_repl(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("audiobench.chat.chat_repl.ChatREPL", mock)
    return mock

from audiobench.cli.commands.memory_cmd import SearchSessionState


@pytest.fixture(autouse=True)
def _isolate_panel_db(test_db):
    """Ensure search panel tests never touch production databases."""
    pass


def test_q_exits_panel_cleanly(mock_input_returns, mock_engine, capsys):
    mock_input_returns(["q"])
    mock_engine.search.return_value = make_query_result()
    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())

def test_invalid_input_reprompts(mock_input_returns, mock_engine, capsys):
    mock_input_returns(["x", "q"])
    mock_engine.search.return_value = make_query_result()
    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())
    out = capsys.readouterr().out
    # Single-char non-slash input is treated as "too short for a query", not unknown command
    assert "Too short" in out or "Unknown command" in out

def test_e_triggers_expanded_synthesis_not_new_search(mock_input_returns, mock_engine):
    # 'e' opens reader, reader takes inputs. We'll mock input for reader as well:
    # 1. 'e' (enters reader)
    # 2. 'q' (quits reader, goes back to panel)
    # 3. 'q' (quits panel)
    mock_input_returns(["e", "q", "q"])
    mock_engine.search.return_value = make_query_result(sources=3)
    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())
    assert mock_engine.search.call_count == 1

def test_c_opens_chat_repl_with_preloaded_fragments(mock_input_returns, mock_engine, mock_repl):
    mock_input_returns(["c"])
    mock_engine.search.return_value = make_query_result(query="bike incident", sources=3)
    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())
    # mock_repl should be instantiated and run called
    assert mock_repl.return_value.run.call_count == 1
