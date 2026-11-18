from unittest.mock import MagicMock

import pytest


def make_query_result(query="test", sources=1, source_file=""):
    from audiobench.memory.query_engine import ResearchResult
    from audiobench.memory.rrf_fusion import FusedResult
    result = ResearchResult(query=query)
    result.sources = [
        FusedResult(
            segment_id=i,
            start_time=10.0,
            end_time=20.0,
            text=f"fragment {i}",
            source_file=source_file or f"file_{i}.mp3",
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
    mock_input_returns(["c", "q"])
    mock_engine.search.return_value = make_query_result(query="bike incident", sources=3)
    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())
    # mock_repl should be instantiated and run called
    assert mock_repl.return_value.run.call_count == 1


def test_space_pauses_when_playing(mock_input_returns, mock_engine, monkeypatch, capsys):
    mock_input_returns(["\x00SPACE", "q"])
    mock_engine.search.return_value = make_query_result(query="philosophy", sources=3)
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {
        "playing": True,
        "paused": False,
        "state": "playing",
        "position": 120.0,
    }
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())

    mock_client.playback_pause.assert_called_once()
    mock_client.playback_resume.assert_not_called()
    mock_client.playback_play.assert_not_called()
    out = capsys.readouterr().out
    assert "Playback paused" in out


def test_space_resumes_when_paused(mock_input_returns, mock_engine, monkeypatch, capsys):
    mock_input_returns(["\x00SPACE", "q"])
    mock_engine.search.return_value = make_query_result(query="philosophy", sources=3)
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {
        "playing": True,
        "paused": True,
        "state": "paused",
        "position": 120.0,
    }
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())

    mock_client.playback_resume.assert_called_once()
    mock_client.playback_pause.assert_not_called()
    mock_client.playback_play.assert_not_called()
    out = capsys.readouterr().out
    assert "Resumed playback at 00:02:00" in out


def test_space_starts_citation_1_when_stopped(mock_input_returns, mock_engine, monkeypatch, capsys):
    mock_input_returns(["\x00SPACE", "q"])
    res = make_query_result(query="philosophy", sources=3, source_file="/fake/audio.mp3")
    mock_engine.search.return_value = res
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {
        "playing": False,
        "paused": False,
        "state": "stopped",
        "position": 0.0,
    }
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    from audiobench.cli.commands.memory_cmd import _run_search_loop
    _run_search_loop(mock_engine, "test", SearchSessionState())

    mock_client.playback_play.assert_called_once()
    mock_client.playback_pause.assert_not_called()
    mock_client.playback_resume.assert_not_called()
    out = capsys.readouterr().out
    assert "Playing citation [1]" in out

