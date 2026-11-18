from unittest.mock import MagicMock
import pytest
from rich.console import Console

from audiobench.cli.commands.memory_cmd import SearchSessionState, _open_fragment_reader
from audiobench.memory.query_engine import ResearchResult
from audiobench.memory.rrf_fusion import FusedResult


@pytest.fixture(autouse=True)
def _isolate_reader_db(test_db):
    """Ensure reader tests never touch production databases."""
    pass


def make_test_result(query="test query", count=3):
    res = ResearchResult(query=query)
    res.sources = [
        FusedResult(
            segment_id=i + 1,
            start_time=float(i * 30),
            end_time=float(i * 30 + 15),
            text=f"Transcript hit text {i+1} for {query}",
            rrf_score=0.25,
            source_file="/tmp/fake_audio.mp3",
            transcription_id=1,
            stream_contributions=(("fts5", 1),),
        )
        for i in range(count)
    ]
    res.answer = f"Synthesized answer for {query}"
    return res


def test_reader_playback_dispatch(monkeypatch):
    """Pressing 'p' dispatches playback_play without NameError."""
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"state": "idle"}
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    # Sequence of keypresses: 'p' (play), 'q' (quit)
    keys = iter(["p", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))

    console = Console(record=True)
    res = make_test_result()
    _open_fragment_reader(console, fragments=res.sources, initial_idx=0)

    assert mock_client.playback_play.call_count == 1
    call_args, call_kwargs = mock_client.playback_play.call_args
    assert call_args[0] == "/tmp/fake_audio.mp3"
    assert call_kwargs["start_pos"] == 0.0


def test_reader_pause_resume_toggle(monkeypatch):
    """Pressing Space toggles pause when playing and plays when idle."""
    mock_client = MagicMock()
    # First check: playing -> pause
    mock_client.playback_status.return_value = {"state": "playing"}
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    keys = iter([" ", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))

    console = Console(record=True)
    res = make_test_result()
    _open_fragment_reader(console, fragments=res.sources, initial_idx=0)

    assert mock_client.playback_pause.call_count == 1


def test_reader_hit_navigation_and_autoseek(monkeypatch):
    """Stepping hits with 'n' auto-seeks if playback was playing."""
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"state": "playing"}
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    # Press 'n' (step hit 1 -> 2), then 'b' (step hit 2 -> 1), then 'q'
    keys = iter(["n", "b", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))

    console = Console(record=True)
    res = make_test_result(count=3)
    _open_fragment_reader(console, fragments=res.sources, initial_idx=0)

    # Playback should have auto-sought twice: once for hit 2 (30.0), once back to hit 1 (0.0)
    assert mock_client.playback_play.call_count == 2
    positions = [k["start_pos"] for _, k in mock_client.playback_play.call_args_list]
    assert positions == [30.0, 0.0]


def test_reader_search_hopping(monkeypatch):
    """Pressing Tab cycles between multiple searches in the session."""
    state = SearchSessionState()
    res1 = make_test_result("search 1", count=2)
    res2 = make_test_result("search 2", count=2)
    state.search_results = {1: res1, 2: res2}
    state.search_count = 2

    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    # Sequence: Tab (switches S1 -> S2 or S2 -> S1), then 'q'
    keys = iter(["\t", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))

    console = Console(record=True)
    ret_seq = _open_fragment_reader(console, state=state, active_seq=1, initial_idx=0)
    # Switched to S2
    assert ret_seq == 2


def test_reader_pin_toggle(monkeypatch):
    """Pressing '*' toggles fragment pin in state.pinned_fragments."""
    state = SearchSessionState()
    res = make_test_result()
    state.search_results = {1: res}
    state.search_count = 1

    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    # Pin then quit
    keys = iter(["*", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))

    console = Console(record=True)
    _open_fragment_reader(console, state=state, active_seq=1, initial_idx=0)

    assert 1 in state.pinned_fragments
    assert 1 in state.pinned_expr_ids


def test_reader_command_bar(monkeypatch):
    """Pressing ':' opens command prompt and processes command."""
    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"state": "idle"}
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._resolve_audio_file_id", lambda tid: 10)
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._fetch_adjacent", lambda tid, s, e: ([], []))
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._get_max_end_time", lambda tid: 120.0)

    keys = iter([":", "q"])
    monkeypatch.setattr("audiobench.cli.commands.memory_cmd._read_single_key", lambda: next(keys))
    # Command entered: speed 1.5
    monkeypatch.setattr("builtins.input", lambda *args: "speed 1.5")

    console = Console(record=True)
    res = make_test_result()
    _open_fragment_reader(console, fragments=res.sources, initial_idx=0)

    assert mock_client.playback_speed.call_count == 1
    mock_client.playback_speed.assert_called_with(1.5)
