"""Unit tests for SearchAutoSuggest and SearchSemanticCompleter."""

import time
from unittest.mock import MagicMock
from prompt_toolkit.document import Document

from audiobench.cli.commands.memory_cmd import SearchSessionState
from audiobench.memory.search_completer import SearchAutoSuggest


def test_search_autosuggest_non_blocking(monkeypatch):
    """SearchAutoSuggest.get_suggestion must return immediately without blocking UI thread."""
    state = SearchSessionState(session_id=1, preset="fts5")
    mock_client = MagicMock()

    # Simulate a slow daemon call (0.5s)
    def slow_autocomplete(text, top_k=1):
        time.sleep(0.1)
        return [{"text": f"{text} is a great continuation of audio"}]

    mock_client.autocomplete.side_effect = slow_autocomplete
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    suggest = SearchAutoSuggest(state)
    mock_buffer = MagicMock()
    mock_buffer.history.get_strings.return_value = []

    doc = Document("what about audio", cursor_position=16)

    # 1. First call: not cached yet -> must return None almost instantly (well under 50ms)
    t0 = time.perf_counter()
    res1 = suggest.get_suggestion(mock_buffer, doc)
    elapsed = time.perf_counter() - t0

    assert res1 is None
    assert elapsed < 0.05, f"get_suggestion took {elapsed:.4f}s, expected < 0.05s"

    # 2. Wait slightly for background thread to complete cache population
    time.sleep(0.2)

    # 3. Second call: now cached -> instant suggestion
    t0 = time.perf_counter()
    res2 = suggest.get_suggestion(mock_buffer, doc)
    elapsed2 = time.perf_counter() - t0

    assert res2 is not None
    assert "continuation" in res2.text
    assert elapsed2 < 0.01
