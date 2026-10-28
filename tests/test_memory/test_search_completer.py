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


# ---------------------------------------------------------------------------
# Multi-Bucket SearchSemanticCompleter Tests
# ---------------------------------------------------------------------------

def test_clean_passage_snippet():
    """_clean_passage_snippet must wrap in quotes and truncate cleanly at word boundaries."""
    from audiobench.memory.search_completer import _clean_passage_snippet

    short_text = "Knowledge begins with self-awareness."
    assert _clean_passage_snippet(short_text, max_chars=50) == '"Knowledge begins with self-awareness."'

    long_text = "I suspect at least two such causes. First, academic psychology, while it is admirable and useful..."
    res = _clean_passage_snippet(long_text, max_chars=45)
    assert res.startswith('"')
    assert res.endswith('…"')
    # Must not cut mid-word into 'academic'
    assert not res.endswith('acad…"')


def test_search_semantic_completer_yields_all_four_buckets(monkeypatch):
    """SearchSemanticCompleter must yield [History], [Author], [Book], [School], and [Passage]."""
    from prompt_toolkit.document import Document
    from prompt_toolkit.completion import CompleteEvent
    from unittest.mock import patch
    from audiobench.memory.search_completer import SearchSemanticCompleter

    state = SearchSessionState(session_id=1, preset="deep")
    mock_client = MagicMock()
    mock_client.autocomplete.return_value = [
        {
            "text": "Academic psychology, while it is admirable and useful as a list of principles...",
            "work_title": "Poor Charlie's Almanack",
            "work_author": "Charles T. Munger",
            "source_type": "audio_transcript",
        }
    ]
    monkeypatch.setattr("audiobench.daemon.factory.get_daemon_client", lambda: mock_client)

    # Mock database queries for history, works, communities
    mock_db = MagicMock()
    mock_db.execute.side_effect = [
        # 1. search_queries
        MagicMock(fetchall=lambda: [("what really is psychology at its core",)]),
        # 2. works (authors)
        MagicMock(fetchall=lambda: [("Charles T. Munger", 2)]),
        # 3. works (titles)
        MagicMock(fetchall=lambda: [("The Psychology of Money", "Morgan Housel")]),
        # 4. expression_communities
        MagicMock(fetchall=lambda: [(12, "Interpersonal Dynamics and Conversational Psychology")]),
    ]
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    completer = SearchSemanticCompleter(state)
    doc = Document("psychology", cursor_position=10)
    event = CompleteEvent(completion_requested=True)

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx):
        completions = list(completer.get_completions(doc, event))

    assert len(completions) >= 5

    meta_tags = [c.display_meta_text.strip() for c in completions]
    assert "[History]" in meta_tags
    assert any("[Author]" in m for m in meta_tags)
    assert any("[Book]" in m for m in meta_tags)
    assert "[School]" in meta_tags
    assert any("[Passage]" in m for m in meta_tags)

    # Verify insertion texts
    texts = [c.text for c in completions]
    assert "what really is psychology at its core" in texts
    assert 'author:"Charles T. Munger"' in texts
    assert 'work:"The Psychology of Money"' in texts
    assert "school:12" in texts
    assert any("Academic psychology" in t for t in texts)


def test_search_semantic_completer_skips_slash_commands(monkeypatch):
    """SearchSemanticCompleter must skip slash commands (handled by SearchSlashCompleter)."""
    from prompt_toolkit.document import Document
    from prompt_toolkit.completion import CompleteEvent
    from audiobench.memory.search_completer import SearchSemanticCompleter

    state = SearchSessionState(session_id=1, preset="deep")
    completer = SearchSemanticCompleter(state)
    doc = Document("/set deep", cursor_position=9)
    event = CompleteEvent(completion_requested=True)

    completions = list(completer.get_completions(doc, event))
    assert len(completions) == 0

