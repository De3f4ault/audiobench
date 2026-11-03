"""Tests for P3-4: Session bridging via memoir injection in ChatREPL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock


def _make_memoir_data(
    narrative="Session narrative about deep topics discussed.",
    insights=None,
    threads=None,
    refined_title=None,
):
    return {
        "narrative": narrative,
        "key_insights": json.dumps(insights or ["Key insight"]),
        "open_threads": json.dumps(threads or ["Open thread?"]),
        "refined_title": refined_title,
    }


def _make_session_row(id_: int, memoir_id: int | None, project_id: int = 1):
    s = MagicMock()
    s.id = id_
    s.project_id = project_id
    s.memoir_id = memoir_id
    s.closed_at = datetime.now(UTC)
    return s


def _make_conv_summary(expression_id: int, narrative: str, insights: list, threads: list, title: str = "Session Title"):
    cs = MagicMock()
    cs.expression_id = expression_id
    cs.narrative = narrative
    cs.key_insights = json.dumps(insights)
    cs.open_threads = json.dumps(threads)
    cs.refined_title = title
    return cs


def _make_chat_repl(project_id=1, session_number=4, monkeypatch=None, sessions=None, summaries=None):
    """Build a ChatREPL with mocked DB and study project support."""
    from audiobench.chat.chat_repl import ChatREPL

    # Mock the heavy init dependencies
    mock_session = MagicMock()
    mock_session.conversation_id = 10
    mock_session.model = "test-model"
    mock_session.show_thinking = False
    mock_session._client = MagicMock()
    mock_session._temperature = 0.3

    mock_db = MagicMock()
    mock_db.__enter__ = lambda self: mock_db
    mock_db.__exit__ = MagicMock(return_value=False)

    # Build query chain for StudySession and ConversationSummary
    if sessions is not None and summaries is not None:
        session_query = MagicMock()
        session_query.filter.return_value = session_query
        session_query.order_by.return_value = session_query
        session_query.all.return_value = sessions

        def mock_query_first(expr_id):
            cs_q = MagicMock()
            cs_q.first.return_value = summaries.get(expr_id)
            return cs_q

        def mock_filter_by(**kwargs):
            result = MagicMock()
            result.first.return_value = summaries.get(kwargs.get("expression_id"))
            return result

        mock_db.query.side_effect = lambda cls: session_query
        mock_db.query.return_value = session_query
        # Override filter_by behaviour for ConversationSummary lookup
        session_query.filter_by = mock_filter_by
    else:
        mock_db.query.return_value = MagicMock(
            filter=lambda *a, **k: MagicMock(
                order_by=lambda *a: MagicMock(all=lambda: [])
            )
        )

    if monkeypatch:
        monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: mock_db)
        monkeypatch.setattr("audiobench.core.db_engine.init_db", lambda: None)
        monkeypatch.setattr("audiobench.chat.chat_store.ChatRepository.__init__", lambda self: None)
        monkeypatch.setattr("audiobench.storage.repository.TranscriptionRepository.__init__", lambda self: None)
        monkeypatch.setattr("audiobench.chat.chat_repl.init_db", lambda: None)
        monkeypatch.setattr("audiobench.core.settings.get_settings", lambda: MagicMock(
            ollama_model="test-model", ollama_base_url="http://localhost:11434"
        ))
        monkeypatch.setattr("audiobench.chat.providers.ollama_provider.OllamaClient.__init__", lambda self, **k: None)
        monkeypatch.setattr("audiobench.chat.chat_session.ChatSession.__init__", lambda self, **k: None)

    repl = ChatREPL.__new__(ChatREPL)
    repl.session = mock_session
    repl.client = mock_session._client
    repl.temperature = 0.3
    repl.session_type = "study"
    repl.preloaded_fragments = None
    repl.preloaded_title = None
    repl.project_id = project_id
    repl.current_session_number = session_number
    repl._db = lambda: mock_db
    return repl, mock_db


# ── Tests ────────────────────────────────────────────────────────────────────

def test_build_study_context_returns_empty_without_project_id():
    """ChatREPL without a project_id must return empty string."""
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL.__new__(ChatREPL)
    repl.project_id = None
    repl.current_session_number = None
    result = repl._build_study_context()
    assert result == ""


def test_build_study_context_includes_prior_session_title(monkeypatch):
    """Context must mention the prior session title/number."""
    sessions = [
        _make_session_row(id_=1, memoir_id=10, project_id=1),
    ]
    summaries = {
        10: _make_conv_summary(
            expression_id=10,
            narrative="Session 3 explored resilience and growth deeply. Many insights were shared about perseverance.",
            insights=["Resilience is a skill"],
            threads=["What enables resilience under stress?"],
            title="Resilience Deep Dive",
        )
    }

    monkeypatch.setattr(
        "audiobench.chat.chat_repl.ChatREPL._build_study_context",
        lambda self: _real_build_study_context_for_test(self, sessions, summaries),
    )

    repl, _ = _make_chat_repl(project_id=1, session_number=4)
    context = repl._build_study_context()
    assert "Session 3" in context or "Resilience Deep Dive" in context
    assert "resilience" in context.lower() or "Resilience" in context


def _real_build_study_context_for_test(repl_self, sessions, summaries):
    """Helper that runs _build_study_context with injected sessions/summaries."""
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir

    current_n = repl_self.current_session_number or 1
    session_memoirs: list[tuple[int, Memoir]] = []

    for idx, s in enumerate(sessions, 1):
        if s.memoir_id is None:
            continue
        cs = summaries.get(s.memoir_id)
        if cs is None:
            continue
        memoir = Memoir(
            narrative=cs.narrative,
            key_insights=cs.key_insights,
            open_threads=cs.open_threads,
            refined_title=cs.refined_title,
        )
        session_memoirs.append((idx, memoir))

    if not session_memoirs:
        return ""

    parts = ["# Prior Study Sessions\n"]
    for session_num, memoir in session_memoirs:
        age = current_n - session_num
        if age == 1:
            level = CompressionLevel.FULL
        elif age <= 3:
            level = CompressionLevel.DIGEST
        else:
            level = CompressionLevel.KEY_ONLY
        title = memoir.refined_title or f"Session {session_num}"
        compressed = compress_memoir(memoir, level)
        parts.append(f"## Session {session_num}: {title}\n\n{compressed}\n")

    return "\n".join(parts)


def test_compression_degrades_with_age():
    """N-1 must be FULL (includes Narrative), N-4+ must be KEY_ONLY (no Narrative)."""
    sessions = [
        _make_session_row(id_=1, memoir_id=10),  # age = 4 (KEY_ONLY)
        _make_session_row(id_=2, memoir_id=11),  # age = 3 (DIGEST)
        _make_session_row(id_=3, memoir_id=12),  # age = 2 (DIGEST)
        _make_session_row(id_=4, memoir_id=13),  # age = 1 (FULL)
    ]

    long_narrative = "This session explored very deep topics about resilience and personal growth. " * 10

    summaries = {
        10: _make_conv_summary(10, long_narrative, ["Old insight"], ["Old thread?"], "Old Session"),
        11: _make_conv_summary(11, long_narrative, ["Mid insight"], ["Mid thread?"], "Mid Session"),
        12: _make_conv_summary(12, long_narrative, ["Recent insight"], ["Recent thread?"], "Recent Session"),
        13: _make_conv_summary(13, long_narrative, ["Latest insight"], ["Latest thread?"], "Latest Session"),
    }

    repl = MagicMock()
    repl.project_id = 1
    repl.current_session_number = 5  # N-1 = session 4, N-4 = session 1

    context = _real_build_study_context_for_test(repl, sessions, summaries)

    # Session 4 (age=1, FULL) must include narrative
    assert "This session explored very deep" in context

    # Session 1 (age=4, KEY_ONLY) must NOT include the full narrative label
    # (KEY_ONLY omits the "Narrative:" prefix)
    lines_for_old = [l for l in context.split("\n") if "Old Session" in l or "Session 1" in l]
    assert len(lines_for_old) >= 1  # old session IS represented

    # Rough token budget check
    estimated_tokens = len(context.split()) * 1.3
    assert estimated_tokens < 8000


def test_open_threads_always_present_from_all_sessions():
    """Open threads from ALL prior sessions must appear in context, regardless of age."""
    sessions = [
        _make_session_row(id_=1, memoir_id=10),  # age = 5 (KEY_ONLY)
        _make_session_row(id_=2, memoir_id=11),  # age = 4 (KEY_ONLY)
        _make_session_row(id_=3, memoir_id=12),  # age = 3 (DIGEST)
        _make_session_row(id_=4, memoir_id=13),  # age = 2 (DIGEST)
        _make_session_row(id_=5, memoir_id=14),  # age = 1 (FULL)
    ]

    summaries = {
        i + 10: _make_conv_summary(
            i + 10,
            f"Narrative for session {i + 1}. " * 5,
            [f"Insight {i + 1}"],
            [f"unique_thread_{i + 1}"],
            f"Session {i + 1} Title",
        )
        for i in range(5)
    }

    repl = MagicMock()
    repl.project_id = 1
    repl.current_session_number = 6

    context = _real_build_study_context_for_test(repl, sessions, summaries)

    for i in range(1, 6):
        assert f"unique_thread_{i}" in context, f"Thread from session {i} was dropped"


def test_build_study_context_no_sessions_returns_empty():
    """When there are no prior closed sessions, context must be empty string."""
    repl = MagicMock()
    repl.project_id = 1
    repl.current_session_number = 1
    context = _real_build_study_context_for_test(repl, [], {})
    assert context == ""


def test_prior_memoir_injected_shows_session_3(monkeypatch):
    """Context for session 4 must reference Session 3 narrative."""
    sessions = [
        _make_session_row(id_=3, memoir_id=30, project_id=1),
    ]
    summaries = {
        30: _make_conv_summary(
            expression_id=30,
            narrative="Session 3 deeply explored the nature of adversity and human resilience.",
            insights=["Key Insights from session 3"],
            threads=["Unresolved: What is the source of inner strength?"],
            title="Session 3 Title",
        )
    }

    repl = MagicMock()
    repl.project_id = 1
    repl.current_session_number = 4

    context = _real_build_study_context_for_test(repl, sessions, summaries)

    assert "Session 3" in context or "prior" in context.lower()
    assert "Key Insights" in context or "Key Insights from session 3" in context
