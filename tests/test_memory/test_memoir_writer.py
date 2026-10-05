"""Tests for MemoirWriter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_conversation(id_=1, title="Test Session", session_type="study", transcript_ids="[]"):
    conv = MagicMock()
    conv.id = id_
    conv.title = title
    conv.session_type = session_type
    conv.transcript_ids = transcript_ids
    return conv


def _make_session(id_=1, project_id=1):
    s = MagicMock()
    s.id = id_
    s.project_id = project_id
    s.memoir_id = None
    return s


def _make_messages(texts: list[str]):
    msgs = []
    for i, text in enumerate(texts):
        m = MagicMock()
        m.id = i + 1
        m.role = "user" if i % 2 == 0 else "assistant"
        m.content = text
        msgs.append(m)
    return msgs


VALID_MEMOIR_JSON = json.dumps({
    "narrative": (
        "This session explored deep questions about resilience and suffering. "
        "The participant reflected on past challenges and how they shaped current thinking. "
        "Several key concepts emerged around growth mindset and deliberate practice."
    ),
    "key_insights": ["Suffering can be a catalyst for growth", "Deliberate practice matters"],
    "open_threads": ["What is the role of suffering in meaning-making?"],
    "refined_title": "Resilience and Growth Session",
})


# ── compress_memoir tests ────────────────────────────────────────────────────

def test_compress_full_includes_narrative():
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir
    m = Memoir(
        narrative="Long narrative text here.",
        key_insights=json.dumps(["Insight A"]),
        open_threads=json.dumps(["Thread 1"]),
    )
    result = compress_memoir(m, CompressionLevel.FULL)
    assert "Long narrative text here." in result
    assert "Insight A" in result
    assert "Thread 1" in result


def test_compress_digest_truncates_narrative():
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir
    long_narrative = "X" * 500
    m = Memoir(
        narrative=long_narrative,
        key_insights=json.dumps(["Insight"]),
        open_threads=json.dumps([]),
    )
    result = compress_memoir(m, CompressionLevel.DIGEST)
    assert "..." in result
    # Truncated to 300 chars + ellipsis
    narrative_part = [p for p in result.split("\n\n") if p.startswith("Summary:")][0]
    assert len(narrative_part) < 400  # much less than 500


def test_compress_key_only_omits_narrative():
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir
    m = Memoir(
        narrative="This should NOT appear.",
        key_insights=json.dumps(["Key thing"]),
        open_threads=json.dumps(["Unresolved thread"]),
    )
    result = compress_memoir(m, CompressionLevel.KEY_ONLY)
    assert "This should NOT appear." not in result


def test_compress_key_only_preserves_open_threads():
    """Open threads must ALWAYS survive any compression level."""
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir
    m = Memoir(
        narrative="Narrative.",
        key_insights=json.dumps([]),
        open_threads=json.dumps(["What is the role of suffering?"]),
    )
    result = compress_memoir(m, CompressionLevel.KEY_ONLY)
    assert "What is the role of suffering?" in result


# ── MemoirWriter tests ───────────────────────────────────────────────────────

def _make_mock_messages_query(msgs):
    """Build mock db.query(...).filter_by(...).order_by(...).all() chain."""
    mock_q = MagicMock()
    mock_q.filter_by.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.all.return_value = msgs
    mock_q.first.return_value = None  # No existing ConversationSummary
    return mock_q


def test_memoir_has_all_required_sections(monkeypatch):
    """generate() must return a Memoir with narrative, key_insights, open_threads."""
    from audiobench.memory.memoir_writer import MemoirWriter

    conv = _make_conversation()
    session = _make_session()
    msgs = _make_messages(["Tell me about resilience", "Resilience is key to growth"])

    # Mock db_session
    mock_db = MagicMock()
    mock_db.__enter__ = lambda self: mock_db
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_db.query.return_value = _make_mock_messages_query(msgs)
    mock_db.commit = MagicMock()
    mock_db.add = MagicMock()
    mock_db.get = MagicMock(return_value=session)

    monkeypatch.setattr("audiobench.memory.memoir_writer.db_session", lambda: mock_db)
    monkeypatch.setattr(
        "audiobench.memory.memoir_writer.MemoirWriter._call_llm",
        lambda self, prompt, title: __import__("audiobench.memory.memoir_writer", fromlist=["Memoir"]).Memoir(
            narrative="This session explored resilience. The participant reflected on challenges and growth. Multiple insights emerged during this meaningful conversation.",
            key_insights=json.dumps(["Resilience matters", "Growth requires challenge"]),
            open_threads=json.dumps(["What enables resilience under extreme stress?"]),
            refined_title="Resilience Session",
        )
    )

    # Mock ExpressionRepository.register
    mock_expr = MagicMock()
    mock_expr.id = 99
    monkeypatch.setattr(
        "audiobench.storage.expression_repository.ExpressionRepository.register",
        lambda self, **kw: mock_expr,
    )

    memoir = MemoirWriter().generate(conv, session)

    assert len(memoir.narrative) > 50
    insights = json.loads(memoir.key_insights)
    threads = json.loads(memoir.open_threads)
    assert len(insights) >= 1
    assert isinstance(threads, list)


def test_memoir_is_stored_in_db(monkeypatch):
    """generate() must write a ConversationSummary to the DB."""
    from audiobench.memory.memoir_writer import Memoir, MemoirWriter

    conv = _make_conversation()
    session = _make_session()
    msgs = _make_messages(["Hello", "World"])

    added_objects = []
    mock_db = MagicMock()
    mock_db.__enter__ = lambda self: mock_db
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_db.query.return_value = _make_mock_messages_query(msgs)
    mock_db.commit = MagicMock()
    mock_db.add = lambda obj: added_objects.append(obj)
    mock_db.get = MagicMock(return_value=session)

    monkeypatch.setattr("audiobench.memory.memoir_writer.db_session", lambda: mock_db)
    monkeypatch.setattr(
        "audiobench.memory.memoir_writer.MemoirWriter._call_llm",
        lambda self, prompt, title: Memoir(
            narrative="Session narrative about meaningful topics and important reflections that happened during the conversation.",
            key_insights=json.dumps(["Insight 1"]),
            open_threads=json.dumps([]),
            refined_title="Test Session",
        )
    )
    mock_expr = MagicMock()
    mock_expr.id = 42
    monkeypatch.setattr(
        "audiobench.storage.expression_repository.ExpressionRepository.register",
        lambda self, **kw: mock_expr,
    )

    MemoirWriter().generate(conv, session)

    # At least one ConversationSummary was added
    from audiobench.storage.models import ConversationSummary
    summary_adds = [o for o in added_objects if isinstance(o, ConversationSummary)]
    assert len(summary_adds) == 1
    assert summary_adds[0].conversation_id == conv.id


def test_memoir_registered_as_expression(monkeypatch):
    """generate() must call ExpressionRepository.register with source_type='session_memoir'."""
    from audiobench.memory.memoir_writer import Memoir, MemoirWriter

    conv = _make_conversation()
    session = _make_session()
    msgs = _make_messages(["Question", "Answer"])

    mock_db = MagicMock()
    mock_db.__enter__ = lambda self: mock_db
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_db.query.return_value = _make_mock_messages_query(msgs)
    mock_db.commit = MagicMock()
    mock_db.add = MagicMock()
    mock_db.get = MagicMock(return_value=session)

    monkeypatch.setattr("audiobench.memory.memoir_writer.db_session", lambda: mock_db)
    monkeypatch.setattr(
        "audiobench.memory.memoir_writer.MemoirWriter._call_llm",
        lambda self, prompt, title: Memoir(
            narrative="Meaningful session narrative with sufficient length for testing purposes and validation.",
            key_insights=json.dumps(["Key thing"]),
            open_threads=json.dumps([]),
            refined_title=title,
        )
    )

    register_calls = []
    mock_expr = MagicMock()
    mock_expr.id = 77

    def mock_register(self, *, content, source_type, **kw):
        register_calls.append({"content": content, "source_type": source_type, **kw})
        return mock_expr

    monkeypatch.setattr(
        "audiobench.storage.expression_repository.ExpressionRepository.register",
        mock_register,
    )

    MemoirWriter().generate(conv, session)

    assert any(c["source_type"] == "session_memoir" for c in register_calls)
    memoir_call = next(c for c in register_calls if c["source_type"] == "session_memoir")
    assert len(memoir_call["content"]) > 50


def test_memoir_open_threads_survive_compression(monkeypatch):
    """Open threads from a memoir must appear after KEY_ONLY compression."""
    from audiobench.memory.memoir_writer import CompressionLevel, Memoir, compress_memoir

    old_session_memoir = Memoir(
        narrative="Old session narrative.",
        key_insights=json.dumps(["Insight"]),
        open_threads=json.dumps(["What is the role of suffering?"]),
    )
    compressed = compress_memoir(old_session_memoir, CompressionLevel.KEY_ONLY)
    assert "What is the role of suffering?" in compressed


def test_fallback_memoir_is_valid():
    """_fallback_memoir must return a valid Memoir even when LLM is broken."""
    from audiobench.memory.memoir_writer import MemoirWriter
    writer = MemoirWriter()
    memoir = writer._fallback_memoir("Test Session Title")
    assert isinstance(memoir.narrative, str)
    assert len(memoir.narrative) > 30
    # Must be valid JSON
    assert isinstance(json.loads(memoir.key_insights), list)
    assert isinstance(json.loads(memoir.open_threads), list)


def test_parse_llm_response_handles_valid_json():
    from audiobench.memory.memoir_writer import MemoirWriter
    writer = MemoirWriter()
    memoir = writer._parse_llm_response(VALID_MEMOIR_JSON, "Test Session")
    assert len(memoir.narrative.split()) >= 20
    insights = json.loads(memoir.key_insights)
    assert len(insights) >= 1


def test_parse_llm_response_handles_markdown_fence():
    from audiobench.memory.memoir_writer import MemoirWriter
    writer = MemoirWriter()
    fenced = f"```json\n{VALID_MEMOIR_JSON}\n```"
    memoir = writer._parse_llm_response(fenced, "Test Session")
    assert len(memoir.narrative) > 50


def test_parse_llm_response_falls_back_on_garbage():
    from audiobench.memory.memoir_writer import MemoirWriter
    writer = MemoirWriter()
    memoir = writer._parse_llm_response("not JSON at all {{{{ broken", "Fallback Session")
    # Must not raise; must return a valid Memoir
    assert isinstance(memoir.narrative, str)
    assert len(memoir.narrative) > 0
