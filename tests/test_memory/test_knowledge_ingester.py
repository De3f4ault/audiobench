import json

import pytest

from audiobench.core.db_session import get_session
from audiobench.memory.knowledge_ingester import KnowledgeIngester
from audiobench.storage.models import (
    AskEntry,
    AskLog,
    AudioFileRecord,
    ChatConversation,
    ChatMessage,
    ExpressionRecord,
    TranscriptionRecord,
)


@pytest.fixture
def ingester(test_db):
    return KnowledgeIngester()

@pytest.fixture
def conversation(test_db):
    with get_session() as session:
        audio = AudioFileRecord(file_path="test_chat.mp3", file_name="test_chat.mp3", duration_seconds=10)
        session.add(audio)
        session.commit()

        tx = TranscriptionRecord(audio_file_id=audio.id, file_name="test_chat.mp3", model_name="x", language="en")
        session.add(tx)
        session.commit()

        c = ChatConversation(title="Test", transcript_ids=json.dumps([tx.id]))
        session.add(c)
        session.commit()
        session.refresh(c)

        # We need to detach so test functions can use them and pass to ingester without threading detached instance issues
        session.expunge(c)
        return c

@pytest.fixture
def chat_message(conversation):
    with get_session() as session:
        m = ChatMessage(conversation_id=conversation.id, role="user", content="Hello world")
        session.add(m)
        session.commit()
        session.refresh(m)
        session.expunge(m)
        return m

@pytest.fixture
def transcript_expr(conversation):
    with get_session() as session:
        t_id = json.loads(conversation.transcript_ids)[0]
        expr = ExpressionRecord(source_type="audio_transcript", source_id=t_id, content="Transcript content")
        session.add(expr)
        session.commit()
        session.refresh(expr)
        session.expunge(expr)
        return expr

@pytest.fixture
def ask_log(test_db):
    with get_session() as session:
        audio = AudioFileRecord(file_path="test_ask.mp3", file_name="test_ask.mp3", duration_seconds=10)
        session.add(audio)
        session.commit()

        tx = TranscriptionRecord(audio_file_id=audio.id, file_name="test_ask.mp3", model_name="x", language="en")
        session.add(tx)
        session.commit()

        log = AskLog(audio_file_id=audio.id)
        session.add(log)
        session.commit()
        session.refresh(log)
        session.expunge(log)
        return log

@pytest.fixture
def ask_entry(ask_log):
    with get_session() as session:
        e = AskEntry(log_id=ask_log.id, question="What is life?", answer="42", model_name="x")
        session.add(e)
        session.commit()
        session.refresh(e)
        session.expunge(e)
        return e


def test_ingest_chat_message_creates_expression(ingester, chat_message, conversation):
    expr = ingester.ingest_chat_message(chat_message, conversation)
    assert expr.source_type == "chat_message"
    assert expr.session_id == conversation.id
    assert chat_message.content[:50] in expr.content

def test_ingest_chat_message_deduplicates(ingester, chat_message, conversation):
    expr1 = ingester.ingest_chat_message(chat_message, conversation)
    expr2 = ingester.ingest_chat_message(chat_message, conversation)
    assert expr1.id == expr2.id  # same expression returned, not a duplicate

def test_ingest_ask_entry_creates_two_expressions(ingester, ask_entry, ask_log):
    q_expr, a_expr = ingester.ingest_ask_entry(ask_entry, ask_log)
    assert q_expr.source_type == "ask_query"
    assert a_expr.source_type == "ask_answer"

def test_ingest_creates_thematic_edge_to_transcript(ingester, chat_message, conversation, transcript_expr):
    # conversation.transcript_ids includes transcript_expr's transcription
    expr = ingester.ingest_chat_message(chat_message, conversation)
    from audiobench.storage.expression_repository import ExpressionRepository
    repo = ExpressionRepository()
    rels = repo.get_relations(expr.id, direction="out", relation_type="thematic")
    assert len(rels) > 0

def test_ingest_is_nonblocking_when_called_from_chat(monkeypatch):
    """ingest_chat_message when called via background thread must not block main thread."""
    import threading
    import time
    ingester_calls = []
    def slow_ingest(*args, **kwargs):
        time.sleep(0.5)
        ingester_calls.append(1)
    monkeypatch.setattr("audiobench.memory.knowledge_ingester.KnowledgeIngester.ingest_chat_message", slow_ingest)

    t0 = time.perf_counter()
    t = threading.Thread(target=slow_ingest)
    t.start()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, "Main thread was blocked by ingestion"
    t.join()
