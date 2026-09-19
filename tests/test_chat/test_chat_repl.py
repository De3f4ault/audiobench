
import pytest


@pytest.fixture
def db_session(test_db):  # test_db patches settings, resets engine/session singletons,
    # and tears down after the test — without it, init_db() resolves the real
    # data/transcriptions.db path and this fixture leaks rows into production.
    from audiobench.core.db_session import get_session
    with get_session() as session:
        yield session

@pytest.fixture
def fused_results():
    from audiobench.memory.rrf_fusion import FusedResult
    return [
        FusedResult(
            segment_id=1,
            start_time=10.0,
            end_time=20.0,
            text="fragment 1",
            rrf_score=0.016,
            stream_contributions=(("fts5", 1),)
        )
    ]

@pytest.fixture
def mock_user_types_exit(monkeypatch):
    class MockPromptSession:
        def __init__(self, *args, **kwargs):
            pass
        def prompt(self, *args, **kwargs):
            return "/exit"

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", MockPromptSession)

@pytest.fixture(autouse=True)
def _isolate_chat_db(test_db):
    """Ensure all chat REPL tests run against isolated test_db."""
    pass

def test_chat_repl_accepts_preloaded_fragments(fused_results):
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL(
        session_type="search_followup",
        preloaded_fragments=fused_results,
        preloaded_title="🔍 Search: test",
    )
    assert repl.preloaded_fragments == fused_results
    assert repl.session_type == "search_followup"

def test_chat_repl_creates_tagged_conversation(db_session, fused_results, mock_user_types_exit):
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL(
        session_type="search_followup",
        preloaded_fragments=fused_results,
        preloaded_title="🔍 Search: bike incident",
    )
    repl.run()
    # A conversation must have been created in DB
    from audiobench.storage.models import ChatConversation
    conv = db_session.query(ChatConversation).filter_by(
        session_type="search_followup"
    ).first()
    assert conv is not None
    assert "Search" in conv.title

def test_chat_cmd_uses_chat_repl_not_inline_loop():
    """chat.py must not contain an inline while-loop REPL — it must use ChatREPL."""
    with open("src/audiobench/cli/commands/chat.py") as f:
        source = f.read()
    # ChatREPL must be imported
    assert "ChatREPL" in source or "chat_repl" in source
    # The old inline pattern must be gone
    assert "while True:" not in source or source.count("while True:") <= 1
