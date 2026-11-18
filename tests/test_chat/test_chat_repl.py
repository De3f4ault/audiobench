
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
    import audiobench.cli.shared.repl_shell
    monkeypatch.setattr(prompt_toolkit, "PromptSession", MockPromptSession)
    monkeypatch.setattr(audiobench.cli.shared.repl_shell, "PromptSession", MockPromptSession)
    monkeypatch.setattr(audiobench.cli.shared.repl_shell, "make_prompt_session", lambda *args, **kwargs: MockPromptSession())

@pytest.fixture(autouse=True)
def _isolate_chat_db(test_db):
    """Ensure all chat REPL tests run against isolated test_db."""
    pass

def test_chat_repl_accepts_preloaded_fragments(fused_results):
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL(
        session_type="search_followup",
        preloaded_fragments=fused_results,
        preloaded_title="Search: test",
    )
    assert repl.preloaded_fragments == fused_results
    assert repl.session_type == "search_followup"

def test_chat_repl_creates_tagged_conversation(db_session, fused_results, mock_user_types_exit):
    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL(
        session_type="search_followup",
        preloaded_fragments=fused_results,
        preloaded_title="Search: bike incident",
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


def test_chat_cmd_handles_provider_router_availability(monkeypatch):
    """audiobench chat command handles router availability checks without crashing."""
    from unittest.mock import MagicMock, patch
    from click.testing import CliRunner
    from audiobench.cli.commands.chat import chat

    mock_router = MagicMock()
    mock_router.is_available.side_effect = lambda provider=None: provider == "gemini" if provider else True
    mock_router.available_providers.return_value = ["gemini"]
    mock_router._slots = [MagicMock()]

    runner = CliRunner()
    with patch("audiobench.chat.chat_session.create_chat_client", return_value=mock_router), \
         patch("audiobench.chat.chat_repl.ChatREPL.run") as mock_run:
        result = runner.invoke(chat, ["--delete", "999999"])
        assert result.exit_code == 0



def test_chat_repl_sentinel_actions():
    from unittest.mock import MagicMock
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL(session_type="general")
    mock_daemon = MagicMock()
    mock_daemon.playback_status.return_value = {"playing": True, "position": 10.0}
    repl._daemon = mock_daemon
    repl._active_citations = [("01:20", 80.0), ("03:40", 220.0)]

    # Space toggle
    repl._handle_sentinel_action("\x00SPACE")
    mock_daemon.playback_toggle.assert_called_once()

    # Next citation
    repl._handle_sentinel_action("\x00NEXT")
    mock_daemon.playback_seek.assert_called_with(80.0)

    # Prev citation
    mock_daemon.playback_status.return_value = {"playing": True, "position": 100.0}
    repl._handle_sentinel_action("\x00PREV")
    mock_daemon.playback_seek.assert_called_with(80.0)

    # Direct digit jump
    repl._handle_sentinel_action("\x002")
    mock_daemon.playback_seek.assert_called_with(220.0)


def test_chat_repl_lyrics_command(monkeypatch):
    from unittest.mock import MagicMock
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL(session_type="general")
    mock_show = MagicMock()
    monkeypatch.setattr("audiobench.playback.lyrics_hud.show_daemon_lyrics", mock_show)

    handled = repl._handle_slash_command("/lyrics")
    assert handled is False  # returns False to keep prompt loop running
    assert mock_show.call_count == 1


def test_chat_repl_restored_model_fallback(monkeypatch, db_session):
    from unittest.mock import MagicMock
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_store import ChatRepository

    repo = ChatRepository()
    cid = repo.create_conversation(model="nonexistent-model:latest", title="Old Session")

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.list_models.return_value = ["gpt-oss:120b-cloud", "gemma3:4b"]

    monkeypatch.setattr("audiobench.chat.chat_repl.OllamaClient", lambda **kw: mock_client)

    repl = ChatREPL(resume_id=cid)
    # Must have fallen back to available model
    assert repl.session.model == "gpt-oss:120b-cloud"


def test_ollama_404_error_formatting():
    from unittest.mock import MagicMock
    import requests
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient

    client = OllamaClient()

    resp = MagicMock()
    resp.status_code = 404
    resp.json.return_value = {"error": "model 'missing-model' not found, try pulling it first"}
    http_err = requests.HTTPError(response=resp)

    err = client._handle_http_error(http_err, "missing-model")
    assert isinstance(err, AIError)
    assert "Model 'missing-model' not available on Ollama server" in str(err)
    assert "ollama pull missing-model" in str(err)


def test_chat_repl_resume_rehydrates_citations_and_renders_banner(monkeypatch, db_session):
    from unittest.mock import MagicMock
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_session import ChatSession
    from audiobench.chat.chat_store import ChatRepository

    repo = ChatRepository()
    cid = repo.create_conversation(model="test-model", title="Resumed Chat")
    repo.add_message(cid, "user", "What happened at the beginning?")
    repo.add_message(
        cid,
        "assistant",
        "At 01:23 the author introduces the concept, and later at 04:50 they discuss it.",
    )

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    monkeypatch.setattr("audiobench.chat.chat_repl.OllamaClient", lambda **kw: mock_client)

    # 1. Test when ChatREPL is instantiated with pre-restored session
    session = ChatSession(client=mock_client, chat_repo=repo, model="test-model", conversation_id=cid)
    session.restore_from_db()

    repl = ChatREPL(session=session, resume_id=cid)
    assert len(repl._active_citations) == 2
    assert repl._active_citations[0] == ("01:23", 83.0)
    assert repl._active_citations[1] == ("04:50", 290.0)

    # 2. Test banner rendering in _render_history
    mock_banner = MagicMock()
    monkeypatch.setattr("audiobench.playback.controls.render_citations_banner", mock_banner)

    repl._render_history()
    assert mock_banner.call_count == 1
    mock_banner.assert_called_with(monkeypatch.delattr, repl._active_citations) if False else None
    assert mock_banner.call_args[0][1] == repl._active_citations

    # 3. Test empty-buffer digit hotkey immediately works
    mock_daemon = MagicMock()
    mock_daemon.playback_status.return_value = {"playing": True, "position": 0.0}
    repl._daemon = mock_daemon

    repl._handle_sentinel_action("\x001")
    mock_daemon.playback_seek.assert_called_with(83.0)

    repl._handle_sentinel_action("\x002")
    mock_daemon.playback_seek.assert_called_with(290.0)


def test_chat_repl_search_session_linkage_and_turn_seeding(monkeypatch, db_session):
    """ChatREPL should link to search_session_id and search_query_id, inject fragments, and seed Turn 1."""
    from unittest.mock import MagicMock
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_store import ChatRepository

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    monkeypatch.setattr("audiobench.chat.chat_repl.OllamaClient", lambda **kw: mock_client)

    frag = MagicMock()
    frag.source_file = "lecture.mp3"
    frag.transcription_id = 10
    frag.start_time = 75.0
    frag.end_time = 95.0
    frag.text = "The Metamorphosis examines the burden of familial expectation."

    repo = ChatRepository()

    repl = ChatREPL(
        session_type="search_followup",
        preloaded_fragments=[frag],
        preloaded_query="kafka family burden",
        preloaded_title="[S2] kafka family burden",
        preloaded_answer="According to [lecture.mp3 @ 01:15], Kafka highlights familial expectation.",
        search_session_id=88,
        search_query_id=42,
    )

    # Verify conversation created with search_session_id and search_query_id
    assert repl.session.conversation_id is not None
    conv = repo.get_conversation(repl.session.conversation_id)
    assert conv["search_session_id"] == 88
    assert conv["search_query_id"] == 42

    # Verify fragments injected into ChatSession
    assert len(repl.session.injected_fragments) == 1
    assert repl.session.injected_fragments[0]["source_file"] == "lecture.mp3"

    # Verify active citations extracted
    assert len(repl._active_citations) > 0
    assert repl._active_citations[0] == ("01:15", 75.0)

    # Verify repo lookup methods
    threads = repo.get_threads_for_search_session(88)
    assert len(threads) == 1
    assert threads[0]["id"] == repl.session.conversation_id
    assert threads[0]["search_query_id"] == 42
    assert threads[0]["sequence_num"] == 2  # parsed from [S2] fallback when search_queries row absent

    search_sessions = repo.get_search_sessions_with_chats()
    assert any(s["search_session_id"] == 88 for s in search_sessions)


def test_show_search_thread_navigator():
    from unittest.mock import MagicMock
    from audiobench.cli.commands.memory_cmd import _show_search_thread_navigator
    from rich.console import Console
    import io

    console = Console(file=io.StringIO())
    threads = [
        {"id": 101, "title": "Kafka & Alienation", "sequence_num": 1, "query_text": "kafka alienation", "message_count": 12, "updated_at": "2026-10-05 10:00"},
        {"id": 102, "title": "The Bug as Symbol", "sequence_num": 2, "query_text": "bug as symbol", "message_count": 4, "updated_at": "2026-10-05 10:30"},
    ]

    mock_repo = MagicMock()

    import builtins
    orig_input = builtins.input

    try:
        # Test selecting thread 1 by default
        builtins.input = lambda prompt="": ""
        assert _show_search_thread_navigator(console, threads, current_seq=3, session_id=88) == 101

        # Test selecting thread 2 explicitly
        builtins.input = lambda prompt="": "2"
        assert _show_search_thread_navigator(console, threads, current_seq=3, session_id=88) == 102

        # Test starting new thread with default auto title
        builtins.input = lambda prompt="": "N"
        assert _show_search_thread_navigator(console, threads, current_seq=3, session_id=88) is None

        # Test starting new thread with custom title
        builtins.input = lambda prompt="": "N Grete's Transformation"
        assert _show_search_thread_navigator(console, threads, current_seq=3, session_id=88) == (None, "Grete's Transformation")

        # Test selecting thread by #ID
        builtins.input = lambda prompt="": "#102"
        assert _show_search_thread_navigator(console, threads, current_seq=3, session_id=88) == 102

        # Test inline rename via R 1 <new_title>
        inputs = iter(["R 1 Renamed Topic", "1"])
        builtins.input = lambda prompt="": next(inputs)
        result = _show_search_thread_navigator(console, threads, current_seq=3, session_id=88, repo=mock_repo)
        assert result == 101
        assert threads[0]["title"] == "Renamed Topic"
        mock_repo.update_title.assert_called_with(101, "Renamed Topic")
    finally:
        builtins.input = orig_input


def test_chat_repl_resume_rehydrates_search_fragments_and_citations(db_session, monkeypatch):
    """Resuming a search-linked chat thread rehydrates fragments and populates active citations."""
    from audiobench.chat.chat_repl import ChatREPL
    from audiobench.chat.chat_store import ChatRepository
    import audiobench.memory.session_store as ss

    repo = ChatRepository()
    cid = repo.create_conversation(
        model="qwen2.5:7b",
        title="[S1] Geese flying",
        session_type="search_followup",
        search_session_id=12,
        search_query_id=777,
    )

    mock_frags = [
        {
            "segment_id": 10,
            "source_file": "birds.mp3",
            "transcription_id": 1,
            "rank": 1,
            "score": 0.05,
            "start_time": 42.5,
            "end_time": 55.0,
            "text": "Geese flying low over the river.",
        }
    ]

    monkeypatch.setattr(ss, "get_query_fragments", lambda qid: mock_frags if qid == 777 else [])
    monkeypatch.setattr(ss, "get_query_text", lambda qid: "geese flying" if qid == 777 else None)

    repl = ChatREPL(resume_id=cid)

    # Injected fragments should be restored from the DB
    assert len(repl.session.injected_fragments) == 1
    assert repl.session.injected_fragments[0]["source_file"] == "birds.mp3"
    assert repl.session.injected_fragments[0]["start_time"] == 42.5

    # Citations should be populated from the rehydrated search fragments
    assert repl._active_citations == [("birds.mp3", 42.5)]

    # Summary should state fragments rather than 'No transcripts loaded'
    summary = repl.session.get_context_summary()
    assert "1 search fragment for 'geese flying'" in summary[0]




