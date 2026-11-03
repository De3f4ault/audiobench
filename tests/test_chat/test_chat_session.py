"""Unit tests for ChatSession context building and timestamp formatting."""

from unittest.mock import MagicMock
import pytest

from audiobench.chat.chat_session import ChatSession, _format_transcript_context_text


def test_format_transcript_context_text_with_timestamps():
    """Segments with non-zero timestamps should format as [MM:SS] lines."""
    transcript = {
        "id": 101,
        "file_name": "interview.mp3",
        "word_count": 25,
        "segments": [
            {"start": 0.0, "end": 12.5, "text": "Welcome to the show.", "speaker": "Host"},
            {"start": 13.0, "end": 28.4, "text": "Delighted to be here.", "speaker": "Guest"},
            {"start": 65.2, "end": 90.0, "text": "Let us discuss the core idea.", "speaker": None},
        ],
        "full_text": "Welcome to the show. Delighted to be here. Let us discuss the core idea.",
    }

    formatted, has_ts = _format_transcript_context_text(transcript)
    assert has_ts is True
    assert "[00:00] [Host]: Welcome to the show." in formatted
    assert "[00:13] [Guest]: Delighted to be here." in formatted
    assert "[01:05] Let us discuss the core idea." in formatted


def test_format_transcript_context_text_fallback_when_all_zero():
    """Segments with all 0.0 timestamps should fall back to full_text without fake [00:00]."""
    transcript = {
        "id": 102,
        "file_name": "unaligned.mp3",
        "word_count": 10,
        "segments": [
            {"start": 0.0, "end": 0.0, "text": "First segment.", "speaker": None},
            {"start": 0.0, "end": 0.0, "text": "Second segment.", "speaker": None},
        ],
        "full_text": "First segment. Second segment.",
    }

    formatted, has_ts = _format_transcript_context_text(transcript)
    assert has_ts is False
    assert formatted == "First segment. Second segment."
    assert "[00:00]" not in formatted


def test_format_transcript_context_text_fallback_when_no_segments():
    """Transcripts without segments key should fall back to full_text."""
    transcript = {
        "id": 103,
        "file_name": "plain.mp3",
        "word_count": 4,
        "full_text": "Plain text transcript without segments.",
    }

    formatted, has_ts = _format_transcript_context_text(transcript)
    assert has_ts is False
    assert formatted == "Plain text transcript without segments."


def test_chat_session_rebuilds_system_prompt_with_timestamps():
    """ChatSession.load_transcripts should inject timestamped lines into _system_prompt."""
    client = MagicMock()
    repo = MagicMock()
    session = ChatSession(client=client, chat_repo=repo)

    transcript = {
        "id": 55,
        "file_name": "talk.mp3",
        "word_count": 20,
        "segments": [
            {"start": 5.0, "end": 15.0, "text": "First point.", "speaker": "Alice"},
            {"start": 75.0, "end": 90.0, "text": "Second point.", "speaker": "Bob"},
        ],
        "full_text": "First point. Second point.",
    }

    session.load_transcripts([transcript])

    assert "--- Transcript #55: talk.mp3" in session._system_prompt
    assert "[00:05] [Alice]: First point." in session._system_prompt
    assert "[01:15] [Bob]: Second point." in session._system_prompt


def test_chat_session_truncation_preserves_lines():
    """When context exceeds MAX_CONTEXT_WORDS, timestamped lines should be cleanly truncated."""
    client = MagicMock()
    repo = MagicMock()
    session = ChatSession(client=client, chat_repo=repo)

    # Create a transcript with many segments
    segments = [
        {"start": float(i * 10), "end": float(i * 10 + 9), "text": f"Sentence number {i} with some words.", "speaker": None}
        for i in range(1200)
    ]
    transcript = {
        "id": 77,
        "file_name": "long.mp3",
        "word_count": len(segments) * 6,
        "segments": segments,
        "full_text": " ".join(s["text"] for s in segments),
    }

    session.load_transcripts([transcript])

    assert "[... truncated]" in session._system_prompt
    assert "[00:00]" in session._system_prompt


def test_chat_session_rebuilds_system_prompt_with_injected_fragments_only():
    """ChatSession with injected fragments and no transcripts should build full system prompt."""
    client = MagicMock()
    repo = MagicMock()
    session = ChatSession(client=client, chat_repo=repo)

    # Mock fragment object (similar to FusedResult)
    frag = MagicMock()
    frag.source_file = "kafka_essay.mp3"
    frag.transcription_id = 42
    frag.start_time = 125.0
    frag.end_time = 145.0
    frag.text = "Gregor Samsa woke up to find himself transformed."

    session.inject_search_fragments([frag], query="kafka metamorphosis")

    assert "No transcripts are loaded yet" not in session._system_prompt
    assert "You are AudioBench's research companion" in session._system_prompt
    assert "--- INJECTED SEARCH FRAGMENTS ---" in session._system_prompt
    assert "kafka_essay.mp3 @ 2:05 → 2:25" in session._system_prompt
    assert "Gregor Samsa woke up to find himself transformed." in session._system_prompt

    summary = session.get_context_summary()
    assert len(summary) == 1
    assert "1 search fragment for 'kafka metamorphosis'" in summary[0]


def test_chat_session_seed_initial_turn():
    """ChatSession.seed_initial_turn should add user and assistant messages and persist them."""
    client = MagicMock()
    repo = MagicMock()
    repo.create_conversation.return_value = 99
    session = ChatSession(client=client, chat_repo=repo)

    session.seed_initial_turn("What is Kafka about?", "It is about alienation.")

    assert session.turn_count == 1
    assert len(session.messages) == 2
    assert session.messages[0] == {"role": "user", "content": "What is Kafka about?"}
    assert session.messages[1] == {"role": "assistant", "content": "It is about alienation."}
    assert repo.add_message.call_count == 3
    repo.add_message.assert_any_call(99, "system", session._system_prompt)
    repo.add_message.assert_any_call(99, "user", "What is Kafka about?")
    repo.add_message.assert_any_call(99, "assistant", "It is about alienation.")


def test_chat_session_preserves_step_prefix_on_auto_rename():
    """ChatSession._generate_title should preserve [S#] step prefix if present."""
    client = MagicMock()
    client.generate.return_value = "Micro Mirrors Macro Insight"
    repo = MagicMock()
    repo.get_conversation.return_value = {
        "id": 78,
        "title": "[S3] back to the idea of the micro contained in the macro",
    }

    session = ChatSession(
        client=client,
        chat_repo=repo,
        conversation_id=78,
        title="[S3] back to the idea of the micro contained in the macro",
    )

    session._generate_title("what is the main idea", "The microcosm mirrors...")

    repo.update_title.assert_called_once_with(78, "[S3] Micro Mirrors Macro Insight")


def test_chat_session_custom_title_prevents_auto_rename():
    """ChatSession with custom_title=True should mark title as generated and not overwrite."""
    client = MagicMock()
    repo = MagicMock()
    session = ChatSession(
        client=client,
        chat_repo=repo,
        conversation_id=79,
        title="[S4] Custom Handcrafted Title",
        custom_title=True,
    )

    assert session._title_generated is True
    # Calling generate_title_async should be a no-op because _title_generated is True
    session.generate_title_async("any message", "any response")
    repo.update_title.assert_not_called()


def test_restore_from_db_rehydrates_search_fragments(monkeypatch):
    """restore_from_db should rehydrate search fragments when search_query_id is present."""
    repo = MagicMock()
    repo.get_conversation.return_value = {
        "id": 99,
        "title": "[S1] geese migration patterns",
        "model": "qwen2.5:7b",
        "transcript_ids": [],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about the geese."},
            {"role": "assistant", "content": "They fly in V formation."},
        ],
        "search_session_id": 12,
        "search_query_id": 45,
    }

    mock_frags = [
        {
            "segment_id": 1,
            "source_file": "nature_doc.mp3",
            "transcription_id": 10,
            "rank": 1,
            "score": 0.033,
            "start_time": 12.0,
            "end_time": 25.0,
            "text": "Geese fly above the water in a graceful V formation.",
        },
        {
            "segment_id": 2,
            "source_file": "nature_doc.mp3",
            "transcription_id": 10,
            "rank": 2,
            "score": 0.028,
            "start_time": 30.0,
            "end_time": 45.0,
            "text": "The lead goose cuts the air resistance for the followers.",
        },
    ]

    import audiobench.memory.session_store as ss
    monkeypatch.setattr(ss, "get_query_fragments", lambda qid: mock_frags if qid == 45 else [])
    monkeypatch.setattr(ss, "get_query_text", lambda qid: "geese migration patterns" if qid == 45 else None)

    session = ChatSession(
        client=MagicMock(),
        chat_repo=repo,
        conversation_id=99,
    )

    success = session.restore_from_db()
    assert success is True
    assert len(session.injected_fragments) == 2
    assert session.injected_fragments[0]["source_file"] == "nature_doc.mp3"
    assert session.injected_fragments[0]["start_time"] == 12.0

    summary = session.get_context_summary()
    assert len(summary) == 1
    assert "2 search fragments for 'geese migration patterns'" in summary[0]
    assert "INJECTED SEARCH FRAGMENTS" in session._system_prompt
    assert "Geese fly above the water" in session._system_prompt


def test_restore_from_db_resolves_legacy_search_query_fallback(monkeypatch):
    """restore_from_db should resolve search_query_id from search_session_id and [S#] title."""
    repo = MagicMock()
    repo.get_conversation.return_value = {
        "id": 101,
        "title": "[S2] the metamorphosis essence",
        "model": "qwen2.5:7b",
        "transcript_ids": [],
        "messages": [],
        "search_session_id": 88,
        "search_query_id": None,  # Legacy thread without query_id
    }

    mock_frags = [
        {
            "segment_id": 99,
            "source_file": "kafka.mp3",
            "transcription_id": 5,
            "rank": 1,
            "score": 0.05,
            "start_time": 5.0,
            "end_time": 15.0,
            "text": "Gregor Samsa woke up transformed into an insect.",
        }
    ]

    import audiobench.memory.session_store as ss
    monkeypatch.setattr(
        ss,
        "resolve_query_id_for_session",
        lambda sid, sequence_num, title: 202 if sid == 88 and sequence_num == 2 else None,
    )
    monkeypatch.setattr(ss, "get_query_fragments", lambda qid: mock_frags if qid == 202 else [])
    monkeypatch.setattr(ss, "get_query_text", lambda qid: "the metamorphosis essence" if qid == 202 else None)

    session = ChatSession(
        client=MagicMock(),
        chat_repo=repo,
        conversation_id=101,
    )

    success = session.restore_from_db()
    assert success is True
    assert len(session.injected_fragments) == 1
    # Check that repo was updated with the resolved query ID
    repo.update_search_session_id.assert_called_once_with(101, 88, 202)
    assert "Gregor Samsa woke up" in session._system_prompt


