"""Unit tests for voice mode and audio/segment coupling in Audiobench Chat."""

import pytest
from audiobench.storage.models import (
    AudioFileRecord,
    ChatConversation,
    ChatMessage,
    SegmentRecord,
    TranscriptionRecord,
)
from audiobench.chat.chat_store import ChatRepository
from audiobench.storage.repository import TranscriptionRepository
from audiobench.chat.command_meta import get_command_meta, COMMAND_REGISTRY
from audiobench.core.db_session import get_session


@pytest.fixture(autouse=True)
def _isolate_db(test_db):
    """Ensure all tests run against isolated test_db."""
    pass


def test_voice_fields_in_chat_models(test_db):
    """Test that voice coupling columns persist and map correctly in Chat models."""
    from audiobench.core.db_session import get_session

    with get_session() as session:
        # Create audio file and transcription
        audio = AudioFileRecord(file_path="/tmp/fake_voice.wav", file_name="fake_voice.wav", duration_seconds=12.0)
        session.add(audio)
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=audio.id,
            source="voice_chat",
            file_name="Voice Chat #1",
            duration_seconds=12.0,
        )
        session.add(tx)
        session.flush()

        seg = SegmentRecord(
            transcription_id=tx.id,
            segment_index=0,
            text="Hello from voice",
            start_time=0.0,
            end_time=3.5,
            speaker="User",
        )
        session.add(seg)
        session.flush()

        # Create conversation with links
        conv = ChatConversation(
            title="Voice Test Session",
            model_name="gemini-2.5-flash",
            audio_file_id=audio.id,
            transcription_id=tx.id,
        )
        session.add(conv)
        session.flush()

        # Create message with voice metadata
        msg = ChatMessage(
            conversation_id=conv.id,
            role="user",
            content="Hello from voice",
            modality="voice",
            duration_seconds=3.5,
            audio_path="/tmp/fake_voice.wav",
            interrupted=0,
            voice_name="Puck",
            segment_id=seg.id,
        )
        session.add(msg)
        session.commit()

        # Query back and verify
        saved_msg = session.query(ChatMessage).filter_by(id=msg.id).first()
        assert saved_msg is not None
        assert saved_msg.modality == "voice"
        assert saved_msg.duration_seconds == 3.5
        assert saved_msg.audio_path == "/tmp/fake_voice.wav"
        assert saved_msg.interrupted == 0
        assert saved_msg.voice_name == "Puck"
        assert saved_msg.segment_id == seg.id
        assert saved_msg.segment.text == "Hello from voice"

        saved_conv = session.query(ChatConversation).filter_by(id=conv.id).first()
        assert saved_conv.audio_file_id == audio.id
        assert saved_conv.transcription_id == tx.id


def test_chat_repository_voice_methods(test_db):
    """Test ChatRepository add_message voice arguments and link_voice_session."""
    with get_session() as session:
        audio = AudioFileRecord(
            file_path="/tmp/session_1.wav",
            file_name="session_1.wav",
            file_hash="hash_session_1",
            file_size_bytes=1024,
            duration_seconds=5.2,
        )
        session.add(audio)
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=audio.id,
            source="voice_chat",
            file_name="Voice Chat #1",
            duration_seconds=5.2,
        )
        session.add(tx)
        session.flush()

        seg = SegmentRecord(
            transcription_id=tx.id,
            segment_index=0,
            text="Spoken answer from AI",
            start_time=0.0,
            end_time=5.2,
            speaker="Assistant",
        )
        session.add(seg)
        session.commit()

        audio_id = audio.id
        tx_id = tx.id
        seg_id = seg.id

    chat_repo = ChatRepository()
    conv_id = chat_repo.create_conversation(model="gemini-2.5-flash", title="Voice Repo Test")

    msg_id = chat_repo.add_message(
        conversation_id=conv_id,
        role="assistant",
        content="Spoken answer from AI",
        modality="voice_live",
        duration_seconds=5.2,
        audio_path="/tmp/session_1.wav",
        interrupted=True,
        voice_name="Puck",
        segment_id=seg_id,
    )
    assert msg_id > 0

    chat_repo.link_voice_session(
        conversation_id=conv_id,
        audio_file_id=audio_id,
        transcription_id=tx_id,
    )

    conv_data = chat_repo.get_conversation(conv_id)
    assert conv_data is not None
    assert conv_data["audio_file_id"] == audio_id
    assert conv_data["transcription_id"] == tx_id
    assert len(conv_data["messages"]) == 1

    msg_data = conv_data["messages"][0]
    assert msg_data["modality"] == "voice_live"
    assert msg_data["duration_seconds"] == 5.2
    assert msg_data["audio_path"] == "/tmp/session_1.wav"
    assert msg_data["interrupted"] is True
    assert msg_data["voice_name"] == "Puck"
    assert msg_data["segment_id"] == seg_id


def test_save_voice_chat_session_in_repository(test_db, tmp_path):
    """Test TranscriptionRepository.save_voice_chat_session creates all records and links."""
    # Create a real small WAV file on disk to test AudioFileRecord creation
    fake_wav = tmp_path / "voice_session_99.wav"
    fake_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    tx_repo = TranscriptionRepository()
    turns = [
        {"role": "user", "text": "What is the capital of Kenya?", "start_time": 0.0, "end_time": 2.5, "speaker": "User"},
        {"role": "assistant", "text": "The capital is Nairobi.", "start_time": 3.0, "end_time": 5.5, "speaker": "Assistant"},
    ]

    audio_fid, tx_id, seg_ids = tx_repo.save_voice_chat_session(
        conversation_id=99,
        audio_file_path=str(fake_wav),
        turns=turns,
        duration_seconds=5.5,
        model_name="gemini-2.5-flash",
    )

    assert audio_fid is not None
    assert tx_id > 0
    assert len(seg_ids) == 2

    # Verify transcription
    record = tx_repo.get_by_id(tx_id, session=None)
    assert record is not None
    assert record["source"] == "voice_chat"
    assert "Voice Chat #99" in record["file_name"]
    assert record["segment_count"] == 2

    # Verify segments
    segments = record["segments"]
    assert len(segments) == 2
    assert segments[0]["text"] == "What is the capital of Kenya?"
    assert segments[0]["speaker"] == "User"
    assert segments[1]["text"] == "The capital is Nairobi."
    assert segments[1]["speaker"] == "Assistant"


def test_voice_command_registered_in_meta():
    """Verify /voice slash command is registered in command metadata."""
    cmd = get_command_meta("/voice")
    assert cmd is not None
    assert "/v" in cmd.aliases
    assert cmd.name == "/voice"
    assert "live" in cmd.choices
    assert "status" in cmd.choices


def test_chat_repl_voice_status_command(test_db):
    """Verify /voice status executes cleanly without errors."""
    from audiobench.chat.chat_repl import ChatREPL

    repl = ChatREPL()
    # Executing /voice status should return False (do not exit REPL)
    res = repl._handle_slash_command("/voice status")
    assert res is False
