"""Tests verifying architectural invariants across STT model upgrades and segment purges."""

from __future__ import annotations

import io
from pathlib import Path
import re
import pytest
from rich.console import Console
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.cli.commands.info_cmd import render_file_dossier
from audiobench.chat.chat_session import ChatSession
from audiobench.chat.chat_store import ChatRepository
from audiobench.memory.knowledge_ingester import KnowledgeIngester
from audiobench.storage.models import (
    AudioFileRecord,
    ChatConversation,
    ChatMessage,
    NoteCapture,
    NoteCollection,
    SegmentRecord,
    TranscriptionRecord,
)
from audiobench.storage.repository import TranscriptionRepository


@pytest.fixture
def setup_upgrade_scenario(test_db):
    """
    Setup an audio file with an active transcription, notes, and chat conversations.
    """
    with get_session() as session:
        audio = AudioFileRecord(
            id=2001,
            file_path="/tmp/fake_library/philosophy_audiobook.mp3",
            file_name="philosophy_audiobook.mp3",
            file_size_bytes=50000,
            duration_seconds=7200.0,
        )
        session.add(audio)
        session.flush()

        tx = TranscriptionRecord(
            id=7001,
            audio_file_id=audio.id,
            file_name="philosophy_audiobook.mp3",
            full_text="Existence precedes essence. Camus and the myth of Sisyphus.",
            raw_text="Existence precedes essence. Camus and the myth of Sisyphus.",
            duration_seconds=7200.0,
            pipeline_phase="complete",
        )
        session.add(tx)
        session.flush()

        s1 = SegmentRecord(
            id=12001,
            transcription_id=tx.id,
            segment_index=0,
            start_time=150.0,
            end_time=165.0,
            text="Existence precedes essence in existential philosophy.",
        )
        s2 = SegmentRecord(
            id=12002,
            transcription_id=tx.id,
            segment_index=1,
            start_time=320.0,
            end_time=340.0,
            text="Camus and the myth of Sisyphus define the absurd.",
        )
        session.add_all([s1, s2])
        session.flush()

        # Create NoteCollection anchored to audio_file_id
        col = NoteCollection(
            id=301,
            title="Reflections on Existentialism",
            audio_file_id=audio.id,
        )
        session.add(col)
        session.flush()

        # Create NoteCapture with segment_id=None (detached after upgrade)
        note = NoteCapture(
            id=401,
            collection_id=col.id,
            segment_id=None,
            body="Crucial note: Camus and the myth of Sisyphus define the absurd hero.",
        )
        session.add(note)

        # Create ChatConversation bound to audio_file_id but with empty transcript_ids
        chat = ChatConversation(
            id=501,
            title="Existential Inquiry",
            audio_file_id=audio.id,
            transcript_ids="[]",
            message_count=2,
        )
        session.add(chat)
        session.flush()

        # Create ChatMessage with no segment_id and conversational intro
        msg = ChatMessage(
            id=601,
            conversation_id=chat.id,
            role="assistant",
            content="As discussed, existence precedes essence in existential philosophy.",
            audio_path=audio.file_path,
            segment_id=None,
        )
        session.add(msg)
        session.commit()

    return {
        "audio_id": 2001,
        "tx_id": 7001,
        "note_id": 401,
        "chat_id": 501,
        "msg_id": 601,
    }


def test_info_cmd_notes_survive_segment_purge(setup_upgrade_scenario):
    """Verify that notes with segment_id=None still appear in render_file_dossier with resolved timestamp."""
    sio = io.StringIO()
    test_console = Console(file=sio, force_terminal=False, color_system=None)

    render_file_dossier(str(setup_upgrade_scenario["audio_id"]), target_console=test_console)
    output = sio.getvalue()

    # Note #401 must appear despite segment_id being None
    assert "Note #401" in output
    assert "Crucial note: Camus and the myth" in output
    # And timestamp was approximated from active transcript via sliding window probe
    assert "@~05:20" in output


def test_info_cmd_chats_scoped_by_audio_file_id(setup_upgrade_scenario):
    """Verify that chats referencing audio_file_id appear in render_file_dossier even if transcript_ids is []."""
    sio = io.StringIO()
    test_console = Console(file=sio, force_terminal=False, color_system=None)

    render_file_dossier(str(setup_upgrade_scenario["audio_id"]), target_console=test_console)
    output = sio.getvalue()

    assert "Chat #501" in output
    assert "Existential Inquiry" in output


def test_chat_session_restore_adapts_to_latest_transcript(setup_upgrade_scenario):
    """Verify that ChatSession.restore_from_db adapts to latest transcript if original transcript_ids was lost."""
    tx_repo = TranscriptionRepository()
    chat_session = ChatSession(model="test-model")

    # Point chat 501 to a nonexistent transcript ID 99999
    with get_session() as session:
        c = session.query(ChatConversation).filter_by(id=setup_upgrade_scenario["chat_id"]).first()
        c.transcript_ids = "[99999]"
        session.commit()

    # Restore session
    chat_session._conversation_id = setup_upgrade_scenario["chat_id"]
    success = chat_session.restore_from_db(tx_repo=tx_repo)
    assert success is True

    # Check that it adapted to transcript #7001 via audio_file_id
    assert 7001 in chat_session._transcript_ids
    assert len(chat_session._transcripts) == 1
    assert chat_session._transcripts[0]["id"] == 7001


def test_knowledge_ingester_inherits_tier_from_audio_file(setup_upgrade_scenario):
    """Verify KnowledgeIngester infers privacy tier via audio_file_id when transcript_ids is empty."""
    ingester = KnowledgeIngester()

    with get_session() as session:
        chat = session.query(ChatConversation).filter_by(id=setup_upgrade_scenario["chat_id"]).first()
        msg = session.query(ChatMessage).filter_by(id=setup_upgrade_scenario["msg_id"]).first()

        # Set privacy tier on the active transcription segments
        session.execute(
            sql_text("UPDATE segments SET privacy_tier = 2 WHERE transcription_id = :tx_id"),
            {"tx_id": setup_upgrade_scenario["tx_id"]},
        )
        session.commit()

        # Ingest message
        expr = ingester.ingest_chat_message(msg, chat)
        assert expr is not None
        assert expr.privacy_tier == 2


def test_play_msg_content_probe_recovers_position(setup_upgrade_scenario):
    """Verify that /play msg content probe recovers start_time when segment_id is dead."""
    msg_id = setup_upgrade_scenario["msg_id"]
    with get_session() as db:
        msg = db.query(ChatMessage).filter_by(id=msg_id).first()
        assert msg.segment_id is None
        audio_path = msg.audio_path
        msg_content = msg.content

        # Probe logic matching chat_repl
        audio_rec = db.query(AudioFileRecord).filter_by(file_path=audio_path).first()
        assert audio_rec is not None

        active_tx = (
            db.query(TranscriptionRecord)
            .filter(
                TranscriptionRecord.audio_file_id == audio_rec.id,
                TranscriptionRecord.pipeline_phase.in_(["complete", "completed"]),
            )
            .order_by(TranscriptionRecord.id.desc())
            .first()
        )
        assert active_tx is not None

        matched_seg = None
        # 1. Quotes first
        quotes = re.findall(r'["\']([^"\']{10,})["\']', msg_content)
        for q in quotes:
            q_words = [w for w in re.sub(r"[^\w\s]", " ", q).split() if len(w) > 3][:4]
            if len(q_words) >= 2:
                pat = "%" + "%".join(q_words) + "%"
                matched_seg = (
                    db.query(SegmentRecord)
                    .filter(
                        SegmentRecord.transcription_id == active_tx.id,
                        SegmentRecord.text.like(pat),
                    )
                    .order_by(SegmentRecord.start_time)
                    .first()
                )
                if matched_seg:
                    break

        # 2. Sliding word windows
        if not matched_seg:
            clean_probe = re.sub(r"[^\w\s]", " ", msg_content.strip()[:120]).strip()
            words = [w for w in clean_probe.split() if len(w) > 3]
            if len(words) >= 3:
                for i in range(min(5, len(words) - 2)):
                    pat = "%" + "%".join(words[i : i + 3]) + "%"
                    matched_seg = (
                        db.query(SegmentRecord)
                        .filter(
                            SegmentRecord.transcription_id == active_tx.id,
                            SegmentRecord.text.like(pat),
                        )
                        .order_by(SegmentRecord.start_time)
                        .first()
                    )
                    if matched_seg:
                        break

        assert matched_seg is not None
        assert matched_seg.id == 12001
        assert matched_seg.start_time == 150.0

        # Self-heal
        msg.segment_id = matched_seg.id
        db.commit()

    # Verify self-healing persisted
    with get_session() as db:
        healed_msg = db.query(ChatMessage).filter_by(id=msg_id).first()
        assert healed_msg.segment_id == 12001
