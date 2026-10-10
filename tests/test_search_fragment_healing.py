"""Tests for multi-tier search fragment healing and self-repairing SearchSessionSweep."""

from __future__ import annotations

import pytest
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.search_session_sweep import SearchSessionSweep
from audiobench.memory.enums import RelationType, SourceType
from audiobench.memory.search_ingester import SearchIngester
from audiobench.storage.expression_repository import ExpressionRepository


@pytest.fixture
def setup_healing_scenario(test_db):
    """
    Set up an audiobook, an upgraded transcription, expressions, and an outdated fragment.
    """
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord, SegmentRecord

    with get_session() as session:
        # 1. Create AudioFile
        audio = AudioFileRecord(
            id=1001,
            file_path="/path/to/the_prince.mp3",
            file_name="the_prince.mp3",
            file_size_bytes=10000,
            duration_seconds=3600.0,
        )
        session.add(audio)
        session.flush()

        # 2. Create Active Transcription
        tx = TranscriptionRecord(
            id=5001,
            audio_file_id=audio.id,
            file_name="the_prince.mp3",
            full_text="Full transcript text for The Prince.",
            raw_text="Full transcript text for The Prince.",
            duration_seconds=3600.0,
            pipeline_phase="complete",
        )
        session.add(tx)
        session.flush()

        # 3. Create active Segments
        s1 = SegmentRecord(
            id=9001,
            transcription_id=tx.id,
            segment_index=0,
            start_time=100.0,
            end_time=115.0,
            text="Machiavelli insisted on the originality of his approach.",
        )
        s2 = SegmentRecord(
            id=9002,
            transcription_id=tx.id,
            segment_index=1,
            start_time=200.0,
            end_time=215.0,
            text="Kamu and his notion of Sisyphusian futility.",
        )
        session.add_all([s1, s2])
        session.flush()
        # 4. Create Expressions for Transcription 5001
        # Chunk 1 (matches 9001 text)
        session.execute(
            sql_text(
                "INSERT INTO expressions (id, content, source_type, source_id, graph_role, privacy_tier) "
                "VALUES (8001, 'Machiavelli insisted on the originality of his approach to virtue.', 'audio_transcript', 5001, 'sweep_chunk', 0)"
            )
        )
        # Chunk 2 (matches 9002 acoustically at 200s, but has STT difference)
        session.execute(
            sql_text(
                "INSERT INTO expressions (id, content, source_type, source_id, graph_role, privacy_tier) "
                "VALUES (8002, 'Discussion on Camus and the philosophy of absurdity.', 'audio_transcript', 5001, 'sweep_chunk', 0)"
            )
        )
        # Document expression
        session.execute(
            sql_text(
                "INSERT INTO expressions (id, content, source_type, source_id, graph_role, privacy_tier) "
                "VALUES (8003, 'Full document for The Prince.', 'audio_transcript', 5001, 'sweep_document', 0)"
            )
        )
        # 5. Populate expression_segment_map for acoustic alignment
        session.execute(
            sql_text(
                "INSERT INTO expression_segment_map (expression_id, segment_id, created_at) "
                "VALUES (8001, 9001, '2026-10-10 00:00:00'), (8002, 9002, '2026-10-10 00:00:00')"
            )
        )
        # 6. Create Search Session and Query
        session.execute(
            sql_text(
                "INSERT INTO search_sessions (id, title) VALUES (301, 'Philosophy Session')"
            )
        )
        session.execute(
            sql_text(
                "INSERT INTO search_queries (id, session_id, query_text, sequence_num, preset) "
                "VALUES (401, 301, 'What did Machiavelli say about virtue?', 1, 'balanced')"
            )
        )
        # 7. Create Expressions for Query and Synthesis
        session.execute(
            sql_text(
                "INSERT INTO expressions (id, content, source_type, source_id, graph_role, privacy_tier) "
                "VALUES (8100, 'What did Machiavelli say about virtue?', 'search_query', 401, 'search_query', 0)"
            )
        )
        session.execute(
            sql_text(
                "INSERT INTO expressions (id, content, source_type, source_id, graph_role, privacy_tier) "
                "VALUES (8200, 'Machiavelli redefined virtue as political effectiveness.', 'search_synthesis', 401, 'search_synthesis', 0)"
            )
        )
        session.commit()

    return {
        "audio_id": 1001,
        "audio_path": "/path/to/the_prince.mp3",
        "tx_id": 5001,
        "query_id": 401,
        "synth_id": 8200,
    }


def test_resolve_fragment_tier1_direct_segment(setup_healing_scenario):
    """Tier 1: Segment ID is active in segments table -> instant resolution."""
    ingester = SearchIngester()
    # Segment 9001 exists in segments table
    expr_id, seg_id = ingester.resolve_fragment_expression(
        segment_id=9001,
        source_file="/path/to/the_prince.mp3",
        start_time=100.0,
        fragment_text="Machiavelli insisted on the originality",
    )
    assert expr_id == 8001
    assert seg_id == 9001


def test_resolve_fragment_tier2_active_tx_text_probe(setup_healing_scenario):
    """Tier 2: Old segment 1234 was purged, but text probe matches chunk in active tx."""
    ingester = SearchIngester()
    # Segment 1234 does NOT exist in segments table
    expr_id, healed_seg_id = ingester.resolve_fragment_expression(
        segment_id=1234,  # Dead old ID
        source_file="/path/to/the_prince.mp3",
        start_time=100.2,
        fragment_text="Machiavelli insisted on the originality",
    )
    assert expr_id == 8001
    assert healed_seg_id == 9001


def test_resolve_fragment_tier3_acoustic_alignment(setup_healing_scenario):
    """Tier 3: Old segment 1235 was purged, text probe differs (Camus vs Kamu), but timeline aligns."""
    ingester = SearchIngester()
    # Old text said "Camus and his notion" which won't match "Kamu" in segment text,
    # and the expression content also differs. But timestamp is 200.0s.
    expr_id, healed_seg_id = ingester.resolve_fragment_expression(
        segment_id=1235,  # Dead old ID
        source_file="/path/to/the_prince.mp3",
        start_time=200.5,
        fragment_text="Unique unmatched probe that fails text match completely",
    )
    assert expr_id == 8002
    assert healed_seg_id == 9002


def test_resolve_fragment_tier4_document_fallback(setup_healing_scenario):
    """Tier 4: No chunk text match, no timestamp match, falls back to sweep_document."""
    ingester = SearchIngester()
    expr_id, healed_seg_id = ingester.resolve_fragment_expression(
        segment_id=1236,
        source_file="/path/to/the_prince.mp3",
        start_time=3500.0,  # Far outside segments 9001/9002
        fragment_text="Unmatched text beyond any segment chunk",
    )
    assert expr_id == 8003
    assert healed_seg_id is None


def test_search_session_sweep_heals_and_terminates_loop(setup_healing_scenario):
    """Test that SearchSessionSweep links unlinked syntheses, updates fragments, and stops looping."""
    scenario = setup_healing_scenario
    query_id = scenario["query_id"]
    synth_id = scenario["synth_id"]

    # Insert an outdated fragment citing dead segment ID 7777
    with get_session() as session:
        session.execute(
            sql_text(
                "INSERT INTO search_query_fragments "
                "(query_id, segment_id, source_file, rank, start_time, end_time, fragment_text) "
                "VALUES (:qid, 7777, :path, 1, 100.1, 115.0, 'Machiavelli insisted on the originality')"
            ),
            {"qid": query_id, "path": scenario["audio_path"]},
        )
        session.commit()

    sweep = SearchSessionSweep()

    # Pass 1: Should repair the thematic link and self-heal the fragment
    sweep._repair_thematic_sync()

    # Check that THEMATIC link was created in expression_relations
    with get_session() as session:
        link = session.execute(
            sql_text(
                "SELECT to_expression_id, relation_type FROM expression_relations "
                "WHERE from_expression_id = :sid AND relation_type = 'thematic'"
            ),
            {"sid": synth_id},
        ).fetchone()
        assert link is not None
        assert link[0] == 8001
        assert link[1] == "thematic"

        # Check that search_query_fragments was self-healed from 7777 to 9001
        frag = session.execute(
            sql_text(
                "SELECT segment_id FROM search_query_fragments WHERE query_id = :qid"
            ),
            {"qid": query_id},
        ).fetchone()
        assert frag[0] == 9001

    # Pass 2: Running sweep again should report NO links to repair (loop eliminated!)
    sweep._repair_thematic_sync()
    with get_session() as session:
        pending = session.execute(
            sql_text(
                "SELECT e.id FROM expressions e "
                "WHERE e.source_type = 'search_synthesis' "
                "  AND NOT EXISTS ( "
                "      SELECT 1 FROM expression_relations er "
                "      WHERE er.from_expression_id = e.id "
                "        AND er.relation_type = 'thematic' "
                "  )"
            )
        ).fetchall()
        assert len(pending) == 0
