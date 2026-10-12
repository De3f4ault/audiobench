"""Tests for SearchSessionSweep knowledge gap deduplication and cooldown hardening."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.search_session_sweep import SearchSessionSweep
from audiobench.memory.enums import SourceType
from audiobench.storage.models import ExpressionRecord


@pytest.fixture
def sweep_fixture(test_db):
    """Seed search queries and corpus expressions to trigger gap detection."""
    now = time.time()

    with get_session() as session:
        # Seed 10 search queries within the 14-day window
        for i in range(10):
            session.add(
                ExpressionRecord(
                    source_type=SourceType.SEARCH_QUERY.value,
                    content=f"search query {i} on existentialism",
                    created_at=datetime.fromtimestamp(now - 3600 * (i + 1), tz=timezone.utc).replace(tzinfo=None),
                )
            )

        # Seed 5 corpus transcript expressions
        for i in range(5):
            session.add(
                ExpressionRecord(
                    source_type=SourceType.AUDIO_TRANSCRIPT.value,
                    graph_role="sweep_chunk",
                    content=f"transcript chunk {i} on gardening",
                    created_at=datetime.fromtimestamp(now - 3600 * (i + 1), tz=timezone.utc).replace(tzinfo=None),
                )
            )
        session.commit()


    # Create dummy orthogonal vectors: search along dimension 0, corpus along dimension 1
    # Dot product = 0.0 -> distance = 1.0 - 0.0 = 1.0 > 0.20
    v_search = [1.0] + [0.0] * 767
    v_corpus = [0.0, 1.0] + [0.0] * 766

    def mock_get_vectors(ids):
        vecs = {}
        for eid in ids:
            # Check if this id belongs to search or corpus
            vecs[eid] = v_search if eid <= 10 else v_corpus
        return vecs

    mock_store = MagicMock()
    mock_store.get_vectors.side_effect = mock_get_vectors

    mock_server_store = MagicMock()
    def fake_add_expression(source_type, content, inference_status):
        with get_session() as s:
            s.add(
                ExpressionRecord(
                    source_type=source_type,
                    content=content,
                    inference_status=inference_status,
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            s.commit()

    mock_server_store.add_expression.side_effect = fake_add_expression

    with patch("audiobench.memory.memory_store.MemoryStore", return_value=mock_store), \
         patch("audiobench.daemon.server._get_store", return_value=mock_server_store), \
         patch("audiobench.daemon.intelligence.gap_naming.name_knowledge_gap", return_value=("Test Gap", "Missing Topic", "Read Book", "mock_source")):
        yield SearchSessionSweep()


def test_dedup_skips_when_knowledge_gap_already_proposed(sweep_fixture):
    """An active proposed knowledge gap suppresses re-emission."""
    now = time.time()
    with get_session() as session:
        session.add(
            ExpressionRecord(
                source_type="daemon_proposal",
                inference_status="proposed",
                content=json.dumps({"proposal_type": "knowledge_gap", "title": "Existing Gap"}),
                created_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()

    count_before = _count_proposals()
    sweep_fixture._knowledge_gap_sync()
    count_after = _count_proposals()

    assert count_after == count_before


def test_dedup_skips_when_knowledge_gap_deferred(sweep_fixture):
    """A deferred knowledge gap suppresses re-emission within the window."""
    now = time.time()
    with get_session() as session:
        session.add(
            ExpressionRecord(
                source_type="daemon_proposal",
                inference_status="deferred",
                content=json.dumps({"proposal_type": "knowledge_gap", "title": "Deferred Gap"}),
                created_at=datetime.fromtimestamp(now - 86400, tz=timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()

    count_before = _count_proposals()
    sweep_fixture._knowledge_gap_sync()
    count_after = _count_proposals()

    assert count_after == count_before


def test_dedup_skips_when_knowledge_gap_recently_confirmed(sweep_fixture):
    """A confirmed knowledge gap within 7-day cooldown suppresses re-emission."""
    now = time.time()
    with get_session() as session:
        session.add(
            ExpressionRecord(
                source_type="daemon_proposal",
                inference_status="confirmed",
                content=json.dumps({"proposal_type": "knowledge_gap", "title": "Confirmed Gap"}),
                created_at=datetime.fromtimestamp(now - 2 * 86400, tz=timezone.utc).replace(tzinfo=None),  # 2 days ago
            )
        )
        session.commit()

    count_before = _count_proposals()
    sweep_fixture._knowledge_gap_sync()
    count_after = _count_proposals()

    assert count_after == count_before


def test_reverse_polarity_pending_operator_does_not_suppress_knowledge_gap(sweep_fixture):
    """A pending operator proposal must NOT suppress knowledge gap detection."""
    now = time.time()
    with get_session() as session:
        session.add(
            ExpressionRecord(
                source_type="daemon_proposal",
                inference_status="proposed",
                content=json.dumps({"proposal_type": "operator", "operator_template": "LectureModeOperator"}),
                created_at=datetime.fromtimestamp(now - 3600, tz=timezone.utc).replace(tzinfo=None),
            )
        )
        session.commit()

    count_before = _count_proposals()
    sweep_fixture._knowledge_gap_sync()
    count_after = _count_proposals()

    # Knowledge gap was NOT suppressed by the operator proposal
    assert count_after == count_before + 1


def test_dedup_emits_when_confirmed_beyond_cooldown(sweep_fixture):
    """A confirmed knowledge gap older than 7 days allows new emission."""
    now = time.time()
    with get_session() as session:
        session.add(
            ExpressionRecord(
                source_type="daemon_proposal",
                inference_status="confirmed",
                content=json.dumps({"proposal_type": "knowledge_gap", "title": "Old Confirmed Gap"}),
                created_at=datetime.fromtimestamp(now - 8 * 86400, tz=timezone.utc).replace(tzinfo=None),  # 8 days ago
            )
        )
        session.commit()

    count_before = _count_proposals()
    sweep_fixture._knowledge_gap_sync()
    count_after = _count_proposals()

    assert count_after == count_before + 1


def _count_proposals() -> int:
    with get_session() as session:
        return session.execute(
            sql_text("SELECT count(*) FROM expressions WHERE source_type = 'daemon_proposal'")
        ).scalar()
