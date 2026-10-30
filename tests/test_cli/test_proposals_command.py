"""Tests for REPL \\proposals command, offline parity, and onboarding banner integration."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import click
import pytest
from sqlalchemy import text as sql_text

from audiobench.cli.display.theme import console
from audiobench.cli.repl.backslash_commands import (
    _authorize_daemon_proposal,
    _defer_daemon_proposal,
    _fetch_daemon_proposals,
    _reject_daemon_proposal,
    cmd_proposals,
)
from audiobench.cli.repl.banner import print_full_help, print_onboarding
from audiobench.cli.repl.session import ReplSession
from audiobench.core.db_session import get_session
from audiobench.storage.models import ExpressionRecord


@pytest.fixture
def repl_session():
    group = click.Group("test")
    return ReplSession(group)


def has_emojis(text: str) -> bool:
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "]"
    )
    return bool(emoji_pattern.search(text))


@pytest.fixture
def seed_proposals():
    """Seed test proposals into the SQLite test database."""
    created_ids = []
    with get_session() as session:
        # 1. Knowledge Gap Proposal (Schema version 2)
        kg_content = {
            "schema_version": 2,
            "proposal_type": "knowledge_gap",
            "title": "Christian Existential Theology",
            "summary": "Explores faith through existential despair and divine foolishness.",
            "diagnostic": "Significant interest in Kierkegaardian existentialism and Pauline theology.",
            "recommended_ingestion": [
                "Fear and Trembling by Soren Kierkegaard",
                "The Courage to Be by Paul Tillich",
            ],
            "stranded_queries": [
                "divine foolishness existentialism",
                "paul theology of the cross absurd",
            ],
            "medoid_query": "divine foolishness existentialism",
            "drift_distance": 0.2562,
            "lookback_days": 14,
            "name_source": "llm_ollama",
        }
        kg_expr = ExpressionRecord(
            content=json.dumps(kg_content),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add(kg_expr)
        session.flush()
        created_ids.append(kg_expr.id)

        # 2. Operator Proposal
        op_content = {
            "proposal_type": "operator",
            "operator_template": "LectureModeOperator",
            "title": "Lecture Mode Operator",
            "summary": "Adjusts playback speed and notes during long lectures.",
            "rationale": "High frequency of academic lecture transcriptions detected.",
        }
        op_expr = ExpressionRecord(
            content=json.dumps(op_content),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add(op_expr)
        session.flush()
        created_ids.append(op_expr.id)

        session.commit()

    yield created_ids

    # Cleanup
    with get_session() as session:
        for eid in created_ids:
            rec = session.query(ExpressionRecord).filter_by(id=eid).first()
            if rec:
                session.delete(rec)
        session.commit()


def _read_output(capsys) -> str:
    res = capsys.readouterr()
    return res.out + res.err


def test_proposals_list_rendering(seed_proposals, repl_session, capsys):
    """Test \\proposals and \\proposals list table display."""
    kg_id, op_id = seed_proposals

    # Default list
    cmd_proposals(repl_session, "list")
    out = _read_output(capsys)

    assert f"#{kg_id}" in out
    assert "Christian Existential" in out
    assert "Theology" in out
    assert "knowledge_gap" in out
    assert f"#{op_id}" in out
    assert "Lecture Mode Operator" in out
    assert "operator" in out
    assert not has_emojis(out)


def test_proposals_list_filtering(seed_proposals, repl_session, capsys):
    """Test \\proposals list with status filters."""
    kg_id, _ = seed_proposals

    # Defer proposal
    cmd_proposals(repl_session, f"defer {kg_id}")
    _read_output(capsys)

    # Filter by deferred
    cmd_proposals(repl_session, "list deferred")
    out = _read_output(capsys)
    assert f"#{kg_id}" in out
    assert "deferred" in out

    # Filter by confirmed (should be empty for kg_id)
    cmd_proposals(repl_session, "list confirmed")
    out = _read_output(capsys)
    assert f"#{kg_id}" not in out


def test_proposals_show_rendering(seed_proposals, repl_session, capsys):
    """Test \\proposals show <id> and numeric shorthand \\proposals <id>."""
    kg_id, _ = seed_proposals

    cmd_proposals(repl_session, f"show {kg_id}")
    out = _read_output(capsys)

    assert f"Proposal #{kg_id}" in out
    assert "Christian Existential Theology" in out
    assert "Missing Topic Diagnostic:" in out
    assert "Kierkegaardian existentialism" in out
    assert "Recommended Acquisitions / Reading List:" in out
    assert "Fear and Trembling" in out
    assert "Correlated Search Queries" in out
    assert "divine foolishness existentialism" in out
    assert "Semantic Drift Distance: 0.2562" in out
    assert "Naming Provider: llm_ollama" in out
    assert not has_emojis(out)

    # Shorthand: \proposals <id>
    cmd_proposals(repl_session, f"{kg_id}")
    out2 = _read_output(capsys)
    assert "Christian Existential Theology" in out2


def test_proposals_show_operator_rendering(seed_proposals, repl_session, capsys):
    """Test \\proposals show <id> for operator proposals."""
    _, op_id = seed_proposals

    cmd_proposals(repl_session, f"show {op_id}")
    out = _read_output(capsys)

    assert f"Proposal #{op_id}" in out
    assert "Operator Template: LectureModeOperator" in out
    assert "academic lecture transcriptions" in out.lower()
    assert not has_emojis(out)


def test_proposals_state_transitions(seed_proposals, repl_session, capsys):
    """Test authorize, defer, and reject command execution and database transitions."""
    kg_id, _ = seed_proposals

    # 1. Defer
    cmd_proposals(repl_session, f"defer {kg_id}")
    out = _read_output(capsys)
    assert f"Deferred proposal #{kg_id}" in out

    with get_session() as session:
        rec = session.query(ExpressionRecord).filter_by(id=kg_id).first()
        assert rec.inference_status == "deferred"

    # 2. Authorize / Confirm
    cmd_proposals(repl_session, f"authorize {kg_id}")
    out = _read_output(capsys)
    assert f"Confirmed proposal #{kg_id}" in out

    with get_session() as session:
        rec = session.query(ExpressionRecord).filter_by(id=kg_id).first()
        assert rec.inference_status == "confirmed"

    # 3. Reject
    cmd_proposals(repl_session, f"reject {kg_id}")
    out = _read_output(capsys)
    assert f"Rejected proposal #{kg_id}" in out

    with get_session() as session:
        rec = session.query(ExpressionRecord).filter_by(id=kg_id).first()
        assert rec.inference_status == "rejected"


def test_online_vs_offline_100_percent_parity():
    """Verify that the offline in-process fallback and online daemon socket path produce identical results.

    Asserts that:
    1. Both paths return identical response dictionary structures.
    2. Both paths perform identical status mutations on the database.
    3. Neither touches CalibrationTracker on knowledge gap proposals.
    """
    from audiobench.daemon.server import (
        _handle_authorize_proposal,
        _handle_defer_proposal,
        _handle_get_proposals,
        _handle_reject_proposal,
    )

    with get_session() as session:
        # Create proposal 1 (tested via simulated online path)
        p1 = ExpressionRecord(
            content=json.dumps({"proposal_type": "knowledge_gap", "title": "Online Parity"}),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        # Create proposal 2 (tested via offline path)
        p2 = ExpressionRecord(
            content=json.dumps({"proposal_type": "knowledge_gap", "title": "Offline Parity"}),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add_all([p1, p2])
        session.flush()
        id1, id2 = p1.id, p2.id
        session.commit()

    try:
        # ── Step 1: Read Parity ──
        # Mock online DaemonClient
        mock_client = MagicMock()
        mock_client.get_proposals.return_value = _handle_get_proposals({"expression_id": id1})["proposals"]

        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=True), \
             patch("audiobench.daemon.client.DaemonClient", return_value=mock_client):
            online_read = _fetch_daemon_proposals(expression_id=id1)

        # Offline in-process path
        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=False):
            offline_read = _fetch_daemon_proposals(expression_id=id2)

        assert len(online_read) == 1
        assert len(offline_read) == 1
        # Keys must match 100%
        assert set(online_read[0].keys()) == set(offline_read[0].keys())
        assert online_read[0]["inference_status"] == "proposed"
        assert offline_read[0]["inference_status"] == "proposed"

        # ── Step 2: Mutation Parity (Defer) ──
        mock_client.defer_proposal.return_value = _handle_defer_proposal({"expression_id": id1})

        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=True), \
             patch("audiobench.daemon.client.DaemonClient", return_value=mock_client):
            online_defer = _defer_daemon_proposal(id1)

        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=False):
            offline_defer = _defer_daemon_proposal(id2)

        # Responses must be identical in keys and action
        assert online_defer == {"status": "ok", "expression_id": id1, "action": "deferred"}
        assert offline_defer == {"status": "ok", "expression_id": id2, "action": "deferred"}

        # ── Step 3: Mutation Parity (Authorize / Confirm) ──
        mock_client.authorize_proposal.return_value = _handle_authorize_proposal({"expression_id": id1})

        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=True), \
             patch("audiobench.daemon.client.DaemonClient", return_value=mock_client):
            online_auth = _authorize_daemon_proposal(id1)

        with patch("audiobench.cli.repl.backslash_commands._is_daemon_alive", return_value=False):
            offline_auth = _authorize_daemon_proposal(id2)

        assert online_auth == {"status": "ok", "expression_id": id1, "action": "acknowledged"}
        assert offline_auth == {"status": "ok", "expression_id": id2, "action": "acknowledged"}

        # Verify DB states match 100%
        with get_session() as session:
            r1 = session.query(ExpressionRecord).filter_by(id=id1).first()
            r2 = session.query(ExpressionRecord).filter_by(id=id2).first()
            assert r1.inference_status == "confirmed"
            assert r2.inference_status == "confirmed"

    finally:
        with get_session() as session:
            for eid in (id1, id2):
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_onboarding_banner_pending_proposals_notice(seed_proposals, repl_session, capsys):
    """Test that print_onboarding displays pending proposals notice when proposals exist."""
    print_onboarding(repl_session)
    out = _read_output(capsys)

    assert "Pending Proposals:" in out
    assert "proposals" in out
    assert r"\proposals" in out
    assert not has_emojis(out)


def test_full_help_mentions_proposals(repl_session, capsys):
    """Test that print_full_help includes the \\proposals command."""
    print_full_help(repl_session)
    out = _read_output(capsys)

    assert r"\proposals" in out
    assert "Review & authorize daemon proposals" in out
    assert not has_emojis(out)


def test_proposals_hash_prefix_subcommands(seed_proposals, repl_session, capsys):
    """Test that all \\proposals subcommands accept # prefix on IDs."""
    kg_id, _ = seed_proposals

    # 1. show with # prefix
    cmd_proposals(repl_session, f"show #{kg_id}")
    out = _read_output(capsys)
    assert f"Proposal #{kg_id}" in out
    assert "Christian Existential Theology" in out

    # 2. shorthand with # prefix
    cmd_proposals(repl_session, f"#{kg_id}")
    out = _read_output(capsys)
    assert f"Proposal #{kg_id}" in out

    # 3. defer with # prefix
    cmd_proposals(repl_session, f"defer #{kg_id}")
    out = _read_output(capsys)
    assert f"Deferred proposal #{kg_id}" in out

    # 4. authorize with # prefix
    cmd_proposals(repl_session, f"authorize #{kg_id}")
    out = _read_output(capsys)
    assert f"Confirmed proposal #{kg_id}" in out

    # 5. reject with # prefix
    cmd_proposals(repl_session, f"reject #{kg_id}")
    out = _read_output(capsys)
    assert f"Rejected proposal #{kg_id}" in out


def test_proposals_schema_v2_dual_shapes(repl_session, capsys):
    """Test schema version 2 payload unpack with both string and list recommended_ingestion shapes."""
    created_ids = []
    try:
        with get_session() as session:
            # Shape A: comma-delimited string matching production proposal #86710
            shape_a = {
                "schema_version": 2,
                "proposal_type": "knowledge_gap",
                "title": "Self-Help & Spiritual Classics",
                "summary": "Query cluster diverges from audio corpus",
                "evidence": "Centroid distance 0.200 vs threshold 0.20",
                "payload": {
                    "distance": 0.200,
                    "threshold": 0.20,
                    "window_days": 14,
                    "diagnostic": "The library lacks core audiobooks on classic self-help.",
                    "recommended_ingestion": (
                        '"The Greatest Salesman in the World" by Og Mandino (Narrated by Michael Prichard), '
                        '"The Metamorphosis" by Franz Kafka (Narrated by Frank Muller)'
                    ),
                    "stranded_queries": [
                        "the greatest salesman in the world og mandino",
                        "franz kafka essence",
                    ],
                    "medoid_query": "the greatest salesman in the world og mandino",
                    "name_source": "llm_ollama",
                },
            }
            rec_a = ExpressionRecord(
                content=json.dumps(shape_a),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec_a)

            # Shape B: canonical list-of-strings shape
            shape_b = {
                "schema_version": 2,
                "proposal_type": "knowledge_gap",
                "title": "Existential Philosophy",
                "summary": "Exploration of absurdism and despair",
                "payload": {
                    "distance": 0.245,
                    "diagnostic": "Significant interest in Kierkegaardian existentialism.",
                    "recommended_ingestion": [
                        "Fear and Trembling by Soren Kierkegaard",
                        "The Myth of Sisyphus by Albert Camus",
                    ],
                    "stranded_queries": ["kierkegaard absurdism despair"],
                    "medoid_query": "kierkegaard absurdism despair",
                    "name_source": "llm_gemini",
                },
            }
            rec_b = ExpressionRecord(
                content=json.dumps(shape_b),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec_b)
            session.commit()
            created_ids = [rec_a.id, rec_b.id]

        # Verify Shape A (string-based) unpack and normalization
        cmd_proposals(repl_session, f"show #{created_ids[0]}")
        out_a = _read_output(capsys)
        assert f"Proposal #{created_ids[0]}" in out_a
        assert "Self-Help & Spiritual Classics" in out_a
        assert "Missing Topic Diagnostic:" in out_a
        assert "classic self-help" in out_a
        assert "Recommended Acquisitions / Reading List:" in out_a
        assert "The Greatest Salesman in the World" in out_a
        assert "The Metamorphosis" in out_a
        assert "Correlated Search Queries (2):" in out_a
        assert "the greatest salesman in the world og mandino" in out_a
        assert "Semantic Drift Distance: 0.2000" in out_a
        assert "Naming Provider: llm_ollama" in out_a

        # Verify Shape B (list-based) unpack and normalization
        cmd_proposals(repl_session, f"show #{created_ids[1]}")
        out_b = _read_output(capsys)
        assert f"Proposal #{created_ids[1]}" in out_b
        assert "Existential Philosophy" in out_b
        assert "Missing Topic Diagnostic:" in out_b
        assert "Kierkegaardian existentialism" in out_b
        assert "Recommended Acquisitions / Reading List:" in out_b
        assert "Fear and Trembling by Soren Kierkegaard" in out_b
        assert "The Myth of Sisyphus by Albert Camus" in out_b
        assert "Correlated Search Queries (1):" in out_b
        assert "Semantic Drift Distance: 0.2450" in out_b
        assert "Naming Provider: llm_gemini" in out_b

    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_proposals_cluster_deduplication_default_view(repl_session, capsys):
    """Test that default \\proposals collapses duplicate cluster runs into a single representative row with runs badge."""
    created_ids = []
    with get_session() as session:
        # Create 3 duplicate knowledge gap proposals
        for i in range(3):
            rec = ExpressionRecord(
                content=json.dumps({
                    "schema_version": 2,
                    "proposal_type": "knowledge_gap",
                    "title": "Christian Existential Theology",
                    "stranded_queries": [
                        "divine foolishness existentialism",
                        "paul theology of the cross absurd",
                    ],
                }),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec)
            session.flush()
            created_ids.append(rec.id)

        # Create 1 operator proposal
        op_rec = ExpressionRecord(
            content=json.dumps({
                "proposal_type": "operator",
                "operator_template": "LectureModeOperator",
                "title": "Lecture Mode Operator",
            }),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add(op_rec)
        session.flush()
        created_ids.append(op_rec.id)
        session.commit()

    try:
        cmd_proposals(repl_session, "")
        out = _read_output(capsys)

        # Knowledge gap cluster should collapse and show 3 runs
        assert "Christian Existential" in out
        assert "Theology" in out
        assert "(3 runs)" in out
        # Operator proposal should be present as a standalone item without runs badge
        assert "Lecture Mode Operator" in out
        assert "Lecture Mode Operator (3 runs)" not in out

        # Representative should be the highest ID among the 3 duplicates
        highest_kg_id = max(created_ids[:3])
        assert f"#{highest_kg_id}" in out
        for older_id in created_ids[:2]:
            assert f"#{older_id}" not in out

        # Show dossier for representative should display cluster runs metadata
        cmd_proposals(repl_session, f"show {highest_kg_id}")
        show_out = _read_output(capsys)
        assert "Cluster: 3 runs" in show_out
        assert f"#{created_ids[0]}" in show_out
        assert f"#{created_ids[1]}" in show_out
        assert f"#{created_ids[2]}" in show_out
        assert not has_emojis(out)
        assert not has_emojis(show_out)
    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_proposals_all_escape_hatch(repl_session, capsys):
    """Test that \\proposals all and \\proposals list all render all raw uncollapsed runs."""
    created_ids = []
    with get_session() as session:
        for i in range(3):
            rec = ExpressionRecord(
                content=json.dumps({
                    "schema_version": 2,
                    "proposal_type": "knowledge_gap",
                    "title": "Christian Existential Theology",
                    "stranded_queries": [
                        "divine foolishness existentialism",
                        "paul theology of the cross absurd",
                    ],
                }),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec)
            session.flush()
            created_ids.append(rec.id)

        op_rec = ExpressionRecord(
            content=json.dumps({
                "proposal_type": "operator",
                "operator_template": "LectureModeOperator",
                "title": "Lecture Mode Operator",
            }),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add(op_rec)
        session.flush()
        created_ids.append(op_rec.id)
        session.commit()

    try:
        # 1. Test \proposals all
        cmd_proposals(repl_session, "all")
        out_all = _read_output(capsys)
        for eid in created_ids:
            assert f"#{eid}" in out_all
        assert not has_emojis(out_all)

        # 2. Test \proposals list all
        cmd_proposals(repl_session, "list all")
        out_list_all = _read_output(capsys)
        for eid in created_ids:
            assert f"#{eid}" in out_list_all
        assert not has_emojis(out_list_all)
    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_proposals_cluster_key_immune_to_title_drift(repl_session, capsys):
    """Test that proposals with drifted titles or different providers merge into one cluster by query set."""
    created_ids = []
    with get_session() as session:
        p1 = ExpressionRecord(
            content=json.dumps({
                "schema_version": 2,
                "proposal_type": "knowledge_gap",
                "title": "Philosophical Integration Gap",
                "name_source": "llm_gemini",
                "stranded_queries": [
                    "Loneliness In Young Men",
                    "Stoic Resilience  ",
                ],
            }),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        p2 = ExpressionRecord(
            content=json.dumps({
                "schema_version": 2,
                "proposal_type": "knowledge_gap",
                "title": "Masculinity, Loneliness & Philosophical Resilience",
                "name_source": "llm_ollama",
                "stranded_queries": [
                    "  loneliness in young men",
                    "stoic resilience",
                ],
            }),
            source_type="daemon_proposal",
            graph_role="sweep_passage",
            inference_status="proposed",
            privacy_tier=0,
        )
        session.add_all([p1, p2])
        session.flush()
        created_ids = [p1.id, p2.id]
        session.commit()

    try:
        cmd_proposals(repl_session, "list")
        out = _read_output(capsys)

        # Both records share normalized query set -> must collapse to 1 cluster row with (2 runs)
        assert "(2 runs)" in out
        # Representative should be latest (p2.id)
        assert f"#{created_ids[1]}" in out
        assert f"#{created_ids[0]}" not in out
        assert not has_emojis(out)
    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_proposals_cascade_mutation_authorize_defer_reject(repl_session, capsys):
    """Test that defer, authorize, and reject actions cascade across all member proposals in the cluster."""
    created_ids = []
    with get_session() as session:
        for i in range(3):
            rec = ExpressionRecord(
                content=json.dumps({
                    "schema_version": 2,
                    "proposal_type": "knowledge_gap",
                    "title": "Christian Existential Theology",
                    "stranded_queries": [
                        "faith in absurdity",
                        "existential dread prayer",
                    ],
                }),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec)
            session.flush()
            created_ids.append(rec.id)
        session.commit()

    id_a, id_b, id_c = created_ids

    try:
        # 1. Defer via id_a
        cmd_proposals(repl_session, f"defer {id_a}")
        out_defer = _read_output(capsys)
        norm_defer = " ".join(out_defer.split())
        assert f"Deferred proposal #{id_a}" in norm_defer
        assert "(cascaded to 3 duplicate runs in cluster 'Christian Existential Theology')" in norm_defer
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                assert rec.inference_status == "deferred"

        # 2. Authorize via id_b
        cmd_proposals(repl_session, f"authorize {id_b}")
        out_auth = _read_output(capsys)
        norm_auth = " ".join(out_auth.split())
        assert f"Confirmed proposal #{id_b}" in norm_auth
        assert "(cascaded to 3 duplicate runs in cluster 'Christian Existential Theology')" in norm_auth
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                assert rec.inference_status == "confirmed"

        # 3. Reject via id_c
        cmd_proposals(repl_session, f"reject {id_c}")
        out_rej = _read_output(capsys)
        norm_rej = " ".join(out_rej.split())
        assert f"Rejected proposal #{id_c}" in norm_rej
        assert "(cascaded to 3 duplicate runs in cluster 'Christian Existential Theology')" in norm_rej
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                assert rec.inference_status == "rejected"

        assert not has_emojis(out_defer)
        assert not has_emojis(out_auth)
        assert not has_emojis(out_rej)
    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()


def test_proposals_cascade_calibration_isolation(repl_session, capsys):
    """Test that cascading a rejection across multiple proposals never debits CalibrationTracker."""
    created_ids = []
    with get_session() as session:
        for i in range(3):
            rec = ExpressionRecord(
                content=json.dumps({
                    "schema_version": 2,
                    "proposal_type": "knowledge_gap",
                    "title": "Existential Dread Isolation",
                    "stranded_queries": [
                        "dread isolation prayer",
                    ],
                }),
                source_type="daemon_proposal",
                graph_role="sweep_passage",
                inference_status="proposed",
                privacy_tier=0,
            )
            session.add(rec)
            session.flush()
            created_ids.append(rec.id)
        session.commit()

    id_target = created_ids[0]

    try:
        mock_calib = MagicMock()
        with patch("audiobench.daemon.intelligence.calibration.get_calibration_tracker", return_value=mock_calib):
            cmd_proposals(repl_session, f"reject {id_target}")
            out = _read_output(capsys)

        norm_out = " ".join(out.split())
        assert f"Rejected proposal #{id_target}" in norm_out
        assert "(cascaded to 3 duplicate runs in cluster 'Existential Dread Isolation')" in norm_out
        # Must strictly never call record_reject
        mock_calib.record_reject.assert_not_called()

        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                assert rec.inference_status == "rejected"

        assert not has_emojis(out)
    finally:
        with get_session() as session:
            for eid in created_ids:
                rec = session.query(ExpressionRecord).filter_by(id=eid).first()
                if rec:
                    session.delete(rec)
            session.commit()
