"""
Integration tests for OperatorRegistry.

Each test uses the `test_db` fixture from conftest.py, which creates a fresh
isolated SQLite DB per test and resets the session engine singleton after.
This means hardcoded expression IDs are safe — the DB is empty at test start.
"""

import json
import pytest

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.operator_registry import OperatorRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_proposal(session, expression_id: int, template: str, params: dict,
                     schema_version: int = 2, status: str = "proposed") -> None:
    """Seed a daemon_proposal expression directly into the test DB."""
    content = json.dumps({
        "operator_template": template,
        "parameters": params,
        "schema_version": schema_version,
        "region_id": "test_file:all",
    })
    session.execute(
        sql_text("""
        INSERT INTO expressions (id, source_type, content, inference_status, created_at)
        VALUES (:id, 'daemon_proposal', :content, :status, CURRENT_TIMESTAMP)
        """),
        {"id": expression_id, "content": content, "status": status}
    )
    session.commit()


def _get_status(session, expression_id: int) -> str | None:
    row = session.execute(
        sql_text("SELECT inference_status FROM expressions WHERE id = :eid"),
        {"eid": expression_id}
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Tests — all take `test_db` to get a fresh isolated DB per test
# ---------------------------------------------------------------------------

def test_authorize_proposal_sets_confirmed_status_and_instantiates_operator(test_db):
    """
    authorize() must atomically:
      1. Write inference_status='confirmed' to DB.
      2. Instantiate the operator so it is resident in dynamic_operators.

    Verifies that neither half is skipped: a confirm in DB that never loads
    is the silent correctness gap this fix closes.
    """
    registry = OperatorRegistry()
    with get_session() as session:
        _insert_proposal(session, expression_id=1,
                         template="LectureModeOperator",
                         params={"frequency_multiplier": 1.5})

    registry.authorize(1)

    with get_session() as session:
        assert _get_status(session, 1) == "confirmed"

    assert 1 in registry.dynamic_operators
    assert registry.dynamic_operators[1].frequency_multiplier == 1.5


def test_load_from_db_instantiates_confirmed_operators(test_db):
    """
    load_from_db() must instantiate all confirmed proposals found in the DB.
    This covers the daemon-startup path: proposals authorized in a previous
    session become live operators after restart.
    """
    registry = OperatorRegistry()
    with get_session() as session:
        _insert_proposal(session, expression_id=1,
                         template="LectureModeOperator",
                         params={"frequency_multiplier": 2.0},
                         status="confirmed")

    registry.load_from_db()

    assert 1 in registry.dynamic_operators
    assert registry.dynamic_operators[1].frequency_multiplier == 2.0


def test_deactivate_revokes_operator(test_db):
    """
    deactivate() must:
      1. Remove the operator from dynamic_operators immediately.
      2. Write inference_status='revoked' to DB.

    Both halves are checked to catch a partial implementation where the
    in-memory removal happens but the DB write is skipped (or vice-versa).
    """
    registry = OperatorRegistry()
    with get_session() as session:
        _insert_proposal(session, expression_id=1,
                         template="LectureModeOperator",
                         params={"frequency_multiplier": 1.0},
                         status="confirmed")

    registry.load_from_db()
    assert 1 in registry.dynamic_operators

    registry.deactivate(1)

    assert 1 not in registry.dynamic_operators
    with get_session() as session:
        assert _get_status(session, 1) == "revoked"


def test_unknown_template_sets_rejected_not_crash(test_db):
    """
    A proposal referencing a template name not in _TEMPLATE_CLASSES must:
      1. Write inference_status='rejected' to DB (not silently return with status unchanged).
      2. Not raise an exception — the registry continues loading other operators.

    The log message for this failure kind ("rejected: unknown operator_template '...'")
    is distinct from the ValidationError message ("Validation failed for proposal ...")
    so the ProposalsPanel can surface a traceable reason string to the human reviewer.
    """
    registry = OperatorRegistry()
    with get_session() as session:
        _insert_proposal(session, expression_id=1,
                         template="NonExistentOperator",
                         params={},
                         status="confirmed")

    # Must not raise
    registry.load_from_db()

    assert 1 not in registry.dynamic_operators
    with get_session() as session:
        assert _get_status(session, 1) == "rejected"


def test_non_operator_plain_text_proposal_skipped_without_error(test_db):
    """
    Confirmed daemon_proposal expressions with plain-text content (such as
    SearchSessionSweep knowledge gap proposals) must be skipped gracefully
    without raising JSONDecodeError or logging errors.
    """
    registry = OperatorRegistry()
    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (33216, 'daemon_proposal',
                    'Knowledge gap detected (distance=0.256, threshold=0.2).\nYour search queries cluster around topics not covered.',
                    'confirmed', CURRENT_TIMESTAMP)
            """)
        )
        session.commit()

    registry.load_from_db()
    assert 33216 not in registry.dynamic_operators
    with get_session() as session:
        # Status should remain confirmed (not rejected), as it's a valid user advisory proposal
        assert _get_status(session, 33216) == "confirmed"


def test_hot_register_non_operator_plain_text_skipped_without_error(test_db):
    """Hot-registering a plain-text proposal should gracefully return without crashing."""
    registry = OperatorRegistry()
    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (33217, 'daemon_proposal',
                    'Plain text advisory message',
                    'confirmed', CURRENT_TIMESTAMP)
            """)
        )
        session.commit()

    registry.hot_register(33217)
    assert 33217 not in registry.dynamic_operators


def test_load_from_db_skips_json_knowledge_gap_proposal(test_db):
    """Confirmed knowledge gap proposals in JSON envelope format must be skipped by OperatorRegistry."""
    registry = OperatorRegistry()
    content = json.dumps({
        "proposal_type": "knowledge_gap",
        "schema_version": 2,
        "title": "Existential Philosophy",
        "summary": "Distance 0.256",
        "evidence": "distance=0.256",
        "payload": {
            "distance": 0.256,
            "threshold": 0.20,
            "diagnostic": "Under-covered topics in existentialism"
        }
    })
    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (33218, 'daemon_proposal', :content, 'confirmed', CURRENT_TIMESTAMP)
            """),
            {"content": content}
        )
        session.commit()

    registry.load_from_db()
    assert 33218 not in registry.dynamic_operators
    with get_session() as session:
        assert _get_status(session, 33218) == "confirmed"


def test_load_from_db_instantiates_envelope_operator_proposal(test_db):
    """Operator proposals in DaemonProposalEnvelope format must instantiate correctly."""
    registry = OperatorRegistry()
    content = json.dumps({
        "proposal_type": "operator",
        "schema_version": 2,
        "title": "Enable Lecture Mode",
        "summary": "Multiplier 1.5x",
        "evidence": "confirm_rate=0.85",
        "payload": {
            "region_id": "file:speaker",
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2
        }
    })
    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (33219, 'daemon_proposal', :content, 'confirmed', CURRENT_TIMESTAMP)
            """),
            {"content": content}
        )
        session.commit()

    registry.load_from_db()
    assert 33219 in registry.dynamic_operators
    assert registry.dynamic_operators[33219].frequency_multiplier == 1.5



@pytest.mark.asyncio
async def test_proposal_generator_output_is_loadable_by_operator_registry(test_db):
    """Proposals generated by ProposalGenerator must be valid and executable by OperatorRegistry."""
    from unittest.mock import MagicMock, patch
    import time
    from audiobench.daemon.intelligence.calibration import RegionStats
    from audiobench.daemon.intelligence.proposal_generator import ProposalGenerator

    now = time.time()

    # Seed calibration row so ProposalGenerator finds >= 30 days
    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, calibration_key, created_at)
            VALUES (101, 'daemon_calibration', '{"stats": {}}', 'tracker_stats', datetime('now', '-35 days'))
            """)
        )
        session.commit()

    tracker = MagicMock()
    tracker.stats = {"test_region": RegionStats(confirms=25, rejects=5, inferences=30, samples=[])}

    def mock_add_expression(content, source_type, inference_status=None, **kwargs):
        with get_session() as s:
            s.execute(
                sql_text("""
                INSERT INTO expressions (source_type, content, inference_status, created_at)
                VALUES (:st, :c, :status, CURRENT_TIMESTAMP)
                """),
                {"st": source_type, "c": content, "status": inference_status}
            )
            s.commit()

    mock_store = MagicMock()
    mock_store.add_expression.side_effect = mock_add_expression

    with patch("audiobench.daemon.intelligence.proposal_generator._get_store", return_value=mock_store), \
         patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker", return_value=tracker):
        generator = ProposalGenerator()
        await generator.run()

    with get_session() as session:
        proposal_row = session.execute(
            sql_text("SELECT id, content FROM expressions WHERE source_type = 'daemon_proposal' LIMIT 1")
        ).fetchone()

    assert proposal_row is not None
    prop_id, content = proposal_row
    data = json.loads(content)
    assert data["proposal_type"] == "operator"
    assert data["payload"]["operator_template"] == "LectureModeOperator"
    assert data["schema_version"] == 2

    # Now authorize and verify OperatorRegistry hot-registers it
    registry = OperatorRegistry()
    registry.authorize(prop_id)

    assert prop_id in registry.dynamic_operators
    op = registry.dynamic_operators[prop_id]
    assert op.frequency_multiplier == 1.5


def test_server_handle_authorize_proposal_routes_by_type(test_db):
    """_handle_authorize_proposal routes operator proposals to registry and acknowledges knowledge gaps."""
    from audiobench.daemon.server import _handle_authorize_proposal
    from audiobench.daemon.intelligence.operator_registry import get_operator_registry

    # 1. Operator proposal
    op_content = json.dumps({
        "proposal_type": "operator",
        "schema_version": 2,
        "title": "Lecture Mode",
        "summary": "1.5x",
        "evidence": "confirm_rate=0.85",
        "payload": {
            "region_id": "file:speaker",
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2
        }
    })
    # 2. Knowledge gap proposal
    kg_content = json.dumps({
        "proposal_type": "knowledge_gap",
        "schema_version": 2,
        "title": "Philosophy Gap",
        "summary": "Distance 0.25",
        "evidence": "distance=0.25",
        "payload": {"distance": 0.25, "threshold": 0.20}
    })
    # 3. Legacy plain-text proposal
    legacy_content = "Knowledge gap detected (distance=0.25, threshold=0.2).\nCluster not covered."

    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (101, 'daemon_proposal', :op, 'proposed', CURRENT_TIMESTAMP),
                   (102, 'daemon_proposal', :kg, 'proposed', CURRENT_TIMESTAMP),
                   (103, 'daemon_proposal', :legacy, 'proposed', CURRENT_TIMESTAMP)
            """),
            {"op": op_content, "kg": kg_content, "legacy": legacy_content}
        )
        session.commit()

    # Authorize operator proposal
    res_op = _handle_authorize_proposal({"expression_id": 101})
    assert res_op["action"] == "authorized"
    assert 101 in get_operator_registry().dynamic_operators

    # Authorize knowledge gap proposal
    res_kg = _handle_authorize_proposal({"expression_id": 102})
    assert res_kg["action"] == "acknowledged"
    assert 102 not in get_operator_registry().dynamic_operators
    with get_session() as session:
        assert _get_status(session, 102) == "confirmed"

    # Authorize legacy proposal
    res_legacy = _handle_authorize_proposal({"expression_id": 103})
    assert res_legacy["action"] == "acknowledged"
    assert 103 not in get_operator_registry().dynamic_operators
    with get_session() as session:
        assert _get_status(session, 103) == "confirmed"


def test_server_handle_defer_proposal(test_db):
    """_handle_defer_proposal updates proposal status to 'deferred'."""
    from audiobench.daemon.server import _handle_defer_proposal

    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (201, 'daemon_proposal', '{}', 'proposed', CURRENT_TIMESTAMP)
            """)
        )
        session.commit()

    res = _handle_defer_proposal({"expression_id": 201})
    assert res["status"] == "ok"
    assert res["action"] == "deferred"
    with get_session() as session:
        assert _get_status(session, 201) == "deferred"


def test_server_handle_reject_proposal(test_db):
    """_handle_reject_proposal marks proposal status 'rejected' without touching calibration."""
    from audiobench.daemon.server import _handle_reject_proposal
    from audiobench.daemon.intelligence.calibration import get_calibration_tracker

    tracker = get_calibration_tracker()
    tracker_events_before = tracker._unflushed_events

    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (301, 'daemon_proposal', '{}', 'proposed', CURRENT_TIMESTAMP)
            """)
        )
        session.commit()

    res = _handle_reject_proposal({"expression_id": 301})
    assert res["status"] == "ok"
    assert res["action"] == "rejected"
    with get_session() as session:
        assert _get_status(session, 301) == "rejected"

    # Calibration tracker was NOT invoked
    assert tracker._unflushed_events == tracker_events_before
    assert "unknown:all" not in tracker.stats


@pytest.mark.asyncio
async def test_proposal_generator_sweeps_stale_operator_proposals_on_run(test_db):
    """ProposalGenerator.run() must sweep stale operator proposals while preserving fresh ones."""
    from unittest.mock import MagicMock, patch
    import time
    from audiobench.daemon.intelligence.calibration import RegionStats
    from audiobench.daemon.intelligence.proposal_generator import ProposalGenerator

    # 1. Stale proposal: 8 days old (past 7-day grace period), region has < 5 samples
    stale_content = json.dumps({
        "proposal_type": "operator",
        "schema_version": 2,
        "title": "Stale Operator",
        "summary": "1.5x",
        "evidence": "confirm_rate=0.85",
        "payload": {
            "region_id": "stale_region",
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2
        }
    })

    # 2. Fresh proposal: 1 hour old (within 7-day grace period), region has 0 samples
    fresh_content = json.dumps({
        "proposal_type": "operator",
        "schema_version": 2,
        "title": "Fresh Operator",
        "summary": "1.5x",
        "evidence": "confirm_rate=0.85",
        "payload": {
            "region_id": "fresh_region",
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2
        }
    })

    with get_session() as session:
        session.execute(
            sql_text("""
            INSERT INTO expressions (id, source_type, content, inference_status, created_at)
            VALUES (401, 'daemon_proposal', :stale, 'proposed', datetime('now', '-8 days')),
                   (402, 'daemon_proposal', :fresh, 'proposed', datetime('now', '-1 hour'))
            """),
            {"stale": stale_content, "fresh": fresh_content}
        )
        session.commit()

    tracker = MagicMock()
    # stale_region has high confirm_rate but total_since = 1 (< 5)
    stale_stats = MagicMock()
    stale_stats.confirm_rate = 0.85
    stale_stats.total_since.return_value = 1

    # fresh_region has high confirm_rate and total_since = 0
    fresh_stats = MagicMock()
    fresh_stats.confirm_rate = 0.85
    fresh_stats.total_since.return_value = 0

    def mock_get_region_stats(region):
        if region == "stale_region":
            return stale_stats
        return fresh_stats

    tracker.get_region_stats.side_effect = mock_get_region_stats
    tracker.stats = {}

    with patch("audiobench.daemon.intelligence.operator_registry.get_calibration_tracker", return_value=tracker), \
         patch("audiobench.daemon.intelligence.proposal_generator.get_calibration_tracker", return_value=tracker), \
         patch("audiobench.daemon.intelligence.proposal_generator._get_store"):
        generator = ProposalGenerator()
        await generator.run()

    with get_session() as session:
        # 401 is 8 days old and has 1 sample -> expired
        assert _get_status(session, 401) == "expired"
        # 402 is 1 hour old and within grace period -> preserved as proposed
        assert _get_status(session, 402) == "proposed"



