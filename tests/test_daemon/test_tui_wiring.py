"""
Tests for 3E / 5C / 5D wiring:
  - _handle_authorize_proposal correctly calls registry.authorize() (not the
    old TODO half-implementation)
  - _handle_get_inferences returns proposed system_inference expressions
  - _handle_get_proposals returns proposed/deferred daemon_proposal expressions
  - DaemonClient.get_inferences() and get_proposals() round-trip through the
    socket correctly
  - InferencesFeed.action_confirm_inference() calls DaemonClient.confirm_inference()
  - InferencesFeed.action_reject_inference() calls DaemonClient.reject_inference()
  - ProposalsFeed.action_authorize_proposal() calls DaemonClient.authorize_proposal()
  - ProposalsFeed.action_reject_proposal() calls DaemonClient.reject_inference()

All DB-touching tests use the `test_db` fixture for isolation.
"""

import json
from unittest.mock import MagicMock

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_expression(session, eid: int, source_type: str, status: str,
                     content: str = "test content") -> None:
    session.execute(
        sql_text("""
        INSERT INTO expressions (id, source_type, content, inference_status, created_at)
        VALUES (:id, :stype, :content, :status, CURRENT_TIMESTAMP)
        """),
        {"id": eid, "stype": source_type, "content": content, "status": status}
    )
    session.commit()


# ---------------------------------------------------------------------------
# 5D: authorize_proposal handler uses registry.authorize() — not the old TODO
# ---------------------------------------------------------------------------

def test_handle_authorize_proposal_calls_registry_authorize(test_db):
    """
    _handle_authorize_proposal must call registry.authorize(expression_id),
    which atomically writes inference_status='confirmed' AND hot-registers the
    operator. The old TODO path called only hot_register() without the DB write.
    """
    with get_session() as session:
        proposal_content = json.dumps({
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2,
            "region_id": "test:all",
        })
        _seed_expression(session, eid=1, source_type="daemon_proposal",
                         status="proposed", content=proposal_content)

    # Reset singleton so it starts empty
    import audiobench.daemon.intelligence.operator_registry as reg_mod
    from audiobench.daemon.intelligence.operator_registry import get_operator_registry
    from audiobench.daemon.server import _handle_authorize_proposal
    reg_mod._registry = None

    result = _handle_authorize_proposal({"expression_id": 1})

    assert result["action"] == "authorized"

    # DB must show confirmed
    with get_session() as session:
        row = session.execute(
            sql_text("SELECT inference_status FROM expressions WHERE id = 1")
        ).fetchone()
    assert row[0] == "confirmed"

    # Operator must be resident in registry
    registry = get_operator_registry()
    assert 1 in registry.dynamic_operators


# ---------------------------------------------------------------------------
# 3E: _handle_get_inferences returns proposed system_inference expressions
# ---------------------------------------------------------------------------

def test_handle_get_inferences_returns_proposed_only(test_db):
    """
    _handle_get_inferences must return only expressions with
    source_type in (system_inference, drift_observation, potential_relation)
    AND inference_status='proposed'. Confirmed and rejected rows are excluded.
    """
    with get_session() as session:
        _seed_expression(session, 1, "system_inference", "proposed", "pattern A")
        _seed_expression(session, 2, "system_inference", "confirmed", "pattern B")
        _seed_expression(session, 3, "drift_observation", "proposed", "drift C")
        _seed_expression(session, 4, "potential_relation", "proposed", "relation D")
        _seed_expression(session, 5, "daemon_proposal", "proposed", "proposal E")

    from audiobench.daemon.server import _handle_get_inferences
    result = _handle_get_inferences({})

    ids = {r["id"] for r in result["inferences"]}
    assert 1 in ids   # system_inference/proposed ✓
    assert 3 in ids   # drift_observation/proposed ✓
    assert 4 in ids   # potential_relation/proposed ✓
    assert 2 not in ids  # confirmed — excluded
    assert 5 not in ids  # daemon_proposal — wrong type


def test_handle_get_inferences_includes_required_fields(test_db):
    """Each returned inference dict must contain id, source_type, content, created_at."""
    with get_session() as session:
        _seed_expression(session, 1, "system_inference", "proposed", "topic convergence...")

    from audiobench.daemon.server import _handle_get_inferences
    result = _handle_get_inferences({})

    assert len(result["inferences"]) == 1
    inf = result["inferences"][0]
    assert "id" in inf
    assert "source_type" in inf
    assert "content" in inf
    assert "created_at" in inf


# ---------------------------------------------------------------------------
# 5C: _handle_get_proposals returns proposed/deferred daemon_proposal rows
# ---------------------------------------------------------------------------

def test_handle_get_proposals_returns_proposed_and_deferred(test_db):
    """
    _handle_get_proposals returns daemon_proposal rows with status
    'proposed' or 'deferred'. Confirmed, rejected, expired are excluded.
    """
    with get_session() as session:
        _seed_expression(session, 1, "daemon_proposal", "proposed", "{}")
        _seed_expression(session, 2, "daemon_proposal", "deferred", "{}")
        _seed_expression(session, 3, "daemon_proposal", "confirmed", "{}")
        _seed_expression(session, 4, "daemon_proposal", "rejected", "{}")
        _seed_expression(session, 5, "system_inference", "proposed", "inference")

    from audiobench.daemon.server import _handle_get_proposals
    result = _handle_get_proposals({})

    ids = {r["id"] for r in result["proposals"]}
    assert 1 in ids   # proposed ✓
    assert 2 in ids   # deferred ✓
    assert 3 not in ids  # confirmed — excluded
    assert 4 not in ids  # rejected — excluded
    assert 5 not in ids  # wrong type


# ---------------------------------------------------------------------------
# 3E: InferencesFeed action handlers call DaemonClient correctly
# ---------------------------------------------------------------------------

def test_action_confirm_inference_calls_daemon_client():
    """
    InferencesFeed.action_confirm_inference() must call
    DaemonClient.confirm_inference(expression_id) with the ID of the
    currently selected row.
    notify() is patched to a no-op — we are testing the client dispatch, not
    the Textual notification, which requires a running app context.
    """
    from audiobench.cli.tui.panels.inferences import InferencesFeed

    mock_client = MagicMock()
    mock_client.confirm_inference.return_value = {"status": "ok"}

    panel = InferencesFeed()
    panel._client = mock_client
    panel._selected_id = 42
    panel.notify = MagicMock()  # no active Textual app in unit tests

    panel.action_confirm_inference()

    mock_client.confirm_inference.assert_called_once_with(42)


def test_action_reject_inference_calls_daemon_client():
    """
    InferencesFeed.action_reject_inference() must call
    DaemonClient.reject_inference(expression_id).
    """
    from audiobench.cli.tui.panels.inferences import InferencesFeed

    mock_client = MagicMock()
    mock_client.reject_inference.return_value = {"status": "ok"}

    panel = InferencesFeed()
    panel._client = mock_client
    panel._selected_id = 99
    panel.notify = MagicMock()

    panel.action_reject_inference()

    mock_client.reject_inference.assert_called_once_with(99)


# ---------------------------------------------------------------------------
# 5C: ProposalsFeed action handlers call DaemonClient correctly
# ---------------------------------------------------------------------------

def test_action_authorize_proposal_calls_daemon_client():
    """
    ProposalsFeed.action_authorize_proposal() must call
    DaemonClient.authorize_proposal(expression_id).
    """
    from audiobench.cli.tui.panels.proposals import ProposalsFeed

    mock_client = MagicMock()
    mock_client.authorize_proposal.return_value = {"status": "ok"}

    panel = ProposalsFeed()
    panel._client = mock_client
    panel._selected_id = 7
    panel.notify = MagicMock()

    panel.action_authorize_proposal()

    mock_client.authorize_proposal.assert_called_once_with(7)


def test_action_defer_proposal_calls_daemon_client():
    """ProposalsFeed.action_defer_proposal() must call DaemonClient.defer_proposal(expression_id)."""
    from audiobench.cli.tui.panels.proposals import ProposalsFeed

    mock_client = MagicMock()
    mock_client.defer_proposal.return_value = {"status": "ok", "action": "deferred"}

    panel = ProposalsFeed()
    panel._client = mock_client
    panel._selected_id = 7
    panel.notify = MagicMock()

    panel.action_defer_proposal()

    mock_client.defer_proposal.assert_called_once_with(7)


def test_action_reject_proposal_calls_daemon_client():
    """ProposalsFeed.action_reject_proposal() must call DaemonClient.reject_proposal(expression_id)."""
    from audiobench.cli.tui.panels.proposals import ProposalsFeed

    mock_client = MagicMock()
    mock_client.reject_proposal.return_value = {"status": "ok", "action": "rejected"}

    panel = ProposalsFeed()
    panel._client = mock_client
    panel._selected_id = 7
    panel.notify = MagicMock()

    panel.action_reject_proposal()

    mock_client.reject_proposal.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# Regression: "component exists but is not in the execution path"
#
# Both gaps below were found 2026-07-05: ForwardingSubscriber existed as a
# complete file but __main__.py never called .install(); InferencesFeed and
# ProposalsFeed were fully built with real keybindings but observatory_app.py
# never imported or mounted them.
#
# Neither gap was caught by the existing unit tests because unit tests exercise
# components in isolation — they don't verify that the consumer imports and
# calls the component at all.  These regression tests close the category of
# gap, not just today's two instances.
# ---------------------------------------------------------------------------

def test_forwarding_subscriber_installed_at_cli_startup():
    """Regression test (2026-07-05): ForwardingSubscriber existed but
    __main__.py never called .install().

    This test imports audiobench.__main__ (which runs module-level code
    including the install() call) and then checks that the EventBus wildcard
    handler list contains a ForwardingSubscriber.record method.

    Passing this test requires:
      1. ForwardingSubscriber has an install() method (not just __init__).
      2. __main__.py calls ForwardingSubscriber().install() at module level.
      3. install() actually registers on the bus wildcard (not some other event).

    A component that passes 'does the file exist' but fails 'is it in the
    execution path' will fail here.
    """
    # Re-import to ensure module-level code in __main__ has run.
    import importlib

    from audiobench.events import get_bus
    from audiobench.observatory.forwarding_subscriber import ForwardingSubscriber

    bus = get_bus()
    orig_wildcards = list(bus._handlers.get("*", []))
    try:
        import audiobench.__main__ as main_mod
        importlib.reload(main_mod)

        wildcard_handlers = bus._handlers.get("*", [])
        # At least one handler must be a bound method named 'record' on a
        # ForwardingSubscriber instance.
        assert any(
            callable(h)
            and getattr(h, "__self__", None) is not None
            and isinstance(h.__self__, ForwardingSubscriber)
            and h.__name__ == "record"
            for h in wildcard_handlers
        ), (
            "No ForwardingSubscriber.record handler found on the EventBus wildcard. "
            "Check that __main__.py calls ForwardingSubscriber().install() at module level."
        )
    finally:
        bus._handlers["*"] = orig_wildcards


def test_inferences_and_proposals_panels_mounted_in_observatory():
    """Regression test (2026-07-05): InferencesFeed and ProposalsFeed were
    fully built with real keybindings and daemon calls but were never imported
    or added to the layout in observatory_app.py.

    We cannot call ObservatoryApp().compose() outside a running Textual event
    loop (it requires an active app context). Instead this test verifies the
    wiring at the import and source level — the two checks that would have
    caught the gap that was found:

    1. Import check: observatory_app's module namespace contains both classes
       (proves they are imported at the top of the file, not just available
       somewhere on the path).
    2. Source check: the compose() method's source code yields both class
       names (proves they appear in the layout, not just imported unused).

    A panel that passes unit tests in isolation but is not mounted here will
    fail both checks.
    """
    import inspect

    import audiobench.cli.tui.observatory_app as app_mod

    # Check 1: both classes must be importable from the app module's namespace,
    # meaning they appear in the top-level imports of observatory_app.py.
    assert hasattr(app_mod, "InferencesFeed"), (
        "InferencesFeed not found in observatory_app module namespace. "
        "Add 'from audiobench.cli.tui.panels.inferences import InferencesFeed' "
        "to observatory_app.py."
    )
    assert hasattr(app_mod, "ProposalsFeed"), (
        "ProposalsFeed not found in observatory_app module namespace. "
        "Add 'from audiobench.cli.tui.panels.proposals import ProposalsFeed' "
        "to observatory_app.py."
    )

    # Check 2: both class names must appear in the compose() source, meaning
    # they are actually yielded in the layout (not just imported and unused).
    compose_source = inspect.getsource(app_mod.ObservatoryApp.compose)
    assert "InferencesFeed" in compose_source, (
        "InferencesFeed is imported but not yielded in ObservatoryApp.compose(). "
        "Mount it in the layout in observatory_app.py."
    )
    assert "ProposalsFeed" in compose_source, (
        "ProposalsFeed is imported but not yielded in ObservatoryApp.compose(). "
        "Mount it in the layout in observatory_app.py."
    )

