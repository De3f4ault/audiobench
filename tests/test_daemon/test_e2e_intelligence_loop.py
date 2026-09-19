"""
Gap 3 — End-to-end intelligence loop test.

Proves the complete cycle:
  CLI events → daemon socket → get_proposals() → authorize_proposal() → DB confirmed

The daemon runs as a real subprocess with isolated DB/socket paths.
The test thread connects to the socket via DaemonClient exactly as the TUI does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session


@pytest.fixture
def live_daemon_client(test_db, tmp_data_dir):
    """
    Spins up a live daemon server subprocess with fully isolated paths.
    Uses the same DB the test_db fixture already initialised, so tables exist.
    """
    from audiobench.daemon.client import DaemonClient

    # Use a per-test socket path that is guaranteed not to collide with the
    # production daemon.
    socket_path = tmp_data_dir / "test_daemon.sock"
    pid_path = tmp_data_dir / "test_daemon.pid"

    if socket_path.exists():
        socket_path.unlink()

    # The daemon reads settings via env vars (env_prefix="AUDIOBENCH_").
    # We must inject the tmp DB, tmp socket, and tmp PID so the subprocess
    # runs in our isolated environment, not the production one.
    env = os.environ.copy()
    env["AUDIOBENCH_DATABASE_URL"] = str(test_db.database_url)
    env["AUDIOBENCH_DATA_DIR"] = str(tmp_data_dir)
    env["AUDIOBENCH_MODELS_DIR"] = str(test_db.models_dir)
    env["AUDIOBENCH_OFFLINE_MODE"] = "1"
    env["AUDIOBENCH_DAEMON_SOCKET_PATH"] = str(socket_path)
    env["AUDIOBENCH_DAEMON_PID_PATH"] = str(pid_path)
    env["AUDIOBENCH_LANCEDB_PATH"] = str(tmp_data_dir / "lancedb")

    process = subprocess.Popen(
        [sys.executable, "-m", "audiobench", "daemon", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # ML models take ~25-35s to warm on good machines, but much longer when loaded. Give the subprocess a generous 300s budget.
    start_wait = time.time()
    while not socket_path.exists():
        if time.time() - start_wait > 300.0:
            process.terminate()
            out, err = process.communicate(timeout=3)
            raise RuntimeError(
                f"Daemon subprocess did not produce socket within 300s.\n"
                f"STDOUT:\n{out.decode()}\n"
                f"STDERR:\n{err.decode()}"
            )
        if process.poll() is not None:
            out, err = process.communicate()
            raise RuntimeError(
                f"Daemon subprocess exited early (code={process.returncode}).\n"
                f"STDOUT:\n{out.decode()}\n"
                f"STDERR:\n{err.decode()}"
            )
        time.sleep(0.2)

    # Point DaemonClient at the test socket (it reads daemon_socket_path from settings).
    # We patch the module-level get_settings cache so client picks up the tmp path.
    from audiobench.core.settings import get_settings
    get_settings.cache_clear()
    old_sock = os.environ.get("AUDIOBENCH_DAEMON_SOCKET_PATH")
    os.environ["AUDIOBENCH_DAEMON_SOCKET_PATH"] = str(socket_path)
    get_settings.cache_clear()

    client = DaemonClient()

    yield client

    # Restore env + teardown
    if old_sock is not None:
        os.environ["AUDIOBENCH_DAEMON_SOCKET_PATH"] = old_sock
    else:
        os.environ.pop("AUDIOBENCH_DAEMON_SOCKET_PATH", None)
    get_settings.cache_clear()

    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
    for p in (socket_path, pid_path):
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------

def test_end_to_end_intelligence_loop(live_daemon_client):
    """
    Gap 3 seal: exercises the full proposal cycle over the real unix socket.

    Steps:
      1. Seed a 'proposed' daemon_proposal row into the isolated test DB.
      2. Call get_proposals() via the live DaemonClient — proves the server can
         query the DB and serialize proposals across the socket.
      3. Call authorize_proposal(id) — proves the authorization command reaches
         the server and is acknowledged.
      4. Read the DB directly from the test process — proves the row was marked
         'confirmed', sealing the authorization guarantee.
    """
    # 1. Seed a daemon_proposal into the DB (simulates ProposalGenerator having run).
    with get_session() as session:
        content = json.dumps({
            "operator_template": "LectureModeOperator",
            "parameters": {"frequency_multiplier": 1.5},
            "schema_version": 2,
            "region_id": "test:all",
        })
        session.execute(
            sql_text(
                "INSERT INTO expressions (source_type, content, inference_status) "
                "VALUES ('daemon_proposal', :content, 'proposed')"
            ),
            {"content": content},
        )
        session.commit()
        proposal_id = session.execute(
            sql_text("SELECT id FROM expressions WHERE source_type='daemon_proposal'")
        ).fetchone()[0]

    # 2. Retrieve proposals via the live socket.
    # get_proposals() returns list[dict] directly (already unwrapped by DaemonClient).
    proposals = live_daemon_client.get_proposals()
    assert isinstance(proposals, list), f"Expected list from get_proposals, got: {type(proposals)}"
    assert len(proposals) >= 1, "No proposals returned — seeded row not visible to daemon"
    ids = [p["id"] for p in proposals]
    assert proposal_id in ids, f"Seeded proposal {proposal_id} not found in {ids}"
    seeded = next(p for p in proposals if p["id"] == proposal_id)
    # The server filters for 'proposed'/'deferred' but doesn't return the column,
    # so finding the ID is sufficient proof it was returned.

    # 3. Authorize via socket (the `[a]` action in the TUI).
    # authorize_proposal() returns the raw _send() dict with status/action keys.
    auth = live_daemon_client.authorize_proposal(proposal_id)
    assert auth.get("status") == "ok", f"authorize_proposal failed: {auth}"

    # 4. Verify the database row was marked 'confirmed'.
    with get_session() as session:
        row = session.execute(
            sql_text("SELECT inference_status FROM expressions WHERE id = :pid"),
            {"pid": proposal_id},
        ).fetchone()
    assert row is not None
    assert row[0] == "confirmed", f"Expected 'confirmed', got '{row[0]}'"
