"""Unit tests for CommunityNamingTask daemon intelligence task."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from audiobench.daemon.intelligence.community_naming import CommunityNamingTask
from audiobench.daemon.intelligence.scheduler import IntelligenceTask


def test_community_naming_task_satisfies_protocol():
    """Verify CommunityNamingTask conforms to IntelligenceTask protocol."""
    task = CommunityNamingTask()
    assert isinstance(task, IntelligenceTask)
    assert task.INTERVAL_SECONDS == 600
    assert task.BATCH_SIZE == 5
    assert task.IS_INFERENCE_TASK is True


@pytest.mark.asyncio
async def test_community_naming_task_run_delegates():
    """Verify async run() executes sync sweep in a thread and returns upgraded count."""
    task = CommunityNamingTask()
    with patch("audiobench.community.naming.run_naming_sweep", return_value=3) as mock_sweep:
        res = await task.run()
        assert res == 3
        mock_sweep.assert_called_once_with(batch_size=5)


def test_partial_progress_and_batch_cap():
    """Verify that batch capping and per-community commits work independently."""
    from audiobench.community.naming import run_naming_sweep

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE expression_communities (
            community_id          INTEGER PRIMARY KEY,
            name                  TEXT,
            name_source           TEXT NOT NULL DEFAULT 'fallback',
            is_active             INTEGER NOT NULL DEFAULT 1,
            medoid_expr_id        INTEGER,
            node_count            INTEGER NOT NULL DEFAULT 0,
            naming_attempted_at   DATETIME,
            naming_failure_count  INTEGER NOT NULL DEFAULT 0,
            updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE TABLE expressions (id INTEGER PRIMARY KEY, content TEXT, work_id INTEGER, community_id INTEGER)")
    cur.execute("CREATE TABLE works (id INTEGER PRIMARY KEY, title TEXT, author TEXT)")

    # Insert 10 active fallback communities
    for i in range(1, 11):
        cur.execute("INSERT INTO works VALUES (?, ?, ?)", (i, f"Work {i}", f"Author {i}"))
        cur.execute(
            """
            INSERT INTO expression_communities (community_id, name, name_source, is_active, medoid_expr_id, node_count)
            VALUES (?, ?, 'fallback', 1, ?, ?)
            """,
            (i, f"Fallback {i}", i, 100 - i),
        )
        cur.execute("INSERT INTO expressions VALUES (?, 'Sample excerpt text', ?, ?)", (i, i, i))
    conn.commit()

    # Simulate: Community 1 succeeds, Community 2 fails, Community 3 succeeds, Community 4 fails, Community 5 succeeds
    def mock_name_community(c, cid):
        if cid in (1, 3, 5):
            return f"Conceptual Movement {cid}", "llm_gemini"
        return f"Fallback {cid}", "fallback"

    with patch("audiobench.community.naming.name_community", side_effect=mock_name_community):
        # Run sweep with batch_size=5
        upgraded = run_naming_sweep(conn, batch_size=5)
        assert upgraded == 3  # Communities 1, 3, 5 upgraded

    # Verify that only the top 5 by node_count (1, 2, 3, 4, 5) were touched
    cur.execute("SELECT community_id, name, name_source, naming_failure_count FROM expression_communities ORDER BY community_id")
    rows = cur.fetchall()

    # Communities 1, 3, 5 upgraded
    assert rows[0] == (1, "Conceptual Movement 1", "llm_gemini", 0)
    assert rows[2] == (3, "Conceptual Movement 3", "llm_gemini", 0)
    assert rows[4] == (5, "Conceptual Movement 5", "llm_gemini", 0)

    # Communities 2 and 4 recorded failure and stayed fallback
    assert rows[1] == (2, "Fallback 2", "fallback", 1)
    assert rows[3] == (4, "Fallback 4", "fallback", 1)

    # Communities 6-10 were never touched (batch cap respected)
    for r in rows[5:]:
        assert r[1].startswith("Fallback")
        assert r[2] == "fallback"
        assert r[3] == 0

    conn.close()
