"""Tests for proposal search query correlation boundaries and data snapshots."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from audiobench.daemon.intelligence.gap_naming import get_queries_in_window


def test_correlation_boundaries_synthetic():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE expressions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            content TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    proposal_ts = "2026-09-15 12:00:00"

    # Insert test cases:
    records = [
        # Outside lookback floor (> 14 days prior)
        ("search_query", "too_old_query", "2026-08-31 11:59:00"),
        # Valid queries within window
        ("search_query", "valid_oldest_query", "2026-09-01 12:01:00"),
        ("search_query", "valid_mid_query", "2026-09-08 12:00:00"),
        ("search_query", "valid_recent_query", "2026-09-15 11:59:50"),
        # Future queries relative to proposal_ts (must be excluded)
        ("search_query", "future_query", "2026-09-15 12:00:01"),
        ("search_query", "way_future_query", "2026-09-16 12:00:00"),
        # Non-search source types (must be excluded)
        ("chat_message", "chat_about_philosophy", "2026-09-10 12:00:00"),
        ("system_inference", "inference_result", "2026-09-10 12:00:00"),
        ("daemon_proposal", "existing_proposal", "2026-09-10 12:00:00"),
        # Empty/whitespace or slash commands (must be excluded)
        ("search_query", "", "2026-09-10 12:00:00"),
        ("search_query", "   ", "2026-09-10 12:00:00"),
        ("search_query", "\\help", "2026-09-10 12:00:00"),
        ("search_query", "/help", "2026-09-10 12:00:00"),
    ]

    for st, content, ts in records:
        cur.execute(
            "INSERT INTO expressions (source_type, content, created_at) VALUES (?, ?, ?)",
            (st, content, ts),
        )
    conn.commit()

    results = get_queries_in_window(conn, proposal_ts, window_days=14)

    assert len(results) == 3
    # DESC order: recent first
    assert results == [
        "valid_recent_query",
        "valid_mid_query",
        "valid_oldest_query",
    ]


def test_correlation_production_snapshots():
    db_path = Path("data/transcriptions.db")
    if not db_path.exists():
        pytest.skip("Production transcriptions.db not present")

    conn = sqlite3.connect(db_path)

    # Cluster 3 (Oct 10, 2026)
    q_oct = get_queries_in_window(conn, "2026-10-10 17:09:50", window_days=14)
    assert len(q_oct) == 14
    assert any("Og Mandino" in q for q in q_oct)

    # Cluster 2 (Sept 19, 2026): 20 raw rows minus 1 slash command (\set) = 19
    q_sep = get_queries_in_window(conn, "2026-09-19 10:36:14", window_days=14)
    assert len(q_sep) == 19
    assert any("Masculinity" in q or "Relationship" in q for q in q_sep)

    # Cluster 1 (Aug 26, 2026)
    q_aug = get_queries_in_window(conn, "2026-08-26 04:33:42", window_days=14)
    assert len(q_aug) == 48
    assert any("foolishness of God" in q for q in q_aug)

    # Verify future-leakage guard on production proposals
    for ts in ["2026-08-26 04:33:42", "2026-09-19 10:36:14", "2026-10-10 17:09:50"]:
        post_queries = conn.execute(
            """
            SELECT count(*) FROM expressions
            WHERE source_type = 'search_query'
              AND created_at > datetime(?)
              AND created_at <= datetime(?, '+1 hour')
            """,
            (ts, ts),
        ).fetchone()[0]
        assert post_queries == 0
