"""Comprehensive test suite for Community Detection service and schema.

Tests:
1. Migration 047 applies cleanly.
2. Canonical clustering runs and produces macro-communities.
3. Micro-islands (<50 nodes) remain community_id = NULL.
4. Canonical clustering reconciles existing IDs via Jaccard >= 0.50.
5. Incremental assignment plurality voting by weight sum.
6. Incremental tie-breaking (epsilon comparison) leaves ties NULL.
7. Incremental assignment with unlinked neighbors stays NULL.
8. Staleness check triggers medoid refresh on >=20% and >=25 node growth.
9. Soft-weighted medoid selects dominant author over minority star-hub.
10. Fallback naming produces clean author/work title without LLM.
11. Naming sweep promotes fallback rows when LLM succeeds.
12. Incremental assignment preserves existing community_ids on older expressions.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch
import pytest

from audiobench.community import (
    assign_incremental,
    compute_soft_weighted_medoid,
    get_community_summary,
    populate_medoids,
    run_canonical_clustering,
)
from audiobench.community.naming import (
    generate_fallback_name,
    name_community,
    run_naming_sweep,
)


@pytest.fixture
def test_db():
    """Create a fully-migrated in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Base schema required for expressions and relations
    cur.execute("""
        CREATE TABLE works (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE expressions (
            id INTEGER PRIMARY KEY,
            content TEXT,
            work_id INTEGER REFERENCES works(id),
            community_id INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE expression_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_expression_id INTEGER NOT NULL,
            to_expression_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL DEFAULT 'semantic',
            weight REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE expression_communities (
            community_id          INTEGER PRIMARY KEY,
            name                  TEXT,
            name_source           TEXT CHECK(name_source IN ('llm_gemini','llm_ollama','fallback','pending'))
                                  NOT NULL DEFAULT 'fallback',
            is_active             INTEGER NOT NULL DEFAULT 1,
            medoid_expr_id        INTEGER REFERENCES expressions(id) ON DELETE SET NULL,
            node_count            INTEGER NOT NULL DEFAULT 0,
            medoid_at_node_count  INTEGER NOT NULL DEFAULT 0,
            resolution            REAL NOT NULL DEFAULT 0.6,
            seed                  INTEGER NOT NULL DEFAULT 42,
            indexed_at            DATETIME,
            updated_at            DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            naming_attempted_at   DATETIME,
            naming_failure_count  INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("INSERT INTO schema_version (version) VALUES (50)")
    conn.commit()
    yield conn
    conn.close()


def test_migration_047_applies_cleanly(test_db):
    """Test that migration 047 schema objects exist."""
    cur = test_db.cursor()
    cur.execute("PRAGMA table_info(expression_communities)")
    cols = {r[1] for r in cur.fetchall()}
    assert "community_id" in cols
    assert "name" in cols
    assert "name_source" in cols
    assert "is_active" in cols
    assert "medoid_expr_id" in cols
    assert "medoid_at_node_count" in cols
    assert "naming_attempted_at" in cols
    assert "naming_failure_count" in cols

    cur.execute("PRAGMA table_info(expressions)")
    expr_cols = {r[1] for r in cur.fetchall()}
    assert "community_id" in expr_cols


def test_canonical_clustering_runs(test_db):
    """Test canonical clustering creates macro-communities and assigns expressions."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Book A', 'Author A'), (2, 'Book B', 'Author B')")

    # Create 60 expressions in Cluster 1 and connect them densely
    for i in range(1, 61):
        cur.execute("INSERT INTO expressions (id, content, work_id) VALUES (?, ?, 1)", (i, f"Content {i}"))
    for i in range(1, 60):
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, ?, 0.85)", (i, i + 1))
        # Add extra cross-links
        if i + 5 <= 60:
            cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, ?, 0.82)", (i, i + 5))

    test_db.commit()

    summary = run_canonical_clustering(test_db, resolution=0.6, seed=42, min_size=10)
    assert summary["n_communities"] >= 1
    assert summary["n_assigned"] >= 10

    cur.execute("SELECT community_id, name, name_source, is_active, node_count, medoid_expr_id FROM expression_communities")
    rows = cur.fetchall()
    assert len(rows) >= 1
    cid, name, source, active, count, med_id = rows[0]
    assert active == 1
    assert count >= 10
    assert med_id is not None
    assert source == "fallback"
    assert "Author A" in name


def test_micro_islands_are_null(test_db):
    """Test that connected components with <50 nodes remain community_id = NULL."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Book A', 'Author A')")

    # Create small island of 5 nodes
    for i in range(100, 105):
        cur.execute("INSERT INTO expressions (id, content, work_id) VALUES (?, ?, 1)", (i, f"Island {i}"))
    for i in range(100, 104):
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, ?, 0.90)", (i, i + 1))
    test_db.commit()

    summary = run_canonical_clustering(test_db, resolution=0.6, seed=42)
    assert summary["n_communities"] == 0
    assert summary["n_assigned"] == 0

    cur.execute("SELECT community_id FROM expressions WHERE id >= 100")
    for r in cur.fetchall():
        assert r[0] is None


def test_canonical_clustering_reconciles_existing_ids(test_db):
    """Test that re-clustering reconciles against existing IDs via Jaccard >= 0.50."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Book A', 'Author A')")

    # Initial cluster of 55 nodes
    for i in range(1, 56):
        cur.execute("INSERT INTO expressions (id, content, work_id) VALUES (?, ?, 1)", (i, f"Chunk {i}"))
    for i in range(1, 55):
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, ?, 0.88)", (i, i + 1))
    test_db.commit()

    # First run
    run_canonical_clustering(test_db, resolution=0.6, seed=42, min_size=10)
    cur.execute("SELECT community_id FROM expression_communities WHERE is_active = 1")
    initial_cid = cur.fetchone()[0]

    # Add 5 more nodes connected to the same cluster (slight modification, >90% Jaccard)
    for i in range(56, 61):
        cur.execute("INSERT INTO expressions (id, content, work_id) VALUES (?, ?, 1)", (i, f"New Chunk {i}"))
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, ?, 0.88)", (i, 1))
    test_db.commit()

    # Re-run canonical clustering
    run_canonical_clustering(test_db, resolution=0.6, seed=42, min_size=10)
    cur.execute("SELECT community_id, is_active FROM expression_communities WHERE community_id = ?", (initial_cid,))
    row = cur.fetchone()
    assert row is not None
    assert row[1] == 1  # ID preserved and still active!


def test_incremental_assignment_weighted_voting(test_db):
    """Test that new chunk votes for community with highest edge weight sum."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Book A', 'Author A')")
    cur.execute("INSERT INTO expression_communities (community_id, name, is_active, node_count) VALUES (1, 'Comm 1', 1, 50), (2, 'Comm 2', 1, 50)")

    # Neighbors:
    # 2 edges to Comm 1 with weight 0.95 each (sum = 1.90)
    # 3 edges to Comm 2 with weight 0.50 each (sum = 1.50)
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (10, 'N1', 1), (11, 'N2', 1), (20, 'N3', 2), (21, 'N4', 2), (22, 'N5', 2)")
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (99, 'Target', NULL)")

    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 10, 0.95)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 11, 0.95)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 20, 0.80)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 21, 0.80)")
    # Comm 1 sum = 1.90, Comm 2 sum = 1.60
    test_db.commit()

    assigned = assign_incremental(test_db, [99])
    assert assigned == 1

    cur.execute("SELECT community_id FROM expressions WHERE id = 99")
    assert cur.fetchone()[0] == 1  # Comm 1 wins on weight sum!


def test_incremental_tie_broken_by_weight_sum(test_db):
    """Test that equal weight sum (< 1e-6 diff) stays NULL."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO expression_communities (community_id, name, is_active, node_count) VALUES (1, 'Comm 1', 1, 50), (2, 'Comm 2', 1, 50)")
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (10, 'N1', 1), (20, 'N2', 2), (99, 'Target', NULL)")

    # Identical weights to Comm 1 and Comm 2
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 10, 0.85)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (99, 20, 0.85)")
    test_db.commit()

    assigned = assign_incremental(test_db, [99])
    assert assigned == 0

    cur.execute("SELECT community_id FROM expressions WHERE id = 99")
    assert cur.fetchone()[0] is None  # True tie remains NULL


def test_incremental_no_neighbors_stays_null(test_db):
    """Test that unlinked expression stays NULL."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (100, 'Unlinked', NULL)")
    test_db.commit()

    assigned = assign_incremental(test_db, [100])
    assert assigned == 0

    cur.execute("SELECT community_id FROM expressions WHERE id = 100")
    assert cur.fetchone()[0] is None


def test_incremental_staleness_triggers_medoid_refresh(test_db):
    """Test that >=20% and >=25 node growth triggers medoid refresh and updates medoid_at_node_count."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Book A', 'Author A')")
    cur.execute(
        """
        INSERT INTO expression_communities (
            community_id, name, is_active, node_count, medoid_at_node_count
        ) VALUES (1, 'Comm 1', 1, 100, 100)
        """
    )

    # Add 25 new expressions connected to an existing node in Comm 1
    cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (1, 'Anchor', 1, 1)")
    new_eids = []
    for i in range(101, 126):  # exactly 25 new nodes
        cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (?, ?, 1, NULL)", (i, f"New {i}"))
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, 1, 0.90)", (i,))
        new_eids.append(i)
    test_db.commit()

    assigned = assign_incremental(test_db, new_eids)
    assert assigned == 25

    cur.execute("SELECT node_count, medoid_at_node_count FROM expression_communities WHERE community_id = 1")
    node_cnt, med_cnt = cur.fetchone()
    assert node_cnt == 125
    assert med_cnt == 125  # medoid_at_node_count was refreshed!


def test_medoid_soft_weighted_selects_dominant_author(test_db):
    """Test that Soft-Weighted selection crowns dominant author over minority star-hub."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Novel A', 'Charles Dickens'), (2, 'Essay B', 'Alan Watts')")

    # Dickens has 8 chunks (80% share)
    for i in range(1, 9):
        cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (?, ?, 1, 1)", (i, f"Dickens passage {i}"))
    # Watts has 2 chunks (20% share)
    cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (9, 'Watts bridge', 2, 1), (10, 'Watts 2', 2, 1)")

    # Watts node 9 has 4 links to Dickens (raw strength = 4 * 0.85 = 3.40)
    for i in range(1, 5):
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (9, ?, 0.85)", (i,))

    # Dickens node 1 has 3 links to other Dickens nodes (raw strength = 3 * 0.85 = 2.55)
    for i in range(2, 5):
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (1, ?, 0.85)", (i,))

    test_db.commit()

    # Raw unconstrained degree would pick Watts (3.40 vs 2.55)
    # But Soft-Weighted:
    # Watts: 3.40 * sqrt(0.20) = 3.40 * 0.447 = 1.52
    # Dickens: 2.55 * sqrt(0.80) = 2.55 * 0.894 = 2.28
    medoid = compute_soft_weighted_medoid(test_db, 1)
    assert medoid == 1  # Dickens wins!


def test_naming_fallback_no_llm(test_db):
    """Test fallback naming produces clean author/work title without LLM."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Out of Your Mind - Essential Lectures', 'Alan Watts')")
    cur.execute("INSERT INTO expression_communities (community_id, name, is_active, node_count) VALUES (1, NULL, 1, 10)")
    for i in range(1, 11):
        cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (?, 'Content', 1, 1)", (i,))
    test_db.commit()

    title, source = name_community(test_db, 1)
    assert source == "fallback"
    assert "Alan Watts" in title
    assert "Out of Your Mind" in title


def test_naming_sweep_promotes_fallback(test_db):
    """Test that naming sweep promotes fallback rows when LLM succeeds."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (1, 'Can''t Hurt Me', 'David Goggins')")
    cur.execute(
        """
        INSERT INTO expression_communities (
            community_id, name, name_source, is_active, medoid_expr_id, node_count
        ) VALUES (1, 'David Goggins — Can''t Hurt Me', 'fallback', 1, 1, 50)
        """
    )
    cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (1, 'Medoid passage on pain and running reps', 1, 1)")
    test_db.commit()

    # Mock Gemini naming to return a synthesized conceptual title
    with patch("audiobench.community.naming.name_community_with_gemini", return_value="Mental Toughness and Radical Discipline"):
        upgraded = run_naming_sweep(test_db)
        assert upgraded == 1

        cur.execute("SELECT name, name_source, naming_attempted_at, naming_failure_count FROM expression_communities WHERE community_id = 1")
        name, source, attempted_at, fail_cnt = cur.fetchone()
        assert name == "Mental Toughness and Radical Discipline"
        assert source == "llm_gemini"
        assert attempted_at is not None
        assert fail_cnt == 0


def test_naming_sweep_records_failure_and_backs_off(test_db):
    """Test that failed LLM calls record failure timestamp and back off without infinite retry."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO works (id, title, author) VALUES (10, 'Hard Times', 'Charles Dickens')")
    cur.execute(
        """
        INSERT INTO expression_communities (
            community_id, name, name_source, is_active, medoid_expr_id, node_count
        ) VALUES (10, 'Charles Dickens — Hard Times', 'fallback', 1, 10, 50)
        """
    )
    cur.execute("INSERT INTO expressions (id, content, work_id, community_id) VALUES (10, 'Passage on industrial factory town', 10, 10)")
    test_db.commit()

    # Both LLM tiers return None (e.g. quota exhausted and daemon offline)
    with patch("audiobench.community.naming.name_community_with_gemini", return_value=None), \
         patch("audiobench.community.naming.name_community_with_ollama", return_value=None):
        upgraded = run_naming_sweep(test_db)
        assert upgraded == 0

        # Row must record attempted_at and increment failure count, while preserving fallback title
        cur.execute("SELECT name, name_source, naming_attempted_at, naming_failure_count FROM expression_communities WHERE community_id = 10")
        name, source, attempted_at, fail_cnt = cur.fetchone()
        assert name == "Charles Dickens — Hard Times"
        assert source == "fallback"
        assert attempted_at is not None
        assert fail_cnt == 1

        # Immediate follow-up sweep must skip community 10 due to 30-minute backoff window
        upgraded_again = run_naming_sweep(test_db)
        assert upgraded_again == 0


def test_assign_incremental_staleness_resets_naming_state(test_db):
    """Test that assign_incremental clears naming_attempted_at and failure count when medoids drift."""
    cur = test_db.cursor()
    cur.execute(
        """
        INSERT INTO expression_communities (
            community_id, name, name_source, is_active, medoid_expr_id, node_count, medoid_at_node_count,
            naming_attempted_at, naming_failure_count
        ) VALUES (5, 'Existing Title', 'llm_gemini', 1, 5, 50, 50, '2026-10-15T00:00:00Z', 3)
        """
    )
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (5, 'Existing medoid', 5)")
    # Add an anchor node in comm 5
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (50, 'Anchor node', 5)")

    # Add 26 new unassigned nodes connected to comm 5 (exceeding STALENESS_MIN_NEW_NODES = 25)
    new_ids = list(range(100, 126))
    for nid in new_ids:
        cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (?, 'New chunk', NULL)", (nid,))
        cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (?, 50, 0.90)", (nid,))
    test_db.commit()

    assigned = assign_incremental(test_db, new_ids)
    assert assigned == 26

    # Verify staleness reset: name_source -> fallback, attempted_at -> NULL, failure_count -> 0
    cur.execute("SELECT name_source, naming_attempted_at, naming_failure_count FROM expression_communities WHERE community_id = 5")
    source, attempted_at, fail_cnt = cur.fetchone()
    assert source == "fallback"
    assert attempted_at is None
    assert fail_cnt == 0


def test_community_id_stable_after_incremental(test_db):
    """Test that assign_incremental never mutates existing community assignments."""
    cur = test_db.cursor()
    cur.execute("INSERT INTO expression_communities (community_id, name, is_active, node_count) VALUES (1, 'Comm 1', 1, 50), (2, 'Comm 2', 1, 50)")

    # Expr 1 belongs to Comm 1
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (1, 'Existing', 1)")
    # Expr 2 is new, connected to Comm 2
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (20, 'Comm 2 node', 2)")
    cur.execute("INSERT INTO expressions (id, content, community_id) VALUES (2, 'New node', NULL)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (2, 20, 0.90)")
    test_db.commit()

    assign_incremental(test_db, [2])

    # Expr 1 must remain Comm 1!
    cur.execute("SELECT community_id FROM expressions WHERE id = 1")
    assert cur.fetchone()[0] == 1
    # Expr 2 gets Comm 2
    cur.execute("SELECT community_id FROM expressions WHERE id = 2")
    assert cur.fetchone()[0] == 2
