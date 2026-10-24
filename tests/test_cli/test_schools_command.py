"""Unit tests for \\schools and \\school REPL commands and tab completions."""

from __future__ import annotations

import re
import sqlite3
from unittest.mock import patch

import click
from prompt_toolkit.document import Document
import pytest
from rich.console import Console

from audiobench.cli.repl.backslash_commands import cmd_school, cmd_schools
from audiobench.cli.repl.completion import AudioBenchCompleter
from audiobench.cli.repl.session import ReplSession


@pytest.fixture
def mock_db():
    """Create in-memory SQLite database populated with test schools and expressions."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

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
            updated_at            DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        )
    """)

    # Populate sample works
    cur.execute("INSERT INTO works VALUES (1, 'Out of Your Mind', 'Alan Watts')")
    cur.execute("INSERT INTO works VALUES (2, 'Never Split the Difference', 'Chris Voss')")
    cur.execute("INSERT INTO works VALUES (3, 'No Excuses', 'Robert C. Solomon')")
    cur.execute("INSERT INTO works VALUES (4, 'Psalms', NULL)")

    # Populate expressions
    # Community 12 (Watts, Voss, Solomon)
    cur.execute("INSERT INTO expressions VALUES (101, 'Medoid passage on realization and conversational awareness across teachers.', 1, 12)")
    cur.execute("INSERT INTO expressions VALUES (102, 'Negotiation requires emotional empathy and calibrated questions rather than force.', 2, 12)")
    cur.execute("INSERT INTO expressions VALUES (103, 'Existential choice demands taking responsibility for human freedom and action.', 3, 12)")
    cur.execute("INSERT INTO expressions VALUES (104, 'Additional Watts discourse on presence and non-dual understanding.', 1, 12)")

    # Community 1 (Psalms)
    cur.execute("INSERT INTO expressions VALUES (201, 'The Lord is my shepherd, I shall not want. He makes me lie down.', 4, 1)")
    cur.execute("INSERT INTO expressions VALUES (202, 'God is our refuge and strength, an ever-present help in trouble.', 4, 1)")

    # Edge relations for community 12
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (101, 102, 0.85)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (101, 103, 0.82)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (102, 103, 0.80)")
    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (101, 104, 0.88)")

    # Communities table
    cur.execute("""
        INSERT INTO expression_communities (community_id, name, name_source, is_active, medoid_expr_id, node_count, resolution, seed)
        VALUES (1, 'Biblical Psalms and Divine Refuge', 'fallback', 1, 201, 2, 0.6, 42)
    """)
    cur.execute("""
        INSERT INTO expression_communities (community_id, name, name_source, is_active, medoid_expr_id, node_count, resolution, seed)
        VALUES (12, 'Interpersonal Dynamics and Conversational Psychology', 'fallback', 1, 101, 4, 0.6, 42)
    """)

    conn.commit()
    return conn


@pytest.fixture
def repl_session():
    group = click.Group("test")
    return ReplSession(group)


def has_emojis(text: str) -> bool:
    """Return True if text contains characters from common emoji unicode blocks."""
    # Matches common emoji ranges: Emoticons, Misc Symbols & Pictographs, Transport/Map, Supplemental Symbols
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F64F"  # Emoticons, symbols, pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "]"
    )
    return bool(emoji_pattern.search(text))


def test_schools_command_renders_table(mock_db, repl_session):
    test_console = Console(record=True, color_system=None, width=120)

    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db):
        cmd_schools(repl_session, "")

    output = test_console.export_text()
    assert "Schools of Thought (2 Active Communities)" in output
    assert "#01" in output
    assert "Biblical Psalms and Divine Refuge" in output
    assert "#12" in output
    assert "Interpersonal Dynamics" in output
    assert "Psychology" in output
    assert "Alan Watts" in output
    assert not has_emojis(output)


def test_schools_command_empty(repl_session):
    empty_db = sqlite3.connect(":memory:")
    empty_db.execute("""
        CREATE TABLE expression_communities (
            community_id INTEGER PRIMARY KEY,
            name TEXT,
            node_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1
        )
    """)
    empty_db.commit()

    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=empty_db):
        cmd_schools(repl_session, "")

    output = test_console.export_text()
    assert "No active schools of thought found" in output
    assert not has_emojis(output)


def test_school_command_no_args(repl_session):
    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console):
        cmd_school(repl_session, "")

    output = test_console.export_text()
    assert "Usage: \\school <id>" in output
    assert not has_emojis(output)


def test_school_command_invalid_id(repl_session):
    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console):
        cmd_school(repl_session, "non_numeric")

    output = test_console.export_text()
    assert "Invalid school ID: 'non_numeric'" in output
    assert not has_emojis(output)


def test_school_command_not_found(mock_db, repl_session):
    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db):
        cmd_school(repl_session, "999")

    output = test_console.export_text()
    assert "School #999 not found or inactive" in output
    assert not has_emojis(output)


def test_school_command_valid_dossier(mock_db, repl_session):
    test_console = Console(record=True, color_system=None, width=120)

    # Test with plain integer "12"
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db):
        cmd_school(repl_session, "12")

    output = test_console.export_text()
    assert "School #12: Interpersonal Dynamics and Conversational Psychology" in output
    assert "Size: 4 chunks" in output
    assert "Resolution: 0.60" in output
    assert "Seed: 42" in output
    assert "Medoid: Expr #101" in output
    assert "Medoid Passage (Core Center)" in output
    assert "Alan Watts, Out of Your Mind" in output
    assert "Top Works Breakdown" in output
    assert "Alan Watts — Out of Your Mind" in output
    assert "Chris Voss — Never Split the Difference" in output
    assert "Key Sample Expressions" in output
    assert not has_emojis(output)


def test_school_command_supports_hash_prefix(mock_db, repl_session):
    test_console = Console(record=True, color_system=None, width=120)

    # Test with prefixed "#12"
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db):
        cmd_school(repl_session, "#12")

    output = test_console.export_text()
    assert "School #12: Interpersonal Dynamics and Conversational Psychology" in output
    assert not has_emojis(output)


def test_school_tab_completion(mock_db, repl_session):
    completer = AudioBenchCompleter(repl_session)

    with patch("sqlite3.connect", return_value=mock_db):
        # 1. Backslash command prefix
        sch_comps = [c.text for c in completer.get_completions(Document("\\sch"), None)]
        assert "\\schools" in sch_comps
        assert "\\school" in sch_comps

        # 2. Argument completions after space: \school <space>
        school_arg_comps = list(completer.get_completions(Document("\\school "), None))
        cand_ids = [c.text for c in school_arg_comps]
        assert "1" in cand_ids
        assert "12" in cand_ids
        # Check display meta contains school title
        c12 = next(c for c in school_arg_comps if c.text == "12")
        assert "Interpersonal Dynamics and Conversational Psychology" in c12.display_meta_text

        # 3. Partial argument completion: \school 1
        school_part_comps = list(completer.get_completions(Document("\\school 1"), None))
        part_ids = [c.text for c in school_part_comps]
        assert "1" in part_ids
        assert "12" in part_ids


def test_dispatch_backslash_routing(mock_db, repl_session):
    from audiobench.cli.repl.dispatch import dispatch_backslash

    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db):
        assert dispatch_backslash(repl_session, "schools") is True
        assert dispatch_backslash(repl_session, "school 12") is True
        assert dispatch_backslash(repl_session, "school") is True


def test_school_command_zero_clean_samples_all_boilerplate(repl_session):
    """When a community has zero clean non-medoid candidates (all others are boilerplate),
    it must degrade gracefully and display '(No additional sample expressions found)'."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE works (id INTEGER PRIMARY KEY, title TEXT, author TEXT)")
    cur.execute("CREATE TABLE expressions (id INTEGER PRIMARY KEY, content TEXT, work_id INTEGER, community_id INTEGER)")
    cur.execute("CREATE TABLE expression_relations (id INTEGER PRIMARY KEY, from_expression_id INTEGER, to_expression_id INTEGER, relation_type TEXT, weight REAL)")
    cur.execute("""
        CREATE TABLE expression_communities (
            community_id INTEGER PRIMARY KEY, name TEXT, name_source TEXT, is_active INTEGER DEFAULT 1,
            medoid_expr_id INTEGER, node_count INTEGER DEFAULT 0, medoid_at_node_count INTEGER DEFAULT 0,
            resolution REAL DEFAULT 0.6, seed INTEGER DEFAULT 42
        )
    """)
    cur.execute("INSERT INTO works VALUES (1, 'Legit Work', 'Author A')")
    cur.execute("INSERT INTO works VALUES (2, 'YouTube Video', NULL)")

    # 1 legit medoid
    cur.execute("INSERT INTO expressions VALUES (1, 'Substantive philosophical medoid chunk that has enough length and depth.', 1, 99)")
    # 3 boilerplate chunks (e.g. outro signatures)
    cur.execute("INSERT INTO expressions VALUES (2, 'Alright guys, I guess we will go ahead and shut this one off. Thanks for watching.', 2, 99)")
    cur.execute("INSERT INTO expressions VALUES (3, 'Don not forget to like and subscribe to our channel for more videos. Thank you.', 2, 99)")
    cur.execute("INSERT INTO expressions VALUES (4, 'Short', 2, 99)")

    cur.execute("""
        INSERT INTO expression_communities (community_id, name, name_source, is_active, medoid_expr_id, node_count)
        VALUES (99, 'Ephemeral Island', 'fallback', 1, 1, 4)
    """)
    conn.commit()

    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=conn):
        cmd_school(repl_session, "99")

    output = test_console.export_text()
    assert "School #99: Ephemeral Island" in output
    assert "Medoid Passage (Core Center)" in output
    assert "(No additional sample expressions found)" in output
    assert not has_emojis(output)


def test_school_command_partial_clean_samples(repl_session):
    """When a community has only 1 clean candidate besides the medoid,
    it must display exactly that 1 sample and not crash trying to fetch 3."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE works (id INTEGER PRIMARY KEY, title TEXT, author TEXT)")
    cur.execute("CREATE TABLE expressions (id INTEGER PRIMARY KEY, content TEXT, work_id INTEGER, community_id INTEGER)")
    cur.execute("CREATE TABLE expression_relations (id INTEGER PRIMARY KEY, from_expression_id INTEGER, to_expression_id INTEGER, relation_type TEXT, weight REAL)")
    cur.execute("""
        CREATE TABLE expression_communities (
            community_id INTEGER PRIMARY KEY, name TEXT, name_source TEXT, is_active INTEGER DEFAULT 1,
            medoid_expr_id INTEGER, node_count INTEGER DEFAULT 0, medoid_at_node_count INTEGER DEFAULT 0,
            resolution REAL DEFAULT 0.6, seed INTEGER DEFAULT 42
        )
    """)
    cur.execute("INSERT INTO works VALUES (1, 'Philosophical Discourse', 'Thinker A')")
    cur.execute("INSERT INTO works VALUES (2, 'Second Book', 'Thinker B')")

    cur.execute("INSERT INTO expressions VALUES (1, 'First substantive chunk that serves as the cluster medoid center.', 1, 77)")
    cur.execute("INSERT INTO expressions VALUES (2, 'The only other substantive thought that exists in this sparse community structure.', 2, 77)")
    cur.execute("INSERT INTO expressions VALUES (3, 'This is a LibriVox recording. All LibriVox recordings are in the public domain.', 2, 77)")

    cur.execute("INSERT INTO expression_relations (from_expression_id, to_expression_id, weight) VALUES (1, 2, 0.85)")

    cur.execute("""
        INSERT INTO expression_communities (community_id, name, name_source, is_active, medoid_expr_id, node_count)
        VALUES (77, 'Sparse Dialogue', 'fallback', 1, 1, 3)
    """)
    conn.commit()

    test_console = Console(record=True, color_system=None, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=conn):
        cmd_school(repl_session, "77")

    output = test_console.export_text()
    assert "School #77: Sparse Dialogue" in output
    assert "Expr #2" in output
    assert "The only other substantive thought" in output
    assert "LibriVox" not in output
    assert not has_emojis(output)


