"""Unit and integration tests for Phase 5 Community Scoping, Playback, and REPL Integration."""

from __future__ import annotations

import re
import sqlite3
from unittest.mock import MagicMock, patch

import click
import numpy as np
from prompt_toolkit.document import Document
import pytest
from rich.console import Console

from audiobench.cli.repl.backslash_commands import (
    cmd_focus,
    cmd_next,
    cmd_play,
    cmd_prev,
    cmd_scope,
    cmd_search,
    cmd_synthesize,
    cmd_unfocus,
)
from audiobench.cli.repl.completion import AudioBenchCompleter
from audiobench.cli.repl.dispatch import dispatch_backslash, print_context_summary
from audiobench.cli.repl.dot_commands import dot_search
from audiobench.cli.repl.session import ReplSession
from audiobench.core.focused_entity import FocusedEntity
from audiobench.memory.synthesis_retriever import (
    SynthesisFragment,
    retrieve_synthesis_context,
)


@pytest.fixture
def mock_db():
    """Create in-memory SQLite database populated with works, transcripts, expressions, and communities."""
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
        CREATE TABLE audio_files (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            duration_seconds REAL
        )
    """)
    cur.execute("""
        CREATE TABLE transcriptions (
            id INTEGER PRIMARY KEY,
            audio_file_id INTEGER REFERENCES audio_files(id),
            full_text TEXT,
            file_name TEXT,
            language TEXT,
            duration REAL,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            transcription_id INTEGER REFERENCES transcriptions(id),
            segment_index INTEGER,
            start_time REAL,
            end_time REAL,
            text TEXT,
            speaker TEXT,
            privacy_tier INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE expressions (
            id INTEGER PRIMARY KEY,
            content TEXT,
            work_id INTEGER REFERENCES works(id),
            community_id INTEGER,
            source_type TEXT DEFAULT 'transcription_chunk',
            source_id INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE expression_segment_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expression_id INTEGER REFERENCES expressions(id),
            segment_id INTEGER REFERENCES segments(id)
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
    cur.execute("INSERT INTO works VALUES (4, 'Psalms', 'King David')")

    # Populate audio files
    cur.execute("INSERT INTO audio_files VALUES (1, 'watts_lecture.mp3', '/media/watts_lecture.mp3', 3600.0)")
    cur.execute("INSERT INTO audio_files VALUES (2, 'voss_negotiation.mp3', '/media/voss_negotiation.mp3', 2400.0)")
    cur.execute("INSERT INTO audio_files VALUES (3, 'solomon_existentialism.mp3', '/media/solomon_existentialism.mp3', 1800.0)")
    cur.execute("INSERT INTO audio_files VALUES (4, 'psalms_recitation.mp3', '/media/psalms_recitation.mp3', 1200.0)")

    # Populate transcriptions
    cur.execute("INSERT INTO transcriptions VALUES (10, 1, 'Watts on realization and presence in everyday life.', 'watts_lecture.mp3', 'en', 3600.0, 500, '2026-01-01')")
    cur.execute("INSERT INTO transcriptions VALUES (20, 2, 'Negotiation tactics and empathy in hostage situations.', 'voss_negotiation.mp3', 'en', 2400.0, 450, '2026-01-01')")
    cur.execute("INSERT INTO transcriptions VALUES (30, 3, 'Existential responsibility and radical freedom in modern thought.', 'solomon_existentialism.mp3', 'en', 1800.0, 350, '2026-01-01')")
    cur.execute("INSERT INTO transcriptions VALUES (40, 4, 'The Lord is my shepherd, a psalm of sacred refuge.', 'psalms_recitation.mp3', 'en', 1200.0, 200, '2026-01-01')")

    # Populate segments
    cur.execute("INSERT INTO segments VALUES (1001, 10, 0, 14.5, 30.0, 'Medoid passage on realization and conversational awareness across teachers.', 'Alan Watts', 0)")
    cur.execute("INSERT INTO segments VALUES (1002, 20, 0, 45.0, 65.0, 'Negotiation requires emotional empathy and calibrated questions rather than force.', 'Chris Voss', 0)")
    cur.execute("INSERT INTO segments VALUES (1003, 30, 0, 80.0, 110.0, 'Existential choice demands taking responsibility for human freedom and action.', 'Robert C. Solomon', 0)")
    cur.execute("INSERT INTO segments VALUES (1004, 40, 0, 5.0, 25.0, 'The Lord is my shepherd, I shall not want. He makes me lie down.', 'Reader', 0)")

    # Populate expressions
    # Community 12 (Watts, Voss, Solomon)
    cur.execute("INSERT INTO expressions VALUES (101, 'Medoid passage on realization and conversational awareness across teachers.', 1, 12, 'transcription_chunk', 10)")
    cur.execute("INSERT INTO expressions VALUES (102, 'Negotiation requires emotional empathy and calibrated questions rather than force.', 2, 12, 'transcription_chunk', 20)")
    cur.execute("INSERT INTO expressions VALUES (103, 'Existential choice demands taking responsibility for human freedom and action.', 3, 12, 'transcription_chunk', 30)")
    cur.execute("INSERT INTO expressions VALUES (104, 'Additional Watts discourse on presence and non-dual understanding.', 1, 12, 'transcription_chunk', 10)")

    # Community 1 (Psalms)
    cur.execute("INSERT INTO expressions VALUES (201, 'The Lord is my shepherd, I shall not want. He makes me lie down.', 4, 1, 'transcription_chunk', 40)")
    cur.execute("INSERT INTO expressions VALUES (202, 'God is our refuge and strength, an ever-present help in trouble.', 4, 1, 'transcription_chunk', 40)")

    # Map expressions to segments
    cur.execute("INSERT INTO expression_segment_map (expression_id, segment_id) VALUES (101, 1001)")
    cur.execute("INSERT INTO expression_segment_map (expression_id, segment_id) VALUES (102, 1002)")
    cur.execute("INSERT INTO expression_segment_map (expression_id, segment_id) VALUES (103, 1003)")
    cur.execute("INSERT INTO expression_segment_map (expression_id, segment_id) VALUES (201, 1004)")

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
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F64F"  # Emoticons, symbols, pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
        "]"
    )
    return bool(emoji_pattern.search(text))


# ── Focus & Scope Command Tests ──────────────────────────────


def test_focus_school_s_prefix(mock_db, repl_session):
    """\\focus s12 should focus on community 12 with FocusedEntity(type='school')."""
    with patch("audiobench.core.db_session.get_session") as mock_sess:
        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                sql = str(stmt).replace(":cid", "?")
                res = cur.execute(sql, (params["cid"],)).fetchall()
                mock_res = MagicMock()
                mock_res.fetchone.return_value = res[0] if res else None
                return mock_res

        mock_sess.return_value = SessionContext()

        test_console = Console(record=True, width=120)
        with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
             patch("audiobench.cli.repl.dispatch.console", test_console):
            cmd_focus(repl_session, "s12")

        assert repl_session.focus is not None
        assert repl_session.focus.type == "school"
        assert repl_session.focus.id == 12
        assert "School #12" in repl_session.focus.label

        out = test_console.export_text()
        assert "Focused on school: #12" in out
        assert "Interpersonal Dynamics" in out
        assert not has_emojis(out)


def test_focus_school_full_prefix(mock_db, repl_session):
    """\\focus school:12 should focus on community 12."""
    with patch("audiobench.core.db_session.get_session") as mock_sess:
        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                sql = str(stmt).replace(":cid", "?")
                res = cur.execute(sql, (params["cid"],)).fetchall()
                mock_res = MagicMock()
                mock_res.fetchone.return_value = res[0] if res else None
                return mock_res

        mock_sess.return_value = SessionContext()

        test_console = Console(record=True, width=120)
        with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
             patch("audiobench.cli.repl.dispatch.console", test_console):
            cmd_focus(repl_session, "school:12")

        assert repl_session.focus is not None
        assert repl_session.focus.type == "school"
        assert repl_session.focus.id == 12


def test_focus_school_not_found(mock_db, repl_session):
    """\\focus s999 should report School #999 not found."""
    with patch("audiobench.core.db_session.get_session") as mock_sess:
        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                mock_res = MagicMock()
                mock_res.fetchone.return_value = None
                return mock_res

        mock_sess.return_value = SessionContext()

        test_console = Console(record=True, width=120)
        with patch("audiobench.cli.repl.backslash_commands.console", test_console):
            cmd_focus(repl_session, "s999")

        assert repl_session.focus is None
        out = test_console.export_text()
        assert "School #999 not found" in out
        assert not has_emojis(out)


def test_scope_command_school_and_clear(mock_db, repl_session):
    """\\scope school:12 should set school focus, and \\scope clear should clear it."""
    with patch("audiobench.core.db_session.get_session") as mock_sess:
        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                sql = str(stmt).replace(":cid", "?")
                res = cur.execute(sql, (params["cid"],)).fetchall()
                mock_res = MagicMock()
                mock_res.fetchone.return_value = res[0] if res else None
                return mock_res

        mock_sess.return_value = SessionContext()

        test_console = Console(record=True, width=120)
        with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
             patch("audiobench.cli.repl.dispatch.console", test_console):
            # 1. Scope to school:12
            cmd_scope(repl_session, "school:12")
            assert repl_session.focus is not None
            assert repl_session.focus.type == "school"
            assert repl_session.focus.id == 12

            # 2. Scope bare: prints summary
            cmd_scope(repl_session, "")
            out1 = test_console.export_text()
            assert "Focused on school: #12" in out1

            # 3. Scope clear
            cmd_scope(repl_session, "clear")
            assert repl_session.focus is None
            out2 = test_console.export_text()
            assert "Focus cleared" in out2
            assert not has_emojis(out2)


def test_scope_command_no_scope_message(repl_session):
    """Bare \\scope with no focus displays helpful usage."""
    test_console = Console(record=True, width=120)
    with patch("audiobench.cli.repl.backslash_commands.console", test_console):
        cmd_scope(repl_session, "")

    out = test_console.export_text()
    assert "No active scope" in out
    assert "\\scope school:<id>" in out
    assert not has_emojis(out)


# ── Retrieval & Synthesis Scoping Tests ───────────────────────


def test_retrieve_synthesis_context_scoped(mock_db):
    """retrieve_synthesis_context with community_id=12 must only return expressions from community 12."""
    mock_hnsw = MagicMock()
    # Mock query returns hits from both community 12 (101, 102) and community 1 (201)
    mock_hnsw.query.return_value = [(101, 0.15), (201, 0.20), (102, 0.25)]
    mock_hnsw.has_item.return_value = True
    v = np.random.randn(384).astype(np.float32)
    v /= np.linalg.norm(v)
    mock_hnsw.get_items.return_value = [v]

    with patch("audiobench.memory.synthesis_retriever.ensure_hnsw_ready", return_value=mock_hnsw), \
         patch("audiobench.memory.synthesis_retriever.embed_query", return_value=v), \
         patch("audiobench.memory.synthesis_retriever.get_session") as mock_sess:

        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                sql = str(stmt)
                # Translate named parameters
                if ":cid" in sql:
                    cid = params["cid"]
                    p_keys = [k for k in params.keys() if k.startswith("id")]
                    eids = [params[k] for k in p_keys]
                    placeholders = ",".join("?" * len(eids))
                    query_sql = f"""
                        SELECT e.id, e.work_id, e.content, w.title, w.author, e.source_type, NULL, NULL
                        FROM expressions e
                        LEFT JOIN works w ON e.work_id = w.id
                        WHERE e.id IN ({placeholders}) AND e.community_id = ?
                    """
                    rows = cur.execute(query_sql, (*eids, cid)).fetchall()
                else:
                    rows = []
                mock_r = MagicMock()
                mock_r.fetchall.return_value = rows
                return mock_r

        mock_sess.return_value = SessionContext()

        frags = retrieve_synthesis_context("interpersonal negotiation", top_k=2, community_id=12)

        assert len(frags) > 0
        for f in frags:
            # Check expression belongs to community 12
            cur = mock_db.cursor()
            comm = cur.execute("SELECT community_id FROM expressions WHERE id = ?", (f.expression_id,)).fetchone()
            assert comm[0] == 12


def test_cmd_synthesize_school_scoped(repl_session):
    """\\synthesize when focused on a school should pass community_id and print scoped banner."""
    repl_session.focus = FocusedEntity(
        type="school",
        id=12,
        label="School #12: Interpersonal Dynamics",
    )

    from audiobench.chat.synthesis_engine import SynthesisResponse
    mock_response = SynthesisResponse(
        query="conflict resolution",
        synthesis_text="Synthesized dialectic on conflict and communication [1].",
        citations={1: 101},
        fragments=[
            SynthesisFragment(
                expression_id=101,
                work_id=1,
                work_title="Out of Your Mind",
                author="Alan Watts",
                content="Sample content",
                q_sim=0.88,
                edge_weight=1.0,
                effective_rel=0.88,
                mmr_score=0.88,
                source_mode="ANN",
            )
        ],
        elapsed_seconds=0.12,
        model_used="mock_model",
    )

    test_console = Console(record=True, width=120)
    with patch("audiobench.chat.synthesis_engine.generate_synthesis", return_value=mock_response) as mock_gen, \
         patch("audiobench.cli.repl.backslash_commands.console", test_console):
        cmd_synthesize(repl_session, "conflict resolution")

    mock_gen.assert_called_once_with("conflict resolution", community_id=12)
    out = test_console.export_text()
    assert "‹Scoped to School #12: Interpersonal Dynamics›" in out
    assert "Synthesized dialectic on conflict" in out
    assert not has_emojis(out)


# ── Search Scoping Tests ─────────────────────────────────────


def test_cmd_search_school_scoped(repl_session):
    """\\search when focused on a school should pass community_id to repository."""
    repl_session.focus = FocusedEntity(
        type="school",
        id=12,
        label="School #12: Interpersonal Dynamics",
    )

    test_console = Console(record=True, width=120)
    mock_repo = MagicMock()
    mock_repo.search.return_value = [
        {"id": 10, "file_name": "watts_lecture.mp3", "text_preview": "Watts on realization"}
    ]

    with patch("audiobench.storage.repository.TranscriptionRepository", return_value=mock_repo), \
         patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("builtins.input", return_value="q"):
        cmd_search(repl_session, "realization")

    mock_repo.search.assert_called_once_with("realization", limit=10, session=repl_session, community_id=12, work_ids=None)
    out = test_console.export_text()
    assert "‹Scoped to School #12: Interpersonal Dynamics›" in out
    assert "watts_lecture.mp3" in out
    assert not has_emojis(out)


# ── Playback & Queue Navigation Tests ─────────────────────────


def test_play_school_sequential_queue(mock_db, repl_session):
    """\\play school 12 should build a playlist, start track 1, and \\next / \\prev should navigate."""
    test_console = Console(record=True, width=120)
    mock_client = MagicMock()

    with patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db), \
         patch("audiobench.core.db_session.get_session") as mock_sess, \
         patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(14.5, 30.0, "Speaker 1")), \
         patch("audiobench.cli.repl.backslash_commands.console", test_console):

        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                eid = params["eid"]
                sql = """
                    SELECT e.id, e.source_id, e.work_id, e.content,
                           t.audio_file_id, af.file_name, af.file_path,
                           w.title as work_title, w.author as work_author
                    FROM expressions e
                    LEFT JOIN transcriptions t ON e.source_id = t.id
                    LEFT JOIN audio_files af ON t.audio_file_id = af.id
                    LEFT JOIN works w ON e.work_id = w.id
                    WHERE e.id = ?
                """
                row = cur.execute(sql, (eid,)).fetchone()
                mock_r = MagicMock()
                mock_r.fetchone.return_value = row
                return mock_r

        mock_sess.return_value = SessionContext()

        # 1. Play school 12
        cmd_play(repl_session, "school 12")

        assert repl_session.school_playlist is not None
        assert repl_session.school_playlist["community_id"] == 12
        assert len(repl_session.school_playlist["items"]) >= 1
        assert repl_session.school_playlist["index"] == 0

        # Verify first track played
        mock_client.playback_play.assert_called()
        track0 = repl_session.school_playlist["items"][0]
        assert mock_client.playback_play.call_args[0][0] == track0["file_path"]

        out1 = test_console.export_text()
        assert "Playing School #12" in out1
        assert not has_emojis(out1)

        # 2. Advance to next track with \next
        if len(repl_session.school_playlist["items"]) > 1:
            cmd_next(repl_session, "")
            assert repl_session.school_playlist["index"] == 1
            track1 = repl_session.school_playlist["items"][1]
            assert mock_client.playback_play.call_args[0][0] == track1["file_path"]

            # 3. Return to previous track with \prev
            cmd_prev(repl_session, "")
            assert repl_session.school_playlist["index"] == 0
            assert mock_client.playback_play.call_args[0][0] == track0["file_path"]


def test_play_school_not_found(mock_db, repl_session):
    """\\play school 999 should display error message."""
    test_console = Console(record=True, width=120)
    mock_client = MagicMock()

    with patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db), \
         patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.cli.repl.backslash_commands.console", test_console):
        cmd_play(repl_session, "school 999")

    out = test_console.export_text()
    assert "School #999 not found" in out
    assert not has_emojis(out)


# ── Tab Completion Tests ─────────────────────────────────────


def test_school_scoping_completions(mock_db, repl_session):
    """Completer should suggest school IDs for \\scope school: and \\play school."""
    with patch("audiobench.core.settings.get_settings") as mock_set:
        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite:///:memory:"
        mock_set.return_value = mock_settings

        with patch("sqlite3.connect", return_value=mock_db):
            completer = AudioBenchCompleter(repl_session)

            # 1. \scope bare tokens -> school:, work:, clear
            doc_scope = Document("\\scope ")
            completions_scope = list(completer.get_completions(doc_scope, None))
            texts_scope = [c.text for c in completions_scope]
            assert "school:" in texts_scope
            assert "work:" in texts_scope
            assert "clear" in texts_scope

            # 2. \scope school: partial -> school:1, school:12
            doc_school_prefix = Document("\\scope school:")
            completions_prefix = list(completer.get_completions(doc_school_prefix, None))
            texts_prefix = [c.text for c in completions_prefix]
            assert "school:1" in texts_prefix
            assert "school:12" in texts_prefix

            # 3. \play school -> 1, 12
            doc_play_school = Document("\\play school ")
            completions_play = list(completer.get_completions(doc_play_school, None))
            texts_play = [c.text for c in completions_play]
            assert "1" in texts_play
            assert "12" in texts_play

            # 4. \focus s -> s1, s12
            doc_focus_s = Document("\\focus s")
            completions_focus_s = list(completer.get_completions(doc_focus_s, None))
            texts_focus_s = [c.text for c in completions_focus_s]
            assert "s1" in texts_focus_s
            assert "s12" in texts_focus_s


# ── Strict Zero-Emoji Output Verification ────────────────────


def test_zero_emojis_across_phase5_outputs(mock_db, repl_session):
    """Verify strictly NO emojis are emitted across all Phase 5 command outputs."""
    test_console = Console(record=True, width=120)

    with patch("audiobench.cli.repl.backslash_commands.console", test_console), \
         patch("audiobench.cli.repl.dispatch.console", test_console), \
         patch("audiobench.cli.repl.backslash_commands._get_db_connection", return_value=mock_db), \
         patch("audiobench.core.db_session.get_session") as mock_disp_sess:

        class SessionContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, stmt, params=None):
                cur = mock_db.cursor()
                sql = str(stmt).replace(":cid", "?")
                res = cur.execute(sql, (params["cid"],)).fetchall()
                mock_res = MagicMock()
                mock_res.fetchone.return_value = res[0] if res else None
                return mock_res

        mock_disp_sess.return_value = SessionContext()

        # Test context summary
        repl_session.focus = FocusedEntity(type="school", id=12, label="School #12")
        print_context_summary(repl_session)

        # Test scope bare & clear
        cmd_scope(repl_session, "")
        cmd_scope(repl_session, "clear")

        all_output = test_console.export_text()
        assert not has_emojis(all_output), f"Emojis found in output: {all_output}"
