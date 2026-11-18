"""Unit and integration tests for School Two-Column Book Mode and Dialectic Relations."""

from __future__ import annotations

import re
import sqlite3
from unittest.mock import MagicMock, patch

import click
from prompt_toolkit.document import Document
import pytest
from rich.console import Console

from audiobench.cli.repl.backslash_commands import (
    cmd_focus,
    cmd_related,
    cmd_school,
)
from audiobench.cli.repl.completion import AudioBenchCompleter
from audiobench.cli.repl.dispatch import _BACKSLASH_HANDLERS
from audiobench.cli.repl.session import ReplSession
from audiobench.community.compute_medoids import format_sample_passage
from audiobench.core.focused_entity import FocusedEntity


def test_format_sample_passage_sentence_boundaries():
    """Verify format_sample_passage snaps to clean sentence boundaries (. ! ?)."""
    text = (
        "This is the first sentence. Here is the second sentence that contains useful thought! "
        "And here is the third sentence that might be cut off if it exceeds the limit."
    )
    formatted = format_sample_passage(text, max_chars=110)
    assert formatted.endswith("!")
    assert "third sentence" not in formatted

    # With question mark
    text_q = "What is the nature of consciousness? It is the silent awareness behind thought. That is all."
    formatted_q = format_sample_passage(text_q, max_chars=50)
    assert formatted_q == "What is the nature of consciousness?"


@pytest.fixture
def school_test_db():
    """Create in-memory SQLite database populated with two connected schools and bridge edges."""
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
            updated_at            DATETIME
        )
    """)

    # Populate works
    cur.execute("INSERT INTO works VALUES (1, 'The Power of Now', 'Eckhart Tolle')")
    cur.execute("INSERT INTO works VALUES (2, 'Out of Your Mind', 'Alan Watts')")
    cur.execute("INSERT INTO works VALUES (3, 'Reflections on the Self', 'Jiddu Krishnamurti')")
    cur.execute("INSERT INTO works VALUES (4, 'Meditations', 'Marcus Aurelius')")
    cur.execute("INSERT INTO works VALUES (5, 'No Excuses', 'Robert Solomon')")
    cur.execute("INSERT INTO works VALUES (6, 'Mans Search for Meaning', 'Viktor Frankl')")
    cur.execute("INSERT INTO works VALUES (7, 'Psalms', 'Unknown Author')")

    # Populate audio files & transcriptions
    for i in range(1, 8):
        cur.execute(
            "INSERT INTO audio_files VALUES (?, ?, ?, 1800.0)",
            (i, f"track_{i:02d}.mp3", f"/media/track_{i:02d}.mp3"),
        )
        cur.execute(
            "INSERT INTO transcriptions VALUES (?, ?, 'Full text', ?, 'en', 1800.0, 300, CURRENT_TIMESTAMP)",
            (i, i, f"track_{i:02d}.mp3"),
        )
        cur.execute(
            "INSERT INTO segments VALUES (?, ?, 0, 10.0, 30.0, 'Segment speech text', 'Speaker A', 0)",
            (i, i),
        )

    # Populate expressions for School #02 (nodes 101 to 106)
    passages = [
        "Emotion arises at the place where mind and body meet. It is the body reaction to your mind. Watch consciousness.",
        "You see, about death, the reality of approaching death, people feel chilly. Life and death are two aspects of one reality.",
        "When one realizes that, faces that, either it brings great despair or true freedom from the known.",
        "Distinguish the material from the causal. In every impulse, give what is right. Keep steady.",
        "It is the constant reminder that we are not being all that we can be. Existential choice matters.",
        "I told him that this was not my way, that I had learned to let fate take its course. Keep your inner dignity.",
    ]
    for idx, (wid, passg) in enumerate(zip(range(1, 7), passages), 1):
        cur.execute(
            "INSERT INTO expressions VALUES (?, ?, ?, 2, 'transcription_chunk', ?)",
            (100 + idx, passg, wid, wid),
        )

    # Populate expressions for School #01 (nodes 201 to 202)
    cur.execute(
        "INSERT INTO expressions VALUES (201, 'The Lord is my shepherd, I shall not want. He makes me lie down in green pastures.', 7, 1, 'transcription_chunk', 7)"
    )
    cur.execute(
        "INSERT INTO expressions VALUES (202, 'Wisdom is a shelter as money is a shelter. But the advantage of knowledge preserves life.', 7, 1, 'transcription_chunk', 7)"
    )

    # Internal edges for School #02
    for u in range(101, 106):
        cur.execute("INSERT INTO expression_relations VALUES (NULL, ?, ?, 'semantic', 0.85, CURRENT_TIMESTAMP)", (u, u + 1))
    cur.execute("INSERT INTO expression_relations VALUES (NULL, 101, 106, 'semantic', 0.88, CURRENT_TIMESTAMP)")

    # Cross-school bridge edges between School #02 and School #01
    cur.execute("INSERT INTO expression_relations VALUES (NULL, 101, 201, 'semantic', 0.84, CURRENT_TIMESTAMP)")
    cur.execute("INSERT INTO expression_relations VALUES (NULL, 103, 202, 'semantic', 0.82, CURRENT_TIMESTAMP)")

    # Insert expression_communities
    cur.execute("""
        INSERT INTO expression_communities VALUES
        (2, 'Watching Consciousness and Ego Transcendence', 'llm_gemini', 1, 101, 6, 6, 0.90, 42, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
        (1, 'Biblical Psalms and Divine Refuge', 'llm_gemini', 1, 201, 2, 2, 0.90, 42, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """)

    conn.commit()
    return conn


def test_school_two_column_book_mode(school_test_db, monkeypatch):
    """Verify \\school renders 6 key sample expressions in two-column Book Mode with single center divider."""
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._get_db_connection",
        lambda: school_test_db,
    )

    test_console = Console(width=120, force_terminal=True, color_system=None, record=True)
    test_console.size = (120, 40)
    monkeypatch.setattr("audiobench.cli.display.theme.console", test_console)
    monkeypatch.setattr("audiobench.cli.repl.backslash_commands.console", test_console)

    cli = click.Group()
    session = ReplSession(cli)

    cmd_school(session, "2")
    output = test_console.export_text()

    # 1. School header and metadata
    assert "School #02: Watching Consciousness and Ego Transcendence" in output
    assert "Medoid: Expr #101" in output
    assert "Medoid Passage" in output

    # 2. Key Sample Expressions header & Book Mode
    assert "Key Sample Expressions" in output
    assert "2-column book mode" in output

    # 3. Single center column divider '│'
    assert "│" in output

    # 4. Expressions present
    assert "Alan Watts" in output
    assert "Jiddu Krishnamurti" in output

    # 5. Continuous single panel: quick actions enclosed inside, no preceding lines before titles
    assert "Quick actions:" in output
    assert "── Alan Watts" not in output
    assert "── Jiddu" not in output
    stripped = output.strip()
    assert stripped.startswith("┌") or stripped.startswith("╭")
    assert stripped.endswith("┘") or stripped.endswith("╯")

    # 6. Zero emojis check
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")
    assert not emoji_pattern.findall(output)


def test_school_narrow_terminal_single_column_fallback(school_test_db, monkeypatch):
    """Verify \\school falls back cleanly to single-column display when terminal width < 100."""
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._get_db_connection",
        lambda: school_test_db,
    )

    test_console = Console(width=80, force_terminal=True, color_system=None, record=True)
    test_console.size = (80, 40)
    monkeypatch.setattr("audiobench.cli.display.theme.console", test_console)
    monkeypatch.setattr("audiobench.cli.repl.backslash_commands.console", test_console)

    cli = click.Group()
    session = ReplSession(cli)

    cmd_school(session, "2")
    output = test_console.export_text()

    assert "single column" in output
    assert "Alan Watts" in output
    # Zero emojis check
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")
    assert not emoji_pattern.findall(output)


def test_related_command_with_school_focus(school_test_db, monkeypatch):
    """Verify \\related with school focus shows macro connected schools and micro bridge pairs in Book Mode."""
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._get_db_connection",
        lambda: school_test_db,
    )

    test_console = Console(width=120, force_terminal=True, color_system=None, record=True)
    test_console.size = (120, 40)
    monkeypatch.setattr("audiobench.cli.display.theme.console", test_console)
    monkeypatch.setattr("audiobench.cli.repl.backslash_commands.console", test_console)

    # Mock DB query for outside audio file
    class MockSdb:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def query(self, model):
            mock_q = MagicMock()
            mock_tx = MagicMock(id=7, audio_file_id=7)
            mock_af = MagicMock(id=7, file_name="track_07.mp3", file_path="/media/track_07.mp3", work=MagicMock(title="Psalms", author="Unknown Author"))
            mock_q.filter_by.return_value.first.side_effect = lambda **kwargs: mock_tx if "id" in kwargs else mock_af
            return mock_q

    monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: MockSdb())
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._resolve_thought_segment",
        lambda sdb, expr_id, tx_id, content: (12.0, 36.0, "Narrator"),
    )

    cli = click.Group()
    session = ReplSession(cli)
    session.focus = FocusedEntity(type="school", id=2, label="School #02: Watching Consciousness")

    cmd_related(session, "")
    output = test_console.export_text()

    # 1. School Dialectic Relations header
    assert "School Dialectic Relations" in output
    assert "School #02" in output

    # 2. Section 1: Connected Schools
    assert "Connected Schools (Macro Coupling)" in output
    assert "School #01: Biblical Psalms and Divine Refuge" in output

    # 3. Section 2: Semantic Bridge Passages (Two-Column Book Mode)
    assert "Semantic Bridge Passages" in output
    assert "2-column book mode" in output
    assert "│" in output
    assert "School #02 (Inside Core)" in output
    assert "Connected Thoughts (Bridge Out)" in output

    # 4. Session registration for downstream actions
    assert len(session.last_related) > 0
    bridge_thought = session.last_related[0]
    assert bridge_thought.work_title == "Psalms"
    assert bridge_thought.audio_file_id == 7

    # 5. Downstream action: \\focus 1
    # Mock repo.get_audio_file for cmd_focus
    mock_repo = MagicMock()
    mock_repo.get_audio_file.return_value = {"id": 7, "file_name": "track_07.mp3", "duration_seconds": 1800.0}
    mock_repo.get_latest_transcript_for_audio_file.return_value = {"id": 7, "word_count": 300}
    mock_repo.get_by_id.return_value = {"id": 7, "word_count": 300}
    monkeypatch.setattr("audiobench.storage.repository.TranscriptionRepository", lambda: mock_repo)

    cmd_focus(session, "1")
    focus_output = test_console.export_text()
    assert "Focus switched to thought #1" in focus_output
    assert session.focus.id == 7
    assert session.focus.type == "file"

    # Zero emojis check
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")
    assert not emoji_pattern.findall(output)


def test_related_school_subcommand_syntax(school_test_db, monkeypatch):
    """Verify \\related school 2 executes school dialectic relations without requiring focus first."""
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._get_db_connection",
        lambda: school_test_db,
    )

    test_console = Console(width=120, force_terminal=True, color_system=None, record=True)
    test_console.size = (120, 40)
    monkeypatch.setattr("audiobench.cli.display.theme.console", test_console)
    monkeypatch.setattr("audiobench.cli.repl.backslash_commands.console", test_console)

    class MockSdb:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def query(self, model):
            mock_q = MagicMock()
            mock_q.filter_by.return_value.first.return_value = None
            return mock_q

    monkeypatch.setattr("audiobench.core.db_session.get_session", lambda: MockSdb())
    monkeypatch.setattr(
        "audiobench.cli.repl.backslash_commands._resolve_thought_segment",
        lambda sdb, expr_id, tx_id, content: (0.0, 0.0, None),
    )

    cli = click.Group()
    session = ReplSession(cli)
    session.focus = None

    cmd_related(session, "school 2")
    output = test_console.export_text()

    assert "School Dialectic Relations" in output
    assert "School #02" in output
    assert "Biblical Psalms and Divine Refuge" in output


def test_lyrics_karaoke_follow_registration_and_completion():
    """Verify \\lyrics, \\karaoke, and \\follow are registered and complete in REPL."""
    for cmd in ("lyrics", "karaoke", "follow"):
        assert cmd in _BACKSLASH_HANDLERS

    cli = click.Group()
    session = ReplSession(cli)
    completer = AudioBenchCompleter(session=session)

    doc = Document("\\lyr", 4)
    completions = [c.text for c in completer.get_completions(doc, None)]
    assert "\\lyrics" in completions

    doc_k = Document("\\kar", 4)
    completions_k = [c.text for c in completer.get_completions(doc_k, None)]
    assert "\\karaoke" in completions_k

    doc_f = Document("\\fol", 4)
    completions_f = [c.text for c in completer.get_completions(doc_f, None)]
    assert "\\follow" in completions_f
