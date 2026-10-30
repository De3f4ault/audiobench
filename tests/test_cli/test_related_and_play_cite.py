"""Tests for refactored \\related command and \\play cite <n> playback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import re
import click
import pytest

from audiobench.cli.repl.session import ReplSession
from audiobench.cli.repl.backslash_commands import (
    cmd_related,
    cmd_play,
)
from audiobench.core.focused_entity import FocusedEntity
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord


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
def repl_session():
    group = click.Group("test")
    session = ReplSession(group)
    session.focus = FocusedEntity(type="transcript", id=592, label="Ulysses Chapter 1")
    return session


# ==============================================================================
# 1. Tests for cmd_related (Work Isolation, KG traversal, HNSW fallback)
# ==============================================================================


def test_cmd_related_work_isolation_cross_work(repl_session):
    """Test that cmd_related returns cross-work connections and excludes same-work chapters."""
    mock_db = MagicMock()

    # 1. tx_info: work_id = 78 (Ulysses)
    mock_tx_info = (592, 253, 78, "Ulysses", "James Joyce", "ulysses_ch1.mp3")
    # 2. chunk_rows
    mock_chunks = [(1001,), (1002,)]
    # 3. sem_edges from KG: returns Great Expectations (work_id=292)
    mock_sem_edges = [
        (49418, 917, 292, "Great Expectations", "Charles Dickens", 0.785, "And then I was recommended to the place as a man who could give another man as good as he brought."),
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
    ]

    mock_tx_record = MagicMock(audio_file_id=427)
    mock_af_record = MagicMock(
        id=427,
        file_name="greatexpectations_29_dickens_64kb.mp3",
        file_path="/audio/greatexpectations.mp3",
        work=MagicMock(title="Great Expectations", author="Charles Dickens"),
    )

    def query_mock(model):
        m = MagicMock()
        if model is TranscriptionRecord:
            m.filter_by.return_value.first.return_value = mock_tx_record
        elif model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = mock_af_record
        return m

    mock_db.query.side_effect = query_mock

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(354.07, 386.95, "Speaker")):
        cmd_related(repl_session, "")

    assert len(repl_session.last_related) >= 1
    for thought in repl_session.last_related:
        # Strict work isolation: never return same work
        assert thought.work_title != "Ulysses"
        assert thought.work_author != "James Joyce"
        assert thought.transcript_id != 592

    top = repl_session.last_related[0]
    assert top.work_title == "Great Expectations"
    assert top.work_author == "Charles Dickens"
    assert top.start_time == 354.07


def test_cmd_related_hnsw_fallback_enforces_work_isolation(repl_session):
    """Test that when KG has < 5 edges, HNSW fallback queries in-core index and filters work_id."""
    mock_db = MagicMock()

    mock_tx_info = (592, 253, 78, "Ulysses", "James Joyce", "ulysses_ch1.mp3")
    mock_chunks = [(1001,)]
    # No KG edges
    mock_sem_edges = []

    # HNSW candidate query results: returns Crime & Punishment (work_id=305)
    mock_hnsw_db_rows = [
        (34711, 850, 305, "Crime and Punishment", "Fyodor Dostoyevsky", "Ah ain't it only ten minutes left see look at the watch right now."),
    ]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: mock_sem_edges),
        MagicMock(fetchall=lambda: mock_hnsw_db_rows),
    ]

    mock_tx_record = MagicMock(audio_file_id=300)
    mock_af_record = MagicMock(
        id=300,
        file_name="crime_and_punishment_35.mp3",
        file_path="/audio/crime_and_punishment_35.mp3",
        work=MagicMock(title="Crime and Punishment", author="Fyodor Dostoyevsky"),
    )

    def query_mock(model):
        m = MagicMock()
        if model is TranscriptionRecord:
            m.filter_by.return_value.first.return_value = mock_tx_record
        elif model is AudioFileRecord:
            m.filter_by.return_value.first.return_value = mock_af_record
        return m

    mock_db.query.side_effect = query_mock

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    mock_hnsw = MagicMock()
    mock_hnsw.has_item.return_value = True
    mock_hnsw.get_items.return_value = [[0.1] * 768]
    # Return two hits: one is same-transcript chunk (1001, should be skipped), other is 34711 (distance 0.25 -> sim 0.75)
    mock_hnsw.query.return_value = [(1001, 0.0), (34711, 0.25)]

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.memory.graph_linker.ensure_hnsw_ready", return_value=mock_hnsw), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(1193.0, 1220.0, None)):
        cmd_related(repl_session, "")

    assert len(repl_session.last_related) == 1
    thought = repl_session.last_related[0]
    assert thought.work_title == "Crime and Punishment"
    assert thought.work_author == "Fyodor Dostoyevsky"
    assert thought.match_percent == 75


def test_cmd_related_graceful_message_when_no_connections(repl_session):
    """Test that cmd_related prints guidance when no connections exist."""
    mock_db = MagicMock()
    mock_tx_info = (592, 253, 78, "Ulysses", "James Joyce", "ulysses_ch1.mp3")
    mock_chunks = [(1001,)]

    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: mock_tx_info),
        MagicMock(fetchall=lambda: mock_chunks),
        MagicMock(fetchall=lambda: []),  # No KG
        MagicMock(fetchall=lambda: []),  # No HNSW matches
    ]

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    mock_hnsw = MagicMock()
    mock_hnsw.has_item.return_value = False

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.memory.graph_linker.ensure_hnsw_ready", return_value=mock_hnsw), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_related(repl_session, "")

        # Verify console output informs user to run \graph seed
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "\\graph seed" in printed


# ==============================================================================
# 2. Tests for cmd_play cite <n>
# ==============================================================================


def test_cmd_play_cite_triggers_playback(repl_session):
    """Test that \\play cite <n> initiates playback at exact passage timestamp."""
    repl_session.last_synthesis_citations = {1: 49418}

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": False}

    mock_db = MagicMock()
    mock_row = (
        49418,  # eid
        917,    # tx_id
        292,    # wid
        "Content text of citation passage",  # content
        427,    # af_id
        "greatexpectations.mp3",  # file_name
        "/audio/greatexpectations.mp3",  # file_path
        "Great Expectations",  # work_title
        "Charles Dickens",     # work_author
    )
    mock_db.execute.return_value.fetchone.return_value = mock_row

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(354.07, 386.95, None)), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_play(repl_session, "cite 1")

        mock_client.playback_play.assert_called_once_with(
            "/audio/greatexpectations.mp3",
            start_pos=354.07,
            audio_file_id=427,
            transcription_id=917,
        )
        mock_client.playback_seek.assert_not_called()

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Playing citation [1]" in printed
        assert "Charles Dickens" in printed


def test_cmd_play_cite_seeks_when_already_playing(repl_session):
    """Test that \\play cite <n> seeks if the cited transcript is currently playing."""
    repl_session.last_synthesis_citations = {1: 49418}

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": True, "transcription_id": 917}

    mock_db = MagicMock()
    mock_row = (
        49418, 917, 292, "Content text of citation", 427,
        "greatexpectations.mp3", "/audio/greatexpectations.mp3",
        "Great Expectations", "Charles Dickens",
    )
    mock_db.execute.return_value.fetchone.return_value = mock_row

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(354.07, 386.95, None)):
        cmd_play(repl_session, "cite 1")

        mock_client.playback_seek.assert_called_once_with(354.07)
        mock_client.playback_play.assert_not_called()


def test_cmd_play_cite_invalid_or_missing_number(repl_session):
    """Test error handling when citation number is missing or invalid."""
    repl_session.last_synthesis_citations = {1: 49418}
    mock_client = MagicMock()

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        # Bare "cite"
        cmd_play(repl_session, "cite")
        mock_client.playback_play.assert_not_called()
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Please specify citation number" in printed

        # Out-of-bounds citation "cite 99"
        mock_print.reset_mock()
        cmd_play(repl_session, "cite 99")
        mock_client.playback_play.assert_not_called()
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Citation [99] not found" in printed


def test_cmd_play_cite_chapter_summary_resolves_and_plays(repl_session):
    """Test that \\play cite <n> for chapter summaries resolves chapter audio and start time."""
    repl_session.last_synthesis_citations = {3: 58784}

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": False}

    mock_db = MagicMock()
    mock_row = (
        58784,  # eid
        0,      # tx_id (0 for chapter_summary)
        12,     # wid
        "Recorded and engineered at Tracktown Records...",  # content
        98,     # af_id
        "Cameron Hanes - Endure.m4b",  # file_name
        "/audio/Cameron Hanes - Endure.m4b",  # file_path
        "Endure: How to Work Hard, Outlast, and Keep Hammering",  # work_title
        "Cameron Hanes",  # work_author
        27707.791,        # chapter_start_time
    )
    mock_db.execute.return_value.fetchone.return_value = mock_row

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(0.0, 0.0, None)), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_play(repl_session, "cite 3")

        mock_client.playback_play.assert_called_once_with(
            "/audio/Cameron Hanes - Endure.m4b",
            start_pos=27707.791,
            audio_file_id=98,
            transcription_id=0,
        )
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Playing citation [3]" in printed
        assert "Cameron Hanes" in printed


def test_cmd_play_dialectic_bridge(repl_session):
    """Test that \\play b1 initiates playback for the bookmarked dialectic bridge."""
    from audiobench.memory.bookmark_service import DialecticBridge

    bridge = DialecticBridge(
        target_expression_id=48202,
        work_title="A Tale of Two Cities",
        author="Charles Dickens",
        cosine_similarity=0.805,
        snippet="That the proof would go back five years...",
    )
    repl_session.last_bookmark_bridges = {1: bridge}

    mock_client = MagicMock()
    mock_client.playback_status.return_value = {"playing": False}

    mock_db = MagicMock()
    mock_row = (
        48202, 852, 245, "That the proof would go back...", 381,
        "tale_of_two_cities.mp3", "/audio/tale_of_two_cities.mp3",
        "A Tale of Two Cities", "Charles Dickens", 0.0,
    )
    mock_db.execute.return_value.fetchone.return_value = mock_row
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.backslash_commands._resolve_thought_segment", return_value=(120.5, 150.0, None)), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_play(repl_session, "b1")

        mock_client.playback_play.assert_called_once_with(
            "/audio/tale_of_two_cities.mp3",
            start_pos=120.5,
            audio_file_id=381,
            transcription_id=852,
        )
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Playing bridge [B1]" in printed
        assert "Charles Dickens" in printed


def test_cmd_focus_dialectic_bridge(repl_session):
    """Test that \\focus b1 switches focus to the bridge's target work/audio file."""
    from audiobench.cli.repl.backslash_commands import cmd_focus
    from audiobench.memory.bookmark_service import DialecticBridge

    bridge = DialecticBridge(
        target_expression_id=48202,
        work_title="A Tale of Two Cities",
        author="Charles Dickens",
        cosine_similarity=0.805,
        snippet="That the proof would go back five years...",
    )
    repl_session.last_bookmark_bridges = {1: bridge}

    mock_db = MagicMock()
    mock_row = (
        48202,  # id
        245,    # work_id
        "A Tale of Two Cities",  # title
        "Charles Dickens",       # author
        381,    # audio_file_id
        "tale_of_two_cities.mp3",  # file_name
    )
    mock_db.execute.return_value.fetchone.return_value = mock_row
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.repl.dispatch.print_context_summary") as mock_summary, \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_focus(repl_session, "b1")

        assert repl_session.focus is not None
        assert repl_session.focus.type == "file"
        assert repl_session.focus.id == 381
        assert repl_session.focus.label == "tale_of_two_cities.mp3"
        mock_summary.assert_called_once()
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Focus switched to dialectic bridge [B1]" in printed
        assert "Charles Dickens" in printed


def test_cmd_bookmark_hermeneutic_codex_rendering(repl_session):
    """Test that \\bookmark renders the Hermeneutic Codex Panel and populates bridge state."""
    from audiobench.cli.repl.backslash_commands import cmd_bookmark
    from audiobench.memory.bookmark_service import DialecticBridge, SmartBookmarkCard

    bridge = DialecticBridge(
        target_expression_id=48202,
        work_title="A Tale of Two Cities",
        author="Charles Dickens",
        cosine_similarity=0.805,
        snippet="A profound excerpt on suffering and destiny.",
        is_clique_partner=True,
    )
    card = SmartBookmarkCard(
        expression_id=30186,
        work_id=10,
        work_title="The Brothers Karamazov",
        author="Fyodor Dostoyevsky",
        timestamp_display="00:21",
        quote="Dangerous witnesses testifying in the court.",
        preceding_context="Prior testimony from the courtroom inquest.",
        following_resolution="The final verdict given by the magistrate.",
        dialectic_bridges=[bridge],
        consensus_triangle=["Dostoyevsky", "Dickens", "Frankl"],
        community_id=14,
        community_name="Courtroom Inquest and Judicial Guilt",
    )

    from audiobench.cli.display.theme import console
    from rich.console import ConsoleDimensions
    from unittest.mock import PropertyMock

    with patch("audiobench.memory.bookmark_service.generate_smart_bookmark_card", return_value=card), \
         patch.object(type(console), "size", new_callable=PropertyMock(return_value=ConsoleDimensions(120, 40))), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_bookmark(repl_session, "30186")

        # Verify bridge state registered in session
        assert 1 in repl_session.last_bookmark_bridges
        assert repl_session.last_bookmark_bridges[1].target_expression_id == 48202

        # Check printed panel
        panels = [arg for call in mock_print.call_args_list for arg in call[0] if hasattr(arg, "__class__") and arg.__class__.__name__ == "Panel"]
        assert len(panels) >= 1
        panel = panels[0]
        # Inspect rendered structure inside Panel Group
        from rich.console import Console
        test_console = Console(width=120)
        with test_console.capture() as capture:
            test_console.print(panel)
        output = capture.get()

        assert "Smart Bookmark: The Brothers Karamazov" in output
        assert "School: #14 Courtroom Inquest and Judicial Guilt" in output
        assert "Focused Excerpt" in output
        assert "Tier 1: Local Narrative Arc" in output
        assert "◄ Context (Ignition)" in output
        assert "► Resolution (Follow-up)" in output
        assert "Tier 2: Dialectic Bridges" in output
        assert "[B1]" in output
        assert "Charles Dickens" in output
        assert "Consensus Partner" in output
        assert "Quick actions:" in output
        assert "\\play b1" in output
        assert "\\focus b1" in output
        assert "\\synthesize \"The Brothers Karamazov\"" in output


def test_smart_bookmark_card_renders_markdown_excerpt(repl_session):
    """Test that Markdown formatting in bookmark excerpt renders without raw asterisks."""
    from audiobench.cli.repl.backslash_commands import cmd_bookmark
    from audiobench.memory.bookmark_service import SmartBookmarkCard

    # Using the exact literal production string circled in screenshot media_1791785385787.png
    markdown_quote = (
        "**Celebrity Endorsement as Social Proof** Warren Buffett's habit of pulling out letters "
        "from admirers in public settings illustrates how elite figures amplify a work's credibility."
    )
    card = SmartBookmarkCard(
        expression_id=64594,
        work_id=None,
        work_title="Unknown Work",
        author="Unknown Author",
        timestamp_display="00:00",
        quote=markdown_quote,
        preceding_context=None,
        following_resolution=None,
        dialectic_bridges=[],
        consensus_triangle=None,
        community_id=None,
        community_name=None,
    )

    from audiobench.cli.display.theme import console
    from rich.console import Console, ConsoleDimensions
    from unittest.mock import PropertyMock

    with patch("audiobench.memory.bookmark_service.generate_smart_bookmark_card", return_value=card), \
         patch.object(type(console), "size", new_callable=PropertyMock(return_value=ConsoleDimensions(120, 40))), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_bookmark(repl_session, "64594")

        panels = [
            arg for call in mock_print.call_args_list
            for arg in call[0] if hasattr(arg, "__class__") and arg.__class__.__name__ == "Panel"
        ]
        assert len(panels) >= 1
        panel = panels[0]

        test_console = Console(width=120)
        with test_console.capture() as capture:
            test_console.print(panel)
        output = capture.get()

        # The bold text should be rendered as formatted content
        assert "Celebrity Endorsement as Social Proof" in output
        # Literal unparsed markdown asterisks must NOT be visible in console output
        assert "**Celebrity Endorsement as Social Proof**" not in output
        assert not has_emojis(output)


# ==============================================================================
# 5. Tests for Epistemic Dual-Typing of Prior Research Syntheses
# ==============================================================================


def test_format_expression_attribution_clean_truncation():
    """Test _format_expression_attribution and word-boundary truncation."""
    from audiobench.memory.synthesis_retriever import (
        _clean_title_truncate,
        _format_expression_attribution,
    )

    # Word boundary truncation
    long_q = "Summarize the arguments made against the modern school system and industrial indoctrination."
    res = _clean_title_truncate(long_q, max_chars=80)
    assert not res.endswith("….")
    assert "…" in res
    assert res == "Summarize the arguments made against the modern school system and industrial…"
    assert not res.endswith("in…")  # Truncates cleanly before 'indoctrination', not mid-word

    # Prior synthesis
    auth, title, playable = _format_expression_attribution(
        "search_synthesis", None, None, long_q, None, "content text"
    )
    assert auth == "[Prior Synthesis]"
    assert playable is False
    assert title.startswith("Summarize the arguments")

    # Past session summary
    auth_s, title_s, playable_s = _format_expression_attribution(
        "search_session_summary", None, None, None, "Late Night Dorm Discussions", "content"
    )
    assert auth_s == "[Past Session]"
    assert playable_s is False
    assert title_s == "Late Night Dorm Discussions"

    # Audio transcript
    auth_a, title_a, playable_a = _format_expression_attribution(
        "audio_transcript", "The Warren Buffett Way", "Robert Hagstrom", None, None, "content"
    )
    assert auth_a == "Robert Hagstrom"
    assert title_a == "The Warren Buffett Way"
    assert playable_a is True


def test_cmd_play_cite_prior_synthesis_informative_guard(repl_session):
    """Test that \\play cite <n> on a prior synthesis gives an informative explanation rather than an error."""
    repl_session.last_synthesis_citations = {1: 33573}

    mock_client = MagicMock()
    mock_db = MagicMock()

    # DB returns source_type = 'search_synthesis'
    mock_db.execute.return_value.fetchone.return_value = ("search_synthesis",)

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None

    with patch("audiobench.daemon.factory.get_daemon_client", return_value=mock_client), \
         patch("audiobench.core.db_session.get_session", return_value=mock_ctx), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_play(repl_session, "cite 1")

        mock_client.playback_play.assert_not_called()
        printed = " ".join(arg for call in mock_print.call_args_list for arg in call[0] if isinstance(arg, str))

        assert "Citation [1] (Expression #33573) is a text-based prior research synthesis, not an audio recording." in printed
        assert "Use [cyan]\\bookmark cite 1[/] to inspect the full text card and dialectic connections." in printed
        assert "Could not locate audio file" not in printed


def test_cmd_synthesize_flywheel_hint_partitioned(repl_session):
    """Test that cmd_synthesize partitions Flywheel hints literally by playability."""
    from audiobench.cli.repl.backslash_commands import cmd_synthesize
    from audiobench.chat.synthesis_engine import SynthesisResponse
    from audiobench.memory.synthesis_retriever import SynthesisFragment

    frag_text1 = SynthesisFragment(
        expression_id=33573,
        work_id=None,
        work_title="Arguments Against Industrial Schooling",
        author="[Prior Synthesis]",
        content="Critique of schooling...",
        q_sim=0.768,
        edge_weight=1.0,
        effective_rel=0.768,
        mmr_score=0.768,
        source_mode="ANN",
        source_type="search_synthesis",
        is_playable=False,
    )
    frag_text2 = SynthesisFragment(
        expression_id=33577,
        work_id=None,
        work_title="Late Night Conversations",
        author="[Prior Synthesis]",
        content="Campus dialogues...",
        q_sim=0.733,
        edge_weight=1.0,
        effective_rel=0.733,
        mmr_score=0.733,
        source_mode="ANN",
        source_type="search_synthesis",
        is_playable=False,
    )
    frag_audio3 = SynthesisFragment(
        expression_id=4001,
        work_id=101,
        work_title="Before You Follow The Wrong Path",
        author="Champions Mentality 365",
        content="Audio passage...",
        q_sim=0.723,
        edge_weight=1.0,
        effective_rel=0.723,
        mmr_score=0.723,
        source_mode="ANN",
        source_type="audio_transcript",
        is_playable=True,
    )
    frag_audio4 = SynthesisFragment(
        expression_id=4002,
        work_id=102,
        work_title="Buffett and Munger Unscripted",
        author="Alex W. Morris",
        content="Audio passage 2...",
        q_sim=0.704,
        edge_weight=1.0,
        effective_rel=0.704,
        mmr_score=0.704,
        source_mode="ANN",
        source_type="chapter_summary",
        is_playable=True,
    )

    mock_resp = SynthesisResponse(
        query="why arent we taught financial literacy",
        synthesis_text="Synthesis content...",
        citations={1: 33573, 2: 33577, 3: 4001, 4: 4002},
        fragments=[frag_text1, frag_text2, frag_audio3, frag_audio4],
        elapsed_seconds=1.23,
        model_used="ollama",
    )

    with patch("audiobench.chat.synthesis_engine.generate_synthesis", return_value=mock_resp), \
         patch("audiobench.cli.display.theme.console.print") as mock_print:
        cmd_synthesize(repl_session, "why arent we taught financial literacy")

        printed = " ".join(arg for call in mock_print.call_args_list for arg in call[0] if isinstance(arg, str))

        # Exact literal assertion on the partitioned hint text
        assert "Flywheel:[/] Type [cyan]\\bookmark cite 1[/] for [Prior Synthesis] to inspect the synthesis card, or [cyan]\\play cite 3 or 4[/] to listen." in printed
        # Ensure \play cite 1 is never promised
        assert "\\play cite 1[/] to listen" not in printed


