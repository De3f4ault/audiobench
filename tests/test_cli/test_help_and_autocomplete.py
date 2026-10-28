"""Tests for REPL help system, autocomplete completions, and ghost suggestions."""

from __future__ import annotations

import click
import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from audiobench.cli.repl.backslash_commands import cmd_help
from audiobench.cli.repl.banner import _TIPS, print_full_help
from audiobench.cli.repl.completion import (
    AudioBenchAutoSuggest,
    AudioBenchCompleter,
    setup_completion,
)
from audiobench.cli.repl.session import ReplSession


@pytest.fixture
def session():
    group = click.Group("test")

    @group.command("transcribe")
    def cmd_transcribe():
        """Transcribe an audio file."""
        pass

    @group.command("search")
    def cmd_search():
        """Search transcriptions."""
        pass

    sess = ReplSession(group)
    setup_completion(sess)
    return sess


@pytest.fixture
def mock_buffer():
    return Buffer(history=InMemoryHistory())


class TestHelpSystem:
    """Test \\help command rendering, detail panels, and fuzzy fallback."""

    def test_top_level_help_renders_categories(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "")
        full_text = "\n".join(output)
        assert "AudioBench Shell — Command Reference" in full_text
        assert "Knowledge Graph & Dialectic" in full_text
        assert "\\synthesize" in full_text
        assert "\\suggest" in full_text
        assert "\\related" in full_text
        assert "\\bookmark" in full_text
        assert "Playback & Library" in full_text
        assert "Tip: Type" in full_text
        assert "\\help <command>" in full_text

    def test_help_synthesize_detail(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "synthesize")
        full_text = "\n".join(output)
        assert "\\synthesize" in full_text
        assert "Synthesize (Dialectic Graph MMR)" in full_text
        assert "\\play cite 1" in full_text
        assert "\\bookmark cite 1" in full_text
        assert "Flywheel Next Steps:" in full_text

    def test_help_synth_alias(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "synth")
        full_text = "\n".join(output)
        assert "\\synthesize" in full_text
        assert "Synthesize (Dialectic Graph MMR)" in full_text

    def test_help_bookmark_detail(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "bookmark")
        full_text = "\n".join(output)
        assert "\\bookmark" in full_text
        assert "Smart Contextual Bookmarking" in full_text
        assert "\\bookmark cite 1" in full_text

    def test_help_suggest_detail(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "suggest")
        full_text = "\n".join(output)
        assert "\\suggest" in full_text
        assert "Next Listen recommendations" in full_text
        assert "Triadic Consensus" in full_text

    def test_help_undocumented_handler_fallback(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "home")
        full_text = "\n".join(output)
        assert "\\home" in full_text
        assert "Clear the entire navigation stack" in full_text

    def test_help_fuzzy_suggestion_fallback(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        cmd_help(session, "synh")
        full_text = "\n".join(output)
        assert "Unknown command: \\synh" in full_text
        assert "Did you mean:" in full_text
        assert ("\\help synth" in full_text or "\\help synthesize" in full_text)


class TestAutocompleteAndCompletions:
    """Test AudioBenchCompleter popup completions for new commands and arguments."""

    def test_complete_backslash_command_with_meta(self, session):
        completer = AudioBenchCompleter(session)
        doc = Document(text="\\synth", cursor_position=6)
        completions = list(completer.get_completions(doc, None))
        cmd_names = [c.text for c in completions]
        assert "\\synthesize" in cmd_names
        assert "\\synth" in cmd_names

        synth_comp = next(c for c in completions if c.text == "\\synthesize")
        assert synth_comp.display_meta_text is not None
        assert "Dialectic synthesis" in synth_comp.display_meta_text

    def test_complete_subcommands_for_bookmark(self, session):
        completer = AudioBenchCompleter(session)
        doc = Document(text="\\bookmark ", cursor_position=10)
        completions = list(completer.get_completions(doc, None))
        sub_names = [c.text for c in completions]
        assert "cite" in sub_names
        assert "list" in sub_names

        cite_comp = next(c for c in completions if c.text == "cite")
        assert cite_comp.display_meta_text is not None
        assert "\\bookmark cite <n>" in cite_comp.display_meta_text

    def test_complete_citation_numbers_dynamic(self, session):
        session.last_synthesis_citations = {1: 1042, 2: 1043, 3: 1044}
        completer = AudioBenchCompleter(session)

        doc = Document(text="\\bookmark cite ", cursor_position=15)
        completions = list(completer.get_completions(doc, None))
        c_nums = [c.text for c in completions]
        assert c_nums == ["1", "2", "3"]

        c1 = completions[0]
        assert "Expr #1042" in c1.display_meta_text

    def test_complete_citation_numbers_partial(self, session):
        session.last_synthesis_citations = {1: 1042, 2: 1043}
        completer = AudioBenchCompleter(session)

        doc = Document(text="\\play cite 1", cursor_position=12)
        completions = list(completer.get_completions(doc, None))
        assert len(completions) == 1
        assert completions[0].text == "1"
        assert "Expr #1042" in completions[0].display_meta_text

    def test_complete_help_command_names(self, session):
        completer = AudioBenchCompleter(session)
        doc = Document(text="\\help syn", cursor_position=9)
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert "synthesize" in names
        assert "synth" in names

    def test_complete_proposals_subcommands(self, session):
        completer = AudioBenchCompleter(session)
        doc = Document(text="\\proposals ", cursor_position=11)
        completions = list(completer.get_completions(doc, None))
        sub_names = [c.text for c in completions]
        assert "show" in sub_names
        assert "all" in sub_names
        assert "list" in sub_names
        assert "authorize" in sub_names
        assert "defer" in sub_names
        assert "reject" in sub_names

    def test_complete_proposals_ids_for_show_and_authorize(self, session, monkeypatch):
        mock_proposals = [
            {
                "id": 86710,
                "inference_status": "proposed",
                "content": '{"proposal_type": "knowledge_gap", "title": "Philosophical Integration Gap"}',
            },
            {
                "id": 37441,
                "inference_status": "deferred",
                "content": '{"proposal_type": "knowledge_gap", "title": "Christian Existential Theology"}',
            },
        ]
        monkeypatch.setattr(
            "audiobench.cli.repl.backslash_commands._fetch_daemon_proposals",
            lambda **kwargs: mock_proposals,
        )

        completer = AudioBenchCompleter(session)

        # Space after show -> list proposal IDs
        doc = Document(text="\\proposals show ", cursor_position=16)
        completions = list(completer.get_completions(doc, None))
        ids = [c.text for c in completions]
        assert "86710" in ids
        assert "37441" in ids

        c1 = next(c for c in completions if c.text == "86710")
        assert "[proposed]" in c1.display_meta_text
        assert "Philosophical Integration Gap" in c1.display_meta_text

        # Partial ID typing: \proposals show 8
        doc_partial = Document(text="\\proposals show 8", cursor_position=17)
        comp_partial = list(completer.get_completions(doc_partial, None))
        assert len(comp_partial) == 1
        assert comp_partial[0].text == "86710"

        # Alias \prop authorize
        doc_auth = Document(text="\\prop authorize ", cursor_position=16)
        comp_auth = list(completer.get_completions(doc_auth, None))
        assert "86710" in [c.text for c in comp_auth]

    def test_complete_context_command(self, session):
        completer = AudioBenchCompleter(session)
        doc = Document(text="\\con", cursor_position=4)
        completions = list(completer.get_completions(doc, None))
        cmd_names = [c.text for c in completions]
        assert "\\context" in cmd_names


class TestGhostAutoSuggest:
    """Test AudioBenchAutoSuggest ghost suggestions."""

    def test_ghost_command_synopsis_proposals(self, session, mock_buffer):
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\proposals ", cursor_position=11)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert "[show <id>|all|authorize <id>|defer <id>|reject <id>]" in suggestion.text

    def test_ghost_subcommand_proposals_show(self, session, mock_buffer):
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\proposals show ", cursor_position=16)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert suggestion.text == "<id>"


    def test_ghost_command_synopsis(self, session, mock_buffer):
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\synthesize ", cursor_position=12)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert suggestion.text == "<thematic query>"

    def test_ghost_command_synopsis_graph(self, session, mock_buffer):
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\graph ", cursor_position=7)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert suggestion.text == "[seed [limit]|days]"

    def test_ghost_subcommand_args(self, session, mock_buffer):
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\graph seed ", cursor_position=12)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert suggestion.text == "[limit]"

    def test_ghost_citation_number_with_session(self, session, mock_buffer):
        session.last_synthesis_citations = {1: 1042, 2: 1043}
        suggester = AudioBenchAutoSuggest(session)
        doc = Document(text="\\play cite ", cursor_position=11)
        suggestion = suggester.get_suggestion(mock_buffer, doc)
        assert suggestion is not None
        assert suggestion.text == "1"


class TestBannerAndTips:
    """Test full help text and rotating tips."""

    def test_print_full_help_includes_knowledge_graph(self, session, monkeypatch):
        output = []
        monkeypatch.setattr("audiobench.cli.display.theme.console.print", lambda *args, **kwargs: output.append(str(args[0])))

        print_full_help(session)
        full_text = "\n".join(output)
        assert "Knowledge Graph & Cross-Work Intelligence" in full_text
        assert "\\synthesize" in full_text
        assert "\\suggest" in full_text
        assert "\\related" in full_text
        assert "\\play cite <n>" in full_text
        assert "\\bookmark cite <n>" in full_text
        assert "\\graph seed" in full_text
        assert "\\help <cmd>" in full_text

    def test_tips_include_new_features(self):
        tips_joined = " ".join(_TIPS)
        assert "\\synthesize" in tips_joined
        assert "\\play cite 1" in tips_joined
        assert "\\bookmark cite 1" in tips_joined
        assert "\\suggest" in tips_joined
        assert "\\help <command>" in tips_joined


class TestAdaptivePromptSession:
    """Test adaptive complete_while_typing and theme styling in repl_shell."""

    def test_make_prompt_session_adaptive_condition(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from audiobench.cli.shared.repl_shell import make_prompt_session
        from prompt_toolkit.filters import Condition

        pt_session = make_prompt_session(
            slash_commands=["/set", "/focus"],
            history_path=tmp_path / "hist.txt",
        )

        assert isinstance(pt_session.complete_while_typing, Condition)

        # Mock app buffer to test condition
        mock_buffer = MagicMock()
        mock_app = MagicMock(current_buffer=mock_buffer)

        with patch("prompt_toolkit.application.current.get_app", return_value=mock_app):
            # 1. Typing slash command -> True
            mock_buffer.text_before_cursor = "/se"
            assert pt_session.complete_while_typing() is True

            # 2. Typing plain query -> False
            mock_buffer.text_before_cursor = "what really is psychology"
            assert pt_session.complete_while_typing() is False

    def test_make_prompt_session_explicit_override(self, tmp_path):
        from audiobench.cli.shared.repl_shell import make_prompt_session

        pt_session = make_prompt_session(
            slash_commands=["/set"],
            history_path=tmp_path / "hist.txt",
            complete_while_typing=False,
        )
        assert pt_session.complete_while_typing is False

