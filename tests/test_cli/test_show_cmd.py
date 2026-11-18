"""Unit tests for the upgraded show CLI command and multi-format views."""

import json
from unittest.mock import MagicMock, patch
import pytest

from audiobench.cli.commands.history import show
from audiobench.cli.shared.transcript_views import (
    format_transcript,
    search_transcript_segments,
    export_transcript_to_file,
)
from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord
from audiobench.core.db_session import get_session


@pytest.fixture
def sample_transcript(test_db):
    """Seed test database with a rich transcript containing segments and chapters."""
    with get_session() as session:
        af = AudioFileRecord(
            file_name="lecture.mp3",
            file_path="/tmp/lecture.mp3",
            file_hash="hash_lec",
            duration_seconds=125.0,
        )
        session.add(af)
        session.flush()

        tx = TranscriptionRecord(
            audio_file_id=af.id,
            file_name="Eastern Philosophy Lecture",
            full_text="First we contemplate silence. Next we study the Way of Zen.",
            raw_text="First we contemplate silence. Next we study the Way of Zen raw.",
            language="en",
            engine="faster-whisper",
            model_name="large-v3-turbo",
            duration_seconds=125.0,
            word_count=10,
            pipeline_phase="complete",
        )
        session.add(tx)
        session.flush()

        seg1 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=1,
            start_time=0.0,
            end_time=4.5,
            speaker="Alan Watts",
            text="First we contemplate silence.",
        )
        seg2 = SegmentRecord(
            transcription_id=tx.id,
            segment_index=2,
            start_time=5.0,
            end_time=9.8,
            speaker="Alan Watts",
            text="Next we study the Way of Zen.",
        )
        session.add_all([seg1, seg2])
        session.commit()

        return tx.id


def test_show_cmd_text_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--raw"])
    assert result.exit_code == 0
    assert "[00:00]" in result.output
    assert "<Alan Watts>" in result.output
    assert "First we contemplate silence." in result.output


def test_show_cmd_no_timestamps_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--no-timestamps", "--raw"])
    assert result.exit_code == 0
    assert "[00:00]" not in result.output
    assert "<Alan Watts>" in result.output
    assert "First we contemplate silence." in result.output


def test_show_cmd_no_speakers_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--no-speakers", "--raw"])
    assert result.exit_code == 0
    assert "<Alan Watts>" not in result.output
    assert "First we contemplate silence." in result.output


def test_show_cmd_srt_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--format", "srt", "--raw"])
    assert result.exit_code == 0
    assert "00:00:00,000 --> 00:00:04,500" in result.output
    assert "First we contemplate silence." in result.output
    assert "00:00:05,000 --> 00:00:09,800" in result.output


def test_show_cmd_vtt_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--format", "vtt", "--raw"])
    assert result.exit_code == 0
    assert "WEBVTT" in result.output
    assert "00:00:00.000 --> 00:00:04.500" in result.output


def test_show_cmd_json_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--format", "json", "--raw"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["id"] == sample_transcript
    assert len(parsed["segments"]) == 2


def test_show_cmd_tsv_raw(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--format", "tsv", "--raw"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "index\tstart\tend\tduration\tspeaker\ttext"
    assert "Alan Watts\tFirst we contemplate silence." in lines[1]


def test_show_cmd_latest_raw(runner, sample_transcript):
    result = runner.invoke(show, ["latest", "--raw"])
    assert result.exit_code == 0
    assert "First we contemplate silence." in result.output


def test_show_cmd_raw_whisper(runner, sample_transcript):
    result = runner.invoke(show, [str(sample_transcript), "--show-raw-whisper", "--raw"])
    assert result.exit_code == 0
    assert "Way of Zen raw" in result.output


def test_show_cmd_nonexistent(runner, test_db):
    result = runner.invoke(show, ["99999", "--raw"])
    assert result.exit_code != 0


def test_show_cmd_interactive_flag(runner, sample_transcript):
    with patch("audiobench.cli.show_repl.start_show_repl") as mock_repl:
        result = runner.invoke(show, ["-i", str(sample_transcript)])
        assert result.exit_code == 0
        assert mock_repl.call_count == 1


def test_search_transcript_segments():
    data = {
        "segments": [
            {"index": 1, "start": 10.0, "speaker": "Speaker 1", "text": "The nature of the universe."},
            {"index": 2, "start": 45.0, "speaker": "Speaker 2", "text": "Exploring everyday reality."},
        ]
    }
    hits = search_transcript_segments(data, "universe")
    assert len(hits) == 1
    assert hits[0]["time_str"] == "00:10"
    assert hits[0]["text"] == "The nature of the universe."

    # No match
    assert len(search_transcript_segments(data, "nonexistent")) == 0


def test_export_transcript_to_file(tmp_path):
    data = {
        "segments": [
            {"index": 1, "start": 0.0, "end": 5.0, "text": "Export testing."}
        ]
    }
    out_file = tmp_path / "test.srt"
    res = export_transcript_to_file(data, "srt", out_file)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:05,000" in content


def test_show_slash_completer_commands():
    from prompt_toolkit.document import Document
    from audiobench.cli.show_repl import ShowSlashCompleter

    completer = ShowSlashCompleter()
    doc = Document("/fo")
    completions = list(completer.get_completions(doc, None))
    assert any(c.text == "/format" for c in completions)


def test_show_slash_completer_format_choices():
    from prompt_toolkit.document import Document
    from audiobench.cli.show_repl import ShowSlashCompleter

    completer = ShowSlashCompleter()
    doc = Document("/format ")
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "srt" in texts
    assert "vtt" in texts
    assert "json" in texts
    assert "txt" in texts


def test_show_slash_completer_open_keywords():
    from prompt_toolkit.document import Document
    from audiobench.cli.show_repl import ShowSlashCompleter

    completer = ShowSlashCompleter()
    doc = Document("/open ")
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "latest" in texts
    assert "last" in texts


def test_show_auto_suggest():
    from prompt_toolkit.document import Document
    from audiobench.cli.show_repl import ShowAutoSuggest

    suggester = ShowAutoSuggest()
    doc = Document("/")
    sug = suggester.get_suggestion(None, doc)
    assert sug is not None
    assert sug.text == "page"

