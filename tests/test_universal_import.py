import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from audiobench.cli.commands.import_cmd import import_cmd, run_import_flow
from audiobench.cli.tui.reimport_tui import ReimportTUI


def test_import_cmd_positional_directory(tmp_path: Path):
    """Test passing a directory routes to auto-detect mode with current_path set."""
    runner = CliRunner()
    sub_dir = tmp_path / "audio_books"
    sub_dir.mkdir()

    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, [str(sub_dir)])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_AUTO"
        assert kwargs["restore_state"]["current_path"] == sub_dir.resolve()


def test_import_cmd_positional_transcript_json(tmp_path: Path):
    """Test passing a .json file routes to single transcript mode with auto-detected state."""
    runner = CliRunner()
    tx_file = tmp_path / "chapter_01.json"
    tx_file.write_text(json.dumps({"segments": [], "text": "hello"}))

    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, [str(tx_file)])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_SINGLE_TX"
        assert kwargs["restore_state"]["selected_tx_file"] == tx_file.resolve()
        assert kwargs["restore_state"]["current_path"] == tmp_path.resolve()


def test_import_cmd_positional_audio_file(tmp_path: Path):
    """Test passing an audio file bypasses file picker and routes to MODE_AUDIO with target_files."""
    runner = CliRunner()
    audio_file = tmp_path / "chapter_01.mp3"
    audio_file.touch()

    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, [str(audio_file)])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_AUDIO"
        assert kwargs["target_files"] == [audio_file.resolve()]


def test_import_cmd_transcript_batch_flags():
    """Test -T --batch flags route to MODE_BATCH_TX."""
    runner = CliRunner()
    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, ["-T", "--batch"])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_BATCH_TX"


def test_reimport_tui_auto_links_adjacent_audio(tmp_path: Path):
    """Test ReimportTUI automatically finds adjacent audio file with matching stem and sets up Screen 3."""
    audio_file = tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"dummy audio content")

    tx_file = tmp_path / "lecture.json"
    tx_data = {
        "text": "Welcome to today's lecture.",
        "engine": "gemini",
        "duration_seconds": 60.0,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Welcome to today's lecture.",
                "words": [{"word": "Welcome", "start": 0.0, "end": 0.5}],
            }
        ],
        "raw_audio": {
            "file_name": "lecture.mp3",
            "duration_seconds": 60.0,
        },
    }
    tx_file.write_text(json.dumps(tx_data))

    shared_state = {
        "selected_tx_file": str(tx_file),
        "current_path": str(tmp_path),
    }

    tui = ReimportTUI(batch=False, auto_detect=False, shared_state=shared_state)
    assert tui.transcript_path == tx_file
    assert tui.audio_path == audio_file
    assert tui.screen == 3


def test_companion_sidecar_detection_and_import(tmp_path: Path, monkeypatch):
    """Test that importing an audio file with an adjacent .json offers companion import and saves transcription."""
    from unittest.mock import MagicMock

    audio_file = tmp_path / "speech.mp3"
    audio_file.write_bytes(b"audio data here")

    companion_json = tmp_path / "speech.json"
    companion_json.write_text(json.dumps({
        "text": "Speech transcript content here",
        "engine": "whisper",
        "duration_seconds": 10.0,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Speech transcript content here",
                "words": [{"word": "Speech", "start": 0.0, "end": 0.5}],
            }
        ],
    }))

    mock_q = MagicMock()
    mock_select = MagicMock()
    # User selects "companion"
    mock_select.ask.return_value = "companion"
    mock_q.select.return_value = mock_select
    mock_q.Choice = lambda title, value: (title, value)
    monkeypatch.setattr("questionary.select", mock_q.select)
    monkeypatch.setattr("questionary.Choice", mock_q.Choice)

    # Mock BackgroundImporter to simulate copy
    dest_audio = tmp_path / "library" / "speech.mp3"
    dest_audio.parent.mkdir(parents=True, exist_ok=True)
    dest_audio.write_bytes(b"audio data here")

    mock_importer = MagicMock()
    def fake_run_import(allocations):
        return [{
            "source": allocations[0]["file"],
            "destination": dest_audio,
            "engine": allocations[0]["engine"],
            "is_duplicate": False,
            "file_hash": "mock_hash_123",
            "companion_file": allocations[0].get("companion_file"),
            "tqs_score": allocations[0].get("tqs_score"),
        }]
    mock_importer.run_import_jobs.side_effect = fake_run_import

    mock_tx_repo = MagicMock()
    mock_tx_repo.save_transcription.return_value = 42

    with patch("audiobench.storage.importer.BackgroundImporter", return_value=mock_importer), \
         patch("audiobench.storage.repository.TranscriptionRepository", return_value=mock_tx_repo), \
         patch("audiobench.core.db_session.get_session") as mock_session_ctx, \
         patch("audiobench.transcribe.transcript_parser.build_audio_metadata") as mock_build_meta:

        mock_session = MagicMock()
        def fake_add(rec):
            if hasattr(rec, "id"):
                rec.id = 1
        mock_session.add.side_effect = fake_add
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session_ctx.return_value.__enter__.return_value = mock_session

        imported_ids = run_import_flow(
            initial_mode="MODE_AUDIO",
            target_files=[audio_file],
        )

        assert 1 in imported_ids
        # Verify save_transcription was called with overwrite=True
        mock_tx_repo.save_transcription.assert_called_once()
        _, save_kwargs = mock_tx_repo.save_transcription.call_args
        assert save_kwargs.get("overwrite") is True
        # Verify companion sidecar was copied to library
        assert (tmp_path / "library" / "speech.json").exists()
