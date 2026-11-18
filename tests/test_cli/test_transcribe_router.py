"""Integration tests for the Transcribe Router Flow."""

from __future__ import annotations

from unittest.mock import patch

from audiobench.cli.commands.transcribe import transcribe
from audiobench.core.db_session import get_session
from audiobench.storage.models import StagingCartItem


@patch("audiobench.jobs.scheduler.drain_slot")
@patch("audiobench.cli.wizard_checkout.prompt_checkout_cart")
@patch("audiobench.cli.tui.library_tui.launch_library_tui")
@patch("audiobench.cli.wizard.prompt_menu")
def test_transcribe_router_foreground(
    mock_prompt_menu, mock_tui, mock_checkout, mock_drain_slot, runner, test_db
):
    """Test staging a file and processing immediately in foreground."""
    # 1. User selects Library from main menu
    # 2. User checks out
    mock_prompt_menu.side_effect = ["library", "checkout"]

    from audiobench.storage.models import AudioFileRecord

    with get_session() as session:
        audio = AudioFileRecord(file_path="/f", file_name="f", file_size_bytes=1, format="mp3")
        session.add(audio)
        session.flush()
        audio_id = audio.id
        session.commit()

    # TUI simulates selecting an item and asking to transcribe
    mock_tui.return_value = {"action": "transcribe", "selected_ids": [audio_id]}

    # Checkout returns "now"
    mock_checkout.return_value = "now"

    result = runner.invoke(transcribe, [])

    assert result.exit_code == 0
    assert "Processing 1 files sequentially in foreground" in result.output
    mock_drain_slot.assert_called_once()

    # Cart should be empty
    with get_session() as session:
        assert session.query(StagingCartItem).count() == 0


@patch("audiobench.jobs.scheduler.ensure_worker")
@patch("audiobench.cli.wizard_checkout.prompt_checkout_cart")
@patch("audiobench.cli.commands.import_cmd.run_import_flow")
@patch("audiobench.cli.wizard.prompt_menu")
def test_transcribe_router_background(
    mock_prompt_menu, mock_import, mock_checkout, mock_ensure_worker, runner, test_db
):
    """Test staging a file via import and processing later in background."""
    mock_prompt_menu.side_effect = ["import", "checkout"]

    from audiobench.storage.models import AudioFileRecord

    with get_session() as session:
        audio = AudioFileRecord(file_path="/f2", file_name="f2", file_size_bytes=1, format="mp3")
        session.add(audio)
        session.flush()
        audio_id = audio.id
        session.commit()

    # Import returns the new audio_file_id
    mock_import.return_value = [audio_id]

    mock_checkout.return_value = "later"

    result = runner.invoke(transcribe, [])

    assert result.exit_code == 0
    assert "Added 1 files to background queue" in result.output
    mock_ensure_worker.assert_called_once()

    # Cart should be empty
    with get_session() as session:
        assert session.query(StagingCartItem).count() == 0
