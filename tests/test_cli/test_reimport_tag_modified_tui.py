import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobench.cli.tui.reimport_tui import ReimportTUI
from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
from audiobench.transcribe.transcription_result import AudioMetadata


def test_reimport_tui_tag_modified_lifecycle(tmp_path, monkeypatch):
    # Set up root folder with 1 JSON transcript and 1 tag-modified MP3
    folder = tmp_path / "Book"
    folder.mkdir()

    audio_file = folder / "Chapter 06 A.mp3"
    audio_file.write_bytes(b"content-identical-audio-A" * 1000)
    actual_hash = AudioMetadata.compute_file_hash(audio_file)

    tx_json = folder / "Chapter 06 A.json"
    tx_data = {
        "schema_version": 1,
        "full_text": "This is chapter 6 audio content.",
        "duration_seconds": 3100.604,
        "engine": "whisper",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "This is chapter 6 audio content."}
        ],
        "audio": {
            "file_name": "Chapter 06 A.mp3",
            "file_hash": "different_id3_edited_hash_value_9999",
            "duration_seconds": 3100.604,
            "file_size_bytes": len(b"content-identical-audio-A" * 1000),
        }
    }
    tx_json.write_text(json.dumps(tx_data), encoding="utf-8")

    from collections import namedtuple
    ProbeResult = namedtuple("ProbeResult", ["duration"])
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=3100.560)
    )

    tui = ReimportTUI()
    tui.is_auto_detect = True
    tui.audio_dir = folder
    tui.transcript_dir = folder

    tui._match_batch()

    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["status"] == "TAG_MODIFIED"
    # Guardrail: unselected by default!
    assert pair["selected"] is False
    assert pair["tag_modified_metrics"] is not None
    assert pair["tag_modified_metrics"]["duration_delta_sec"] == 0.044

    # Test 'A' (select all) does NOT select TAG_MODIFIED
    valid_candidates = [p for p in tui.batch_pairs if p.get("status") in ("READY", "UPGRADE")]
    for p in valid_candidates:
        p["selected"] = True
    assert pair["selected"] is False

    # Test 'T' toggles TAG_MODIFIED
    tui.screen = 22
    tui.ui_selected_idx = 0
    tui._handle_input(ord('t'))
    assert pair["selected"] is True

    # Test toggling again with 'T'
    tui._handle_input(ord('t'))
    assert pair["selected"] is False

    # Turn it back on for import
    pair["selected"] = True

    # Test batch import saves with audit provenance
    stdscr_mock = MagicMock()
    stdscr_mock.getmaxyx.return_value = (24, 80)
    stdscr_mock.getch.return_value = ord('q')

    with patch("curses.color_pair", return_value=0):
        tui._draw_screen_4b(stdscr_mock)

    with get_session() as session:
        tx_rec = session.query(TranscriptionRecord).filter(
            TranscriptionRecord.file_name.like("%Chapter 06 A.mp3%")
        ).order_by(TranscriptionRecord.id.desc()).first()
        assert tx_rec is not None
        assert tx_rec.source == "reimport:hash_override"

        a_rec = session.get(AudioFileRecord, tx_rec.audio_file_id)
        assert a_rec is not None
        tags = json.loads(a_rec.tags)
        assert tags["hash_verification"] == "bypassed_tag_modified"
        assert tags["original_json_hash"] == "different_id3_edited_hash_value_9999"
        assert tags["actual_audio_hash"] == actual_hash
        assert tags["duration_delta_sec"] == 0.044


def test_reimport_tui_interactive_failure_review_force_import(tmp_path, monkeypatch):
    folder = tmp_path / "BookFail"
    folder.mkdir()

    audio_file = folder / "Chapter 06 B.mp3"
    audio_file.write_bytes(b"content-identical-audio-B" * 1000)
    actual_hash = AudioMetadata.compute_file_hash(audio_file)

    tx_json = folder / "Chapter 06 B.json"
    tx_data = {
        "schema_version": 1,
        "full_text": "This is chapter 6 B audio content.",
        "duration_seconds": 3100.604,
        "engine": "whisper",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "This is chapter 6 B audio content."}
        ],
        "audio": {
            "file_name": "Chapter 06 B.mp3",
            "file_hash": "different_id3_edited_hash_value_8888",
            "duration_seconds": 3100.604,
            "file_size_bytes": len(b"content-identical-audio-B" * 1000),
        }
    }
    tx_json.write_text(json.dumps(tx_data), encoding="utf-8")

    from collections import namedtuple
    ProbeResult = namedtuple("ProbeResult", ["duration"])
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=3100.560)
    )

    tui = ReimportTUI()
    tui.is_auto_detect = True
    tui.audio_dir = folder
    tui.transcript_dir = folder

    tui._match_batch()
    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]

    # Leave it not marked TAG_MODIFIED so it fails in regular import (simulate unexpected failure)
    pair["status"] = "READY"
    pair["selected"] = True

    # Simulate keyboard inputs in interactive failure review:
    # 1st getch: ord('f') -> force imports the item
    # 2nd getch: ord('q') -> exits review screen
    key_sequence = [ord('f'), ord('q')]
    stdscr_mock = MagicMock()
    stdscr_mock.getmaxyx.return_value = (24, 80)
    stdscr_mock.getch.side_effect = key_sequence

    with patch("curses.color_pair", return_value=0):
        tui._draw_screen_4b(stdscr_mock)

    with get_session() as session:
        tx_rec = session.query(TranscriptionRecord).filter(
            TranscriptionRecord.file_name.like("%Chapter 06 B.mp3%")
        ).order_by(TranscriptionRecord.id.desc()).first()
        assert tx_rec is not None
        assert tx_rec.source == "reimport:hash_override"

        a_rec = session.get(AudioFileRecord, tx_rec.audio_file_id)
        assert a_rec is not None
        tags = json.loads(a_rec.tags)
        assert tags["hash_verification"] == "bypassed_tag_modified"
        assert tags["original_json_hash"] == "different_id3_edited_hash_value_8888"
        assert tags["actual_audio_hash"] == actual_hash
        assert tags["duration_delta_sec"] == 0.044
