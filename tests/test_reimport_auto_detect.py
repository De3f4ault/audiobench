"""Unit tests for Auto-Detect import guardrails and matching logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobench.cli.tui.reimport_tui import ReimportTUI
from audiobench.transcribe.transcription_result import AudioMetadata


def test_match_batch_ignores_non_transcript_json(tmp_path: Path):
    # Setup test directory with non-transcript JSON and valid transcript JSON
    root = tmp_path / "imports"
    root.mkdir()

    # Non-transcript 1: package.json
    (root / "package.json").write_text(json.dumps({"name": "test-app", "version": "1.0.0"}), encoding="utf-8")

    # Non-transcript 2: transcripts_registry.json
    (root / "transcripts_registry.json").write_text(json.dumps({"entries": [{"id": 1, "path": "abc"}]}), encoding="utf-8")

    # Non-transcript 3: Empty json or list
    (root / "empty.json").write_text("[]", encoding="utf-8")

    # Audio file
    audio = root / "sample.mp3"
    audio.write_bytes(b"dummy audio content 12345678")
    audio_hash = AudioMetadata.compute_file_hash(audio)

    # Valid transcript
    valid_tx = {
        "schema_version": "1",
        "language": "en",
        "duration_seconds": 10.0,
        "text": "Hello world this is a test audio file.",
        "segments": [
            {
                "id": 0,
                "text": "Hello world this is a test audio file.",
                "start": 0.0,
                "end": 3.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.99},
                ]
            }
        ],
        "audio": {
            "file_name": "sample.mp3",
            "file_hash": audio_hash,
            "duration_seconds": 10.0,
        }
    }
    (root / "sample.json").write_text(json.dumps(valid_tx), encoding="utf-8")

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root

    tui._match_batch()

    # Only sample.json should be matched!
    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["tx"].name == "sample.json"
    assert pair["audio"] is not None
    assert pair["audio"].name == "sample.mp3"
    assert pair["status"] == "READY"
    assert pair["selected"] is True


def test_match_batch_flags_missing_audio(tmp_path: Path):
    root = tmp_path / "imports"
    root.mkdir()

    # Valid transcript but NO audio file exists
    valid_tx = {
        "schema_version": "1",
        "text": "No audio exists for this text.",
        "segments": [
            {"id": 0, "text": "No audio exists for this text.", "start": 0.0, "end": 2.0}
        ],
        "audio": {
            "file_name": "missing.mp3",
            "duration_seconds": 10.0,
        }
    }
    (root / "missing.json").write_text(json.dumps(valid_tx), encoding="utf-8")

    # In auto-detect mode, orphaned transcripts without audio are strictly filtered out
    tui_auto = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui_auto.transcript_dir = root
    tui_auto.audio_dir = root
    tui_auto._match_batch()
    assert len(tui_auto.batch_pairs) == 0

    # In batch mode, orphaned transcripts are kept and flagged NO_AUDIO
    tui_batch = ReimportTUI(start_path=str(root), audio_start_path=str(root), batch=True)
    tui_batch.transcript_dir = root
    tui_batch.audio_dir = root
    tui_batch._match_batch()
    assert len(tui_batch.batch_pairs) == 1
    pair = tui_batch.batch_pairs[0]
    assert pair["status"] == "NO_AUDIO"
    assert pair["audio"] is None
    assert pair["selected"] is False


def test_match_batch_flags_hash_mismatch(tmp_path: Path):
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "mismatch.mp3"
    audio.write_bytes(b"actual content")

    valid_tx = {
        "schema_version": "1",
        "text": "Mismatch text.",
        "segments": [
            {"id": 0, "text": "Mismatch text.", "start": 0.0, "end": 2.0}
        ],
        "audio": {
            "file_name": "mismatch.mp3",
            "file_hash": "different_hash_that_does_not_match",
            "duration_seconds": 10.0,
        }
    }
    (root / "mismatch.json").write_text(json.dumps(valid_tx), encoding="utf-8")

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root

    tui._match_batch()

    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["status"] == "HASH_MISMATCH"
    assert pair["selected"] is False


def test_match_batch_detects_upgrade_for_existing_inferior_record(tmp_path: Path):
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "book.mp3"
    audio.write_bytes(b"dummy audio content for book")
    audio_hash = AudioMetadata.compute_file_hash(audio)

    # High quality incoming transcript with word timestamps
    words = [
        {"word": "The", "start": 0.0, "end": 0.5, "probability": 0.98},
        {"word": "Greatest", "start": 0.5, "end": 1.0, "probability": 0.98},
        {"word": "Salesman", "start": 1.0, "end": 1.5, "probability": 0.98},
        {"word": "in", "start": 1.5, "end": 1.8, "probability": 0.98},
        {"word": "the", "start": 1.8, "end": 2.0, "probability": 0.98},
        {"word": "World.", "start": 2.0, "end": 2.5, "probability": 0.98},
    ]
    valid_tx = {
        "schema_version": "1",
        "text": "The Greatest Salesman in the World.",
        "duration_seconds": 10.0,
        "engine": "whisper-large-v3",
        "segments": [
            {
                "id": 0,
                "text": "The Greatest Salesman in the World.",
                "start": 0.0,
                "end": 2.5,
                "words": words,
            }
        ],
        "audio": {
            "file_name": "book.mp3",
            "file_hash": audio_hash,
            "duration_seconds": 10.0,
        }
    }
    (root / "book.json").write_text(json.dumps(valid_tx), encoding="utf-8")

    # Mock DB record: Defective transcript (0 word timestamps, collapsed segment)
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    class MockExistingRecord:
        id = 999
        audio_file_id = 101
        file_name = "book.mp3"
        duration_seconds = 10.0
        engine = "gemini-2.5-flash"
        full_text = "The Greatest Salesman in the World."
        segments = [
            MagicMock(start_time=0.0, end_time=0.0, text="The Greatest Salesman in the World.", words=[]),
        ]
        audio_file = None

    class MockAudioRecord:
        id = 101
        file_name = "book.mp3"
        file_path = str(audio)
        file_hash = audio_hash
        duration_seconds = 10.0

    mock_audio_rec = MockAudioRecord()
    mock_existing_rec = MockExistingRecord()

    def mock_query(model):
        m = MagicMock()
        if model == AudioFileRecord:
            m.all.return_value = [mock_audio_rec]
            m.filter_by.return_value.first.return_value = mock_audio_rec
        elif model == TranscriptionRecord:
            m.all.return_value = [mock_existing_rec]
            m.filter_by.return_value.first.return_value = mock_existing_rec
        else:
            m.all.return_value = []
            m.filter_by.return_value.first.return_value = None
        return m

    mock_session = MagicMock()
    mock_session.query.side_effect = mock_query

    with patch("audiobench.core.db_session.get_session") as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session

        tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
        tui.transcript_dir = root
        tui.audio_dir = root

        tui._match_batch()

        assert len(tui.batch_pairs) == 1
        pair = tui.batch_pairs[0]
        assert pair["status"] == "UPGRADE"
        assert pair["selected"] is True
        assert pair["is_existing"] is True
        assert pair["comparison"].recommendation == "UPGRADE_RECOMMENDED"
        assert pair["comparison"].delta_tqs >= 15.0


def test_draw_screen_22_renders_comparison_inspector():
    tui = ReimportTUI(auto_detect=True)

    from audiobench.transcribe.quality_evaluator import ComparisonResult, FamiliarityScore, QualityScore

    inc_tqs = QualityScore(
        total_score=97.5,
        coverage_score=29.5,
        granularity_score=25.0,
        health_score=25.0,
        confidence_score=18.0,
        degeneration_penalty=0.0,
        duration_seconds=100.0,
        transcribed_span=99.8,
        coverage_ratio=0.998,
        word_count=500,
        segment_count=50,
        wpm=150.0,
        words_with_timestamps_ratio=1.0,
        avg_segment_duration=2.0,
        zero_duration_segments_count=0,
        stretched_segments_count=0,
        inversion_count=0,
        avg_word_probability=0.95,
        repetition_loops_count=0,
        grade="PRODUCTION_GRADE",
    )
    ex_tqs = QualityScore(
        total_score=55.0,
        coverage_score=25.0,
        granularity_score=5.0,
        health_score=10.0,
        confidence_score=15.0,
        degeneration_penalty=0.0,
        duration_seconds=100.0,
        transcribed_span=80.0,
        coverage_ratio=0.80,
        word_count=350,
        segment_count=30,
        wpm=120.0,
        words_with_timestamps_ratio=0.0,
        avg_segment_duration=2.7,
        zero_duration_segments_count=4,
        stretched_segments_count=2,
        inversion_count=0,
        avg_word_probability=0.80,
        repetition_loops_count=0,
        grade="SUBPAR",
    )
    comp = ComparisonResult(
        existing_id=1506,
        existing_tqs=ex_tqs,
        incoming_tqs=inc_tqs,
        familiarity=FamiliarityScore(95.0, 0.95, 0.92, 0.98, 0.90, "DIRECT_MATCH"),
        delta_tqs=42.5,
        recommendation="UPGRADE_RECOMMENDED",
        summary="Upgrade recommended (+42.5 TQS)",
    )

    dummy_audio = MagicMock()
    dummy_audio.name = "audiobook.mp3"
    dummy_audio.stat.return_value.st_size = 50 * 1024 * 1024
    dummy_audio.exists.return_value = True

    dummy_tx = MagicMock()
    dummy_tx.name = "audiobook.json"

    tui.batch_pairs = [
        {
            "tx": dummy_tx,
            "audio": dummy_audio,
            "status": "UPGRADE",
            "selected": True,
            "is_existing": True,
            "existing_record_id": 1506,
            "comparison": comp,
            "incoming_tqs": inc_tqs,
            "existing_tqs": ex_tqs,
            "familiarity": comp.familiarity,
        }
    ]

    mock_stdscr = MagicMock()
    mock_stdscr.getmaxyx.return_value = (30, 100)

    # Call _draw_screen_22
    with patch("curses.color_pair", return_value=0):
        tui._draw_screen_22(mock_stdscr)

    # Verify header, divider, comparison, and footer rendered
    calls = [call[0] for call in mock_stdscr.addstr.call_args_list]
    rendered_strings = [c[2] for c in calls if len(c) >= 3]

    # Verify key UI components
    assert any("Auto-Detect & Quality Inspector" in s for s in rendered_strings)
    assert any("[UPGRADE]" in s for s in rendered_strings)
    assert any("Live Quality & Health Comparison Inspector" in s for s in rendered_strings)
    assert any("INCOMING CANDIDATE" in s for s in rendered_strings)
    assert any("EXISTING DB RECORD (#1506)" in s for s in rendered_strings)
    assert any("UPGRADE_RECOMMENDED" in s for s in rendered_strings)


def test_match_batch_flags_duration_mismatch(tmp_path: Path):
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "preview.mp3"
    audio.write_bytes(b"short preview audio")

    # 1-hour transcript paired with 30s audio
    tx = {
        "schema_version": "1",
        "text": "Extremely long 1 hour book...",
        "duration_seconds": 3600.0,
        "segments": [
            {"id": 0, "text": "Start", "start": 0.0, "end": 10.0},
            {"id": 1, "text": "End", "start": 3500.0, "end": 3600.0},
        ],
        "audio": {
            "file_name": "preview.mp3",
            "duration_seconds": 30.0,  # Audio is only 30s
        }
    }
    (root / "preview.json").write_text(json.dumps(tx), encoding="utf-8")

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root
    tui._match_batch()

    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["status"] == "DURATION_MISMATCH"
    assert pair["selected"] is False


def test_match_batch_flags_empty_transcript(tmp_path: Path):
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "empty_speech.mp3"
    audio.write_bytes(b"audio content")

    tx = {
        "schema_version": "1",
        "text": "",
        "segments": [],
        "audio": {
            "file_name": "empty_speech.mp3",
            "duration_seconds": 10.0,
        }
    }
    (root / "empty_speech.json").write_text(json.dumps(tx), encoding="utf-8")

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root
    tui._match_batch()

    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["status"] == "EMPTY"
    assert pair["selected"] is False


def test_match_batch_recursive_blacklist_and_nested_discovery(tmp_path: Path):
    root = tmp_path / "downloads"
    root.mkdir()

    # Blacklisted folders: node_modules and .git
    nm = root / "node_modules" / "some_pkg"
    nm.mkdir(parents=True)
    (nm / "ignored.json").write_text(json.dumps({"schema_version": "1", "text": "bad", "segments": [{"id": 0, "start": 0, "end": 1, "text": "bad"}]}))

    git = root / ".git" / "hooks"
    git.mkdir(parents=True)
    (git / "hook.json").write_text(json.dumps({"schema_version": "1", "text": "bad", "segments": [{"id": 0, "start": 0, "end": 1, "text": "bad"}]}))

    # Legitimate nested directory
    nested_tx = root / "audiobook_transcripts" / "deep_folder"
    nested_tx.mkdir(parents=True)
    nested_au = root / "audiobook_files" / "deep_folder"
    nested_au.mkdir(parents=True)

    audio = nested_au / "nested_book.mp3"
    audio.write_bytes(b"audio data for nested book")

    tx = {
        "schema_version": "1",
        "text": "Deeply nested transcript content.",
        "segments": [
            {"id": 0, "text": "Deeply nested transcript content.", "start": 0.0, "end": 5.0}
        ],
        "audio": {
            "file_name": "nested_book.mp3",
            "duration_seconds": 5.0,
        }
    }
    (nested_tx / "nested_book.json").write_text(json.dumps(tx), encoding="utf-8")

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root
    tui._match_batch()

    # Only nested_book should be found; node_modules and .git must be pruned
    assert len(tui.batch_pairs) == 1
    pair = tui.batch_pairs[0]
    assert pair["tx"].name == "nested_book.json"
    assert pair["audio"].name == "nested_book.mp3"
    assert pair["status"] == "READY"
    assert pair["selected"] is True


def test_match_batch_flags_identical_existing_transcript_as_current(tmp_path: Path):
    """Verify that incoming files with identical/similar quality to existing records are flagged CURRENT and NOT reimported."""
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "already_imported.mp3"
    audio.write_bytes(b"dummy audio content for existing file")
    audio_hash = AudioMetadata.compute_file_hash(audio)

    # Identical transcript
    words = [
        {"word": "Already", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": "imported", "start": 0.5, "end": 1.0, "probability": 0.99},
        {"word": "speech.", "start": 1.0, "end": 1.5, "probability": 0.99},
    ]
    valid_tx = {
        "schema_version": "1",
        "text": "Already imported speech.",
        "duration_seconds": 10.0,
        "engine": "whisper-large-v3",
        "segments": [
            {
                "id": 0,
                "text": "Already imported speech.",
                "start": 0.0,
                "end": 1.5,
                "words": words,
            }
        ],
        "audio": {
            "file_name": "already_imported.mp3",
            "file_hash": audio_hash,
            "duration_seconds": 10.0,
        }
    }
    (root / "already_imported.json").write_text(json.dumps(valid_tx), encoding="utf-8")

    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    class MockExistingRecord:
        id = 2001
        audio_file_id = 501
        file_name = "already_imported.mp3"
        duration_seconds = 10.0
        engine = "whisper-large-v3"
        full_text = "Already imported speech."
        segments = [
            MagicMock(start_time=0.0, end_time=1.5, text="Already imported speech.", words=words),
        ]
        audio_file = None

    class MockAudioRecord:
        id = 501
        file_name = "already_imported.mp3"
        file_path = str(audio)
        file_hash = audio_hash
        duration_seconds = 10.0

    mock_audio = MockAudioRecord()
    mock_tx = MockExistingRecord()

    def mock_query(model):
        m = MagicMock()
        if model == AudioFileRecord:
            m.all.return_value = [mock_audio]
            m.filter_by.return_value.first.return_value = mock_audio
        elif model == TranscriptionRecord:
            m.all.return_value = [mock_tx]
            m.filter_by.return_value.first.return_value = mock_tx
        else:
            m.all.return_value = []
            m.filter_by.return_value.first.return_value = None
        return m

    mock_session = MagicMock()
    mock_session.query.side_effect = mock_query

    with patch("audiobench.core.db_session.get_session") as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session

        tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
        tui.transcript_dir = root
        tui.audio_dir = root
        tui._match_batch()

        assert len(tui.batch_pairs) == 1
        pair = tui.batch_pairs[0]

        # 1. Must be recognized as an existing record
        assert pair["is_existing"] is True
        assert pair["existing_record_id"] == 2001

        # 2. Recommendation must NOT recommend upgrade
        assert pair["comparison"].recommendation in ("KEEP_EXISTING", "SIMILAR_QUALITY")

        # 3. Status must be CURRENT and selected must be False (unchecked!)
        assert pair["status"] == "CURRENT"
        assert pair["selected"] is False

        # 4. Pressing 'A' (Select All) MUST NOT check CURRENT items!
        tui.screen = 22
        tui._handle_input(ord('a'))
        assert pair["selected"] is False  # Still False!

        # 5. Pressing 'S' (Start Batch) must filter out unselected items -> 0 items to import
        tui._handle_input(ord('s'))
        assert len(tui.batch_pairs) == 0  # Excluded! Zero files reimported!


def test_smart_fit_name_preserves_title_on_narrow_and_wide_columns():
    long_name = "David Thomas, Andrew Hunt - The Pragmatic Programmer 20th Anniversary, 2nd Editiony.json"

    # Narrow column (30 chars) - title must be prioritized rather than chopped by author prefix
    narrow_res = ReimportTUI._smart_fit_name(long_name, 30)
    assert "Pragmatic" in narrow_res

    # Medium column (45 chars)
    med_res = ReimportTUI._smart_fit_name(long_name, 45)
    assert "The Pragmatic Programmer" in med_res

    # Wide column (70 chars) - both author and title fit
    wide_res = ReimportTUI._smart_fit_name(long_name, 70)
    assert "David Thomas" in wide_res
    assert "The Pragmatic Programmer" in wide_res


def test_match_batch_deduplicates_identical_audio_candidates(tmp_path):
    # Setup root with loose transcript and nested transcript targeting same audio
    root = tmp_path / "downloads"
    nested = root / "whisper_json_transcripts"
    nested.mkdir(parents=True)

    audio_file = root / "01 - A Mind for Numbers.mp3"
    audio_file.write_bytes(b"dummy mp3 data" * 100)

    tx_loose = root / "01 - A Mind for Numbers.json"
    tx_nested = nested / "01 - A Mind for Numbers.json"

    dummy_payload = json.dumps({
        "schema_version": 1,
        "duration_seconds": 120.0,
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Segment one", "words": [{"word": "Segment", "start": 0.0, "end": 1.0, "probability": 0.99}]},
            {"start": 60.0, "end": 120.0, "text": "Segment two", "words": [{"word": "two", "start": 60.0, "end": 61.0, "probability": 0.99}]}
        ],
        "audio": {
            "file_name": "01 - A Mind for Numbers.mp3",
            "file_size_bytes": 1400,
            "duration_seconds": 120.0
        }
    })
    tx_loose.write_text(dummy_payload)
    tx_nested.write_text(dummy_payload)

    with patch("audiobench.core.db_session.get_session") as mock_session_ctx:
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = []
        mock_session_ctx.return_value.__enter__.return_value = mock_session

        with patch("audiobench.transcribe.quality_evaluator.evaluate_transcript_quality") as mock_eval:
            mock_score = MagicMock()
            mock_score.total_score = 95.0
            mock_score.grade = "PRODUCTION_GRADE"
            mock_eval.return_value = mock_score

            tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
            tui.transcript_dir = root
            tui.audio_dir = root
            tui._match_batch()

            # Must deduplicate to exactly 1 candidate pair
            assert len(tui.batch_pairs) == 1
            # Preferred the deeper/organized copy in whisper_json_transcripts
            assert "whisper_json_transcripts" in str(tui.batch_pairs[0]["tx"])


def test_match_batch_flags_physical_audio_duration_mismatch_even_if_json_metadata_matches_segments(tmp_path: Path):
    """Regression test: if a transcript's audio.duration_seconds matches its own segments (e.g. 2358s),
    but the physical audio file on disk is longer (e.g. 3823s), Auto-Detect must probe the disk
    file and flag DURATION_MISMATCH rather than falsely reporting READY.
    """
    root = tmp_path / "imports"
    root.mkdir()

    audio = root / "Final Spin-Part04.mp3"
    with open(audio, "wb") as f:
        f.truncate(30590108)

    # Transcript where internal audio metadata says 2358.9s (matching segments),
    # but disk audio probe returns 3823.4s
    tx = {
        "schema_version": "1",
        "text": "Partial chapter dialogue...",
        "duration_seconds": 2358.909,
        "segments": [
            {"id": 0, "text": "Start dialogue", "start": 0.0, "end": 10.0},
            {"id": 1, "text": "End dialogue", "start": 2350.0, "end": 2358.88},
        ],
        "audio": {
            "file_name": "Final Spin-Part04.mp3",
            "file_size_bytes": 30590108,
            "duration_seconds": 2358.909,  # Internally consistent with segments!
            "file_hash": "dummy_hash_1234",
        }
    }
    (root / "Final Spin-Part04.json").write_text(json.dumps(tx), encoding="utf-8")

    # Mock audio_converter.probe to return the real physical duration of 3823.4s
    mock_probe_info = MagicMock()
    mock_probe_info.duration = 3823.38

    with patch("audiobench.transcribe.audio_converter.probe", return_value=mock_probe_info):
        tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
        tui.transcript_dir = root
        tui.audio_dir = root
        tui._match_batch()

        assert len(tui.batch_pairs) == 1
        pair = tui.batch_pairs[0]
        # Must catch duration mismatch against physical audio!
        assert pair["status"] == "DURATION_MISMATCH"
        assert pair["selected"] is False
        assert pair["audio_duration"] == 3823.38
        assert pair["tx_duration"] == 2358.909

