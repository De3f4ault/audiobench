import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from click.testing import CliRunner

from audiobench.cli.commands.import_cmd import (
    find_companion_transcript,
    import_cmd,
    normalize_stem,
    run_import_flow,
    run_post_import_launchpad,
    token_similarity,
)
from audiobench.storage.importer import BackgroundImporter
from audiobench.storage.models import AudioFileRecord, StagingCartItem, TranscriptionRecord


def test_stem_normalization_and_token_similarity():
    """Verify stem normalization and fuzzy token similarity computation."""
    assert normalize_stem("01 - Beyond Good and Evil (Nietzsche)") == "01 beyond good and evil nietzsche"
    assert normalize_stem("01_beyond_good_and_evil") == "01 beyond good and evil"

    score = token_similarity(
        "01 - Friedrich Nietzsche - Beyond Good and Evil",
        "01_beyond_good_and_evil_nietzsche"
    )
    assert score >= 0.70

    unrelated_score = token_similarity("speech_welcome", "lecture_biology_101")
    assert unrelated_score < 0.30


def test_find_companion_transcript_exact_subfolder_fuzzy(tmp_path: Path):
    """Verify exact, subfolder, and fuzzy companion transcript discovery."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    tx_dir = audio_dir / "transcripts"
    tx_dir.mkdir()

    # 1. Exact match in same directory
    f1 = audio_dir / "interview.mp3"
    f1.touch()
    tx1 = audio_dir / "interview.json"
    tx1.touch()
    matched1, score1 = find_companion_transcript(f1)
    assert matched1 == tx1
    assert score1 == 1.0

    # 2. Match in transcripts/ subfolder
    f2 = audio_dir / "keynote.m4a"
    f2.touch()
    tx2 = tx_dir / "keynote.srt"
    tx2.touch()
    matched2, score2 = find_companion_transcript(f2)
    assert matched2 == tx2
    assert score2 == 1.0

    # 3. Fuzzy match in audio directory
    f3 = audio_dir / "01 - Introduction to Quantum.mp3"
    f3.touch()
    tx3 = audio_dir / "01_introduction_quantum.vtt"
    tx3.touch()
    matched3, score3 = find_companion_transcript(f3)
    assert matched3 == tx3
    assert score3 >= 0.70


def test_import_cmd_multiple_paths(tmp_path: Path):
    """Verify passing multiple paths routes to MODE_AUDIO with collected files."""
    runner = CliRunner()
    f1 = tmp_path / "track1.mp3"
    f2 = tmp_path / "track2.mp3"
    f1.touch()
    f2.touch()

    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, [str(f1), str(f2)])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_AUDIO"
        assert len(kwargs["target_files"]) == 2
        assert f1.resolve() in kwargs["target_files"]
        assert f2.resolve() in kwargs["target_files"]


def test_import_cmd_glob_and_flags(tmp_path: Path):
    """Verify glob pattern, -e engine, -m model, and -y non-interactive flag."""
    runner = CliRunner()
    f1 = tmp_path / "partA.mp3"
    f2 = tmp_path / "partB.mp3"
    f1.touch()
    f2.touch()

    glob_pattern = str(tmp_path / "*.mp3")
    with patch("audiobench.cli.commands.import_cmd.run_import_flow") as mock_flow:
        result = runner.invoke(import_cmd, [glob_pattern, "-e", "whisper", "-m", "large-v3", "-y"])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _, kwargs = mock_flow.call_args
        assert kwargs["initial_mode"] == "MODE_AUDIO"
        assert len(kwargs["target_files"]) == 2
        assert kwargs["default_engine"] == "whisper"
        assert kwargs["default_model"] == "large-v3"
        assert kwargs["interactive"] is False


def test_batch_allocation_hybrid_strategy(tmp_path: Path, monkeypatch):
    """Verify 1-Click batch allocation selects hybrid strategy without asking per file."""
    f1 = tmp_path / "ch1.mp3"
    f2 = tmp_path / "ch2.mp3"
    f1.write_bytes(b"audio1")
    f2.write_bytes(b"audio2")

    # Companion for f1 only
    tx1 = tmp_path / "ch1.json"
    tx1.write_text(json.dumps({
        "text": "Chapter one content",
        "engine": "gemini",
        "duration_seconds": 12.0,
        "segments": [{"start": 0.0, "end": 12.0, "text": "Chapter one content"}],
    }))

    mock_q = MagicMock()
    mock_select = MagicMock()
    # Batch strategy selected: "hybrid_gemini"
    mock_select.ask.return_value = "hybrid_gemini"
    mock_q.select.return_value = mock_select
    mock_q.Choice = lambda title, value: (title, value)
    monkeypatch.setattr("questionary.select", mock_q.select)
    monkeypatch.setattr("questionary.Choice", mock_q.Choice)

    mock_importer = MagicMock()
    captured_allocations = []
    def fake_run(allocs):
        captured_allocations.extend(allocs)
        return [{
            "source": a["file"],
            "destination": tmp_path / a["file"].name,
            "engine": a["engine"],
            "is_duplicate": False,
            "file_hash": f"hash_{a['file'].name}",
            "companion_file": a.get("companion_file"),
            "tqs_score": a.get("tqs_score"),
            "copy_method": "reflink",
        } for a in allocs]

    mock_importer.run_import_jobs.side_effect = fake_run

    with patch("audiobench.storage.importer.BackgroundImporter", return_value=mock_importer), \
         patch("audiobench.core.db_session.get_session") as mock_session_ctx, \
         patch("audiobench.cli.commands.import_cmd.run_post_import_launchpad"):

        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        imported_ids = run_import_flow(
            initial_mode="MODE_AUDIO",
            target_files=[f1, f2],
            interactive=True,
        )

        # Batch menu asked once for the whole batch
        assert len(captured_allocations) == 2
        alloc1 = next(a for a in captured_allocations if a["file"] == f1)
        alloc2 = next(a for a in captured_allocations if a["file"] == f2)

        assert alloc1["engine"].startswith("companion:")
        assert alloc1["companion_file"] == tx1
        assert alloc2["engine"] == "gemini"


def test_batch_allocation_non_interactive_whisper(tmp_path: Path):
    """Verify non-interactive mode with default_engine bypasses prompts entirely."""
    f1 = tmp_path / "audio_a.mp3"
    f2 = tmp_path / "audio_b.mp3"
    f1.write_bytes(b"data1")
    f2.write_bytes(b"data2")

    mock_importer = MagicMock()
    captured_allocations = []
    def fake_run(allocs):
        captured_allocations.extend(allocs)
        return [{
            "source": a["file"],
            "destination": tmp_path / a["file"].name,
            "engine": a["engine"],
            "is_duplicate": False,
            "file_hash": f"hash_{a['file'].name}",
            "copy_method": "reflink",
        } for a in allocs]
    mock_importer.run_import_jobs.side_effect = fake_run

    with patch("audiobench.storage.importer.BackgroundImporter", return_value=mock_importer), \
         patch("audiobench.core.db_session.get_session") as mock_session_ctx, \
         patch("audiobench.cli.commands.import_cmd.run_post_import_launchpad"):

        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        run_import_flow(
            initial_mode="MODE_AUDIO",
            target_files=[f1, f2],
            default_engine="whisper",
            default_model="medium",
            interactive=False,
        )

        assert len(captured_allocations) == 2
        for alloc in captured_allocations:
            assert alloc["engine"] == "whisper"
            assert alloc["model"] == "medium"


def test_cow_reflink_and_streaming_copy(tmp_path: Path):
    """Verify BackgroundImporter CoW reflink attempt and stream fallback."""
    importer = BackgroundImporter()
    src = tmp_path / "sample.mp3"
    src.write_bytes(b"0123456789" * 100)
    dst = tmp_path / "library" / "sample.mp3"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 1. Test streaming fallback when reflink fails
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value.returncode = 1  # reflink unsupported
        success, method = importer._copy_file_streaming(src, dst, src.name)
        assert success is True
        assert method == "stream_copy"
        assert dst.exists()
        assert dst.read_bytes() == src.read_bytes()

    # 2. Test reflink when subprocess cp succeeds
    dst2 = tmp_path / "library" / "sample2.mp3"
    with patch("subprocess.run") as mock_sub:
        def fake_cp(cmd, **kwargs):
            # simulate cp creating the destination
            Path(cmd[4]).write_bytes(src.read_bytes())
            res = MagicMock()
            res.returncode = 0
            return res
        mock_sub.side_effect = fake_cp
        success2, method2 = importer._copy_file_streaming(src, dst2, "sample2.mp3")
        assert success2 is True
        assert method2 == "reflink"
        assert dst2.exists()

    # 3. Test Windows platform fallback: does NOT invoke subprocess cp
    dst3 = tmp_path / "library" / "sample3.mp3"
    with patch("sys.platform", "win32"), \
         patch("audiobench.storage.importer._clone_refs_windows", return_value=False), \
         patch("subprocess.run") as mock_sub:
        success3, method3 = importer._copy_file_streaming(src, dst3, "sample3.mp3")
        assert success3 is True
        assert method3 == "stream_copy"
        assert dst3.exists()
        mock_sub.assert_not_called()

    # 4. Test Windows ReFS block cloning success
    dst4 = tmp_path / "library" / "sample4.mp3"
    with patch("sys.platform", "win32"), \
         patch("audiobench.storage.importer._clone_refs_windows") as mock_refs:
        def fake_refs(s, d):
            d.write_bytes(s.read_bytes())
            return True
        mock_refs.side_effect = fake_refs
        success4, method4 = importer._copy_file_streaming(src, dst4, "sample4.mp3")
        assert success4 is True
        assert method4 == "reflink"
        assert dst4.exists()

    # 5. Test macOS native clonefile
    dst5 = tmp_path / "library" / "sample5.mp3"
    with patch("os.clonefile", create=True) as mock_clone, \
         patch("subprocess.run") as mock_sub:
        def fake_clone(s, d):
            Path(d).write_bytes(Path(s).read_bytes())
        mock_clone.side_effect = fake_clone
        success5, method5 = importer._copy_file_streaming(src, dst5, "sample5.mp3")
        assert success5 is True
        assert method5 == "reflink"
        assert dst5.exists()
        mock_sub.assert_not_called()


def test_post_import_launchpad_transcribe_action(tmp_path: Path, monkeypatch):
    """Verify Post-Import Launchpad enqueues transcription jobs when requested."""
    f_rec = AudioFileRecord(
        id=101,
        file_path=str(tmp_path / "lecture.mp3"),
        file_name="lecture.mp3",
        file_size_bytes=1024,
        format="mp3",
        file_hash="mock_hash",
        tags='["engine_preference: whisper", "model_preference: large-v3"]',
    )

    mock_q = MagicMock()
    mock_select = MagicMock()
    mock_select.ask.return_value = "transcribe"
    mock_q.select.return_value = mock_select
    mock_q.Choice = lambda title, value: (title, value)
    monkeypatch.setattr("questionary.select", mock_q.select)
    monkeypatch.setattr("questionary.Choice", mock_q.Choice)

    with patch("audiobench.core.db_session.get_session") as mock_session_ctx, \
         patch("audiobench.jobs.scheduler.enqueue") as mock_enqueue:

        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            [f_rec],  # AudioFileRecord
            [],       # No TranscriptionRecord (pending)
        ]
        mock_session.query.return_value.get.return_value = f_rec

        run_post_import_launchpad([101], [], interactive=True)

        mock_enqueue.assert_called_once()
        _, enq_kwargs = mock_enqueue.call_args
        assert enq_kwargs["job_type"] == "transcribe"
        assert "--audio-id" in enq_kwargs["args"]
        assert "101" in enq_kwargs["args"]
        assert "--engine" in enq_kwargs["args"]
        assert "whisper" in enq_kwargs["args"]
        assert "--model" in enq_kwargs["args"]
        assert "large-v3" in enq_kwargs["args"]


def test_post_import_launchpad_cart_action(tmp_path: Path, monkeypatch):
    """Verify Post-Import Launchpad adds imported files to staging cart when requested."""
    f_rec = AudioFileRecord(
        id=202,
        file_path=str(tmp_path / "audio.mp3"),
        file_name="audio.mp3",
        file_size_bytes=2048,
        format="mp3",
        file_hash="mock_hash_2",
        tags='["engine_preference: gemini"]',
    )

    mock_q = MagicMock()
    mock_select = MagicMock()
    mock_select.ask.return_value = "cart"
    mock_q.select.return_value = mock_select
    mock_q.Choice = lambda title, value: (title, value)
    monkeypatch.setattr("questionary.select", mock_q.select)
    monkeypatch.setattr("questionary.Choice", mock_q.Choice)

    with patch("audiobench.core.db_session.get_session") as mock_session_ctx:
        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.all.side_effect = [
            [f_rec],
            [],
        ]
        # Cart query returns None (not in cart yet)
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.query.return_value.get.return_value = f_rec

        run_post_import_launchpad([202], [], interactive=True)

        mock_session.add.assert_called_once()
        added_item = mock_session.add.call_args[0][0]
        assert isinstance(added_item, StagingCartItem)
        assert added_item.audio_file_id == 202
        assert added_item.engine == "gemini"
