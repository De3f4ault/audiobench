"""Tests for memory and I/O efficiency during batch transcript imports."""

import gc
import json
import tracemalloc
from pathlib import Path

from audiobench.cli.tui.reimport_tui import ReimportTUI
from audiobench.transcribe.quality_evaluator import (
    evaluate_familiarity,
    evaluate_transcript_quality,
)
from audiobench.transcribe.transcription_result import AudioMetadata


def test_streaming_tqs_generator_input():
    """Verify that evaluate_transcript_quality accepts lazy generators with O(1) memory."""
    def segment_generator():
        for i in range(50):
            words = [
                {"word": f"word_{i}_{w}", "start": i * 2.0 + w * 0.2, "end": i * 2.0 + w * 0.2 + 0.18, "probability": 0.95}
                for w in range(10)
            ]
            yield {
                "id": i,
                "text": f"segment number {i} with ten spoken words here today",
                "start": i * 2.0,
                "end": i * 2.0 + 1.9,
                "words": words,
            }

    score = evaluate_transcript_quality(segment_generator(), total_duration=100.0, engine="whisper-large-v3")
    assert score.total_score >= 85.0
    assert score.grade == "PRODUCTION_GRADE"
    assert score.segment_count == 50
    assert score.word_count == 450  # 9 words per segment text * 50
    assert score.words_with_timestamps_ratio == 1.0


def test_stratified_familiarity_large_documents():
    """Verify that evaluate_familiarity efficiently and accurately discriminates large texts."""
    base_text = "The quick brown fox jumps over the lazy dog and explores the vast forest. " * 3000  # ~45k words
    identical_text = base_text
    abridged_text = base_text[:int(len(base_text) * 0.85)]  # Cut 15%
    unrelated_text = "Call me Ishmael. Some years ago never mind how long precisely. " * 3000

    fam_identical = evaluate_familiarity(base_text, identical_text)
    assert fam_identical.classification == "DIRECT_MATCH"
    assert fam_identical.score == 100.0

    fam_abridged = evaluate_familiarity(base_text, abridged_text)
    assert fam_abridged.classification == "DIRECT_MATCH"
    assert fam_abridged.score >= 75.0
    assert fam_abridged.containment >= 0.90

    fam_unrelated = evaluate_familiarity(base_text, unrelated_text)
    assert fam_unrelated.classification == "MISMATCH"
    assert fam_unrelated.score < 20.0


def test_batch_scan_peak_memory_and_flyweight_retention(tmp_path: Path):
    """Verify that batch directory scan remains strictly under 20 MB peak memory."""
    root = tmp_path / "batch_test"
    root.mkdir()

    # Generate 15 synthetic audiobook candidates
    for i in range(15):
        audio_file = root / f"book_{i:02d}.mp3"
        audio_file.write_bytes(b"dummy audio content " * 100)
        audio_hash = AudioMetadata.compute_file_hash(audio_file)
        audio_size = audio_file.stat().st_size

        segments = []
        for s_idx in range(100):
            seg_words = [
                {"word": f"word_{s_idx}_{w}", "start": s_idx * 2.0 + w * 0.2, "end": s_idx * 2.0 + w * 0.2 + 0.18, "probability": 0.95}
                for w in range(10)
            ]
            segments.append({
                "id": s_idx,
                "text": f"This is segment {s_idx} of synthetic audiobook {i} speaking words.",
                "start": s_idx * 2.0,
                "end": s_idx * 2.0 + 1.9,
                "words": seg_words,
            })

        tx_data = {
            "schema_version": "1",
            "language": "en",
            "duration_seconds": 200.0,
            "engine": "whisper-large-v3",
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "audio": {
                "file_name": audio_file.name,
                "file_size_bytes": audio_size,
                "file_hash": audio_hash,
                "duration_seconds": 200.0,
            }
        }
        (root / f"book_{i:02d}.json").write_text(json.dumps(tx_data), encoding="utf-8")

    gc.collect()
    tracemalloc.start()

    tui = ReimportTUI(start_path=str(root), audio_start_path=str(root), auto_detect=True)
    tui.transcript_dir = root
    tui.audio_dir = root
    tui._match_batch()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory assertion: 15 books should consume < 15 MB peak heap memory
    assert peak_mem < 15 * 1024 * 1024, f"Peak memory was {peak_mem / (1024*1024):.2f} MB, expected < 15 MB"

    # Candidate structure assertions
    assert len(tui.batch_pairs) == 15
    for pair in tui.batch_pairs:
        # Crucial: Full Pydantic Transcript and raw audio dict must NOT be retained in memory!
        assert "parsed_tx" not in pair
        assert "raw_audio" not in pair

        # Lightweight scalars and comparison must be present
        assert pair["status"] in ("READY", "UPGRADE", "CURRENT")
        assert pair["comparison"] is not None
        assert pair["incoming_tqs"] is not None
        assert pair["incoming_tqs"].total_score >= 80.0
