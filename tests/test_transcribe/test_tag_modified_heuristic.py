import json
from pathlib import Path
import pytest

from audiobench.transcribe.transcript_parser import (
    normalize_stem_strict,
    evaluate_tag_modified_heuristic,
    build_audio_metadata,
    HashMismatchError,
)
from audiobench.transcribe.transcription_result import AudioMetadata, Transcript, Segment


def test_normalize_stem_strict_numeric_tokens():
    # Identical with leading zeroes or spaces or symbols
    assert normalize_stem_strict("Chapter 06") == "chapter 6"
    assert normalize_stem_strict("Chapter 6") == "chapter 6"
    assert normalize_stem_strict("Chapter-06") == "chapter 6"
    assert normalize_stem_strict("01 - Introduction") == "1 introduction"
    assert normalize_stem_strict("1. Introduction") == "1 introduction"

    # Strict isolation: adjacent chapters must NEVER conflate
    assert normalize_stem_strict("Chapter 6") != normalize_stem_strict("Chapter 16")
    assert normalize_stem_strict("Chapter 6") != normalize_stem_strict("Chapter 60")
    assert normalize_stem_strict("Chapter 06") != normalize_stem_strict("Chapter 60")
    assert normalize_stem_strict("Chapter 1") != normalize_stem_strict("Chapter 11")


def test_tag_modified_heuristic_stem_mismatch(tmp_path):
    # Dummy audio file
    audio_path = tmp_path / "Chapter 6.mp3"
    audio_path.write_bytes(b"dummy audio data")

    raw_audio = {
        "file_name": "Chapter 16.mp3",
        "file_hash": "deadbeef" * 8,
        "duration_seconds": 100.0,
        "file_size_bytes": 1000,
    }

    passes, metrics = evaluate_tag_modified_heuristic(audio_path, raw_audio, tx_duration=100.0)
    assert passes is False
    assert "Stem name mismatch" in metrics["reason"]
    assert metrics["stem_matched"] is False


def test_tag_modified_heuristic_duration_and_size_tolerances(monkeypatch, tmp_path):
    audio_path = tmp_path / "Chapter 6.mp3"
    audio_path.write_bytes(b"x" * 50_000_000)  # ~50MB

    # Mock probe to return valid duration
    from collections import namedtuple
    ProbeResult = namedtuple("ProbeResult", ["duration"])
    
    # Case 1: Drift within 0.2% and <= 0.5s (0.044s on 3100s = 0.0014%)
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=3100.560)
    )

    raw_audio = {
        "file_name": "Chapter 6.mp3",
        "file_hash": "original_sha256_hash",
        "duration_seconds": 3100.604,
        "file_size_bytes": 50_000_000,
    }

    passes, metrics = evaluate_tag_modified_heuristic(audio_path, raw_audio)
    assert passes is True
    assert metrics["stem_matched"] is True
    assert metrics["probe_passed"] is True
    assert metrics["duration_passed"] is True
    assert metrics["size_passed"] is True
    assert metrics["duration_delta_sec"] == 0.044
    assert metrics["size_delta_bytes"] == 0

    # Case 2: Duration drift exceeds 0.5s absolute cap (even if <= 0.2%)
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=3101.5)  # delta = 0.896s > 0.50s
    )
    passes, metrics = evaluate_tag_modified_heuristic(audio_path, raw_audio)
    assert passes is False
    assert "Duration drift exceeds tolerance" in metrics["reason"]

    # Case 3: Duration drift exceeds 0.2% relative cap (e.g. short 10s audio with 0.1s delta = 1.0%)
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=10.10)
    )
    short_raw = {
        "file_name": "Chapter 6.mp3",
        "file_hash": "original_sha256_hash",
        "duration_seconds": 10.0,
        "file_size_bytes": 50_000_000,
    }
    passes, metrics = evaluate_tag_modified_heuristic(audio_path, short_raw)
    assert passes is False
    assert "Duration drift exceeds tolerance" in metrics["reason"]

    # Case 4: File size delta exceeds ID3 tag ceiling (max(2MB, 1%))
    monkeypatch.setattr(
        "audiobench.transcribe.audio_converter.probe",
        lambda p: ProbeResult(duration=3100.60)
    )
    size_skewed_raw = {
        "file_name": "Chapter 6.mp3",
        "file_hash": "original_sha256_hash",
        "duration_seconds": 3100.604,
        "file_size_bytes": 45_000_000,  # 5MB delta > 2MB and > 1% (500KB)
    }
    passes, metrics = evaluate_tag_modified_heuristic(audio_path, size_skewed_raw)
    assert passes is False
    assert "exceeds ID3 tag ceiling" in metrics["reason"]


def test_build_audio_metadata_allow_hash_mismatch(monkeypatch, tmp_path):
    audio_path = tmp_path / "Chapter 6.mp3"
    audio_path.write_bytes(b"hello world audio")

    raw_audio = {
        "file_name": "Chapter 6.mp3",
        "file_hash": "differenthask1234567890",
        "duration_seconds": 120.0,
        "file_size_bytes": len(b"hello world audio"),
    }

    # Without allow_hash_mismatch -> must raise HashMismatchError
    with pytest.raises(HashMismatchError):
        build_audio_metadata(audio_path, raw_audio, allow_hash_mismatch=False)

    # With allow_hash_mismatch=True -> succeeds and computes actual local hash
    meta = build_audio_metadata(audio_path, raw_audio, allow_hash_mismatch=True)
    assert isinstance(meta, AudioMetadata)
    assert meta.file_hash == AudioMetadata.compute_file_hash(audio_path)
    assert meta.duration_seconds == 120.0


def test_repository_save_transcription_audit_trail(tmp_path, monkeypatch):
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.core.settings import get_settings
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import TranscriptionRecord, AudioFileRecord

    settings = get_settings()
    audit_log_path = settings.data_dir / "logs" / "reimport_audit.log"
    if audit_log_path.exists():
        audit_log_path.unlink()

    repo = TranscriptionRepository()

    audio_path = tmp_path / "Chapter 6.mp3"
    audio_path.write_bytes(b"mock audio bytes")
    local_hash = AudioMetadata.compute_file_hash(audio_path)

    meta = AudioMetadata(
        file_path=str(audio_path),
        file_name=audio_path.name,
        file_size_bytes=len(b"mock audio bytes"),
        format="mp3",
        duration_seconds=3000.0,
        sample_rate=44100,
        channels=2,
        file_hash=local_hash,
    )

    transcript = Transcript(
        segments=[
            Segment(id=0, text="Welcome to chapter 6.", start=0.0, end=5.0)
        ],
        language="en",
        language_probability=1.0,
        duration_seconds=3000.0,
        engine="whisper",
        model_name="large-v3",
        speaker_map={},
        chapters=[],
    )

    audit_prov = {
        "hash_verification": "bypassed_tag_modified",
        "verified_by": "operator_interactive",
        "original_json_hash": "original_json_hash_value_12345",
        "actual_audio_hash": local_hash,
        "duration_delta_sec": 0.044,
        "size_delta_bytes": 0,
        "source": "reimport:hash_override",
    }

    tx_id = repo.save_transcription(
        transcript,
        meta,
        overwrite=True,
        audit_provenance=audit_prov,
    )

    assert tx_id > 0

    with get_session() as session:
        tx_rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
        assert tx_rec is not None
        assert tx_rec.source == "reimport:hash_override"

        audio_rec = session.query(AudioFileRecord).filter_by(id=tx_rec.audio_file_id).first()
        assert audio_rec is not None
        tags = json.loads(audio_rec.tags)
        assert tags["hash_verification"] == "bypassed_tag_modified"
        assert tags["verified_by"] == "operator_interactive"
        assert tags["original_json_hash"] == "original_json_hash_value_12345"
        assert tags["actual_audio_hash"] == local_hash
        assert tags["duration_delta_sec"] == 0.044
        assert tags["size_delta_bytes"] == 0

    # Verify structured audit log entry in reimport_audit.log
    assert audit_log_path.exists()
    lines = audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    last_entry = json.loads(lines[-1])
    assert last_entry["transcription_id"] == tx_id
    assert last_entry["hash_verification"] == "bypassed_tag_modified"
    assert last_entry["verified_by"] == "operator_interactive"
    assert last_entry["original_json_hash"] == "original_json_hash_value_12345"
    assert last_entry["actual_audio_hash"] == local_hash
    assert last_entry["duration_delta_sec"] == 0.044
