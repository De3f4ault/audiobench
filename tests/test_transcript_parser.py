"""Tests for reverse import transcript parser."""

import json
from pathlib import Path

import pytest

from audiobench.transcribe.transcript_parser import (
    HashMismatchError,
    UnsupportedSchemaVersion,
    build_audio_metadata,
    parse_transcript_file,
)


def test_parse_json(tmp_path: Path):
    data = {
        "schema_version": "1",
        "language": "en",
        "duration_seconds": 10.0,
        "segments": [
            {
                "id": 0,
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99, "duration": 0.5, "midpoint": 0.25},
                    {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.99, "duration": 0.5, "midpoint": 0.75}
                ],
                "duration": 1.0,
                "word_count": 2
            }
        ],
        "audio": {
            "file_name": "test.mp3",
            "file_size_bytes": 100,
            "format": "mp3",
            "duration_seconds": 10.5,
            "sample_rate": 44100,
            "channels": 2,
            "file_hash": "abc123hash"
        },
        "text": "Hello world",
        "word_count": 2,
        "segment_count": 1
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    transcript, raw_audio = parse_transcript_file(p)
    assert transcript.language == "en"
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello world"
    assert len(transcript.segments[0].words) == 2

    assert raw_audio is not None
    assert raw_audio["file_hash"] == "abc123hash"

def test_parse_json_missing_schema_version_defaults_to_1(tmp_path: Path):
    data = {
        "language": "en",
        "duration_seconds": 10.0,
        "segments": []
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    transcript, _ = parse_transcript_file(p)
    assert transcript.language == "en"

def test_parse_json_unsupported_schema_version(tmp_path: Path):
    data = {
        "schema_version": "99",
        "language": "en",
        "duration_seconds": 10.0,
        "segments": []
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersion):
        parse_transcript_file(p)

def test_parse_srt(tmp_path: Path):
    srt_content = '''1
00:00:01,000 --> 00:00:02,500
Hello

2
00:00:03,000 --> 00:00:04,500
World
'''
    p = tmp_path / "test.srt"
    p.write_text(srt_content, encoding="utf-8")

    transcript, raw_audio = parse_transcript_file(p)
    assert raw_audio is None
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == 1.0
    assert transcript.segments[0].end == 2.5
    assert transcript.segments[0].text == "Hello"
    assert transcript.duration_seconds == 4.5

def test_parse_txt(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("Hello World", encoding="utf-8")

    transcript, raw_audio = parse_transcript_file(p)
    assert raw_audio is None
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello World"
    assert transcript.duration_seconds == 0.0

def test_build_audio_metadata_hash_match(tmp_path: Path, monkeypatch):
    import audiobench.transcribe.transcription_result as result_mod
    monkeypatch.setattr(result_mod.AudioMetadata, "compute_file_hash", lambda p: "myhash")

    local_mp3 = tmp_path / "local.mp3"
    local_mp3.write_bytes(b"dummy")

    raw_audio = {
        "file_hash": "myhash",
        "file_name": "orig.mp3",
        "file_size_bytes": 100,
        "format": "mp3",
        "duration_seconds": 10.0,
        "sample_rate": 44100,
        "channels": 2
    }

    metadata = build_audio_metadata(local_mp3, raw_audio)
    assert metadata.file_hash == "myhash"
    assert metadata.file_name == "orig.mp3"
    assert metadata.duration_seconds == 10.0

def test_build_audio_metadata_hash_mismatch(tmp_path: Path, monkeypatch):
    import audiobench.transcribe.transcription_result as result_mod
    monkeypatch.setattr(result_mod.AudioMetadata, "compute_file_hash", lambda p: "myhash")

    local_mp3 = tmp_path / "local.mp3"
    local_mp3.write_bytes(b"dummy")

    raw_audio = {"file_hash": "wronghash"}

    with pytest.raises(HashMismatchError, match="Hash mismatch"):
        build_audio_metadata(local_mp3, raw_audio)


def test_parse_json_auto_heals_chunk_overlap_jumps(tmp_path: Path):
    data = {
        "schema_version": "1",
        "language": "en",
        "duration_seconds": 30.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world from chunk one.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0},
                    {"word": "world", "start": 1.0, "end": 2.0},
                    {"word": "from", "start": 2.0, "end": 3.0},
                    {"word": "chunk", "start": 3.0, "end": 4.0},
                    {"word": "one.", "start": 4.0, "end": 5.0},
                ],
            },
            {
                "id": 1,
                "start": 10.0,
                "end": 15.0,
                "text": "Shared overlap sentence.",
                "words": [
                    {"word": "Shared", "start": 10.0, "end": 11.0},
                    {"word": "overlap", "start": 11.0, "end": 13.0},
                    {"word": "sentence.", "start": 13.0, "end": 15.0},
                ],
            },
            # Backward jump to 10.0s!
            {
                "id": 2,
                "start": 10.0,
                "end": 15.0,
                "text": "Shared overlap sentence.",
                "words": [
                    {"word": "Shared", "start": 10.0, "end": 11.0},
                    {"word": "overlap", "start": 11.0, "end": 13.0},
                    {"word": "sentence.", "start": 13.0, "end": 15.0},
                ],
            },
            {
                "id": 3,
                "start": 16.0,
                "end": 20.0,
                "text": "Continuing in chunk two.",
                "words": [
                    {"word": "Continuing", "start": 16.0, "end": 17.0},
                    {"word": "in", "start": 17.0, "end": 18.0},
                    {"word": "chunk", "start": 18.0, "end": 19.0},
                    {"word": "two.", "start": 19.0, "end": 20.0},
                ],
            },
        ],
    }
    p = tmp_path / "corrupted_chunked.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    transcript, _ = parse_transcript_file(p)
    # Check that backward jump was excised and segments are monotonic
    assert len(transcript.segments) == 3
    for i in range(len(transcript.segments) - 1):
        assert transcript.segments[i + 1].start >= transcript.segments[i].end - 0.1
    assert "Shared overlap sentence" in transcript.text
    assert transcript.text.count("Shared overlap sentence") == 1

