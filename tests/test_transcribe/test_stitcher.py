"""Tests for word-level LCS chunk stitcher and auto-healer."""

import pytest

from audiobench.transcribe.stitcher import (
    clean_word,
    has_chunk_overlap_jumps,
    heal_stitched_transcript,
    stitch_chunk_segments,
)


def test_clean_word_normalization():
    assert clean_word("Hello,") == "hello"
    assert clean_word("World!...") == "world"
    assert clean_word("best-of-the-best") == "bestofthebest"
    assert clean_word("  \"quoted\"  ") == "quoted"


def test_has_chunk_overlap_jumps():
    # Clean monotonic segments
    clean_segs = [
        {"start": 0.0, "end": 5.0, "text": "First segment"},
        {"start": 5.0, "end": 10.0, "text": "Second segment"},
        {"start": 10.5, "end": 15.0, "text": "Third segment"},
    ]
    assert has_chunk_overlap_jumps(clean_segs) is False

    # Corrupted segments with backward time jump (1229s -> 1200s)
    corrupted_segs = [
        {"start": 1220.0, "end": 1229.5, "text": "End of chunk 1"},
        {"start": 1200.0, "end": 1206.0, "text": "Start of chunk 2 (jump!)"},
    ]
    assert has_chunk_overlap_jumps(corrupted_segs) is True

    # Empty / single segment
    assert has_chunk_overlap_jumps([]) is False
    assert has_chunk_overlap_jumps([{"start": 0.0, "end": 1.0}]) is False


def test_stitch_chunk_segments_exact_overlap():
    # Simulating Chunk 1 tail [1190s -> 1230s]
    c1_segs = [
        {
            "id": 0,
            "start": 1190.0,
            "end": 1195.0,
            "text": "Before overlap section.",
            "words": [
                {"word": "Before", "start": 1190.0, "end": 1191.0},
                {"word": "overlap", "start": 1191.0, "end": 1193.0},
                {"word": "section.", "start": 1193.0, "end": 1195.0},
            ],
        },
        {
            "id": 1,
            "start": 1200.0,
            "end": 1215.0,
            "text": "This is shared audio in both chunks.",
            "words": [
                {"word": "This", "start": 1200.0, "end": 1202.0},
                {"word": "is", "start": 1202.0, "end": 1204.0},
                {"word": "shared", "start": 1204.0, "end": 1207.0},
                {"word": "audio", "start": 1207.0, "end": 1209.0},
                {"word": "in", "start": 1209.0, "end": 1210.0},
                {"word": "both", "start": 1210.0, "end": 1212.0},
                {"word": "chunks.", "start": 1212.0, "end": 1215.0},
            ],
        },
        {
            "id": 2,
            "start": 1228.0,
            "end": 1229.5,
            "text": "Hallucinated tail The Crap",
            "words": [
                {"word": "Hallucinated", "start": 1228.0, "end": 1228.5},
                {"word": "tail", "start": 1228.5, "end": 1229.0},
                {"word": "The", "start": 1229.0, "end": 1229.2},
                {"word": "Crap", "start": 1229.2, "end": 1229.5},
            ],
        },
    ]

    # Simulating Chunk 2 head [1200s -> 1250s]
    c2_segs = [
        {
            "id": 0,
            "start": 1200.0,
            "end": 1215.0,
            "text": "This is shared audio in both chunks.",
            "words": [
                {"word": "This", "start": 1200.0, "end": 1202.0},
                {"word": "is", "start": 1202.0, "end": 1204.0},
                {"word": "shared", "start": 1204.0, "end": 1207.0},
                {"word": "audio", "start": 1207.0, "end": 1209.0},
                {"word": "in", "start": 1209.0, "end": 1210.0},
                {"word": "both", "start": 1210.0, "end": 1212.0},
                {"word": "chunks.", "start": 1212.0, "end": 1215.0},
            ],
        },
        {
            "id": 1,
            "start": 1216.0,
            "end": 1225.0,
            "text": "Unique continuation in chunk two.",
            "words": [
                {"word": "Unique", "start": 1216.0, "end": 1218.0},
                {"word": "continuation", "start": 1218.0, "end": 1221.0},
                {"word": "in", "start": 1221.0, "end": 1222.0},
                {"word": "chunk", "start": 1222.0, "end": 1223.0},
                {"word": "two.", "start": 1223.0, "end": 1225.0},
            ],
        },
    ]

    stitched = stitch_chunk_segments(c1_segs, c2_segs, overlap_start=1200.0)

    # 1. Monotonicity check
    for i in range(len(stitched) - 1):
        assert stitched[i + 1]["start"] >= stitched[i]["end"] - 0.1

    # 2. Check that "This is shared audio in both chunks" is NOT duplicated
    full_text = " ".join(s["text"] for s in stitched)
    assert full_text.count("shared audio in both chunks") == 1

    # 3. Check that the hallucinated tail "The Crap" at 1229s was dropped
    assert "The Crap" not in full_text

    # 4. Check that both before and after text are preserved
    assert "Before overlap section" in full_text
    assert "Unique continuation in chunk two" in full_text


def test_heal_stitched_transcript():
    corrupted_data = {
        "schema_version": "1",
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
                "start": 20.0,
                "end": 25.0,
                "text": "Overlap sentence shared by both.",
                "words": [
                    {"word": "Overlap", "start": 20.0, "end": 21.0},
                    {"word": "sentence", "start": 21.0, "end": 22.0},
                    {"word": "shared", "start": 22.0, "end": 23.0},
                    {"word": "by", "start": 23.0, "end": 24.0},
                    {"word": "both.", "start": 24.0, "end": 25.0},
                ],
            },
            # Backward jump to 20.0s from chunk 2!
            {
                "id": 2,
                "start": 20.0,
                "end": 25.0,
                "text": "Overlap sentence shared by both.",
                "words": [
                    {"word": "Overlap", "start": 20.0, "end": 21.0},
                    {"word": "sentence", "start": 21.0, "end": 22.0},
                    {"word": "shared", "start": 22.0, "end": 23.0},
                    {"word": "by", "start": 23.0, "end": 24.0},
                    {"word": "both.", "start": 24.0, "end": 25.0},
                ],
            },
            {
                "id": 3,
                "start": 26.0,
                "end": 30.0,
                "text": "Continuing into chunk two.",
                "words": [
                    {"word": "Continuing", "start": 26.0, "end": 27.0},
                    {"word": "into", "start": 27.0, "end": 28.0},
                    {"word": "chunk", "start": 28.0, "end": 29.0},
                    {"word": "two.", "start": 29.0, "end": 30.0},
                ],
            },
        ],
        "word_count": 18,
        "segment_count": 4,
        "text": "Hello world from chunk one. Overlap sentence shared by both. Overlap sentence shared by both. Continuing into chunk two.",
    }

    healed = heal_stitched_transcript(corrupted_data)

    assert has_chunk_overlap_jumps(healed["segments"]) is False
    assert healed["segment_count"] == 3
    assert healed.get("stitched") is True


def test_pipeline_merge_transcripts_overlapping():
    from audiobench.transcribe.transcriber import TranscriptionPipeline
    from audiobench.transcribe.transcription_result import Transcript, Segment, Word

    pipeline = TranscriptionPipeline()

    t1 = Transcript(
        language="en",
        duration_seconds=25.0,
        segments=[
            Segment(
                id=0,
                text="Chapter 1 start.",
                start=0.0,
                end=10.0,
                words=[
                    Word(word="Chapter", start=0.0, end=1.0),
                    Word(word="1", start=1.0, end=2.0),
                    Word(word="start.", start=2.0, end=3.0),
                ],
            ),
            Segment(
                id=1,
                text="Overlap at the border.",
                start=20.0,
                end=25.0,
                words=[
                    Word(word="Overlap", start=20.0, end=21.0),
                    Word(word="at", start=21.0, end=22.0),
                    Word(word="the", start=22.0, end=23.0),
                    Word(word="border.", start=23.0, end=25.0),
                ],
            ),
        ],
    )

    t2 = Transcript(
        language="en",
        duration_seconds=50.0,
        segments=[
            Segment(
                id=0,
                text="Overlap at the border.",
                start=20.0,
                end=25.0,
                words=[
                    Word(word="Overlap", start=20.0, end=21.0),
                    Word(word="at", start=21.0, end=22.0),
                    Word(word="the", start=22.0, end=23.0),
                    Word(word="border.", start=23.0, end=25.0),
                ],
            ),
            Segment(
                id=1,
                text="Chapter 2 continuation.",
                start=26.0,
                end=35.0,
                words=[
                    Word(word="Chapter", start=26.0, end=27.0),
                    Word(word="2", start=27.0, end=28.0),
                    Word(word="continuation.", start=28.0, end=30.0),
                ],
            ),
        ],
    )

    merged = pipeline._merge_transcripts([t1, t2])
    assert len(merged.segments) == 3
    assert merged.text.count("Overlap at the border") == 1
    assert "Chapter 1 start" in merged.text
    assert "Chapter 2 continuation" in merged.text

