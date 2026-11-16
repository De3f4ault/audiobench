"""Word-level Longest Common Subsequence (LCS) chunk stitcher.

Provides memory-safe, boundary-aware audio chunk stitching for long audiobooks
and speech streams. Slices overlapping chunks at exact matching word sequences
to eliminate:
1. Repetition loops and duplicated paragraphs at chunk boundaries.
2. Backward timestamp regressions (time paradoxes).
3. Ghost hallucinations and truncated phonemes at buffer slice points.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from audiobench.core.logger_factory import get_logger

logger = get_logger("transcribe.stitcher")


def clean_word(w: str) -> str:
    """Normalize word token by stripping punctuation and casing for robust alignment."""
    return re.sub(r"[^\w]", "", w.lower())


def stitch_chunk_segments(
    final_segments: list[dict[str, Any]],
    new_segments: list[dict[str, Any]],
    overlap_start: float,
) -> list[dict[str, Any]]:
    """Stitch two overlapping Whisper chunk segment streams using word-level LCS.

    Args:
        final_segments: Accumulated segments up to (overlap_start + overlap_sec).
                        All segment and word timestamps must be in absolute timeline seconds.
        new_segments:   New chunk segments starting around overlap_start.
                        All segment and word timestamps must be in absolute timeline seconds.
        overlap_start:  The start timestamp of the new chunk on the absolute timeline.

    Returns:
        A unified list of segments with duplicates excised and monotonic timestamps.
    """
    if not final_segments:
        return new_segments
    if not new_segments:
        return final_segments

    overlap_end = final_segments[-1]["end"]

    # 1. Extract all word tokens within the overlap window from both streams
    words1: list[dict[str, Any]] = []
    for s in final_segments:
        if s["end"] >= overlap_start - 2.0:
            for w in s.get("words", []):
                if w["start"] >= overlap_start - 2.0:
                    words1.append(w)

    words2: list[dict[str, Any]] = []
    for s in new_segments:
        if s["start"] <= overlap_end + 2.0:
            for w in s.get("words", []):
                if w["end"] <= overlap_end + 2.0:
                    words2.append(w)

    clean1 = [clean_word(w["word"]) for w in words1]
    clean2 = [clean_word(w["word"]) for w in words2]

    # 2. Find longest contiguous matching word sequence
    matcher = SequenceMatcher(None, clean1, clean2)
    match = matcher.find_longest_match(0, len(clean1), 0, len(clean2))

    # Fallback: if no clear shared words (e.g. ambient silence), cut at midpoint timestamp
    if match.size < 2:
        midpoint = overlap_start + max(0.0, overlap_end - overlap_start) / 2.0
        c1_kept = [s for s in final_segments if s["end"] <= midpoint]
        c2_kept = [s for s in new_segments if s["start"] > midpoint]
        return c1_kept + c2_kept

    # 3. Choose transition pivot in the match (preferring sentence punctuation)
    has_punct = False
    pivot_offset = match.size // 2
    for off in range(match.size):
        if words1[match.a + off]["word"].endswith((".", "?", "!", ";")):
            pivot_offset = off
            has_punct = True
            break

    m1_idx = match.a + pivot_offset
    m2_idx = match.b + pivot_offset

    if has_punct:
        # Cut cleanly immediately AFTER the punctuated sentence-ending word
        target_time = (words1[m1_idx]["end"] + words2[m2_idx]["end"]) / 2.0
    else:
        # Cut at the start of the mid-match word
        target_time = (words1[m1_idx]["start"] + words2[m2_idx]["start"]) / 2.0

    # 4. Trim tail of first stream up to target_time
    c1_final: list[dict[str, Any]] = []
    for s in final_segments:
        if s["end"] <= target_time + 0.05:
            c1_final.append(s)
        elif s["start"] < target_time - 0.05:
            kept_words = [w for w in s.get("words", []) if w["end"] <= target_time + 0.05]
            if kept_words:
                s_copy = dict(s)
                s_copy["words"] = kept_words
                s_copy["end"] = kept_words[-1]["end"]
                s_copy["text"] = " ".join(w["word"] for w in kept_words)
                s_copy["word_count"] = len(kept_words)
                s_copy["duration"] = round(s_copy["end"] - s_copy["start"], 3)
                c1_final.append(s_copy)
            break
        else:
            break

    # 5. Trim head of new stream starting from target_time
    c2_final: list[dict[str, Any]] = []
    for s in new_segments:
        if s["start"] >= target_time - 0.05:
            c2_final.append(s)
        elif s["end"] > target_time + 0.05:
            kept_words = [w for w in s.get("words", []) if w["start"] >= target_time - 0.05]
            if kept_words:
                s_copy = dict(s)
                s_copy["words"] = kept_words
                s_copy["start"] = kept_words[0]["start"]
                s_copy["text"] = " ".join(w["word"] for w in kept_words)
                s_copy["word_count"] = len(kept_words)
                s_copy["duration"] = round(s_copy["end"] - s_copy["start"], 3)
                c2_final.append(s_copy)

    return c1_final + c2_final


def has_chunk_overlap_jumps(segments: list[dict[str, Any]]) -> bool:
    """Check if a segment list contains backward time jumps from unstitched chunk boundaries."""
    if not segments or len(segments) < 2:
        return False
    for i in range(len(segments) - 1):
        if segments[i + 1].get("start", 0.0) < segments[i].get("end", 0.0) - 0.5:
            return True
    return False


def heal_stitched_transcript(data: dict[str, Any]) -> dict[str, Any]:
    """Heal an unstitched or corrupted transcript containing backward time jumps.

    Detects backward jumps, partitions segments into discrete chunks, stitches
    them sequentially using word-level LCS, recalculates word counts, and resets IDs.
    """
    raw_segs = data.get("segments", [])
    if not has_chunk_overlap_jumps(raw_segs):
        return data

    # Partition chunks at backward jump boundaries
    chunk_boundaries = [0]
    for i in range(len(raw_segs) - 1):
        if raw_segs[i + 1].get("start", 0.0) < raw_segs[i].get("end", 0.0) - 0.5:
            chunk_boundaries.append(i + 1)
    chunk_boundaries.append(len(raw_segs))

    raw_chunks = [
        raw_segs[chunk_boundaries[k] : chunk_boundaries[k + 1]]
        for k in range(len(chunk_boundaries) - 1)
    ]

    logger.info(
        "Healing transcript: detected %d overlapping chunks across %d jump points",
        len(raw_chunks),
        len(chunk_boundaries) - 2,
    )

    stitched = raw_chunks[0]
    for k in range(1, len(raw_chunks)):
        next_chunk = raw_chunks[k]
        overlap_start = next_chunk[0]["start"]
        stitched = stitch_chunk_segments(stitched, next_chunk, overlap_start)

    # Re-index segment IDs
    for idx, s in enumerate(stitched):
        s["id"] = idx

    full_text = " ".join(s["text"].strip() for s in stitched if s.get("text"))

    healed_data = dict(data)
    healed_data["segments"] = stitched
    healed_data["segment_count"] = len(stitched)
    healed_data["text"] = full_text
    healed_data["word_count"] = len(full_text.split())
    healed_data["stitched"] = True

    logger.info(
        "Healed transcript: reduced %d segments -> %d segments (-%d duplicates)",
        len(raw_segs),
        len(stitched),
        len(raw_segs) - len(stitched),
    )

    return healed_data
