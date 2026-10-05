from unittest.mock import MagicMock, patch

from audiobench.memory.chunking import (
    AdvancedSemanticChunker,
    _clean_text,
    content_aware_router,
    speaker_turn_splitter,
)


def test_clean_text_whisper_artifacts() -> None:
    """Verify _clean_text removes hallucinated tags and null bytes."""
    raw = "Hello \x00world [BLANK_AUDIO] [Music] (silence) (Applause) the the the end."
    cleaned = _clean_text(raw)

    assert "\x00" not in cleaned
    assert "[BLANK_AUDIO]" not in cleaned
    assert "[Music]" not in cleaned
    assert "(silence)" not in cleaned
    assert "(Applause)" not in cleaned
    assert "the the the" not in cleaned
    assert "the end" in cleaned


def test_clean_text_repetition_loops() -> None:
    """Verify trailing repetition loops are collapsed."""
    raw = (
        "This is a normal sentence. This is another normal sentence. "
        + "I am stuck in a loop. I am stuck in a loop. I am stuck in a loop."
    )

    cleaned = _clean_text(raw)
    # The clean_text function should collapse the last 3 identical sentences down to one
    assert cleaned.count("I am stuck in a loop.") == 1


@patch("audiobench.memory.chunking.get_boundary_embedder")
def test_advanced_semantic_chunker(mock_get_embedder: MagicMock) -> None:
    """Verify chunker logic with mocked distance thresholding."""
    mock_embedder = MagicMock()
    # Mock encode to return 3 dummy vectors
    import numpy as np

    mock_embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),  # Orthogonal to cause a split
    ]
    mock_get_embedder.return_value = mock_embedder

    # Give it 9 sentences, so it forms 3 groups of 3 sentences
    text = "S1. S2. S3. S4. S5. S6. S7. S8. S9."
    chunker = AdvancedSemanticChunker(
        breakpoint_percentile=50.0, max_tokens=350, sentence_group_size=3
    )

    chunks = chunker.chunk(text)

    # Because group 1 and 2 are identical [1.0, 0.0], distance is 0
    # Group 2 and 3 are orthogonal, distance is 1
    # Threshold (percentile 50) of [0.0, 1.0] is 0.5.
    # Therefore, distance > 0.5 happens between group 2 and 3.
    # It should split into two chunks: one with 6 sentences, one with 3.

    assert len(chunks) == 2
    assert "S1" in chunks[0].content
    assert "S6" in chunks[0].content
    assert "S7" in chunks[1].content
    assert "S9" in chunks[1].content


def test_speaker_turn_splitter() -> None:
    """Verify diarized segments are grouped into blocks per speaker."""
    segments = [
        {"speaker": "SPEAKER_00", "text": "Hello, how are you?"},
        {"speaker": "SPEAKER_00", "text": "I hope you are well."},
        {"speaker": "SPEAKER_01", "text": "I am fine, thanks."},
        {"speaker": "SPEAKER_00", "text": "Great."},
    ]

    blocks = speaker_turn_splitter(segments)
    assert len(blocks) == 3
    assert blocks[0].speaker == "SPEAKER_00"
    assert blocks[0].text == "Hello, how are you? I hope you are well."

    assert blocks[1].speaker == "SPEAKER_01"
    assert blocks[1].text == "I am fine, thanks."

    assert blocks[2].speaker == "SPEAKER_00"
    assert blocks[2].text == "Great."


@patch("audiobench.memory.chunking.AdvancedSemanticChunker.chunk")
def test_content_aware_router_short_text(mock_chunk: MagicMock) -> None:
    """Short text (<600 chars) should bypass the chunker."""
    text = "Short sentence."

    with patch("audiobench.memory.chunking.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(chunk_short_threshold=600)

        chunks = content_aware_router(text)
        assert len(chunks) == 1
        assert chunks[0].content == "Short sentence."
        mock_chunk.assert_not_called()


@patch("audiobench.memory.chunking.AdvancedSemanticChunker.chunk")
def test_content_aware_router_diarized(mock_chunk: MagicMock) -> None:
    """Diarized segments should be split by speaker, then chunked."""
    text = "Not used directly when diarization is provided."
    segments = [
        {"speaker": "SPK1", "text": "Sentence one. S2. S3. S4. S5."},
        {"speaker": "SPK2", "text": "Sentence two."},
    ]

    with patch("audiobench.memory.chunking.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(chunk_short_threshold=10)
        # Mock chunk to just return dummy chunks
        mock_chunk.return_value = [MagicMock()]

        chunks = content_aware_router(text, diarized_segments=segments)
        # SPK1 has 5 sentences, SPK2 has 1 sentence. Both are > 10 chars so they get chunked.
        assert mock_chunk.call_count == 2


def test_attribute_chunks_to_segments_monotonic_alignment() -> None:
    """Verify attribute_chunks_to_segments accurately links chunks to physical segments."""
    from audiobench.memory.chunking import Chunk, attribute_chunks_to_segments

    raw_text = "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the forest."
    segments = [
        {"id": 101, "text": "The quick brown fox jumps", "start_time": 0.0, "end_time": 2.5},
        {"id": 102, "text": "over the lazy dog.", "start_time": 2.5, "end_time": 4.8},
        {"id": 103, "text": "It was a sunny afternoon", "start_time": 5.0, "end_time": 7.2},
        {"id": 104, "text": "in the forest.", "start_time": 7.2, "end_time": 9.0},
    ]

    chunk1 = Chunk(content="The quick brown fox jumps over the lazy dog.", uuid="c1", tier=3)
    chunk2 = Chunk(content="It was a sunny afternoon in the forest.", uuid="c2", tier=3)

    attributed = attribute_chunks_to_segments([chunk1, chunk2], raw_text, segments)
    assert len(attributed) == 2

    assert attributed[0].segment_ids == [101, 102]
    assert attributed[0].start_time == 0.0
    assert attributed[0].end_time == 4.8

    assert attributed[1].segment_ids == [103, 104]
    assert attributed[1].start_time == 5.0
    assert attributed[1].end_time == 9.0


def test_content_aware_router_attaches_segments() -> None:
    """Verify content_aware_router populates segment_ids and timestamps when segments are passed."""
    raw_text = "Short text under threshold."
    segments = [
        {"id": 201, "text": "Short text", "start_time": 1.0, "end_time": 2.0},
        {"id": 202, "text": "under threshold.", "start_time": 2.0, "end_time": 3.5},
    ]

    chunks = content_aware_router(raw_text, segments=segments)
    assert len(chunks) == 1
    assert chunks[0].segment_ids == [201, 202]
    assert chunks[0].start_time == 1.0
    assert chunks[0].end_time == 3.5

