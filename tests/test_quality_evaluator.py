"""Unit tests for transcript quality evaluation and content familiarity engine."""

from audiobench.transcribe.quality_evaluator import (
    ComparisonResult,
    FamiliarityScore,
    QualityScore,
    compare_transcripts,
    evaluate_familiarity,
    evaluate_transcript_quality,
)
from audiobench.transcribe.transcription_result import Segment, Transcript, Word


def test_evaluate_quality_perfect_whisper():
    # 10 segments spanning 0s to 50s, 100 timestamped words, ~120 WPM
    segments = []
    word_id = 0
    for i in range(10):
        start = i * 5.0
        end = start + 4.5
        words = []
        for w in range(10):
            w_start = start + w * 0.4
            w_end = w_start + 0.35
            words.append(Word(word=f"word{word_id}", start=w_start, end=w_end, probability=0.95))
            word_id += 1
        seg_text = " ".join(f"word{j}" for j in range(word_id - 10, word_id))
        segments.append(Segment(id=i, text=seg_text, start=start, end=end, words=words))

    score = evaluate_transcript_quality(segments, total_duration=50.0, engine="whisper-large-v3")

    assert score.total_score >= 85.0
    assert score.grade == "PRODUCTION_GRADE"
    assert score.coverage_ratio >= 0.98
    assert score.words_with_timestamps_ratio == 1.0
    assert score.zero_duration_segments_count == 0
    assert score.stretched_segments_count == 0
    assert score.degeneration_penalty == 0.0


def test_evaluate_quality_gemini_defects():
    # Truncated container (ends at 40s instead of 100s, 0.4 coverage)
    # No word timestamps
    # Has zero duration segments and stretched fragments
    segments = [
        # Zero-duration collapsed segment
        Segment(id=0, text="first segment", start=0.0, end=0.0, words=[]),
        Segment(id=1, text="second collapsed", start=0.0, end=0.0, words=[]),
        # Normal segment
        Segment(id=2, text="some normal spoken words here", start=1.0, end=5.0, words=[]),
        # Stretched fragment (< 0.3 words/sec over >= 10s)
        Segment(id=3, text="...", start=5.0, end=35.0, words=[]),
        # Truncated ending: stops at 40s
        Segment(id=4, text="the end before cutoff", start=35.0, end=40.0, words=[]),
    ]

    score = evaluate_transcript_quality(segments, total_duration=100.0, engine="gemini-2.5-flash")

    assert score.total_score < 65.0
    assert score.grade in ("SUBPAR", "DEFECTIVE")
    assert score.words_with_timestamps_ratio == 0.0
    assert score.zero_duration_segments_count == 2
    assert score.stretched_segments_count == 1
    assert score.health_score < 25.0


def test_evaluate_quality_empty_segments():
    score = evaluate_transcript_quality([], total_duration=60.0)
    assert score.total_score == 0.0
    assert score.grade == "DEFECTIVE"
    assert score.segment_count == 0
    assert score.word_count == 0


def test_evaluate_quality_repetition_loops():
    # 4 consecutive identical segments
    segments = [
        Segment(id=0, text="Thank you for watching.", start=0.0, end=2.0),
        Segment(id=1, text="Thank you for watching.", start=2.0, end=4.0),
        Segment(id=2, text="Thank you for watching.", start=4.0, end=6.0),
        Segment(id=3, text="Thank you for watching.", start=6.0, end=8.0),
    ]
    score = evaluate_transcript_quality(segments, total_duration=10.0)
    assert score.repetition_loops_count >= 1
    assert score.degeneration_penalty >= 5.0


def test_evaluate_quality_subtitle_leak():
    segments = [
        Segment(id=0, text="Welcome to the presentation.", start=0.0, end=3.0),
        Segment(id=1, text="Subtitles by the Amara.org community", start=3.0, end=6.0),
    ]
    score = evaluate_transcript_quality(segments, total_duration=10.0)
    assert score.degeneration_penalty >= 5.0


def test_evaluate_familiarity_identical():
    text = "The greatest salesman in the world was Hafid, master of the market in Damascus."
    fam = evaluate_familiarity(text, text)
    assert fam.score == 100.0
    assert fam.classification == "DIRECT_MATCH"
    assert fam.unigram_jaccard == 1.0
    assert fam.bigram_jaccard == 1.0


def test_evaluate_familiarity_truncated():
    full_text = (
        "Chapter one of the greatest salesman in the world. "
        "Hafid stood before the bronze mirror and examined his reflection. "
        "Only the eyes remained young and keen. "
        "He turned and walked slowly across the marble floor to the great window. "
        "The city of Damascus was bustling with merchants and caravans. "
        "He called for Erasmus, his trusted bookkeeper of many decades. "
        "Erasmus, I wish to dispose of all my wealth and transfer my scrolls."
    )
    # Truncate the ending 15%
    truncated_text = (
        "Chapter one of the greatest salesman in the world. "
        "Hafid stood before the bronze mirror and examined his reflection. "
        "Only the eyes remained young and keen. "
        "He turned and walked slowly across the marble floor to the great window. "
        "The city of Damascus was bustling with merchants and caravans. "
        "He called for Erasmus, his trusted bookkeeper of many decades."
    )
    fam = evaluate_familiarity(full_text, truncated_text)
    assert fam.classification == "DIRECT_MATCH"
    assert fam.score >= 75.0
    assert fam.containment >= 0.90


def test_evaluate_familiarity_completely_different():
    text_a = "Moby Dick by Herman Melville. Call me Ishmael. Some years ago never mind how long precisely."
    text_b = "A Tale of Two Cities by Charles Dickens. It was the best of times it was the worst of times."
    fam = evaluate_familiarity(text_a, text_b)
    assert fam.classification == "MISMATCH"
    assert fam.score < 40.0


def test_compare_transcripts_upgrade_recommended():
    # Existing: defective Gemini without timestamps, collapsed segments
    class MockRecord:
        id = 1506
        engine = "gemini-2.5-flash"
        duration_seconds = 100.0
        full_text = "Hafid examined his reflection. Erasmus came in."
        segments = [
            Segment(id=0, text="Hafid examined his reflection.", start=0.0, end=0.0),
            Segment(id=1, text="Erasmus came in.", start=1.0, end=5.0),
        ]

    # Incoming: high quality Whisper with word timestamps
    inc_words = [
        Word(word="Hafid", start=0.1, end=0.6, probability=0.98),
        Word(word="examined", start=0.7, end=1.2, probability=0.97),
        Word(word="his", start=1.3, end=1.5, probability=0.99),
        Word(word="reflection", start=1.6, end=2.2, probability=0.99),
        Word(word="Erasmus", start=2.5, end=3.0, probability=0.96),
        Word(word="came", start=3.1, end=3.4, probability=0.99),
        Word(word="in", start=3.5, end=3.8, probability=0.98),
    ]
    incoming = Transcript(
        text="Hafid examined his reflection. Erasmus came in.",
        engine="whisper-large-v3",
        duration_seconds=100.0,
        segments=[
            Segment(id=0, text="Hafid examined his reflection.", start=0.1, end=2.2, words=inc_words[:4]),
            Segment(id=1, text="Erasmus came in.", start=2.5, end=3.8, words=inc_words[4:]),
        ]
    )

    res = compare_transcripts(MockRecord(), incoming, audio_duration=100.0)
    assert res.recommendation == "UPGRADE_RECOMMENDED"
    assert res.delta_tqs >= 15.0


def test_compare_transcripts_new_import():
    incoming = Transcript(
        text="New audiobook content.",
        engine="whisper-large-v3",
        duration_seconds=60.0,
        segments=[Segment(id=0, text="New audiobook content.", start=1.0, end=3.0)]
    )
    res = compare_transcripts(None, incoming, audio_duration=60.0)
    assert res.recommendation == "NEW_IMPORT"
    assert res.existing_id is None


def test_evaluate_quality_mature_reading_paragraphs():
    # 4 mature reading paragraphs, 25s each, spanning 100s
    segments = [
        Segment(id=0, text="This is the first mature paragraph with several sentences that flow naturally together and provide rich context.", start=0.0, end=25.0),
        Segment(id=1, text="Continuing onto the second paragraph where the narrative explores deeper themes and nuanced character development.", start=25.0, end=50.0),
        Segment(id=2, text="The third paragraph explains key lessons and actionable principles clearly and concisely without interruption.", start=50.0, end=75.0),
        Segment(id=3, text="Finally the fourth paragraph summarizes the chapter and concludes with memorable insights for the reader.", start=75.0, end=100.0),
    ]
    score = evaluate_transcript_quality(segments, total_duration=100.0)
    # Granularity segment duration must get full 5.0 marks for 25.0s avg duration
    assert score.avg_segment_duration == 25.0
    assert score.granularity_score >= 5.0
    assert score.micro_segments_count == 0
    assert score.micro_segment_ratio == 0.0


def test_evaluate_quality_micro_fragmentation_penalty():
    # 20 micro-segments, 1.0s each with 2 words (e.g. prompt-saturated Whisper)
    segments = [
        Segment(id=i, text=f"word {i}", start=i * 1.0, end=(i + 1) * 1.0)
        for i in range(20)
    ]
    score = evaluate_transcript_quality(segments, total_duration=20.0)
    assert score.micro_segments_count == 20
    assert score.micro_segment_ratio == 1.0
    # Degeneration penalty must include maximum micro-fragmentation penalty
    assert score.degeneration_penalty >= 20.0
    assert score.grade in ("SUBPAR", "DEFECTIVE")


def test_evaluate_windowed_fidelity_clean_agreement():
    # 600 seconds (2 windows of 300s)
    # Existing: 100 segments of 6s each
    # Incoming: 20 segments of 30s each with identical wording
    base_text = "The quick brown fox jumps over the lazy dog and explores the peaceful countryside every morning. "
    ex_segs = [
        Segment(id=i, text=f"Segment {i}: {base_text}", start=i * 6.0, end=(i + 1) * 6.0)
        for i in range(100)
    ]
    # Incoming grouped into 30s paragraphs
    inc_segs = []
    for p in range(20):
        p_text = " ".join(f"Segment {p*5 + j}: {base_text}" for j in range(5))
        inc_segs.append(Segment(id=p, text=p_text, start=p * 30.0, end=(p + 1) * 30.0))

    from audiobench.transcribe.quality_evaluator import evaluate_windowed_fidelity
    fidelity = evaluate_windowed_fidelity(ex_segs, inc_segs, total_duration=600.0, window_size_seconds=300.0)

    assert fidelity.window_count == 2
    assert fidelity.flagged_windows_count == 0
    assert fidelity.is_faithful is True


def test_evaluate_windowed_fidelity_synthetic_omission_veto():
    # 600s audio. Existing has text throughout.
    # Incoming has word timestamps and higher TQS, but drops 90 seconds in window 0 (from 60s to 150s).
    words_per_seg = 30
    ex_segs = [
        Segment(id=i, text=f"Continuous audiobook narration passage number {i} with several important sentences.",
                start=i * 10.0, end=(i + 1) * 10.0)
        for i in range(60)
    ]

    class ExistingMock:
        id = 101
        engine = "gemini-2.5-flash"
        duration_seconds = 600.0
        full_text = " ".join(s.text for s in ex_segs)
        segments = ex_segs

    # Incoming with timestamps on retained segments, but omitting 60s to 150s (9 segments omitted = 90 seconds)
    inc_segs = []
    for s in ex_segs:
        if 60.0 <= s.start < 150.0:
            continue  # Omit 90s
        inc_segs.append(Segment(id=s.id, text=s.text, start=s.start, end=s.end, words=[]))

    inc_tx = Transcript(
        text=" ".join(s.text for s in inc_segs),
        engine="whisper-large-v3",
        duration_seconds=600.0,
        segments=inc_segs,
    )

    comp = compare_transcripts(ExistingMock(), inc_tx, audio_duration=600.0)

    assert comp.fidelity is not None
    assert comp.fidelity.flagged_windows_count == 1
    assert comp.fidelity.is_faithful is False
    assert comp.recommendation == "POTENTIAL_OMISSION_DETECTED"
    assert "Upgrade blocked: localized omissions/defects detected in 1 window(s)" in comp.summary
    assert comp.fidelity.flagged_details[0]["window_index"] == 0


def test_evaluate_windowed_fidelity_hallucination_loop_veto():
    # 900s audio (3 windows of 300s).
    # Existing has clean narration.
    # Incoming gets stuck in a hallucination loop in window 1 (300s to 600s), repeating "'de" x 100.
    ex_segs = [
        Segment(id=i, text=f"Narration sentence {i} discussing ancient philosophy and historic events across kingdoms.",
                start=i * 10.0, end=(i + 1) * 10.0)
        for i in range(90)
    ]

    class ExistingMock:
        id = 102
        engine = "whisper"
        duration_seconds = 900.0
        full_text = " ".join(s.text for s in ex_segs)
        segments = ex_segs

    inc_segs = []
    for s in ex_segs:
        if 300.0 <= s.start < 600.0:
            # Replaced with hallucination loop
            inc_segs.append(Segment(id=s.id, text="'de 'de 'de", start=s.start, end=s.end))
        else:
            inc_segs.append(Segment(id=s.id, text=s.text, start=s.start, end=s.end))

    inc_tx = Transcript(
        text=" ".join(s.text for s in inc_segs),
        engine="whisper",
        duration_seconds=900.0,
        segments=inc_segs,
    )

    comp = compare_transcripts(ExistingMock(), inc_tx, audio_duration=900.0)

    assert comp.fidelity is not None
    assert comp.fidelity.flagged_windows_count >= 1
    assert comp.fidelity.is_faithful is False
    assert comp.recommendation == "POTENTIAL_OMISSION_DETECTED"


