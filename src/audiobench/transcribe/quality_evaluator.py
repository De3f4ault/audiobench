"""Transcript Quality Evaluator & Content Familiarity Engine.

Provides objective, mathematical evaluation of transcription quality down to
millisecond precision and compares incoming transcripts with existing database records.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class QualityScore:
    """Quantitative quality assessment of a transcript down to the millisecond."""
    total_score: float             # TQS clamped to [0.0, 100.0]
    coverage_score: float          # S_cov in [0.0, 30.0]
    granularity_score: float       # S_gran in [0.0, 25.0]
    health_score: float            # S_health in [0.0, 25.0]
    confidence_score: float        # S_conf in [0.0, 20.0]
    degeneration_penalty: float    # P_degen >= 0.0

    # Detailed metrics
    duration_seconds: float
    transcribed_span: float
    coverage_ratio: float
    word_count: int
    segment_count: int
    wpm: float
    words_with_timestamps_ratio: float
    avg_segment_duration: float
    zero_duration_segments_count: int
    stretched_segments_count: int
    inversion_count: int
    avg_word_probability: float
    repetition_loops_count: int
    grade: str                     # "PRODUCTION_GRADE", "ACCEPTABLE", "SUBPAR", "DEFECTIVE"
    micro_segments_count: int = 0
    micro_segment_ratio: float = 0.0


@dataclass
class FamiliarityScore:
    """Content similarity and variant equivalence between two transcript texts."""
    score: float                   # F_sim in [0.0, 100.0]%
    unigram_jaccard: float         # [0.0, 1.0]
    bigram_jaccard: float          # [0.0, 1.0]
    containment: float             # [0.0, 1.0]
    length_ratio: float            # min(len_a, len_b) / max(len_a, len_b)
    classification: str            # "DIRECT_MATCH", "PARTIAL_MATCH", "MISMATCH"


@dataclass
class WindowFidelityResult:
    """Timeline-aligned windowed fidelity analysis between two transcripts."""
    window_count: int
    flagged_windows_count: int
    flagged_ratio: float
    flagged_details: list[dict]
    is_faithful: bool
    summary: str


@dataclass
class ComparisonResult:
    """Side-by-side comparative analysis between existing DB record and incoming candidate."""
    existing_id: int | None
    existing_tqs: QualityScore | None
    incoming_tqs: QualityScore
    familiarity: FamiliarityScore
    delta_tqs: float
    recommendation: str            # "UPGRADE_RECOMMENDED", "KEEP_EXISTING", "SIMILAR_QUALITY", "MISMATCH", "NEW_IMPORT", "POTENTIAL_OMISSION_DETECTED"
    summary: str
    fidelity: WindowFidelityResult | None = None


SUBTITLE_LEAK_PATTERNS = [
    re.compile(r"subtitles?\s+by\b", re.IGNORECASE),
    re.compile(r"closed\s+captions?\s+by\b", re.IGNORECASE),
    re.compile(r"downloaded\s+from\s+amara\.org", re.IGNORECASE),
    re.compile(r"transcribed\s+by\s+https?://", re.IGNORECASE),
    re.compile(r"please\s+subscribe\s+to\s+my\s+channel", re.IGNORECASE),
    re.compile(r"like\s+and\s+subscribe", re.IGNORECASE),
]


def evaluate_transcript_quality(
    segments: Iterable[Any],
    total_duration: float | None = None,
    engine: str | None = None,
    full_text: str | None = None,
) -> QualityScore:
    """
    Evaluate transcript quality with millisecond precision across 5 core pillars:
    Coverage (0-30), Granularity (0-25), Timeline Health (0-25), Confidence (0-20),
    and Degeneration Penalty.

    Implemented as a single-pass streaming algorithm with O(1) auxiliary memory.
    """
    t_start = math.inf
    t_end = 0.0
    n_seg = 0
    total_words_count = 0
    timestamped_words_count = 0
    prob_sum = 0.0
    prob_count = 0

    zero_duration_count = 0
    stretched_count = 0
    inversion_count = 0
    micro_segments_count = 0

    # Streaming repetition loop detector
    repetition_loops = 0
    current_repeat = 1
    prev_norm = ""

    # Streaming subtitle leak detector
    leak_detected = False
    if full_text and any(pat.search(full_text) for pat in SUBTITLE_LEAK_PATTERNS):
        leak_detected = True

    prev_start: float | None = None

    if segments:
        for s in segments:
            n_seg += 1
            start = float(_get_attr(s, "start", _get_attr(s, "start_time", 0.0)))
            end = float(_get_attr(s, "end", _get_attr(s, "end_time", 0.0)))
            text = str(_get_attr(s, "text", "")).strip()

            if start < t_start:
                t_start = start
            if end > t_end:
                t_end = end

            dur = max(0.0, end - start)
            words_in_seg = len(text.split()) if text else 0
            total_words_count += words_in_seg

            # Timeline Health: Zero-duration collapsed segment
            if dur <= 0.001:
                zero_duration_count += 1

            # Timeline Health: Stretched ghost fragment (<0.3 words/sec over >=10.0s)
            if dur >= 10.0 and (words_in_seg / dur) < 0.3:
                stretched_count += 1

            # Timeline Health: Non-monotonic backward jump
            if prev_start is not None and start < (prev_start - 0.5):
                inversion_count += 1
            prev_start = start

            # Granularity / Fragmentation: Micro-segments (<1.2s or <=2 words under 2.5s)
            if dur < 1.2 or (words_in_seg <= 2 and dur < 2.5):
                micro_segments_count += 1

            # Word-level granularity
            words = _get_attr(s, "words", None)
            if words and isinstance(words, (list, tuple)):
                for w in words:
                    w_start = _get_attr(w, "start", None)
                    w_end = _get_attr(w, "end", None)
                    if w_start is not None and w_end is not None and float(w_end) >= float(w_start):
                        timestamped_words_count += 1

                    p = _get_attr(w, "probability", None)
                    if p is not None and float(p) > 0.0:
                        prob_sum += float(p)
                        prob_count += 1

            # Streaming repetition loop check
            norm = re.sub(r"\W+", " ", text.lower()).strip()
            if norm:
                if norm == prev_norm:
                    current_repeat += 1
                    if current_repeat >= 3:
                        repetition_loops += 1
                else:
                    current_repeat = 1
                    prev_norm = norm

            # Streaming subtitle leak check
            if not leak_detected and any(pat.search(text) for pat in SUBTITLE_LEAK_PATTERNS):
                leak_detected = True

    if n_seg == 0:
        words = len(full_text.split()) if full_text else 0
        return QualityScore(
            total_score=0.0,
            coverage_score=0.0,
            granularity_score=0.0,
            health_score=0.0,
            confidence_score=0.0,
            degeneration_penalty=0.0,
            duration_seconds=float(total_duration or 0.0),
            transcribed_span=0.0,
            coverage_ratio=0.0,
            word_count=words,
            segment_count=0,
            wpm=0.0,
            words_with_timestamps_ratio=0.0,
            avg_segment_duration=0.0,
            zero_duration_segments_count=0,
            stretched_segments_count=0,
            inversion_count=0,
            avg_word_probability=0.0,
            repetition_loops_count=0,
            grade="DEFECTIVE",
        )

    # 1. Timeline boundaries down to millisecond precision
    transcribed_span = max(0.0, t_end - (t_start if t_start != math.inf else 0.0))
    effective_duration = float(total_duration) if (total_duration and total_duration > 0.0) else max(transcribed_span, t_end, 1.0)
    coverage_ratio = min(1.0, max(0.0, transcribed_span / effective_duration))

    # If full_text provided, override total_words_count if it exceeds sum of segments
    if full_text:
        ft_words = len(full_text.split())
        if ft_words > total_words_count:
            total_words_count = ft_words

    words_ratio = (timestamped_words_count / max(1, total_words_count)) if total_words_count > 0 else 0.0
    words_ratio = min(1.0, words_ratio)

    # 3. Speaking Rate (WPM)
    speech_minutes = max(effective_duration, transcribed_span) / 60.0
    wpm = total_words_count / max(0.01, speech_minutes)

    # Natural speech rate curve: [110, 190] WPM is ideal (1.0)
    if 110.0 <= wpm <= 190.0:
        s_wpm = 1.0
    elif wpm < 110.0:
        s_wpm = max(0.0, wpm / 110.0)
    else:
        s_wpm = max(0.0, 1.0 - (wpm - 190.0) / 100.0)

    # Coverage Pillar (0 - 30 points)
    s_cov = 30.0 * (0.6 * coverage_ratio + 0.4 * s_wpm)

    # Granularity Pillar (0 - 25 points)
    # Ideal segment duration range for speech/audiobook reading: [2.5s, 35.0s]
    # Accommodates both conversational dialogue (3-10s) and mature reading paragraphs (12-35s)
    avg_seg_dur = transcribed_span / max(1, n_seg)
    if 2.5 <= avg_seg_dur <= 35.0:
        s_seg = 1.0
    elif avg_seg_dur < 2.5:
        s_seg = max(0.1, avg_seg_dur / 2.5)
    else:
        s_seg = max(0.2, 1.0 - (avg_seg_dur - 35.0) / 30.0)

    s_gran = (20.0 * words_ratio) + (5.0 * s_seg)

    # Timeline Health Pillar (0 - 25 points)
    p_zero = min(15.0, zero_duration_count * 2.0)
    p_stretch = min(10.0, stretched_count * 2.0)
    p_inv = min(10.0, inversion_count * 3.0)
    s_health = max(0.0, 25.0 - (p_zero + p_stretch + p_inv))

    # Model Confidence Pillar (0 - 20 points)
    if prob_count > 0:
        avg_prob = prob_sum / prob_count
        s_conf = 20.0 * min(1.0, max(0.0, avg_prob))
    else:
        # Fallback heuristic when word probability is absent
        eng_lower = (engine or "").lower()
        if "whisper" in eng_lower or "gemini" in eng_lower:
            avg_prob = 0.88
            s_conf = 17.6
        else:
            avg_prob = 0.80
            s_conf = 16.0

    # Degeneration Deductions
    p_loop = min(20.0, repetition_loops * 5.0)
    p_leak = 5.0 if leak_detected else 0.0

    # Micro-fragmentation penalty: penalizes prompt-saturated Whisper runs with high micro-segment ratios
    # Normal audiobook speech tolerates up to 5% brief interjections/utterances.
    micro_ratio = (micro_segments_count / max(1, n_seg)) if n_seg >= 10 else 0.0
    if micro_ratio > 0.05:
        p_frag = min(20.0, (micro_ratio - 0.05) * 30.0)
    else:
        p_frag = 0.0

    p_degen = p_loop + p_leak + p_frag

    # Total clamped TQS
    total_tqs = max(0.0, min(100.0, s_cov + s_gran + s_health + s_conf - p_degen))

    # Qualitative Grade
    if total_tqs >= 85.0:
        grade = "PRODUCTION_GRADE"
    elif total_tqs >= 70.0:
        grade = "ACCEPTABLE"
    elif total_tqs >= 50.0:
        grade = "SUBPAR"
    else:
        grade = "DEFECTIVE"

    return QualityScore(
        total_score=round(total_tqs, 2),
        coverage_score=round(s_cov, 2),
        granularity_score=round(s_gran, 2),
        health_score=round(s_health, 2),
        confidence_score=round(s_conf, 2),
        degeneration_penalty=round(p_degen, 2),
        duration_seconds=round(effective_duration, 3),
        transcribed_span=round(transcribed_span, 3),
        coverage_ratio=round(coverage_ratio, 4),
        word_count=total_words_count,
        segment_count=n_seg,
        wpm=round(wpm, 1),
        words_with_timestamps_ratio=round(words_ratio, 4),
        avg_segment_duration=round(avg_seg_dur, 3),
        zero_duration_segments_count=zero_duration_count,
        stretched_segments_count=stretched_count,
        inversion_count=inversion_count,
        avg_word_probability=round(avg_prob, 4),
        repetition_loops_count=repetition_loops,
        grade=grade,
        micro_segments_count=micro_segments_count,
        micro_segment_ratio=round(micro_ratio, 4),
    )


def _sample_stratified_tokens(text: str, window_size: int = 1500) -> list[str]:
    """Sample tokens from head, middle, and tail of large text documents."""
    n = len(text)
    char_chunk = window_size * 7  # approx 7 chars per word token
    if n <= char_chunk * 3:
        return _tokenize(text)

    head = text[:char_chunk]
    mid_start = max(0, (n - char_chunk) // 2)
    mid = text[mid_start:mid_start + char_chunk]
    tail = text[-char_chunk:]

    head_tokens = _tokenize(head)[:window_size]
    mid_tokens = _tokenize(mid)[:window_size]
    tail_tokens = _tokenize(tail)[:window_size]

    return head_tokens + mid_tokens + tail_tokens


def evaluate_familiarity(text_a: str, text_b: str) -> FamiliarityScore:
    """
    Measure content similarity between two transcripts using unigram and bigram Jaccard,
    vocabulary containment, and length preservation.
    """
    len_a = len(text_a)
    len_b = len(text_b)

    if len_a <= 25000 and len_b <= 25000:
        tokens_a = _tokenize(text_a)
        tokens_b = _tokenize(text_b)
        total_len_a = len(tokens_a)
        total_len_b = len(tokens_b)
    else:
        # Fast regex word count for length ratio preservation across whole text
        total_len_a = len(re.findall(r"\b\w+\b", text_a))
        total_len_b = len(re.findall(r"\b\w+\b", text_b))

        # Stratified window sampling: head 1500 + mid 1500 + tail 1500 tokens
        tokens_a = _sample_stratified_tokens(text_a, 1500)
        tokens_b = _sample_stratified_tokens(text_b, 1500)

    if not tokens_a and not tokens_b:
        return FamiliarityScore(100.0, 1.0, 1.0, 1.0, 1.0, "DIRECT_MATCH")
    if not tokens_a or not tokens_b:
        return FamiliarityScore(0.0, 0.0, 0.0, 0.0, 0.0, "MISMATCH")

    # Unigram Jaccard
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    unigram_jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))

    # Bigram Jaccard (captures phrase and word-order coherence)
    bigrams_a = set(zip(tokens_a[:-1], tokens_a[1:])) if len(tokens_a) > 1 else {("", tokens_a[0])}
    bigrams_b = set(zip(tokens_b[:-1], tokens_b[1:])) if len(tokens_b) > 1 else {("", tokens_b[0])}
    bigram_jaccard = len(bigrams_a & bigrams_b) / max(1, len(bigrams_a | bigrams_b))

    # Vocabulary Containment (resistant to truncation/abridgement)
    min_vocab = min(len(set_a), len(set_b))
    containment = len(set_a & set_b) / max(1, min_vocab)

    # Length Preservation (using global word counts)
    length_ratio = min(total_len_a, total_len_b) / max(1, max(total_len_a, total_len_b))

    # Combined composite similarity score (0.0 to 100.0%)
    composite = (0.40 * unigram_jaccard + 0.40 * bigram_jaccard + 0.20 * (containment * length_ratio)) * 100.0
    composite = max(0.0, min(100.0, composite))

    if composite >= 75.0 or (containment >= 0.90 and length_ratio >= 0.65):
        classification = "DIRECT_MATCH"
    elif composite >= 45.0 or containment >= 0.60:
        classification = "PARTIAL_MATCH"
    else:
        classification = "MISMATCH"

    return FamiliarityScore(
        score=round(composite, 2),
        unigram_jaccard=round(unigram_jaccard, 4),
        bigram_jaccard=round(bigram_jaccard, 4),
        containment=round(containment, 4),
        length_ratio=round(length_ratio, 4),
        classification=classification,
    )


def evaluate_windowed_fidelity(
    existing_segments: Iterable[Any],
    incoming_segments: Iterable[Any],
    total_duration: float,
    window_size_seconds: float = 300.0,
    min_word_ratio: float = 0.75,
    min_containment: float = 0.80,
) -> WindowFidelityResult:
    """
    Timeline-aligned fidelity check across chronological windows (default 5 minutes).
    Compares word counts and vocabulary containment in each window to detect
    localized omissions, silent gap drops, or hallucination loops that get masked
    by global bag-of-words similarity.

    Honest-pair noise floor: >=85% word ratio, >=91% vocabulary containment.
    Thresholds: word_ratio < 75% OR containment < 80% triggers a window anomaly flag.
    Veto rule: >=2 flagged windows (or >=3% of total duration) hard-blocks UPGRADE_RECOMMENDED.
    """
    if total_duration <= 0.0 or not existing_segments or not incoming_segments:
        return WindowFidelityResult(
            window_count=0,
            flagged_windows_count=0,
            flagged_ratio=0.0,
            flagged_details=[],
            is_faithful=True,
            summary="Insufficient duration or segments for windowed analysis",
        )

    # Slicing with trailing tail absorption (< 120s) to avoid edge boundary bleed on short tail slivers
    min_tail = min(120.0, window_size_seconds * 0.4)
    windows: list[tuple[float, float]] = []
    if total_duration <= window_size_seconds + min_tail:
        windows.append((0.0, total_duration))
    else:
        cur_s = 0.0
        while cur_s < total_duration:
            cur_e = cur_s + window_size_seconds
            if (total_duration - cur_e) < min_tail:
                windows.append((cur_s, total_duration))
                break
            else:
                windows.append((cur_s, cur_e))
                cur_s = cur_e

    num_windows = len(windows)
    flagged: list[dict] = []

    # Pre-extract segment tuples: (start, end, text)
    ex_list = []
    for s in existing_segments:
        st = float(_get_attr(s, "start", _get_attr(s, "start_time", 0.0)))
        en = float(_get_attr(s, "end", _get_attr(s, "end_time", 0.0)))
        txt = str(_get_attr(s, "text", "")).strip()
        if txt:
            ex_list.append((st, en, txt))

    inc_list = []
    for s in incoming_segments:
        st = float(_get_attr(s, "start", _get_attr(s, "start_time", 0.0)))
        en = float(_get_attr(s, "end", _get_attr(s, "end_time", 0.0)))
        txt = str(_get_attr(s, "text", "")).strip()
        if txt:
            inc_list.append((st, en, txt))

    for w_idx, (w_start, w_end) in enumerate(windows):

        ex_text = " ".join(t[2] for t in ex_list if not (t[1] <= w_start or t[0] >= w_end))
        inc_text = " ".join(t[2] for t in inc_list if not (t[1] <= w_start or t[0] >= w_end))

        ex_words = len(ex_text.split()) if ex_text else 0
        inc_words = len(inc_text.split()) if inc_text else 0

        # Skip windows that are essentially pure silence/music in both
        if ex_words < 15 and inc_words < 15:
            continue

        if max(ex_words, inc_words) > 0:
            word_ratio = min(ex_words, inc_words) / max(ex_words, inc_words)
        else:
            word_ratio = 1.0

        tok_ex = set(_tokenize(ex_text))
        tok_inc = set(_tokenize(inc_text))

        if tok_ex and tok_inc:
            containment = len(tok_ex & tok_inc) / len(tok_ex)
        elif not tok_ex and not tok_inc:
            containment = 1.0
        else:
            containment = 0.0

        if word_ratio < min_word_ratio or containment < min_containment:
            flagged.append({
                "window_index": w_idx,
                "start_seconds": round(w_start, 1),
                "end_seconds": round(w_end, 1),
                "existing_words": ex_words,
                "incoming_words": inc_words,
                "word_ratio": round(word_ratio, 4),
                "containment": round(containment, 4),
            })

    flagged_count = len(flagged)
    flagged_ratio = flagged_count / max(1, num_windows)
    is_faithful = flagged_count == 0

    if flagged_count == 0:
        summary = f"All {num_windows} windows verified faithful (0 anomalies)"
    else:
        worst_w = min(flagged, key=lambda x: min(x["word_ratio"], x["containment"]))
        summary = (
            f"{flagged_count}/{num_windows} window(s) flagged "
            f"(worst at {worst_w['start_seconds']/60:.1f}m: ratio {worst_w['word_ratio']:.1%}, cont {worst_w['containment']:.1%})"
        )

    return WindowFidelityResult(
        window_count=num_windows,
        flagged_windows_count=flagged_count,
        flagged_ratio=round(flagged_ratio, 4),
        flagged_details=flagged,
        is_faithful=is_faithful,
        summary=summary,
    )


def compare_transcripts(
    existing_record: Any | None,
    incoming_transcript: Any,
    audio_duration: float | None = None,
) -> ComparisonResult:
    """
    Compare an existing database transcription with an incoming candidate.
    Determines whether an upgrade/overwrite is recommended.
    """
    inc_segments = _get_attr(incoming_transcript, "segments", []) or []
    inc_engine = _get_attr(incoming_transcript, "engine", None)
    inc_text = _get_attr(incoming_transcript, "text", "") or _get_attr(incoming_transcript, "full_text", "") or ""
    inc_dur = audio_duration or _get_attr(incoming_transcript, "duration_seconds", None)

    incoming_tqs = evaluate_transcript_quality(
        segments=inc_segments,
        total_duration=inc_dur,
        engine=inc_engine,
        full_text=inc_text,
    )

    if existing_record is None:
        return ComparisonResult(
            existing_id=None,
            existing_tqs=None,
            incoming_tqs=incoming_tqs,
            familiarity=FamiliarityScore(100.0, 1.0, 1.0, 1.0, 1.0, "DIRECT_MATCH"),
            delta_tqs=round(incoming_tqs.total_score, 2),
            recommendation="NEW_IMPORT",
            summary=f"New recording: ready for initial import (TQS: {incoming_tqs.total_score:.1f})",
        )

    # Evaluate existing record
    existing_id = getattr(existing_record, "id", None)
    ex_engine = getattr(existing_record, "engine", None)
    ex_text = getattr(existing_record, "full_text", "") or ""
    ex_duration = getattr(existing_record, "duration_seconds", None) or audio_duration

    try:
        ex_segments = list(getattr(existing_record, "segments", []) or [])
    except Exception:
        ex_segments = []

    # If existing record has an audio file with a sidecar JSON in the library, load it for full word timestamps
    try:
        audio_rec = getattr(existing_record, "audio_file", None)
        if audio_rec and getattr(audio_rec, "file_path", None):
            from pathlib import Path
            sidecar = Path(audio_rec.file_path).with_suffix(".json")
            if sidecar.exists():
                import json
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if data.get("segments"):
                    ex_segments = data["segments"]
                    ex_engine = data.get("engine") or ex_engine
                    ex_text = data.get("text") or data.get("full_text") or ex_text
    except Exception:
        pass

    existing_tqs = evaluate_transcript_quality(
        segments=ex_segments,
        total_duration=ex_duration,
        engine=ex_engine,
        full_text=ex_text,
    )

    familiarity = evaluate_familiarity(ex_text, inc_text)
    delta_tqs = round(incoming_tqs.total_score - existing_tqs.total_score, 2)

    # Timeline-aligned windowed fidelity check
    fidelity: WindowFidelityResult | None = None
    if ex_segments and inc_segments:
        effective_dur = ex_duration or inc_dur or 0.0
        if effective_dur > 60.0:
            fidelity = evaluate_windowed_fidelity(ex_segments, inc_segments, effective_dur)

    if familiarity.classification == "MISMATCH":
        recommendation = "MISMATCH"
        summary = f"Content mismatch ({familiarity.score:.1f}% similarity). Audio recordings appear distinct."
    elif fidelity and not fidelity.is_faithful:
        recommendation = "POTENTIAL_OMISSION_DETECTED"
        summary = f"Upgrade blocked: localized omissions/defects detected in {fidelity.flagged_windows_count} window(s) ({fidelity.summary})"
    elif delta_tqs >= 15.0:
        recommendation = "UPGRADE_RECOMMENDED"
        reasons = []
        if incoming_tqs.words_with_timestamps_ratio > existing_tqs.words_with_timestamps_ratio + 0.5:
            reasons.append("word timestamps added")
        if incoming_tqs.coverage_ratio > existing_tqs.coverage_ratio + 0.1:
            reasons.append("full ending recovered")
        if incoming_tqs.health_score > existing_tqs.health_score + 5.0:
            reasons.append("zero-duration segments healed")
        if existing_tqs.micro_segment_ratio > 0.15 and incoming_tqs.micro_segment_ratio <= 0.05:
            reasons.append("micro-fragmentation healed")
        if existing_tqs.avg_segment_duration < 3.0 and 12.0 <= incoming_tqs.avg_segment_duration <= 35.0:
            reasons.append("reading paragraphs restored")
        reason_str = f" ({', '.join(reasons)})" if reasons else ""
        summary = f"Upgrade recommended (+{delta_tqs:.1f} TQS): {existing_tqs.grade} → {incoming_tqs.grade}{reason_str}"
    elif delta_tqs <= -10.0:
        recommendation = "KEEP_EXISTING"
        summary = f"Keep existing (+{-delta_tqs:.1f} TQS higher): {existing_tqs.grade} is superior to candidate"
    else:
        if existing_tqs.total_score >= 80.0:
            recommendation = "KEEP_EXISTING"
            summary = f"Existing transcript is already production-grade (TQS {existing_tqs.total_score:.1f})"
        else:
            recommendation = "SIMILAR_QUALITY"
            summary = f"Similar quality (Δ TQS {delta_tqs:+.1f})"

    return ComparisonResult(
        existing_id=existing_id,
        existing_tqs=existing_tqs,
        incoming_tqs=incoming_tqs,
        familiarity=familiarity,
        delta_tqs=delta_tqs,
        recommendation=recommendation,
        summary=summary,
        fidelity=fidelity,
    )


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r"\b[a-z0-9]+(?:'[a-z0-9]+)?\b", text.lower())


def _detect_repetition_loops(seg_texts: list[str]) -> int:
    """Detect repetitive loop hallucinations (e.g. identical phrases repeated 3+ times)."""
    if len(seg_texts) < 3:
        return 0

    loops = 0
    current_repeat = 1
    prev_norm = ""

    for text in seg_texts:
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        if not norm:
            continue
        if norm == prev_norm:
            current_repeat += 1
            if current_repeat >= 3:
                loops += 1
        else:
            current_repeat = 1
            prev_norm = norm

    return loops


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safe getter for object attribute or dictionary key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
