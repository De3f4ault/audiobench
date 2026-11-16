"""
alignment_pipeline.py — Timestamp alignment for Gemini transcripts.

WHY THIS APPROACH
-----------------
Gemini returns segment text but no timestamps.  ctc-forced-aligner (ONNX)
was used previously but leaks C++ heap memory across repeated
generate_emissions() calls — the ONNX runtime accumulates transformer
activation buffers that Python's GC cannot reach.  For a 1h 30m file at
10-second windows (550 windows), the leak grows until SIGABRT (exit 134).
No amount of gc.collect() or window-size reduction fixes a C++ heap leak.

NEW APPROACH: faster-whisper tiny
---------------------------------
faster-whisper uses CTranslate2 (C++) but correctly frees memory after
each transcription call — it does not accumulate state across calls.
The tiny model (~75 MB) runs at ~20× real-time on CPU, so a 1h 30m file
completes in about 4–5 minutes.

Algorithm
---------
1. Run faster-whisper-tiny on the full audio to get N whisper segments,
   each with a real (start, end) timestamp.
2. Build a timeline by cumulative character count over both the Whisper
   segments and the Gemini segments.
3. For each Gemini segment, interpolate its position in the Whisper
   timeline using cumulative character count as a proxy for elapsed time.

The interpolation uses character count rather than word count because it
handles multilingual (Swahili + English) and punctuation-heavy text more
robustly.  It also requires no text-matching heuristics that break when
Gemini and Whisper use slightly different words.
"""

import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from audiobench.transcribe.transcription_result import Transcript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: faster-whisper model size to use for alignment.
#: "tiny" is ~75 MB and runs at ~20× real-time on CPU.
#: "base" is ~150 MB and runs at ~10× real-time — more accurate but slower.
_ALIGN_MODEL_SIZE: str = "tiny"


# ---------------------------------------------------------------------------
# Model loading (singleton inside the worker subprocess)
# ---------------------------------------------------------------------------

def load_align_model(device: str | None = None) -> Any:
    """Load the faster-whisper model used for timestamp alignment.

    Parameters
    ----------
    device:
        'cpu', 'cuda', or None/'auto' (auto-detects CUDA → CPU).

    Returns a ``faster_whisper.WhisperModel`` instance.  The model is
    loaded from the local HuggingFace cache — no network access required.
    """
    import torch
    from faster_whisper import WhisperModel

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "int8"   # INT8 on CPU — fastest and smallest memory footprint
    logger.info(
        "Loading faster-whisper-%s for alignment (device=%s, compute=%s)…",
        _ALIGN_MODEL_SIZE, device, compute_type,
    )
    return WhisperModel(_ALIGN_MODEL_SIZE, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align_transcript(
    transcript: Transcript,
    audio_path: str | Path,
    model: Any,
    on_progress: Callable | None = None,
) -> tuple[Transcript, bool]:
    """Align Gemini segment timestamps using faster-whisper re-transcription.

    Parameters
    ----------
    transcript:
        The Gemini transcript (segments already populated, timestamps = 0).
    audio_path:
        Path to the original audio file.
    model:
        A ``faster_whisper.WhisperModel`` instance from ``load_align_model()``.
    on_progress:
        Optional callback fired for each segment after alignment.

    Returns
    -------
    tuple[Transcript, bool]
        The mutated *transcript* with updated timestamps, and a boolean
        ``has_clamped`` that is ``True`` if any Whisper segment had a
        backward-jumping start time and was clamped forward.  A clamped
        transcript has monotonic but potentially distorted timing in the
        affected region and should be marked ``pipeline_phase='degraded'``
        with ``failure_reason='timeline_clamped'``.
    """
    if not transcript.segments:
        return transcript, False

    audio_path = Path(audio_path)

    # ── Step 1: re-transcribe with the tiny model to get real timestamps ──────
    logger.info(
        "Alignment (faster-whisper-%s): transcribing %s …",
        _ALIGN_MODEL_SIZE, audio_path.name,
    )
    try:
        whisper_segments = _transcribe_for_timestamps(model, audio_path, transcript)
    except Exception as exc:
        logger.error("Alignment: faster-whisper transcription failed: %s", exc)
        return transcript, False

    if not whisper_segments:
        logger.warning("Alignment: faster-whisper returned no segments — keeping proportional timestamps")
        return transcript, False

    logger.info(
        "Alignment: faster-whisper produced %d segments covering %.0f–%.0fs",
        len(whisper_segments),
        whisper_segments[0]["start"],
        whisper_segments[-1]["end"],
    )

    # ── Step 1b: sanitize Whisper timeline (detect backward jumps) ────────────
    tx_id = getattr(transcript, "id", None)
    whisper_segments, has_clamped = _sanitize_whisper_segments(whisper_segments, tx_id)

    # ── Step 2: patch transcript duration from the whisper timeline ───────────
    if whisper_segments:
        whisper_end = whisper_segments[-1]["end"]
        if transcript.duration_seconds == 0.0 or abs(transcript.duration_seconds - whisper_end) > 5.0:
            transcript.duration_seconds = whisper_end

    # ── Step 3: map Gemini segments → Whisper timestamps ─────────────────────
    _map_timestamps(transcript, whisper_segments)

    # ── Step 4: fire progress callbacks ───────────────────────────────────────
    if on_progress:
        for segment in transcript.segments:
            on_progress(segment)

    gc.collect()
    return transcript, has_clamped


# ---------------------------------------------------------------------------
# Internal: faster-whisper transcription pass
# ---------------------------------------------------------------------------

def _transcribe_for_timestamps(
    model: Any,
    audio_path: Path,
    transcript: Transcript,
) -> list[dict]:
    """Run faster-whisper on the full audio and return a list of segment dicts.

    Each dict has keys: ``start``, ``end``, ``text``, ``char_count``.

    We transcribe with ``word_timestamps=False`` (segment-level only) because
    we only need approximate timestamps.  VAD filtering is enabled so silent
    gaps don't inflate the character-time mapping.
    """
    # Infer the primary language from the Gemini transcript
    lang = transcript.language if transcript.language else None

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang,
        task="transcribe",
        word_timestamps=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "threshold": 0.3,
        },
        beam_size=1,           # fastest beam; accuracy is sufficient for alignment
        best_of=1,
        temperature=0.0,
    )

    result = []
    for seg in segments_iter:
        result.append({
            "start": float(seg.start),
            "end":   float(seg.end),
            "text":  seg.text.strip(),
            "char_count": len(seg.text.strip()),
        })

    return result


# ---------------------------------------------------------------------------
# Internal: Whisper timeline sanitization
# ---------------------------------------------------------------------------

def _sanitize_whisper_segments(
    whisper_segments: list[dict],
    tx_id: int | None = None,
) -> tuple[list[dict], bool]:
    """Enforce a monotonically increasing Whisper timeline.

    When faster-whisper outputs a segment whose start time precedes the
    previous segment's end time (a backward jump), this function clamps
    the offending segment's start to the previous end.  The resulting
    timeline is monotonic but the timing for the clamped region is
    distorted — those Gemini segments will receive plausible-but-wrong
    timestamps.

    The caller MUST surface this distortion rather than hiding it:

    * The function logs a WARNING for every clamp, including the segment
      index and the clamp amount in seconds.
    * It returns ``has_clamped=True`` so the alignment worker can mark
      the transcript ``pipeline_phase='degraded'`` with
      ``failure_reason='timeline_clamped'``.

    Parameters
    ----------
    whisper_segments:
        Raw segment dicts from ``_transcribe_for_timestamps``.
    tx_id:
        Transcription ID, included in log messages for traceability.

    Returns
    -------
    tuple[list[dict], bool]
        Sanitized segment list and ``has_clamped`` flag.
    """
    sanitized: list[dict] = []
    last_end = 0.0
    has_clamped = False

    for idx, seg in enumerate(whisper_segments):
        s = float(seg["start"])
        e = float(seg["end"])

        if s < last_end:
            shift = last_end - s
            logger.warning(
                "Alignment: Whisper timeline backward jump on tx=%s whisper_seg_idx=%d "
                "(start=%.3fs < last_end=%.3fs, clamping forward by %.3fs). "
                "Timing in this region will be distorted. Transcript will be flagged degraded.",
                tx_id, idx, s, last_end, shift,
            )
            s = last_end
            has_clamped = True

        if e <= s:
            orig_dur = float(seg["end"]) - float(seg["start"])
            e = s + max(0.1, orig_dur if orig_dur > 0 else 0.5)

        sanitized.append({**seg, "start": s, "end": e})
        last_end = e

    return sanitized, has_clamped


# ---------------------------------------------------------------------------
# Internal: cumulative character-count interpolation
# ---------------------------------------------------------------------------

def _map_timestamps(
    transcript: Transcript,
    whisper_segments: list[dict],
) -> None:
    """Map Gemini segment timestamps from the Whisper timeline.

    Strategy: cumulative character count as a proxy for elapsed time.

    Anchors are 4-tuples ``(c_start, c_end, t_start, t_end)`` representing
    a single Whisper segment's character span and time span.  Interpolation
    for a target character position is performed **inside** the containing
    segment's own spoken interval — never across the silence gap between
    adjacent segments.

    This eliminates the cross-gap interpolation defect (line 315, commit
    963a0b0) that produced inverted timestamps whenever ``hi_start <
    lo_end`` (Whisper segment K+1 starts before segment K ends).  The
    old formula ``lo_end + frac * (hi_start − lo_end)`` had a negative
    slope under that condition; the new formula ``t_start + frac *
    (t_end − t_start)`` is always non-decreasing because ``t_end >=
    t_start`` is guaranteed by ``_sanitize_whisper_segments``.

    This works because:
    - Speech rate is roughly constant in talking-head / lecture audio.
    - Character count per second is approximately uniform for the same
      speaker and language.
    - We only need ±10–30 second accuracy — CTC forced alignment gives
      ±0.1s but at the cost of OOM crashes for long files.

    Handles multilingual content (Swahili + English) gracefully because
    character count is language-agnostic.
    """
    if not whisper_segments or not transcript.segments:
        return

    # Build 4-tuple anchors: (c_start, c_end, t_start, t_end)
    # c_start/c_end are cumulative character positions in the Whisper timeline.
    # t_start/t_end are the segment's audio timestamps (monotonic after sanitization).
    w_anchors: list[tuple[float, float, float, float]] = []
    cum = 0.0
    for seg in whisper_segments:
        c = max(1, seg["char_count"])
        w_anchors.append((cum, cum + c, float(seg["start"]), float(seg["end"])))
        cum += c
    w_total_chars = cum

    # Build cumulative char positions for Gemini segments
    g_total_chars = sum(max(1, len(s.text)) for s in transcript.segments)

    if g_total_chars == 0 or w_total_chars == 0:
        return

    max_time = whisper_segments[-1]["end"]
    prev_start = 0.0
    cum_g = 0.0

    for i, seg in enumerate(transcript.segments):
        n_chars = max(1, len(seg.text))

        # Fractional position of this Gemini segment in [0, 1]
        g_frac_start = cum_g / g_total_chars
        g_frac_end   = (cum_g + n_chars) / g_total_chars

        # Map to absolute position in Whisper char timeline
        w_char_start = g_frac_start * w_total_chars
        w_char_end   = g_frac_end   * w_total_chars

        seg.start = _interpolate_time(w_anchors, w_char_start, max_time)
        seg.end   = _interpolate_time(w_anchors, w_char_end,   max_time)

        # These conditions should not occur after sanitization, but must
        # stay detectable rather than silently corrected.
        if seg.end < seg.start:
            logger.error(
                "Alignment: inverted segment on tx=%s idx=%d start=%.3f end=%.3f — "
                "the formula has a bug that survived sanitization, please report.",
                getattr(transcript, "id", "?"), i, seg.start, seg.end,
            )
        elif i > 0 and seg.start < prev_start:
            logger.error(
                "Alignment: non-monotonic segment on tx=%s idx=%d start=%.3f < prev=%.3f — "
                "the formula has a bug that survived sanitization, please report.",
                getattr(transcript, "id", "?"), i, seg.start, prev_start,
            )

        prev_start = seg.start
        cum_g += n_chars


def _interpolate_time(
    anchors: list[tuple[float, float, float, float]],
    target_chars: float,
    max_time: float,
) -> float:
    """Interpolate a time value from 4-tuple Whisper char-time anchors.

    Each anchor is ``(c_start, c_end, t_start, t_end)`` representing one
    Whisper segment.  The target character position is looked up in the
    anchor whose ``[c_start, c_end)`` interval contains it, and the result
    is linearly interpolated within ``[t_start, t_end]``.

    Because ``t_end >= t_start`` for all anchors (guaranteed by
    ``_sanitize_whisper_segments``), the returned value is always
    non-decreasing as ``target_chars`` increases — the inversion defect
    present in the previous 3-tuple / cross-gap formula cannot occur.

    Parameters
    ----------
    anchors:
        Sorted list of (c_start, c_end, t_start, t_end) from ``_map_timestamps``.
    target_chars:
        The cumulative Whisper character position to look up.
    max_time:
        Upper bound: the end time of the last Whisper segment.
    """
    if not anchors:
        return 0.0

    # Clamp below: target is before all whisper content
    if target_chars <= anchors[0][0]:
        return anchors[0][2]  # t_start of first anchor

    # Clamp above: target is after all whisper content
    if target_chars >= anchors[-1][1]:
        return min(max_time, anchors[-1][3])  # t_end of last anchor

    # Binary search for the anchor whose [c_start, c_end) contains target_chars
    lo, hi = 0, len(anchors) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if anchors[mid][0] <= target_chars:
            lo = mid
        else:
            hi = mid

    c_start, c_end, t_start, t_end = anchors[lo]

    span = c_end - c_start
    if span == 0:
        return t_start

    frac = max(0.0, min(1.0, (target_chars - c_start) / span))
    # Interpolate WITHIN the segment's spoken span — never across the silence gap
    return t_start + frac * (t_end - t_start)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_initial_timestamps(transcript: Transcript) -> None:
    """Evenly distribute segment timestamps proportional to character count.

    Only runs when the engine returned all-zero timestamps (Gemini, etc.).
    Guards against a zero duration so it never silently produces 0:00 for all.
    Must be called AFTER transcript.duration_seconds has been set correctly.

    This provides an instant fallback before alignment runs, so the DB always
    has non-zero timestamps even if the alignment subprocess crashes.
    """
    needs_estimation = all(
        s.start == 0.0 and s.end == 0.0 for s in transcript.segments
    )
    if not needs_estimation:
        return

    duration = transcript.duration_seconds
    if duration <= 0.0:
        logger.warning(
            "Alignment: _estimate_initial_timestamps called with duration=%.1f "
            "— skipping estimation, segments will have zero timestamps.",
            duration,
        )
        return

    total_chars = sum(len(s.text) for s in transcript.segments)
    if total_chars == 0:
        return

    current_time = 0.0
    for segment in transcript.segments:
        char_count   = len(segment.text)
        seg_duration = (char_count / total_chars) * duration
        segment.start = current_time
        segment.end   = min(current_time + seg_duration, duration)
        current_time += seg_duration
