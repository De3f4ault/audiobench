"""SpeechBrain ECAPA-TDNN voiceprint enrollment and biometric tagging.

Architecture
------------
Enrollment
    The user provides an audio sample >= 20 seconds. SpeechBrain loads
    the ECAPA-TDNN model, extracts a 192-dimensional cosine embedding,
    and saves it to ``data/security/voiceprint.npy``.

Biometric pass (macro-batch)
    After each Whisper transcription, a background pass compares every new
    segment's audio slice against the enrolled embedding using cosine
    similarity. Segments that exceed the threshold (default 0.75) are
    tagged with ``privacy_tier = 2`` (Intimate) in the database.

    The pass operates on NumPy float32 arrays extracted directly from the
    waveform — no Python list copies, no GIL overhead on the math.

Performance
    ~5-10 ms/segment on CPU (ECAPA-TDNN is lightweight for inference).
    A 20-file macro-batch (~2000 segments) takes ~20-25 minutes background.

Facade pattern
    The rest of the app never imports PyTorch or SpeechBrain directly.
    Only this module does. All public functions use plain Python types.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

if TYPE_CHECKING:
    pass

logger = get_logger("security.voiceprint")

_ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
_EMBEDDING_DIMS = 192
_MIN_DURATION_S = 20.0  # minimum audio duration for reliable enrollment
_VOICEPRINT_FILENAME = "voiceprint.npy"
_ENROLLMENT_META_FILENAME = "enrollment.json"


# ── Internal helpers ───────────────────────────────────────────────────────

def _voiceprint_path() -> Path:
    d = get_settings().security_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / _VOICEPRINT_FILENAME


def _meta_path() -> Path:
    return get_settings().security_dir / _ENROLLMENT_META_FILENAME


def _load_ecapa() -> object:
    """Lazy-load SpeechBrain ECAPA-TDNN model. Raises ImportError if not installed."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise ImportError(
            "SpeechBrain is required for voiceprint features. "
            "Install it with: pip install speechbrain"
        ) from exc

    return EncoderClassifier.from_hparams(
        source=_ECAPA_MODEL_ID,
        run_opts={"device": "cpu"},
    )


def _extract_embedding(model, waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract a single ECAPA-TDNN embedding from a waveform array.

    Args:
        model:    Loaded SpeechBrain EncoderClassifier.
        waveform: float32 NumPy array, shape (n_samples,). Mono.
        sr:       Sample rate in Hz. Will be resampled to 16 kHz if needed.

    Returns:
        float32 NumPy array, shape (192,). L2-normalised.
    """
    import torch

    # Resample to 16 kHz if needed (ECAPA-TDNN expects 16 kHz)
    if sr != 16000:
        try:
            import torchaudio
            wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
            waveform = wav_tensor.squeeze(0).numpy()
        except ImportError:
            logger.warning("torchaudio not installed — cannot resample. Audio may be degraded.")

    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, n_samples)
    with torch.no_grad():
        embedding = model.encode_batch(wav_tensor)          # (1, 1, 192)
    vec = embedding.squeeze().cpu().numpy().astype(np.float32)

    # L2 normalise so cosine similarity == dot product
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors.

    Because both vectors are already L2-normalised in _extract_embedding,
    this reduces to a simple dot product — O(dims) in C via NumPy.
    """
    return float(np.dot(a, b))


# ── Public API ─────────────────────────────────────────────────────────────

def is_enrolled() -> bool:
    """Return True if a voiceprint embedding file exists."""
    return _voiceprint_path().exists()


def get_enrollment_summary() -> str | None:
    """Return a one-line human summary of the enrolled voiceprint, or None."""
    import json

    if not is_enrolled():
        return None
    meta_p = _meta_path()
    if not meta_p.exists():
        return "Enrolled (no name set)"
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        name = meta.get("name", "Unknown")
        n_tagged = meta.get("segments_tagged", 0)
        enrolled_date = (meta.get("enrolled_at", "")[:10]) or "?"
        return f"{name} ({n_tagged:,} segments tagged · {enrolled_date})"
    except Exception:
        return "Enrolled (metadata unreadable)"


def enroll(audio_path: str | Path, name: str = "Owner") -> dict:
    """Enroll a voiceprint from an audio file.

    Args:
        audio_path: Path to audio file (wav, opus, mp3, flac, etc.).
        name:       Human name for the enrolled voiceprint.

    Returns:
        dict with keys: embedding_shape, duration_seconds, name, segments_tagged.

    Raises:
        ValueError:  Audio too short (<20 s).
        ImportError: SpeechBrain/torchaudio not installed.
    """
    import datetime
    import json

    audio_path = Path(audio_path)
    logger.info("Enrolling voiceprint from: %s", audio_path)

    # Load audio as float32 NumPy array
    waveform, sr = _load_audio(audio_path)
    duration = len(waveform) / sr

    if duration < _MIN_DURATION_S:
        raise ValueError(
            f"Audio is {duration:.1f}s — minimum {_MIN_DURATION_S}s required for reliable enrollment."
        )

    logger.info("Audio: %.1fs @ %d Hz — extracting ECAPA-TDNN embedding...", duration, sr)
    model = _load_ecapa()
    embedding = _extract_embedding(model, waveform, sr)

    # Save embedding
    np.save(str(_voiceprint_path()), embedding)
    logger.info("Voiceprint saved: %s (dims=%d)", _voiceprint_path(), len(embedding))

    # Save metadata
    meta = {
        "name": name,
        "model": _ECAPA_MODEL_ID,
        "embedding_dims": len(embedding),
        "enrolled_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "duration_seconds": duration,
        "segments_tagged": 0,
    }
    _meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "embedding_shape": embedding.shape,
        "duration_seconds": duration,
        "name": name,
        "segments_tagged": 0,
    }


def tag_segments_batch(
    segment_ids: list[int],
    audio_slices: list[np.ndarray],
    sr: int,
    model=None,
) -> list[int]:
    """Tag a batch of segments via biometric matching.

    For each audio slice, extracts an ECAPA-TDNN embedding and computes
    cosine similarity against the enrolled voiceprint. Segments exceeding
    the threshold are marked Tier 2 in the database.

    Args:
        segment_ids:  DB row IDs of the segments.
        audio_slices: List of float32 NumPy arrays (one per segment), mono.
        sr:           Sample rate.
        model:        Optional pre-loaded EncoderClassifier (avoids re-loading
                      per batch — pass from the caller for efficiency).

    Returns:
        List of segment IDs that were tagged (similarity >= threshold).
    """
    if not is_enrolled():
        logger.debug("No voiceprint enrolled — skipping biometric pass.")
        return []

    threshold = get_settings().voiceprint_threshold
    enrolled_vec = np.load(str(_voiceprint_path())).astype(np.float32)

    if model is None:
        model = _load_ecapa()

    tagged_ids: list[int] = []
    for seg_id, audio in zip(segment_ids, audio_slices):
        if audio is None or len(audio) == 0:
            continue
        try:
            vec = _extract_embedding(model, audio, sr)
            sim = _cosine_similarity(enrolled_vec, vec)
            if sim >= threshold:
                tagged_ids.append(seg_id)
        except Exception as exc:
            logger.warning("Biometric inference failed for segment %d: %s", seg_id, exc)

    if tagged_ids:
        _update_segment_tiers(tagged_ids, tier=2)
        logger.info(
            "Biometric pass: %d/%d segments tagged Tier 2 (threshold=%.2f)",
            len(tagged_ids), len(segment_ids), threshold,
        )

    return tagged_ids


# ── Internal utilities ────────────────────────────────────────────────────

def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file to a mono float32 NumPy waveform using ffmpeg."""
    from audiobench.transcribe.audio_converter import load_as_numpy
    # ECAPA-TDNN expects 16kHz mono audio
    target_sr = 16000
    try:
        wav = load_as_numpy(path, target_sr=target_sr, target_channels=1)
        return wav, target_sr
    except Exception as e:
        raise ImportError(f"Failed to load audio with ffmpeg: {e}") from e


def _retroactive_tag(model, enrolled_vec: np.ndarray) -> int:
    """Run biometric pass on all existing DB segments and tag matches.

    Called once after initial enrollment so historical recordings are
    correctly attributed before any new sessions start.

    Returns the number of segments tagged.
    """
    from audiobench.core.db_session import get_session as db_session
    from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord

    threshold = get_settings().voiceprint_threshold
    tagged_ids: list[int] = []

    logger.info("Running retroactive biometric pass on existing segments...")

    try:
        with db_session() as s:
            # Load segments that still have privacy_tier = 0 (untagged)
            segments = (
                s.query(SegmentRecord)
                .filter(SegmentRecord.privacy_tier == 0)
                .order_by(SegmentRecord.id)
                .all()
            )
            total = len(segments)
            logger.info("Retroactive pass: %d segments to evaluate", total)

            # Group by transcription → audio file for waveform loading
            tx_ids = list({seg.transcription_id for seg in segments})
            tx_map: dict[int, str] = {}  # tx_id → audio file path
            for tx_id in tx_ids:
                tx = s.query(TranscriptionRecord).filter_by(id=tx_id).first()
                if tx and tx.audio_file_id:
                    af = s.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                    if af:
                        tx_map[tx_id] = af.file_path

        # Process segments grouped by audio file to avoid reloading

        segments_by_tx = {}
        for seg in segments:
            segments_by_tx.setdefault(seg.transcription_id, []).append(seg)

        for tx_id, tx_segs in segments_by_tx.items():
            audio_path = tx_map.get(tx_id)
            if not audio_path or not Path(audio_path).exists():
                continue
            try:
                waveform, sr = _load_audio(Path(audio_path))
            except Exception as exc:
                logger.warning("Could not load audio for tx %d: %s", tx_id, exc)
                continue

            try:
                for seg in tx_segs:
                    try:
                        start_sample = int(seg.start_time * sr)
                        end_sample = int(seg.end_time * sr)
                        audio_slice = waveform[start_sample:end_sample]
                        if len(audio_slice) < sr * 0.5:  # skip < 0.5s slices
                            continue
                        vec = _extract_embedding(model, audio_slice, sr)
                        sim = _cosine_similarity(enrolled_vec, vec)
                        if sim >= threshold:
                            tagged_ids.append(seg.id)
                    except Exception as exc:
                        logger.debug("Retroactive tag failed for seg %d: %s", seg.id, exc)
            finally:
                del waveform
                import gc
                gc.collect()

    except Exception as exc:
        logger.error("Retroactive biometric pass failed: %s", exc)
        return 0

    if tagged_ids:
        _update_segment_tiers(tagged_ids, tier=2)

    logger.info("Retroactive pass complete: %d/%d segments tagged", len(tagged_ids), total)
    return len(tagged_ids)


def _update_segment_tiers(segment_ids: list[int], tier: int) -> None:
    """Bulk-update privacy_tier for a list of segment IDs and cascade to expressions."""
    if not segment_ids:
        return
    from audiobench.core.db_session import get_session as db_session
    from audiobench.storage.models import ExpressionRecord, SegmentRecord
    try:
        with db_session() as s:
            s.query(SegmentRecord).filter(
                SegmentRecord.id.in_(segment_ids)
            ).update({"privacy_tier": tier}, synchronize_session=False)

            # Cascade tier upgrade to associated audio_transcript expressions
            tx_ids = [
                row[0] for row in s.query(SegmentRecord.transcription_id)
                .filter(SegmentRecord.id.in_(segment_ids))
                .distinct()
                .all()
                if row[0] is not None
            ]
            if tx_ids:
                updated_exprs = s.query(ExpressionRecord).filter(
                    ExpressionRecord.source_type == "audio_transcript",
                    ExpressionRecord.source_id.in_(tx_ids),
                    ExpressionRecord.privacy_tier < tier,
                ).update({"privacy_tier": tier}, synchronize_session=False)
                if updated_exprs:
                    logger.info(
                        "Cascaded privacy_tier=%d to %d expressions for transcripts %s",
                        tier, updated_exprs, tx_ids,
                    )

            s.commit()
    except Exception as exc:
        logger.error("Failed to update segment tiers: %s", exc)
