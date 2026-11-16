"""Transcript parser for reverse import."""

import json
import re
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.stitcher import has_chunk_overlap_jumps, heal_stitched_transcript
from audiobench.transcribe.transcription_result import AudioMetadata, Segment, Transcript, Word

logger = get_logger("transcribe.parser")


class UnsupportedSchemaVersion(Exception):
    pass

class HashMismatchError(Exception):
    pass

SKIP_TRANSCRIPT = {"text", "word_count", "segment_count", "chapters", "audio", "segments"}
SKIP_SEGMENT = {"duration", "word_count"}
SKIP_WORD = {"duration", "midpoint"}

def parse_transcript_file(path: Path) -> tuple[Transcript, dict | None]:
    """Parse a transcript file. Returns (Transcript, raw_audio_dict | None)."""
    ext = path.suffix.lower()
    if ext == ".json":
        return _parse_json(path)
    elif ext == ".srt":
        return _parse_srt(path), None
    elif ext == ".vtt":
        return _parse_vtt(path), None
    elif ext == ".txt":
        return _parse_txt(path), None
    else:
        raise ValueError(f"Unsupported format: {ext}")

def _parse_json(path: Path) -> tuple[Transcript, dict | None]:
    data = json.loads(path.read_text(encoding="utf-8"))

    version = data.get("schema_version", "1")
    if str(version) not in ("1",):
        raise UnsupportedSchemaVersion(f"schema_version '{version}' not supported")

    # Auto-heal backward time jumps / unstitched chunk overlap boundaries
    if has_chunk_overlap_jumps(data.get("segments", [])):
        logger.info("Auto-healing overlapping chunk boundaries in %s", path.name)
        data = heal_stitched_transcript(data)

    segments = []
    for idx, raw_seg in enumerate(data.get("segments", [])):
        words = []
        for w in raw_seg.get("words", []):
            if isinstance(w, dict):
                w_data = {k: v for k, v in w.items() if k not in SKIP_WORD}
                if "end" not in w_data and "start" in w_data:
                    w_data["end"] = w_data["start"]
                words.append(Word(**w_data))
        seg_data = {k: v for k, v in raw_seg.items() if k not in SKIP_SEGMENT | {"words"}}
        if "id" not in seg_data:
            seg_data["id"] = idx
        segments.append(Segment(**seg_data, words=words))

    tx_data = {k: v for k, v in data.items() if k not in SKIP_TRANSCRIPT}
    transcript = Transcript(**tx_data, segments=segments)

    return transcript, data.get("audio") or data.get("raw_audio")

def _srt_time_to_seconds(time_str: str) -> float:
    # 00:12:47,123
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def _parse_srt(path: Path) -> Transcript:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', content.strip())

    segments = []

    for i, block in enumerate(blocks):
        lines = block.splitlines()
        if len(lines) >= 3:
            time_match = re.search(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', lines[1])
            if time_match:
                start = _srt_time_to_seconds(time_match.group(1))
                end = _srt_time_to_seconds(time_match.group(2))
                text = " ".join(lines[2:])
                segments.append(Segment(id=i, text=text, start=start, end=end))

    duration = segments[-1].end if segments else 0.0

    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration_seconds=duration,
        engine="external",
        model_name="unknown",
        speaker_map={},
        chapters=[]
    )

def _parse_vtt(path: Path) -> Transcript:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', content.strip())

    segments = []
    idx = 0

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("WEBVTT") or lines[0].startswith("NOTE"):
            continue

        ts_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_idx = i
                break

        if ts_idx >= 0:
            time_match = re.search(r'([\d:.,]+)\s*-->\s*([\d:.,]+)', lines[ts_idx])
            if time_match:
                s_str = time_match.group(1).replace(",", ".")
                e_str = time_match.group(2).replace(",", ".")
                def _to_secs(t: str) -> float:
                    parts = t.split(":")
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    elif len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(t)

                start = _to_secs(s_str)
                end = _to_secs(e_str)
                text = " ".join(lines[ts_idx + 1:])
                if text:
                    segments.append(Segment(id=idx, text=text, start=start, end=end))
                    idx += 1

    duration = segments[-1].end if segments else 0.0

    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration_seconds=duration,
        engine="external",
        model_name="unknown",
        speaker_map={},
        chapters=[],
    )

def _parse_txt(path: Path) -> Transcript:
    text = path.read_text(encoding="utf-8").strip()
    segments = []
    if text:
        segments.append(Segment(id=0, text=text, start=0.0, end=0.0))

    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration_seconds=0.0,
        engine="external",
        model_name="unknown",
        speaker_map={},
        chapters=[]
    )

def normalize_stem_strict(stem: str) -> str:
    """Normalize a stem name while strictly preserving numeric token boundaries.

    1. Lowercase and replace non-alphanumeric characters with whitespace.
    2. Compress multiple whitespace into a single space.
    3. Normalize leading zeros on standalone numeric tokens (e.g. '06' -> '6', '001' -> '1')
       WITHOUT collapsing adjacent numbers ('6' != '16' != '60').
    """
    s = re.sub(r"[^a-z0-9]+", " ", stem.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\b0+([0-9]+)\b", r"\1", s)


def evaluate_tag_modified_heuristic(
    local_audio_path: Path,
    raw_audio: dict | None,
    tx_duration: float | None = None,
) -> tuple[bool, dict]:
    """Evaluate whether an audio file with a SHA-256 hash mismatch qualifies as tag-modified.

    Requires 4 independent invariants to pass simultaneously:
    1. Exact / strict normalized stem match (e.g. 'Chapter 6' == 'Chapter 06',
       but 'Chapter 6' != 'Chapter 16' or 'Chapter 60').
    2. Relative duration tolerance:
       rel_diff <= 0.002 (0.2%) AND abs_diff <= 0.50s.
    3. File size ceiling:
       abs_size_diff <= max(2 * 1024 * 1024, 0.01 * json_size) (ID3 tag boundary <= 2MB).
    4. Audio stream probe:
       Local file must probe cleanly with duration > 0.

    Returns:
        (passes_heuristic: bool, metrics: dict)
    """
    metrics = {
        "stem_matched": False,
        "duration_passed": False,
        "size_passed": False,
        "probe_passed": False,
        "duration_delta_sec": None,
        "relative_duration_delta": None,
        "size_delta_bytes": None,
        "expected_hash": (raw_audio or {}).get("file_hash") if raw_audio else None,
        "local_hash": None,
        "reason": "",
    }

    if not raw_audio or not local_audio_path.exists():
        metrics["reason"] = "Missing raw_audio metadata or local audio file"
        return False, metrics

    # 1. Stem name invariant
    expected_name = raw_audio.get("file_name", local_audio_path.name)
    norm_local = normalize_stem_strict(local_audio_path.stem)
    norm_expected = normalize_stem_strict(Path(expected_name).stem)
    if norm_local != norm_expected:
        metrics["reason"] = f"Stem name mismatch ('{norm_local}' != '{norm_expected}')"
        return False, metrics
    metrics["stem_matched"] = True

    # 2. Probe local audio
    from audiobench.transcribe.audio_converter import probe

    try:
        info = probe(local_audio_path)
    except Exception as probe_err:
        metrics["reason"] = f"Failed to probe local audio file: {probe_err}"
        return False, metrics

    if not info or info.duration <= 0:
        metrics["reason"] = "Failed to probe local audio file duration"
        return False, metrics
    metrics["probe_passed"] = True
    local_duration = float(info.duration)

    # Expected duration: prefer raw_audio['duration_seconds'], fallback to tx_duration
    expected_duration = float(raw_audio.get("duration_seconds") or tx_duration or 0.0)
    if expected_duration <= 0:
        metrics["reason"] = "Transcript/JSON lacks valid duration_seconds"
        return False, metrics

    # 3. Relative duration tolerance check
    abs_dur_diff = abs(local_duration - expected_duration)
    rel_dur_diff = abs_dur_diff / max(local_duration, expected_duration)
    metrics["duration_delta_sec"] = round(abs_dur_diff, 4)
    metrics["relative_duration_delta"] = round(rel_dur_diff, 6)

    # Invariant: <= 0.2% relative AND <= 0.50s absolute
    if not (rel_dur_diff <= 0.002 and abs_dur_diff <= 0.50):
        metrics["reason"] = (
            f"Duration drift exceeds tolerance: Δ {abs_dur_diff:.3f}s "
            f"({rel_dur_diff:.2%}), allowed <= 0.2% and <= 0.50s"
        )
        return False, metrics
    metrics["duration_passed"] = True

    # 4. File size ceiling check (ID3 tag boundary)
    local_size = local_audio_path.stat().st_size
    expected_size = raw_audio.get("file_size_bytes")
    if expected_size is not None:
        abs_size_diff = abs(local_size - int(expected_size))
        metrics["size_delta_bytes"] = abs_size_diff
        max_allowed_size_diff = max(2 * 1024 * 1024, int(0.01 * int(expected_size)))
        if abs_size_diff > max_allowed_size_diff:
            metrics["reason"] = (
                f"File size delta {abs_size_diff:,} bytes exceeds ID3 tag ceiling "
                f"{max_allowed_size_diff:,} bytes"
            )
            return False, metrics
    metrics["size_passed"] = True

    local_hash = AudioMetadata.compute_file_hash(local_audio_path)
    metrics["local_hash"] = local_hash
    metrics["reason"] = "TAG_MODIFIED_VERIFIED"
    return True, metrics


def build_audio_metadata(
    local_mp3_path: Path,
    raw_audio: dict | None,
    allow_hash_mismatch: bool = False,
) -> AudioMetadata:
    """Construct AudioMetadata for the local file.

    If raw_audio contains a file_hash and it does not match the local file hash:
    - If allow_hash_mismatch is False, raises HashMismatchError.
    - If allow_hash_mismatch is True, accepts the local audio and uses the local file hash.
    """
    local_hash = AudioMetadata.compute_file_hash(local_mp3_path)

    if raw_audio:
        json_hash = raw_audio.get("file_hash")
        if json_hash and json_hash != local_hash and not allow_hash_mismatch:
            raise HashMismatchError(
                f"Hash mismatch!\n"
                f"  JSON:  {json_hash}\n"
                f"  Local: {local_hash}\n"
                f"Wrong file selected."
            )

        duration = raw_audio.get("duration_seconds")
        if duration is None:
            # Fallback to ffprobe if JSON lacks metadata somehow
            from audiobench.transcribe.audio_converter import probe_metadata
            meta = probe_metadata(str(local_mp3_path))
            duration = meta.duration_seconds

        return AudioMetadata(
            file_path=str(local_mp3_path.absolute()),
            file_name=raw_audio.get("file_name", local_mp3_path.name),
            file_size_bytes=local_mp3_path.stat().st_size,
            format=raw_audio.get("format", local_mp3_path.suffix.lstrip(".")),
            duration_seconds=duration,
            sample_rate=raw_audio.get("sample_rate", 44100),
            channels=raw_audio.get("channels", 2),
            file_hash=local_hash,
        )
    else:
        # Complete fallback
        from audiobench.transcribe.audio_converter import probe_metadata
        meta = probe_metadata(str(local_mp3_path))
        return AudioMetadata(
            file_path=str(local_mp3_path.absolute()),
            file_name=local_mp3_path.name,
            file_size_bytes=local_mp3_path.stat().st_size,
            format=local_mp3_path.suffix.lstrip("."),
            duration_seconds=meta.duration_seconds,
            sample_rate=meta.sample_rate,
            channels=meta.channels,
            file_hash=local_hash,
        )

