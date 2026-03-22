"""FFmpeg/FFprobe audio loader -- direct subprocess wrapper.

Replaces pydub with direct ffmpeg calls for:
- Instant metadata via ffprobe (no file loading)
- Piped conversion to float32 numpy (no temp files)
- Optional audio filter graphs (noise reduction, normalization)

System requirements: ffmpeg and ffprobe on PATH.

Usage:
    from audiobench.transcribe.audio_converter import AudioLoader

    with AudioLoader() as loader:
        wav_path, metadata = loader.load("/path/to/meeting.m4a")

    # Metadata only (instant):
    info = probe("/path/to/meeting.m4a")

    # With noise reduction filters:
    with AudioLoader() as loader:
        wav_path, metadata = loader.load(
            "/path/to/noisy.m4a",
            filters=["highpass=f=200", "afftdn=nf=-25", "dynaudnorm=p=0.9"],
        )
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audiobench.core.error_types import AudioLoadError, UnsupportedFormatError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.transcription_result import AudioMetadata

logger = get_logger("core.ffmpeg")

# Default filter preset for --enhance
ENHANCE_FILTERS = [
    "highpass=f=200",  # remove low-frequency rumble (HVAC, traffic)
    "afftdn=nf=-25",  # adaptive noise reduction (FFT-based)
    "loudnorm=I=-16:LRA=11:tp=-1.5",  # EBU R128 loudness normalization
]

# Silence trimming preset for --trim
# Forward trim → reverse → trim → reverse removes both leading AND trailing silence
TRIM_FILTERS = [
    "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-35dB:detection=rms",
    "areverse",
    "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-35dB:detection=rms",
    "areverse",
]

# ── Neural noise reduction (RNNoise / arnndn) ──────────────

RNNOISE_MODEL_URLS = [
    # GitHub API with raw body — most reliable (works even when CDN is blocked)
    "https://api.github.com/repos/richardpl/arnndn-models/contents/bd.rnnn",
    # Direct raw content — faster when CDN is reachable
    "https://raw.githubusercontent.com/richardpl/arnndn-models/master/bd.rnnn",
]
RNNOISE_MODEL_DIR = Path.home() / ".audiobench" / "models" / "rnnoise"
RNNOISE_MODEL_NAME = "bd.rnnn"  # beguiling-drafter — best general-purpose speech model
RNNOISE_MODEL_SIZE = 299693  # expected size in bytes


def _download_rnnoise_model() -> None:
    """Download RNNoise model (~293KB) to ~/.audiobench/models/rnnoise/.

    Tries multiple URLs with a 15-second timeout each.
    Uses GitHub API with raw accept header for reliable download.
    """
    RNNOISE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = RNNOISE_MODEL_DIR / RNNOISE_MODEL_NAME

    errors = []
    for url in RNNOISE_MODEL_URLS:
        logger.info("Downloading RNNoise model from: %s", url)
        try:
            req = urllib.request.Request(url)
            # GitHub API needs Accept header to return raw binary
            if "api.github.com" in url:
                req.add_header("Accept", "application/vnd.github.v3.raw")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) < 10000:
                errors.append(f"{url}: file too small ({len(data)} bytes)")
                continue
            dest.write_bytes(data)
            logger.info("RNNoise model saved: %s (%d bytes)", dest, len(data))
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
            logger.warning("Download failed from %s: %s", url, e)

    # All URLs failed — clean up empty file if present
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()

    raise AudioLoadError(
        "rnnoise",
        f"Failed to download RNNoise model from all sources. "
        f"Errors: {'; '.join(errors)}. "
        f"Manually download a .rnnn model and place at: {dest}",
    )


def get_denoise_filters() -> list[str]:
    """Get arnndn filter list with auto-downloaded RNNoise model.

    Downloads the model (~293KB) on first use.

    Returns:
        Filter list for ffmpeg -af chain.
    """
    model_path = RNNOISE_MODEL_DIR / RNNOISE_MODEL_NAME
    if not model_path.exists():
        _download_rnnoise_model()
    return [f"arnndn=m={model_path}:mix=0.85"]


def build_filter_chain(
    *,
    enhance: bool = False,
    denoise: bool = False,
    trim: bool = False,
    custom: str | None = None,
) -> list[str] | None:
    """Build an optimal ffmpeg audio filter chain.

    Encodes empirically-tested filter ordering rules:
    1. highpass first (when any cleaning is active)
    2. Neural denoise (arnndn) supersedes spectral denoise (afftdn)
    3. Silence trimming after denoising, before normalization
    4. Loudness normalization always last

    Ordering: highpass → [arnndn | afftdn] → silenceremove → loudnorm

    Args:
        enhance: Apply noise reduction + loudness normalization.
        denoise: Apply neural noise reduction (RNNoise arnndn).
        trim: Remove leading/trailing silence.
        custom: Additional user-specified ffmpeg filter graph.

    Returns:
        Filter list for ffmpeg -af, or None if no processing requested.
    """
    filters: list[str] = []

    # ── Stage 1: Highpass (always first when cleaning) ──
    if enhance or denoise:
        filters.append("highpass=f=200")

    # ── Stage 2: Noise reduction (neural wins over spectral) ──
    if denoise:
        # arnndn supersedes afftdn — no double denoising
        filters.extend(get_denoise_filters())
    elif enhance:
        # Spectral cleaning only when neural is not active
        filters.append("afftdn=nf=-25")

    # ── Stage 3: Silence trimming (after denoise, before loudnorm) ──
    if trim:
        filters.extend(TRIM_FILTERS)

    # ── Stage 4: Loudness normalization (always last) ──
    if enhance or denoise:
        filters.append("loudnorm=I=-16:LRA=11:tp=-1.5")

    # ── Custom filters appended at end ──
    if custom:
        filters.extend(custom.split(","))

    return filters if filters else None


# ── Format conversion codec defaults ───────────────────────

CODEC_DEFAULTS: dict[str, list[str]] = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "opus": ["-c:a", "libopus", "-b:a", "64k"],
    "ogg": ["-c:a", "libvorbis", "-q:a", "5"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
    "aac": ["-c:a", "aac", "-b:a", "192k"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
}

# Formats ffmpeg can handle
SUPPORTED_AUDIO_FORMATS = {
    "m4a",
    "mp3",
    "wav",
    "flac",
    "ogg",
    "aac",
    "wma",
    "opus",
    "aiff",
    "webm",
    "amr",
    "oga",
    "spx",
}

SUPPORTED_VIDEO_FORMATS = {
    "mp4",
    "mkv",
    "avi",
    "mov",
    "m4v",
    "webm",
    "flv",
    "wmv",
}

ALL_SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS


def _check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are available on PATH."""
    if not shutil.which("ffmpeg"):
        raise AudioLoadError(
            "ffmpeg",
            "ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html",
        )
    if not shutil.which("ffprobe"):
        raise AudioLoadError(
            "ffprobe",
            "ffprobe not found on PATH. Install it: https://ffmpeg.org/download.html",
        )


@dataclass
class AudioInfo:
    """Metadata from ffprobe."""

    duration: float  # seconds
    sample_rate: int
    channels: int
    codec: str
    bitrate: int  # bits per second (0 if unknown)
    format_name: str  # container format


def probe(file_path: str | Path) -> AudioInfo:
    """Get audio metadata via ffprobe. Instant, no file loading.

    Args:
        file_path: Path to audio/video file.

    Returns:
        AudioInfo with duration, sample rate, channels, codec, bitrate.

    Raises:
        AudioLoadError: If ffprobe fails or no audio stream found.
    """
    _check_ffmpeg()
    file_path = str(file_path)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "a:0",
        file_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioLoadError(file_path, f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AudioLoadError(file_path, f"ffprobe returned invalid JSON: {e}") from e

    streams = data.get("streams", [])
    if not streams:
        raise AudioLoadError(file_path, "No audio stream found")

    stream = streams[0]
    fmt = data.get("format", {})

    # Duration: prefer stream duration, fall back to format duration
    duration = float(stream.get("duration") or fmt.get("duration") or 0)

    return AudioInfo(
        duration=duration,
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        codec=stream.get("codec_name", "unknown"),
        bitrate=int(stream.get("bit_rate") or fmt.get("bit_rate") or 0),
        format_name=fmt.get("format_name", "unknown"),
    )


def load_as_numpy(
    file_path: str | Path,
    filters: list[str] | None = None,
    target_sr: int = 16000,
    target_channels: int = 1,
) -> np.ndarray:
    """Convert any audio/video to a float32 numpy array via ffmpeg pipe.

    No temp files. Pipes ffmpeg stdout directly into memory.

    Args:
        file_path: Path to audio/video file.
        filters: Optional ffmpeg audio filter list (joined with ',').
        target_sr: Target sample rate (default: 16000 for Whisper).
        target_channels: Target channel count (default: 1 mono).

    Returns:
        float32 numpy array, normalized to [-1.0, 1.0].

    Raises:
        AudioLoadError: If ffmpeg fails.
    """
    _check_ffmpeg()
    file_path = str(file_path)

    cmd = [
        "ffmpeg",
        "-i",
        file_path,
        "-vn",  # strip video
        "-f",
        "s16le",  # raw PCM 16-bit little-endian
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        str(target_channels),
    ]

    if filters:
        cmd += ["-af", ",".join(filters)]

    cmd.append("-")  # output to stdout (pipe)

    logger.info("ffmpeg command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        # Extract the last meaningful line from ffmpeg stderr
        lines = [line for line in stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(file_path, f"ffmpeg conversion failed: {error_msg}")

    if not result.stdout:
        raise AudioLoadError(file_path, "ffmpeg produced no output")

    # Convert raw PCM bytes to float32 numpy array
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    logger.info(
        "Loaded %d samples (%.1fs at %dHz)",
        len(audio),
        len(audio) / target_sr,
        target_sr,
    )

    return audio


def embed_subtitles(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    hard_burn: bool = False,
) -> Path:
    """Embed subtitles into a video file using ffmpeg.

    Args:
        video_path: Path to source video file.
        subtitle_path: Path to SRT or VTT subtitle file.
        output_path: Path for output video file.
        hard_burn: If True, render subtitles into the video pixels (permanent).
                   If False, add as a selectable subtitle track (soft embed).

    Returns:
        Path to the output video file.

    Raises:
        AudioLoadError: If ffmpeg fails or input files are invalid.
    """
    _check_ffmpeg()
    video_path = Path(video_path).resolve()
    subtitle_path = Path(subtitle_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.exists():
        raise AudioLoadError(str(video_path), "Video file does not exist")
    if not subtitle_path.exists():
        raise AudioLoadError(str(subtitle_path), "Subtitle file does not exist")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hard_burn:
        # Hard burn: render subtitles into video pixels (permanent, any container)
        # Requires re-encoding the video stream
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={str(subtitle_path)}",
            "-c:a",
            "copy",  # copy audio as-is
            str(output_path),
        ]
    else:
        # Soft embed: mux subtitle track into container (selectable, no re-encode)
        # Works with mp4 (mov_text), mkv (srt/ass), webm (webvtt)
        ext = output_path.suffix.lstrip(".").lower()
        sub_codec = "mov_text" if ext == "mp4" else "srt"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(subtitle_path),
            "-c",
            "copy",  # copy all streams
            "-c:s",
            sub_codec,  # subtitle codec for container
            str(output_path),
        ]

    logger.info("ffmpeg subtitle embed: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(str(video_path), f"Subtitle embedding failed: {error_msg}")

    logger.info(
        "Subtitles embedded: %s → %s (mode=%s)",
        video_path.name,
        output_path.name,
        "hard_burn" if hard_burn else "soft_embed",
    )

    return output_path


# ── Audio analysis ─────────────────────────────────────────


@dataclass
class SilenceRegion:
    """A detected silence region in audio."""

    start: float
    end: float
    duration: float


@dataclass
class AudioAnalysis:
    """Complete audio analysis results."""

    duration: float
    silence_regions: list[SilenceRegion] = field(default_factory=list)
    silence_total: float = 0.0
    silence_percent: float = 0.0
    speech_percent: float = 100.0
    integrated_loudness: float = 0.0  # LUFS
    loudness_range: float = 0.0  # LU
    true_peak: float = 0.0  # dBTP
    loudness_threshold: float = 0.0  # dBTP


def silence_detect(
    file_path: str | Path,
    noise: str = "-35dB",
    duration: float = 0.5,
) -> list[SilenceRegion]:
    """Detect silence regions in an audio file via ffmpeg silencedetect.

    Args:
        file_path: Path to audio/video file.
        noise: Noise threshold (e.g. '-35dB'). Lower = more sensitive.
        duration: Minimum silence duration in seconds.

    Returns:
        List of SilenceRegion with start, end, and duration.
    """
    _check_ffmpeg()
    cmd = [
        "ffmpeg",
        "-i",
        str(file_path),
        "-af",
        f"silencedetect=noise={noise}:d={duration}",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse silencedetect output from stderr
    regions: list[SilenceRegion] = []
    starts: list[float] = []

    for match in re.finditer(r"silence_start:\s*([\d.]+)", stderr):
        starts.append(float(match.group(1)))

    for match in re.finditer(
        r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", stderr
    ):
        end = float(match.group(1))
        dur = float(match.group(2))
        start = starts.pop(0) if starts else end - dur
        regions.append(
            SilenceRegion(start=round(start, 3), end=round(end, 3), duration=round(dur, 3))
        )

    logger.info("Detected %d silence regions in %s", len(regions), file_path)
    return regions


def loudness_analyze(file_path: str | Path) -> dict:
    """Analyze audio loudness using EBU R128 via ffmpeg loudnorm.

    Args:
        file_path: Path to audio/video file.

    Returns:
        Dict with keys: input_i (LUFS), input_lra (LU),
        input_tp (dBTP), input_thresh (dBTP).
    """
    _check_ffmpeg()
    cmd = [
        "ffmpeg",
        "-i",
        str(file_path),
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # loudnorm outputs JSON at the end of stderr
    json_match = re.search(r"\{[^}]*\"input_i\"[^}]*\}", stderr, re.DOTALL)
    if not json_match:
        logger.warning("Could not parse loudnorm output for %s", file_path)
        return {}

    try:
        data = json.loads(json_match.group())
        logger.info(
            "Loudness: %s LUFS, LRA=%s LU, TP=%s dBTP",
            data.get("input_i"),
            data.get("input_lra"),
            data.get("input_tp"),
        )
        return data
    except json.JSONDecodeError:
        logger.warning("Failed to parse loudnorm JSON for %s", file_path)
        return {}


def analyze_audio(file_path: str | Path) -> AudioAnalysis:
    """Full audio analysis: silence detection + loudness statistics.

    Combines silencedetect and loudnorm in a single report.

    Args:
        file_path: Path to audio/video file.

    Returns:
        AudioAnalysis with silence map and loudness stats.
    """
    file_path = Path(file_path).resolve()

    # Get file duration from probe
    info = probe(file_path)
    total_duration = info.duration

    # Silence detection
    regions = silence_detect(file_path)
    silence_total = sum(r.duration for r in regions)
    silence_pct = (silence_total / total_duration * 100) if total_duration > 0 else 0

    # Loudness analysis
    loudness = loudness_analyze(file_path)

    def _safe_float(val: str | None, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return AudioAnalysis(
        duration=total_duration,
        silence_regions=regions,
        silence_total=round(silence_total, 2),
        silence_percent=round(silence_pct, 1),
        speech_percent=round(100 - silence_pct, 1),
        integrated_loudness=_safe_float(loudness.get("input_i")),
        loudness_range=_safe_float(loudness.get("input_lra")),
        true_peak=_safe_float(loudness.get("input_tp")),
        loudness_threshold=_safe_float(loudness.get("input_thresh")),
    )


# ── Format conversion ─────────────────────────────────────


def convert_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bitrate: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    speed: float | None = None,
    filters: list[str] | None = None,
) -> Path:
    """Convert audio between formats with optional processing.

    Automatically selects codec based on output extension.
    Supports speed change via rubberband (preferred) or atempo.

    Args:
        input_path: Source audio/video file.
        output_path: Destination file (format inferred from extension).
        bitrate: Override default bitrate (e.g. '128k').
        sample_rate: Target sample rate in Hz.
        channels: Target channel count.
        speed: Playback speed multiplier (e.g. 1.5 for 50% faster).
        filters: Additional ffmpeg audio filters.

    Returns:
        Path to the output file.

    Raises:
        AudioLoadError: If conversion fails.
    """
    _check_ffmpeg()
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lstrip(".").lower()

    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vn"]

    # Codec selection
    codec_args = CODEC_DEFAULTS.get(ext, [])
    if codec_args:
        cmd.extend(codec_args)

    # Override bitrate if specified
    if bitrate:
        # Remove existing bitrate args and add override
        cmd = [c for i, c in enumerate(cmd) if not (i > 0 and cmd[i - 1] == "-b:a")]
        cmd = [c for c in cmd if c not in ["-b:a"]]
        cmd.extend(["-b:a", bitrate])

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    if channels:
        cmd.extend(["-ac", str(channels)])

    # Build filter chain
    af_filters = list(filters) if filters else []

    if speed and speed != 1.0:
        # Prefer rubberband for quality (confirmed compiled on system)
        af_filters.append(f"rubberband=tempo={speed}")

    if af_filters:
        cmd.extend(["-af", ",".join(af_filters)])

    cmd.append(str(output_path))

    logger.info("ffmpeg convert: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(str(input_path), f"Audio conversion failed: {error_msg}")

    logger.info("Converted: %s → %s", input_path.name, output_path.name)
    return output_path


def concat_audio(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Concatenate multiple audio files using ffmpeg concat demuxer.

    All input files should have compatible codecs/sample rates for
    lossless concatenation. If they differ, ffmpeg will transcode.

    Args:
        input_paths: List of audio file paths (in order).
        output_path: Destination file.

    Returns:
        Path to the concatenated output file.

    Raises:
        AudioLoadError: If concatenation fails.
    """
    _check_ffmpeg()
    if len(input_paths) < 2:
        raise AudioLoadError("concat", "Need at least 2 files to concatenate")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary concat list file
    fd, list_path = tempfile.mkstemp(prefix="audiobench_concat_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            for p in input_paths:
                resolved = Path(p).resolve()
                f.write(f"file '{resolved}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            str(output_path),
        ]

        logger.info("ffmpeg concat: %d files → %s", len(input_paths), output_path.name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            lines = [line for line in result.stderr.splitlines() if line.strip()]
            error_msg = lines[-1] if lines else "unknown error"
            raise AudioLoadError("concat", f"Audio concatenation failed: {error_msg}")
    finally:
        os.unlink(list_path)

    logger.info("Concatenated %d files → %s", len(input_paths), output_path.name)
    return output_path


def split_audio(
    input_path: str | Path,
    chunk_duration: float = 900.0,
    overlap: float = 30.0,
    output_dir: str | Path | None = None,
) -> list[tuple[Path, float]]:
    """Split an audio file into overlapping chunks using ffmpeg.

    Uses stream-copy (-c copy) for speed — no re-encoding.

    Args:
        input_path: Source audio file.
        chunk_duration: Duration of each chunk in seconds (default: 900 = 15 min).
        overlap: Overlap between consecutive chunks in seconds (default: 30).
        output_dir: Directory for chunk files.  If None, uses a temp directory
                    (caller is responsible for cleanup).

    Returns:
        List of (chunk_path, time_offset) tuples, where time_offset is the
        start time of the chunk in the original audio.

    Raises:
        AudioLoadError: If ffmpeg fails.
    """
    _check_ffmpeg()
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise AudioLoadError(str(input_path), "File does not exist")

    info = probe(input_path)
    total_duration = info.duration

    if total_duration <= chunk_duration:
        # No splitting needed — return the original file as-is.
        return [(input_path, 0.0)]

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="audiobench_chunks_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ext = input_path.suffix  # keep original container format
    step = chunk_duration - overlap
    chunks: list[tuple[Path, float]] = []
    idx = 0
    offset = 0.0

    while offset < total_duration:
        # Last chunk: extend to the end of the file.
        remaining = total_duration - offset
        dur = min(chunk_duration, remaining)
        if remaining - chunk_duration < overlap:
            # Not enough room for another chunk — extend this one to EOF.
            dur = remaining

        chunk_path = output_dir / f"chunk_{idx:03d}{ext}"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(offset),
            "-t", str(dur),
            "-i", str(input_path),
            "-c", "copy",
            str(chunk_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
            error_msg = lines[-1] if lines else "unknown error"
            raise AudioLoadError(
                str(input_path),
                f"Audio splitting failed on chunk {idx}: {error_msg}",
            )

        chunks.append((chunk_path, offset))
        logger.info(
            "Split chunk %d: offset=%.1fs, duration=%.1fs → %s",
            idx, offset, dur, chunk_path.name,
        )

        idx += 1
        offset += step

        # If this chunk reached EOF, we're done.
        if offset >= total_duration or dur >= remaining:
            break

    logger.info("Split %s into %d chunks", input_path.name, len(chunks))
    return chunks


# ── Visual inspection ──────────────────────────────────────


def generate_waveform(
    file_path: str | Path,
    output_path: str | Path,
    width: int = 1920,
    height: int = 200,
    color: str = "#4488ff",
) -> Path:
    """Generate a waveform PNG image from an audio file.

    Args:
        file_path: Source audio file.
        output_path: Destination PNG file.
        width: Image width in pixels.
        height: Image height in pixels.
        color: Waveform color (hex).

    Returns:
        Path to the generated image.
    """
    _check_ffmpeg()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-filter_complex",
        f"showwavespic=s={width}x{height}:colors={color}",
        "-frames:v",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(str(file_path), f"Waveform generation failed: {error_msg}")

    logger.info("Waveform generated: %s", output_path)
    return output_path


def generate_spectrogram(
    file_path: str | Path,
    output_path: str | Path,
    width: int = 1920,
    height: int = 512,
) -> Path:
    """Generate a spectrogram PNG image from an audio file.

    Args:
        file_path: Source audio file.
        output_path: Destination PNG file.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Path to the generated image.
    """
    _check_ffmpeg()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-lavfi",
        f"showspectrumpic=s={width}x{height}:legend=1:fscale=log",
        "-frames:v",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(str(file_path), f"Spectrogram generation failed: {error_msg}")

    logger.info("Spectrogram generated: %s", output_path)
    return output_path


class AudioLoader:
    """High-level audio loader with same interface as the old pydub-based one.

    Converts any supported format to 16kHz mono WAV for Whisper.
    Uses direct ffmpeg pipes -- no temp files for the conversion itself,
    but writes a temp WAV for the engine (faster-whisper expects a file path).

    Usage:
        with AudioLoader() as loader:
            wav_path, metadata = loader.load("meeting.m4a")
            wav_path, metadata = loader.load("noisy.mp3", filters=ENHANCE_FILTERS)
    """

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1

    def __init__(self, temp_dir: str | None = None) -> None:
        self._temp_dir = temp_dir
        self._temp_files: list[str] = []

    def load(
        self,
        file_path: str | Path,
        filters: list[str] | None = None,
    ) -> tuple[str, AudioMetadata]:
        """Load an audio file and convert to 16kHz mono WAV.

        Args:
            file_path: Path to audio/video file.
            filters: Optional ffmpeg audio filter list for preprocessing.

        Returns:
            Tuple of (path to WAV file, AudioMetadata).

        Raises:
            AudioLoadError: If the file cannot be loaded.
            UnsupportedFormatError: If the format is not supported.
        """
        file_path = Path(file_path).resolve()
        logger.info("Loading audio: %s", file_path.name)

        # Validate
        if not file_path.exists():
            raise AudioLoadError(str(file_path), "File does not exist")
        if not file_path.is_file():
            raise AudioLoadError(str(file_path), "Path is not a file")

        ext = file_path.suffix.lstrip(".").lower()
        if not ext:
            raise AudioLoadError(str(file_path), "File has no extension")
        if ext not in ALL_SUPPORTED_FORMATS:
            raise UnsupportedFormatError(str(file_path), ext)

        # Probe metadata (instant, no loading)
        info = probe(file_path)

        metadata = AudioMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            format=ext,
            duration_seconds=round(info.duration, 3),
            sample_rate=info.sample_rate,
            channels=info.channels,
            file_hash=AudioMetadata.compute_file_hash(file_path),
        )

        logger.info(
            "Audio probed: %s, duration=%.1fs, codec=%s, rate=%dHz, channels=%d",
            file_path.name,
            info.duration,
            info.codec,
            info.sample_rate,
            info.channels,
        )

        # Convert to numpy via ffmpeg pipe
        audio_array = load_as_numpy(
            file_path,
            filters=filters,
            target_sr=self.TARGET_SAMPLE_RATE,
            target_channels=self.TARGET_CHANNELS,
        )

        # Write temp WAV for faster-whisper (it expects a file path)
        wav_path = self._write_wav(audio_array, file_path.stem)

        logger.info("Converted to 16kHz mono WAV: %s", wav_path)
        return wav_path, metadata

    def _write_wav(self, audio: np.ndarray, stem: str) -> str:
        """Write float32 numpy array to a temp WAV file."""
        import wave

        fd, wav_path = tempfile.mkstemp(
            prefix=f"audiobench_{stem}_",
            suffix=".wav",
            dir=self._temp_dir,
        )
        os.close(fd)

        # Convert float32 back to int16 for WAV
        pcm_data = (audio * 32767).astype(np.int16).tobytes()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.TARGET_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm_data)

        self._temp_files.append(wav_path)
        return wav_path

    def cleanup(self) -> None:
        """Remove temporary WAV files."""
        for path in self._temp_files:
            try:
                os.unlink(path)
                logger.debug("Cleaned up: %s", path)
            except OSError:
                pass
        self._temp_files.clear()

    @staticmethod
    def get_supported_formats() -> dict[str, set[str]]:
        """Return supported formats grouped by type."""
        return {
            "audio": SUPPORTED_AUDIO_FORMATS,
            "video": SUPPORTED_VIDEO_FORMATS,
        }

    def __enter__(self) -> AudioLoader:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()
