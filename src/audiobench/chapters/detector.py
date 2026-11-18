"""Chapter detection and extraction engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from audiobench.chapters.cue_parser import ChapterInfo, CueParser
from audiobench.core.logger_factory import get_logger

logger = get_logger("chapters.detector")


class ChapterDetector:
    """Detects chapters in audio files using ffprobe or .cue sidecars."""

    def __init__(self):
        self.cue_parser = CueParser()

    def detect(self, file_path: Path) -> list[ChapterInfo]:
        """Detect chapters for a given audio file.

        Attempts the following in order:
        1. ffprobe -show_chapters
        2. .cue sidecar file (if ffprobe finds 0 chapters)
        3. Fallback to a single "Full Recording" chapter
        """
        if not file_path.exists():
            logger.error("Cannot detect chapters for non-existent file: %s", file_path)
            return []

        total_duration = self._get_file_duration(file_path)

        # 1. Try ffprobe
        chapters = self._run_ffprobe(file_path)
        if chapters:
            logger.info("Found %d embedded chapters in %s", len(chapters), file_path.name)
            return chapters

        # 2. Try .cue sidecar
        cue_path = file_path.with_suffix(".cue")
        if cue_path.exists():
            chapters = self.cue_parser.parse(cue_path, total_duration)
            if chapters:
                return chapters

        # Also check if there's a cue file with the exact same name but .cue appended
        # e.g. "audio.mp3.cue" instead of "audio.cue"
        alt_cue_path = file_path.with_name(file_path.name + ".cue")
        if alt_cue_path.exists():
            chapters = self.cue_parser.parse(alt_cue_path, total_duration)
            if chapters:
                return chapters

        # 3. Fallback: single chapter (hydrating title from tags if available)
        logger.debug("No chapters found for %s, treating as single chapter", file_path.name)
        title = "Full Recording"
        try:
            from audiobench.transcribe.audio_converter import probe_tags
            tags = probe_tags(file_path)
            raw_title = tags.get("title")
            if raw_title and raw_title.strip():
                clean_title = raw_title.strip()
                if clean_title.lower() not in {"full recording", "untitled", "unknown"}:
                    title = clean_title
        except Exception as e:
            logger.debug("Failed to read format tags for fallback chapter title: %s", e)

        return [
            ChapterInfo(
                index=0,
                title=title,
                start_time=0.0,
                end_time=total_duration,
                is_ghost=False,
            )
        ]

    def _run_ffprobe(self, file_path: Path) -> list[ChapterInfo]:
        """Run ffprobe to extract embedded chapter metadata."""
        cmd = ["ffprobe", "-v", "error", "-show_chapters", "-print_format", "json", str(file_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            raw_chapters = data.get("chapters", [])
            if not raw_chapters:
                return []

            parsed = []
            for i, chap in enumerate(raw_chapters):
                # ffprobe usually gives start_time/end_time as strings representing floats
                start_time = float(chap.get("start_time", 0.0))
                end_time = float(chap.get("end_time", 0.0))

                # Extract title from tags
                tags = chap.get("tags", {})
                title = tags.get("title") or tags.get("TITLE") or f"Chapter {i + 1}"

                is_ghost = end_time <= start_time

                parsed.append(
                    ChapterInfo(
                        index=i,
                        title=title,
                        start_time=start_time,
                        end_time=end_time,
                        is_ghost=is_ghost,
                    )
                )

            return parsed

        except subprocess.CalledProcessError as e:
            logger.error("ffprobe failed for %s: %s", file_path.name, e.stderr)
            return []
        except json.JSONDecodeError as e:
            logger.error("Failed to parse ffprobe JSON for %s: %s", file_path.name, e)
            return []
        except Exception as e:
            logger.error("Unexpected error in ffprobe parsing for %s: %s", file_path.name, e)
            return []

    def _get_file_duration(self, file_path: Path) -> float:
        """Get the total duration of the audio file in seconds."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            # Fallback duration if ffprobe fails
            return 3600.0 * 24  # 24 hours just in case
