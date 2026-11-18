"""CUE sheet parser for fallback chapter detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from audiobench.core.logger_factory import get_logger

logger = get_logger("chapters.cue_parser")


@dataclass
class ChapterInfo:
    """Lightweight representation of a chapter — the universal currency across all layers.

    Used by: ChapterDetector → ChapterRepository → CLI commands → formatters.
    Eliminates the ORM-object vs dict ambiguity that caused most of the inconsistency bugs.
    """

    index: int
    title: str
    start_time: float
    end_time: float
    is_ghost: bool
    id: int | None = None  # DB primary key; populated after save

    @property
    def chapter_index(self) -> int:
        """Alias for index for backwards compatibility with ChapterRecord."""
        return self.index

    @property
    def duration_seconds(self) -> float:
        """Duration of this chapter in seconds."""
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (safe to pass across any boundary)."""
        return {
            "id": self.id,
            "index": self.index,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_ghost": self.is_ghost,
            "duration_seconds": self.duration_seconds,
        }


class CueParser:
    """Parses .cue files to extract chapter boundaries."""

    def parse_disc_title(self, cue_path: Path) -> str | None:
        """Return the top-level TITLE from a .cue file (the disc / album title).

        CUE sheets have a global TITLE line before any TRACK block that names
        the whole disc.  The existing :meth:`parse` state-machine skips it
        because ``current_track`` is ``None`` at that point.  This method reads
        only that header line so rename_service can use the official title
        without touching chapter-level data.

        Returns:
            The disc title string, or ``None`` if absent / unreadable.
        """
        if not cue_path.exists():
            return None

        try:
            content = cue_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = cue_path.read_text(encoding="latin-1")
            except Exception:
                return None
        except Exception:
            return None

        for line in content.splitlines():
            stripped = line.strip()
            # Stop once we hit the first TRACK — everything after is per-track
            if re.match(r"^TRACK\s+\d+\s+AUDIO", stripped, re.IGNORECASE):
                break
            title_match = re.match(r'^TITLE\s+"?([^"]+)"?', stripped, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()

        return None

    def parse(self, cue_path: Path, total_duration: float) -> list[ChapterInfo]:
        """Parse a .cue file into a list of ChapterInfo objects.

        Args:
            cue_path: Path to the .cue file.
            total_duration: Total duration of the audio file in seconds,
                            used to set the end time of the final chapter.

        Returns:
            A list of ChapterInfo objects.
        """
        if not cue_path.exists():
            return []

        try:
            content = cue_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = cue_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.error("Failed to read .cue file %s: %s", cue_path, e)
                return []
        except Exception as e:
            logger.error("Failed to read .cue file %s: %s", cue_path, e)
            return []

        chapters: list[dict] = []
        current_track = None
        current_title = "Untitled"

        # Simple state machine for parsing tracks
        for line in content.splitlines():
            line = line.strip()

            # TRACK 01 AUDIO
            track_match = re.match(r"^TRACK\s+(\d+)\s+AUDIO", line, re.IGNORECASE)
            if track_match:
                if current_track is not None:
                    chapters.append(current_track)
                current_track = {
                    "index": int(track_match.group(1)) - 1,  # 0-indexed
                    "title": f"Track {int(track_match.group(1))}",
                    "start_time": 0.0,
                }
                continue

            if current_track is None:
                continue

            # TITLE "Chapter Title"
            title_match = re.match(r'^TITLE\s+"?([^"]+)"?', line, re.IGNORECASE)
            if title_match:
                current_track["title"] = title_match.group(1)

            # INDEX 01 00:00:00 (MM:SS:FF)
            index_match = re.match(r"^INDEX\s+01\s+(\d+):(\d+):(\d+)", line, re.IGNORECASE)
            if index_match:
                mins, secs, frames = map(int, index_match.groups())
                # 75 frames per second in CD audio
                current_track["start_time"] = mins * 60 + secs + (frames / 75.0)

        if current_track is not None:
            chapters.append(current_track)

        # Convert to ChapterInfo objects and calculate end times
        result = []
        for i, chap in enumerate(chapters):
            start = chap["start_time"]
            # Next chapter's start time, or total duration if it's the last one
            end = chapters[i + 1]["start_time"] if i + 1 < len(chapters) else total_duration

            # Prevent end_time from exceeding total duration (just in case)
            end = min(end, total_duration)

            is_ghost = end <= start

            result.append(
                ChapterInfo(
                    index=i, title=chap["title"], start_time=start, end_time=end, is_ghost=is_ghost
                )
            )

        logger.info("Parsed %d chapters from %s", len(result), cue_path.name)
        return result
