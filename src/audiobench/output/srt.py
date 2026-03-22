"""SubRip (.srt) subtitle formatter."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SrtFormatter(OutputFormatter):
    """Format transcript as SubRip (.srt) subtitles."""

    def format(self, transcript: Transcript) -> str:
        lines: list[str] = []

        for i, seg in enumerate(transcript.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}")
            text = seg.text
            if seg.speaker:
                text = f"[{seg.speaker}] {text}"
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def extension() -> str:
        return "srt"
