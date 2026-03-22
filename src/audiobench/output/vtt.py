"""WebVTT (.vtt) subtitle formatter."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


def _format_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VttFormatter(OutputFormatter):
    """Format transcript as WebVTT (.vtt) subtitles."""

    def format(self, transcript: Transcript) -> str:
        lines: list[str] = ["WEBVTT", ""]

        for seg in transcript.segments:
            lines.append(f"{_format_vtt_time(seg.start)} --> {_format_vtt_time(seg.end)}")
            text = seg.text
            if seg.speaker:
                text = f"<v {seg.speaker}>{text}</v>"
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def extension() -> str:
        return "vtt"
