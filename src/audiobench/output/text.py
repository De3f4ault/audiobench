"""Plain text output formatter."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


class TextFormatter(OutputFormatter):
    """Format transcript as plain text, grouped by speaker if available."""

    def format(self, transcript: Transcript) -> str:
        lines: list[str] = []
        current_speaker = None

        for seg in transcript.segments:
            if seg.speaker and seg.speaker != current_speaker:
                if lines:
                    lines.append("")
                lines.append(f"[{seg.speaker}]")
                current_speaker = seg.speaker
            lines.append(seg.text)

        return "\n".join(lines) + "\n"

    @staticmethod
    def extension() -> str:
        return "txt"
