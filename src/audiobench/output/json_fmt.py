"""JSON output formatter with full metadata."""


from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


class JsonFormatter(OutputFormatter):
    """Format transcript as JSON with all metadata, timestamps, and words."""

    def format(self, transcript: Transcript) -> str:
        return transcript.model_dump_json(indent=2)

    @staticmethod
    def extension() -> str:
        return "json"
