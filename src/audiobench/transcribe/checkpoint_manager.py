import hashlib
import json
from pathlib import Path

from audiobench.core.settings import get_settings
from audiobench.transcribe.transcription_result import Transcript


class CheckpointManager:
    """Manages chapter-level state checkpionts for resumable transcription pipelines."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        # Derive a stable hash key from the absolute file path
        # Using the filepath string instead of the whole file content to be fast,
        # since we only care about resuming a specific file that failed recently.
        key = str(self.file_path.absolute()).encode("utf-8")
        self.hash_prefix = hashlib.sha256(key).hexdigest()[:12]

        self.checkpoint_dir = get_settings().data_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, chapter_index: int) -> Path:
        return self.checkpoint_dir / f"{self.hash_prefix}_ch{chapter_index}.json"

    def has_checkpoint(self, chapter_index: int) -> bool:
        """Check if a fully processed transcript checkpoint exists for this chapter."""
        return self._get_checkpoint_path(chapter_index).exists()

    def load_checkpoint(self, chapter_index: int) -> Transcript | None:
        """Load a checkpointed transcript for a given chapter."""
        cp_path = self._get_checkpoint_path(chapter_index)
        if not cp_path.exists():
            return None
        try:
            with cp_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Support pydantic v1 vs v2 gracefully if parse_obj doesn't exist
            if hasattr(Transcript, "model_validate"):
                return Transcript.model_validate(data)
            return Transcript.parse_obj(data)
        except Exception:
            return None

    def save_checkpoint(self, chapter_index: int, transcript: Transcript) -> None:
        """Save a transcript checkpoint for a chapter to disk."""
        cp_path = self._get_checkpoint_path(chapter_index)
        try:
            if hasattr(transcript, "model_dump_json"):
                json_str = transcript.model_dump_json(indent=2)
            else:
                json_str = transcript.json(indent=2)
            cp_path.write_text(json_str, encoding="utf-8")
        except Exception:
            pass

    def clear_all(self) -> None:
        """Clear all checkpoints associated with this file."""
        for cp_path in self.checkpoint_dir.glob(f"{self.hash_prefix}_ch*.json"):
            try:
                cp_path.unlink()
            except OSError:
                pass
