"""Speaker diarization engine using pyannote.audio.

Identifies who is speaking in each segment of a transcript.
Requires a HuggingFace token for pyannote model access.

Usage:
    from audiobench.diarization.engine import PyannoteDiarizer

    diarizer = PyannoteDiarizer(hf_token="hf_...")
    transcript = diarizer.diarize(audio_path, transcript)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audiobench.core.error_types import DiarizationError
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.transcription_result import Segment, Transcript

logger = get_logger("diarization.engine")


@dataclass
class SpeakerTurn:
    """A time-bounded speaker turn from pyannote."""

    speaker: str  # e.g., "SPEAKER_00"
    start: float  # seconds
    end: float  # seconds


class PyannoteDiarizer:
    """Speaker diarization via pyannote.audio.

    Requires:
    - HuggingFace token with access to pyannote/speaker-diarization-3.1
    - Accept user conditions at https://hf.co/pyannote/speaker-diarization-3.1
    """

    def __init__(self, hf_token: str | None = None) -> None:
        self._hf_token = hf_token
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Lazily load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise DiarizationError(
                "pyannote.audio not installed",
                "Install with: pip install pyannote.audio torch torchaudio\n"
                "Or: pip install -e '.[diarization]'",
            ) from None

        if not self._hf_token:
            raise DiarizationError(
                "HuggingFace token required",
                "Set AUDIOBENCH_HF_TOKEN in .env or pass --hf-token\n"
                "Get a token at: https://huggingface.co/settings/tokens\n"
                "Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1",
            )

        logger.info("Loading pyannote diarization pipeline")

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
            logger.info("Diarization pipeline loaded")
        except Exception as e:
            raise DiarizationError(
                "Failed to load diarization pipeline",
                str(e),
            ) from e

    def get_speaker_turns(self, audio_path: str | Path) -> list[SpeakerTurn]:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV preferred).

        Returns:
            List of speaker turns with timestamps.

        Raises:
            DiarizationError: If diarization fails.
        """
        self._load_pipeline()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise DiarizationError("Audio file not found", str(audio_path))

        logger.info("Running diarization on: %s", audio_path.name)

        try:
            diarization = self._pipeline(str(audio_path))

            turns: list[SpeakerTurn] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end,
                    )
                )

            logger.info(
                "Diarization complete: %d turns, %d speakers",
                len(turns),
                len({t.speaker for t in turns}),
            )
            return turns

        except DiarizationError:
            raise
        except Exception as e:
            raise DiarizationError("Diarization failed", str(e)) from e

    def assign_speakers(
        self,
        transcript: Transcript,
        turns: list[SpeakerTurn],
    ) -> Transcript:
        """Assign speaker labels to transcript segments.

        Uses overlap-based matching: each segment gets the speaker
        label of the turn with the greatest time overlap.

        Args:
            transcript: Transcript with segments to label.
            turns: Speaker turns from diarization.

        Returns:
            Transcript with speaker labels filled in.
        """
        if not turns:
            return transcript

        for segment in transcript.segments:
            best_speaker = self._find_best_speaker(segment, turns)
            if best_speaker:
                segment.speaker = best_speaker

        # Simplify speaker labels (SPEAKER_00 → Speaker 1)
        speakers = sorted({s.speaker for s in transcript.segments if s.speaker})
        speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(speakers)}
        for segment in transcript.segments:
            if segment.speaker and segment.speaker in speaker_map:
                segment.speaker = speaker_map[segment.speaker]

        logger.info(
            "Assigned %d unique speakers to %d segments",
            len(speaker_map),
            len([s for s in transcript.segments if s.speaker]),
        )

        return transcript

    @staticmethod
    def _find_best_speaker(segment: Segment, turns: list[SpeakerTurn]) -> str | None:
        """Find the speaker with the most overlap with a segment."""
        best_speaker = None
        best_overlap = 0.0

        for turn in turns:
            # Calculate overlap between segment and turn
            overlap_start = max(segment.start, turn.start)
            overlap_end = min(segment.end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        return best_speaker

    def diarize(
        self,
        audio_path: str | Path,
        transcript: Transcript,
    ) -> Transcript:
        """Full diarization pipeline: run pyannote then assign speakers.

        Args:
            audio_path: Path to audio file.
            transcript: Transcript to enrich with speaker labels.

        Returns:
            Transcript with speaker assignments.
        """
        turns = self.get_speaker_turns(audio_path)
        return self.assign_speakers(transcript, turns)
