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


class LightweightDiarizer:
    """Fast diarization using Whisper segments, SpeechBrain ECAPA-TDNN, and AHC clustering."""

    def __init__(self, distance_threshold: float = 0.65, device: str | None = None) -> None:
        self.distance_threshold = distance_threshold
        self.device = device

    def diarize(self, audio_path: str | Path, transcript: Transcript) -> Transcript:
        """Run clustering on existing transcript segments."""
        if not transcript.segments:
            return transcript

        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        from audiobench.diarization.verification import SpeakerVerificationEngine

        engine = SpeakerVerificationEngine(device=self.device)
        embeddings = []
        valid_indices = []

        logger.info("Extracting voice prints for %d segments...", len(transcript.segments))
        for i, segment in enumerate(transcript.segments):
            vec = engine.extract_voice_print(audio_path, segment.start, segment.end)
            if vec is not None:
                embeddings.append(vec)
                valid_indices.append(i)

        if not embeddings:
            logger.warning("No valid voice prints extracted. Skipping diarization.")
            return transcript

        # 2. Cluster the valid embeddings
        X = np.array(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.distance_threshold,
        )
        labels = clustering.fit_predict(X)

        # Map valid indices back to segments
        for idx, label in zip(valid_indices, labels):
            transcript.segments[idx].speaker = f"SPEAKER_{label:02d}"

        # 3. Handle None segments by nearest temporal neighbor
        for i, segment in enumerate(transcript.segments):
            if segment.speaker is None:
                # Find closest valid index based on temporal distance
                closest_idx = min(
                    valid_indices,
                    key=lambda vi: min(
                        abs(segment.end - transcript.segments[vi].start),
                        abs(segment.start - transcript.segments[vi].end)
                    )
                )
                segment.speaker = transcript.segments[closest_idx].speaker

        # Apply speaker to words
        for segment in transcript.segments:
            if segment.words and segment.speaker:
                for word in segment.words:
                    word.speaker = segment.speaker

        # 4. Resolve global identities
        self._resolve_identities(transcript, audio_path, engine)
        return transcript

    def _resolve_identities(self, transcript: Transcript, audio_path: str | Path, engine) -> None:
        speakers = sorted({s.speaker for s in transcript.segments if s.speaker})
        speaker_map = {}

        try:
            import uuid

            from audiobench.memory.memory_store import SpeakerProfileStore

            profile_store = SpeakerProfileStore()

            for i, spk in enumerate(speakers):
                spk_segments = [(s.start, s.end) for s in transcript.segments if s.speaker == spk]
                try:
                    vector = engine.extract_mean_voice_print(audio_path, spk_segments)
                    global_name = profile_store.identify_speaker(vector)

                    if global_name:
                        speaker_map[spk] = global_name
                    else:
                        profile_id = str(uuid.uuid4())
                        local_name = f"Speaker {i + 1} ({profile_id[:4]})"
                        speaker_map[spk] = local_name
                        profile_store.save_speaker(profile_id, local_name, vector)
                except Exception as e:
                    logger.warning("Voice print extraction failed for %s: %s", spk, e)
                    speaker_map[spk] = f"Speaker {i + 1}"

        except Exception as e:
            logger.warning("Speaker verification unavailable: %s", e)
            speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(speakers)}

        # Apply mapping
        for segment in transcript.segments:
            if segment.speaker and segment.speaker in speaker_map:
                segment.speaker = speaker_map[segment.speaker]
            if segment.words:
                for word in segment.words:
                    if word.speaker and word.speaker in speaker_map:
                        word.speaker = speaker_map[word.speaker]

        logger.info(
            "Assigned %d unique speakers across %d segments",
            len(speaker_map),
            len(transcript.segments),
        )


class PyannoteDiarizer:
    """Speaker diarization via pyannote.audio.

    Requires:
    - HuggingFace token with access to pyannote/speaker-diarization-3.1
    - Accept user conditions at https://hf.co/pyannote/speaker-diarization-3.1
    """

    def __init__(self, hf_token: str | None = None, device: str | None = None) -> None:
        self._hf_token = hf_token
        self._device = device
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
                token=self._hf_token,
            )
            if self._device:
                import torch
                self._pipeline.to(torch.device(self._device))
                logger.info("Diarization pipeline loaded on %s", self._device)
            else:
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

            # Compatibility check for pyannote.audio 3.3+ / 4.x+
            # Newer versions return a DiarizeOutput object instead of Annotation
            if hasattr(diarization, "itertracks"):
                annotation = diarization
            else:
                annotation = diarization.speaker_diarization

            turns: list[SpeakerTurn] = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
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
        audio_path: str | Path | None = None,
    ) -> Transcript:
        """Assign speaker labels to transcript segments at the word level.

        Strategy:
        1. For every word in a segment, find the speaker turn that covers
           the word's midpoint timestamp (most reliable single-point lookup).
        2. Each word gets its own speaker label stored in word.speaker.
        3. The segment's top-level speaker is set to the majority speaker
           among its words (most words wins). This keeps backwards
           compatibility with formatters that only read segment.speaker.
        4. Falls back to segment-overlap matching for segments with no
           word-level timestamps (e.g. Gemini engine output).

        Args:
            transcript: Transcript with segments (and ideally word timestamps).
            turns: Speaker turns from pyannote diarization.
            audio_path: Path to audio file for voice print extraction.

        Returns:
            Transcript with speaker labels on both words and segments.
        """
        if not turns:
            return transcript

        # Build a sorted list of turns for efficient lookup
        sorted_turns = sorted(turns, key=lambda t: t.start)

        for segment in transcript.segments:
            if segment.words:
                # ── Word-level assignment ──────────────────────────────────
                speaker_votes: dict[str, int] = {}
                for word in segment.words:
                    spk = self._speaker_at(word.midpoint, sorted_turns)
                    if spk:
                        word.speaker = spk
                        speaker_votes[spk] = speaker_votes.get(spk, 0) + 1

                # Majority vote → segment label
                if speaker_votes:
                    segment.speaker = max(speaker_votes, key=lambda s: speaker_votes[s])
            else:
                # ── Fallback: segment-overlap (no word timestamps) ─────────
                segment.speaker = self._find_best_speaker(segment, sorted_turns)

        # Simplify speaker labels (SPEAKER_00 → Speaker 1) or Global Identity
        speakers = sorted({s.speaker for s in transcript.segments if s.speaker})
        speaker_map = {}

        if audio_path:
            try:
                import uuid

                from audiobench.diarization.verification import SpeakerVerificationEngine
                from audiobench.memory.memory_store import SpeakerProfileStore

                verification_engine = SpeakerVerificationEngine()
                profile_store = SpeakerProfileStore()

                for i, spk in enumerate(speakers):
                    spk_turns = [(t.start, t.end) for t in sorted_turns if t.speaker == spk]
                    try:
                        vector = verification_engine.extract_mean_voice_print(audio_path, spk_turns)
                        global_name = profile_store.identify_speaker(vector)

                        if global_name:
                            speaker_map[spk] = global_name
                        else:
                            profile_id = str(uuid.uuid4())
                            # Make the default name slightly unique to avoid overlapping "Speaker 1"s in DB
                            local_name = f"Speaker {i + 1} ({profile_id[:4]})"
                            speaker_map[spk] = local_name
                            profile_store.save_speaker(profile_id, local_name, vector)
                    except Exception as e:
                        logger.warning("Voice print extraction failed for %s: %s", spk, e)
                        speaker_map[spk] = f"Speaker {i + 1}"
            except Exception as e:
                logger.warning("Speaker verification unavailable: %s", e)
                speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(speakers)}
        else:
            speaker_map = {spk: f"Speaker {i + 1}" for i, spk in enumerate(speakers)}

        for segment in transcript.segments:
            if segment.speaker and segment.speaker in speaker_map:
                segment.speaker = speaker_map[segment.speaker]
            for word in segment.words:
                if word.speaker and word.speaker in speaker_map:
                    word.speaker = speaker_map[word.speaker]

        logger.info(
            "Assigned %d unique speakers across %d segments (%d words labeled)",
            len(speaker_map),
            len([s for s in transcript.segments if s.speaker]),
            sum(1 for s in transcript.segments for w in s.words if w.speaker),
        )

        return transcript

    @staticmethod
    def _speaker_at(midpoint: float, sorted_turns: list[SpeakerTurn]) -> str | None:
        """Find the speaker whose turn covers a given timestamp midpoint."""
        for turn in sorted_turns:
            if turn.start <= midpoint <= turn.end:
                return turn.speaker
            if turn.start > midpoint:
                break  # sorted — no need to keep scanning
        return None

    @staticmethod
    def _find_best_speaker(segment: Segment, turns: list[SpeakerTurn]) -> str | None:
        """Fallback: find the speaker with the most overlap with a segment."""
        best_speaker = None
        best_overlap = 0.0

        for turn in turns:
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
        return self.assign_speakers(transcript, turns, audio_path=audio_path)
