"""Speaker verification and voice print extraction.

Uses SpeechBrain ECAPA-TDNN to extract 192-dimensional embeddings
from raw audio segments for cross-session speaker identification.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torchaudio

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("diarization.verification")

_classifiers = {}

def get_speaker_classifier(device: str | None = None):
    """Lazy load the SpeechBrain ECAPA-TDNN classifier on a specific device."""
    global _classifiers

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device not in _classifiers:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise RuntimeError("speechbrain is not installed. Run: pip install speechbrain")

        settings = get_settings()
        is_offline = settings.offline_mode or os.environ.get("HF_HUB_OFFLINE") == "1"

        logger.info("Loading SpeechBrain ECAPA-TDNN model on %s [offline=%s]...", device, is_offline)
        t0 = time.time()

        # Load the model
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        logger.info("SpeechBrain model loaded in %.2fs on %s", time.time() - t0, device)
        _classifiers[device] = classifier

    return _classifiers[device]


class SpeakerVerificationEngine:
    """Extracts voice prints from audio segments and matches them globally."""

    def __init__(self, device: str | None = None):
        self.sample_rate = 16000  # ECAPA-TDNN expects 16kHz
        self.device = device

    def extract_voice_print(self, audio_path: str | Path, start: float, end: float) -> list[float] | None:
        """Extract a 192-D voice print from a specific time segment."""
        # ECAPA-TDNN requires a minimum segment length to produce meaningful embeddings
        if end - start < 1.0:
            return None

        classifier = get_speaker_classifier(self.device)

        # Load audio segment
        signal, fs = torchaudio.load(
            str(audio_path),
            frame_offset=int(start * self.sample_rate),
            num_frames=int((end - start) * self.sample_rate)
        )

        # Resample if necessary
        if fs != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)
            signal = resampler(signal)

        # Ensure mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # Move to model device
        signal = signal.to(classifier.device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)

        # Flatten to 1D list
        vector = embeddings.squeeze().cpu().tolist()
        return vector

    def extract_mean_voice_print(self, audio_path: str | Path, segments: list[tuple[float, float]]) -> list[float]:
        """Extract a stable voice print by averaging across multiple segments."""
        vectors = []
        for start, end in segments:
            # Skip very short segments (less than 1.5s)
            if end - start < 1.5:
                continue
            try:
                vec = self.extract_voice_print(audio_path, start, end)
                if vec is not None:
                    vectors.append(vec)
            except Exception as e:
                logger.warning("Failed to extract vector for segment %.2f-%.2f: %s", start, end, e)

        if not vectors:
            # Fallback to the longest segment if all are too short
            if segments:
                longest = max(segments, key=lambda x: x[1] - x[0])
                vec = self.extract_voice_print(audio_path, longest[0], longest[1])
                if vec is None:
                    raise RuntimeError("Voice print extraction failed for longest segment")
                return vec
            raise ValueError("No valid audio segments to extract voice print from")

        # Calculate mean vector
        import numpy as np
        mean_vector = np.mean(vectors, axis=0)
        # Normalize
        norm_vector = mean_vector / np.linalg.norm(mean_vector)
        return norm_vector.tolist()
