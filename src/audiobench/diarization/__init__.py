"""Speaker diarization module — identify who is speaking.

Provides:
    PyannoteDiarizer — pyannote.audio-based speaker identification
"""

from audiobench.diarization.engine import PyannoteDiarizer

__all__ = ["PyannoteDiarizer"]
