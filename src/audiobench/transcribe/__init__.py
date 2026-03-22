"""Transcribe feature — audio transcription pipeline.

Provides:
    - transcriber: Orchestrates engine selection → transcription → results
    - audio_converter: FFmpeg-based audio loading and conversion
    - audio_filters: Post-processing filters for transcription text
    - transcription_result: Dataclasses for Transcript, Segment, Word
    - engines/: Pluggable transcription engine implementations
"""
