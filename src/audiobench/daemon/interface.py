"""Protocol interface for daemon clients."""

from __future__ import annotations

from typing import Protocol

from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType


class RetrievalClient(Protocol):
    """Shared interface for daemon and local retrieval clients."""

    def search(
        self,
        query: str,
        top_k: int,
        speaker_filter: str | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]: ...

    def autocomplete(self, prefix: str, top_k: int = 10) -> list[dict]: ...

    def embed(
        self, expression_id: int, content: str, source_type: SourceType, speaker: str | None
    ) -> None: ...

    def chunk(self, text: str, audio_file_id: int, diarized: bool) -> list[ChunkResult]: ...

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None: ...

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None: ...

    def embed_query(self, text: str) -> list[float]: ...

    def rerank(self, query: str, docs: list[str]) -> list[float]: ...

    def ping(self) -> bool: ...

    def playback_play(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        audio_file_id: int | None = None,
        transcription_id: int | None = None,
        segments: list[dict] | None = None,
        auto_resume: bool = False,
    ) -> dict: ...

    def playback_sync_session(
        self,
        file_path: str,
        *,
        transcription_id: int | None = None,
        position: float = 0.0,
        audio_file_id: int | None = None,
    ) -> dict: ...

    def playback_seek(self, position: float) -> dict: ...
    def playback_seek_relative(self, offset: float) -> dict: ...
    def playback_pause(self) -> dict: ...
    def playback_resume(self) -> dict: ...
    def playback_toggle(self) -> dict: ...
    def playback_speed(self, speed: float) -> dict: ...
    def playback_stop(self) -> dict: ...
    def playback_status(self) -> dict: ...
    def playback_context(
        self, position: float | None = None, window: int = 3
    ) -> dict: ...

    def whisper_acquire(self, session_id: str, model: str = "base") -> dict: ...
    def whisper_release(self, session_id: str) -> dict: ...
    def whisper_transcribe(
        self, file_path: str, model: str = "base", language: str = "en"
    ) -> dict: ...
