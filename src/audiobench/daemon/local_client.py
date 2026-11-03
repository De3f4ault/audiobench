"""LocalRetrievalClient — in-process fallback when the daemon is unavailable.

Exposes the same interface as DaemonClient but loads all models directly
in the calling process. Used when auto-start fails or is explicitly disabled.
"""

from __future__ import annotations

from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType

logger = get_logger("daemon.local_client")


class LocalRetrievalClient:
    """Loads models in-process on first call (lazy, thread-safe via singletons)."""

    def __init__(self) -> None:
        self._store: Any = None

    def _get_store(self) -> Any:
        """Lazily initialise MemoryStore the first time it's needed."""
        if self._store is None:
            logger.warning(
                "Daemon unavailable — using in-process memory store. "
                "Consider running `audiobench daemon start` for faster responses."
            )
            from audiobench.memory.memory_store import MemoryStore

            self._store = MemoryStore()
        return self._store

    # ------------------------------------------------------------------
    # RetrievalClient interface
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Always True — we are the process."""
        return True

    def chunk(
        self, text: str, audio_file_id: int, diarized: bool, segments: list[dict] | None = None
    ) -> list[ChunkResult]:
        """Run text through the local chunking pipeline."""
        from audiobench.memory.chunking import content_aware_router

        chunks = content_aware_router(text, diarized_segments=segments if diarized else None)
        results = []
        for c in chunks:
            res: dict = {"content": c.content, "uuid": c.uuid, "tier": c.tier}
            if c.speaker:
                res["speaker"] = c.speaker
            results.append(res)  # type: ignore
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        speaker_filter: str | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search over memory store."""
        return self._get_store().search(
            query=query,
            top_k=top_k,
            speaker_filter=speaker_filter,
            hyde_document=hyde_document,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_colbert=use_colbert,
        )

    def embed(
        self,
        expression_id: int,
        content: str,
        source_type: SourceType,
        speaker: str | None = None,
    ) -> None:
        """Embed and persist an expression to LanceDB."""
        self._get_store().write_node(
            expression_id=expression_id,
            content=content,
            source_type=source_type.value,
            speaker=speaker,
        )

    def delete(self, expression_id: int) -> None:
        """Remove an expression from LanceDB."""
        self._get_store().delete_node(expression_id)

    def delete_batch(self, expression_ids: list[int]) -> None:
        """Remove a batch of expressions from LanceDB."""
        self._get_store().delete_node_batch(expression_ids)

    def status(self) -> dict:
        """Return store statistics."""
        store = self._get_store()
        res = {
            "mode": "local",
            "uptime_seconds": 0.0,  # always 0 for local client
            "embedding_model_version": store.model_version,
            "total_nodes": store.count_nodes(),
        }
        try:
            res["playback"] = self.playback_status()
        except Exception:
            pass
        return res

    def _get_cache(self) -> Any:
        """Lazily initialise QueryCacheStore."""
        if not hasattr(self, "_cache"):
            from audiobench.memory.memory_store import QueryCacheStore

            self._cache = QueryCacheStore()
        return self._cache

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check semantic cache locally."""
        return self._get_cache().check_cache(query, distance_threshold)

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write to semantic cache locally."""
        self._get_cache().write_cache(query, answer, hyde_document)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using the primary embedder."""
        from audiobench.memory.singletons import get_primary_embedder, get_primary_inference_lock

        text = str(text)[:12_000]
        prefixed = f"search_query: {text}"
        model = get_primary_embedder()
        with get_primary_inference_lock():
            return model.encode(prefixed).tolist()

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Rerank a list of documents against a query."""
        if not docs:
            return []
        try:
            from audiobench.memory.singletons import get_reranker

            reranker = get_reranker()
            pairs = [[query, d] for d in docs]
            scores = reranker.predict(pairs).tolist()
            return scores
        except Exception as e:
            logger.warning("Local reranker failed: %s", e)
            return [0.0] * len(docs)

    def autocomplete(self, prefix: str, top_k: int = 5) -> list[dict]:
        """Local autocomplete fallback."""
        try:
            from audiobench.daemon.autocomplete import get_autocomplete_index
            from audiobench.memory.singletons import get_primary_embedder, get_primary_inference_lock

            prefix = str(prefix).strip()
            if not prefix:
                return []
            index = get_autocomplete_index()
            if not index.ready:
                return []
            model = get_primary_embedder()
            with get_primary_inference_lock():
                vector = model.encode(f"search_query: {prefix}").tolist()
            results = index.lookup(vector, k=int(top_k))
            expression_ids = [r["expression_id"] for r in results if "expression_id" in r]
            if expression_ids:
                from audiobench.storage.expression_repository import ExpressionRepository

                repo = ExpressionRepository()
                expr_map = repo.get_by_ids(expression_ids, session=None)
                enriched = []
                for r in results:
                    expr_id = r.get("expression_id")
                    if expr_id in expr_map:
                        expr = expr_map[expr_id]
                        r["speaker"] = expr.speaker
                        r["source_type"] = expr.source_type
                        enriched.append(r)
                return enriched
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Universal Playback interface (in-process fallback)
    # ------------------------------------------------------------------

    def _get_playback_service(self) -> Any:
        """Return the shared PlaybackService instance."""
        from audiobench.playback.service import get_playback_service

        return get_playback_service()

    def playback_play(
        self,
        file_path: str,
        *,
        start_pos: float = 0.0,
        speed: float = 1.0,
        audio_file_id: int | None = None,
        transcription_id: int | None = None,
        segments: list[dict[str, Any]] | None = None,
        auto_resume: bool = False,
    ) -> dict[str, Any]:
        """Start playback or seek within the current file."""
        kwargs: dict[str, Any] = {
            "start_pos": float(start_pos or 0.0),
            "speed": float(speed or 1.0),
            "audio_file_id": audio_file_id,
            "transcription_id": transcription_id,
        }
        if segments is not None:
            kwargs["segments"] = segments
        if auto_resume:
            kwargs["auto_resume"] = auto_resume
        return self._get_playback_service().play(file_path, **kwargs)

    def playback_sync_session(
        self,
        file_path: str,
        *,
        transcription_id: int | None = None,
        position: float = 0.0,
        audio_file_id: int | None = None,
    ) -> dict[str, Any]:
        """Sync playback session metadata without spawning a separate mpv process."""
        return self._get_playback_service().sync_session(
            file_path,
            transcription_id=transcription_id,
            position=float(position or 0.0),
            audio_file_id=audio_file_id,
        )

    def playback_seek(self, position: float) -> dict[str, Any]:
        """Seek to absolute position in seconds."""
        return self._get_playback_service().seek(float(position or 0.0))

    def playback_seek_relative(self, offset: float) -> dict[str, Any]:
        """Seek by relative offset in seconds."""
        return self._get_playback_service().seek_relative(float(offset or 0.0))

    def playback_pause(self) -> dict[str, Any]:
        """Pause playback."""
        return self._get_playback_service().pause()

    def playback_resume(self) -> dict[str, Any]:
        """Resume playback."""
        return self._get_playback_service().resume()

    def playback_toggle(self) -> dict[str, Any]:
        """Toggle pause/play."""
        return self._get_playback_service().toggle()

    def playback_speed(self, speed: float) -> dict[str, Any]:
        """Set playback speed."""
        return self._get_playback_service().set_speed(float(speed or 1.0))

    def playback_stop(self) -> dict[str, Any]:
        """Stop playback and return mpv to idle state."""
        return self._get_playback_service().stop()

    def playback_status(self) -> dict[str, Any]:
        """Return current playback status."""
        return self._get_playback_service().status()

    def playback_context(
        self, position: float | None = None, window: int = 3
    ) -> dict[str, Any]:
        """Return transcript segments around current/given position."""
        return self._get_playback_service().get_context_window(
            position=position, window=window
        )

    def playback_loop_segment(self, start: float, end: float) -> dict[str, Any]:
        """Engage A-B loop between start and end seconds."""
        return self._get_playback_service().loop_segment(float(start), float(end))

    def playback_clear_loop(self) -> dict[str, Any]:
        """Cancel active A-B loop."""
        return self._get_playback_service().clear_loop()

    def playback_set_filter(self, filter_str: str) -> dict[str, Any]:
        """Set live DSP audio filter (empty string to clear)."""
        return self._get_playback_service().set_filter(str(filter_str))

    def playback_chapter_telemetry(self, segment_ids: list[int]) -> dict[str, dict]:
        """Map segment IDs from search results to their chapter metadata.

        Returns str-keyed dict — consistent with the daemon IPC path,
        since get_chapter_telemetry() returns str keys canonically.
        """
        from audiobench.playback.service import PlaybackService
        return PlaybackService.get_chapter_telemetry(list(segment_ids))

    def whisper_acquire(self, session_id: str, model: str = "base") -> dict[str, Any]:
        """No-op for local in-process client."""
        return {"ok": True, "local": True}

    def whisper_release(self, session_id: str) -> dict[str, Any]:
        """No-op for local in-process client."""
        return {"ok": True, "local": True}

    def whisper_transcribe(
        self, file_path: str, model: str = "base", language: str = "en"
    ) -> dict[str, Any]:
        """Transcribe directly via local faster-whisper if available."""
        from audiobench.chat.voice_session import get_whisper_model, is_local_stt_available
        if not is_local_stt_available():
            return {"ok": False, "error": "faster-whisper not installed"}
        try:
            m = get_whisper_model(model)
            segments, info = m.transcribe(file_path, language=language)
            text = " ".join(s.text.strip() for s in segments).strip()
            return {"ok": True, "text": text, "language": getattr(info, "language", language)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

