"""Model singletons for memory embedding and reranking.

Handles thread-safe lazy-loading of ML models for semantic boundary detection,
primary embedding generation, and post-retrieval reranking.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Generator
from typing import Any

from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import ConnectionError  # type: ignore[import-untyped]

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.exceptions import AudioBenchError

logger = get_logger("memory.singletons")

# Singleton references
_boundary_embedder: Any = None
_boundary_lock = threading.Lock()

_primary_embedder: Any = None
_primary_lock = threading.Lock()
# Separate lock that serialises *inference* calls (not just loading).
# NomicBert's RoPE cache is lazily extended and not thread-safe: concurrent
# encode() calls from the sweep thread + asyncio executor threads can race
# to write the same cache entry, producing tensor-dimension mismatches.
_primary_inference_lock = threading.RLock()

_reranker: Any = None
_reranker_lock = threading.Lock()

_llm_breaker: Any = None  # kept for backward compat — alias to _gemini_breaker below
_gemini_breaker: Any = None
_gemini_breaker_lock = threading.Lock()

_memory_store: Any = None
_memory_store_lock = threading.Lock()

_segment_store: Any = None
_segment_store_lock = threading.Lock()

_hf_configured = False


def _configure_hf() -> bool:
    """Configure HuggingFace environment variables.

    Returns True if offline mode is active.
    """
    global _hf_configured
    settings = get_settings()

    # 1. Check offline mode
    is_offline = settings.offline_mode or os.environ.get("HF_HUB_OFFLINE") == "1"
    if is_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    _hf_configured = True
    return is_offline


@contextlib.contextmanager
def _force_offline_env() -> Generator[None, None, None]:
    """Context manager that sets HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE for its
    entire duration, then restores the originals.

    This is the correct way to force offline loading: env vars must remain set
    for the full depth of SentenceTransformer.__init__, including all lazily
    invoked sub-loaders such as modeling_hf_nomic_bert.state_dict_from_pretrained.
    Restoring inside a finally{} block that wraps only the top-level call is
    insufficient because those sub-loaders run *after* the finally fires.
    """
    import huggingface_hub.constants as hf_constants
    orig_hf = os.environ.get("HF_HUB_OFFLINE")
    orig_tr = os.environ.get("TRANSFORMERS_OFFLINE")
    orig_const = hf_constants.HF_HUB_OFFLINE

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = orig_const
        if orig_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig_hf
        if orig_tr is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = orig_tr


def _handle_load_error(e: Exception, model_name: str) -> None:
    """Gracefully handle HuggingFace load errors for new users."""
    if isinstance(e, (LocalEntryNotFoundError, ConnectionError, HfHubHTTPError)):
        raise AudioBenchError(
            f"Could not load model '{model_name}'. "
            f"If this is your first run, ensure you are connected to the internet "
            f"and not forcing offline_mode."
        ) from e
    raise


def get_boundary_embedder() -> Any:
    """Get or load the boundary detection embedder.

    Used for chunking audio transcripts into semantic segments.
    """
    global _boundary_embedder
    if _boundary_embedder is None:
        with _boundary_lock:
            if _boundary_embedder is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading boundary embedder (sentence-transformers/all-MiniLM-L6-v2) ...",
                )
                t0 = time.time()
                from sentence_transformers import SentenceTransformer

                # Attempt 1: load from local cache — env vars held for the full
                # depth of SentenceTransformer.__init__ via context manager.
                try:
                    with _force_offline_env():
                        _boundary_embedder = SentenceTransformer(
                            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
                        )
                except Exception as e:
                    logger.debug("Failed to load boundary embedder from cache: %s", e)
                    if is_offline:
                        raise
                    # Attempt 2: online download (first-run path).
                    try:
                        _boundary_embedder = SentenceTransformer(
                            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=False
                        )
                    except Exception as e2:
                        _handle_load_error(e2, "sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Boundary embedder loaded in %.2fs", time.time() - t0)
    return _boundary_embedder


def get_primary_embedder() -> Any:
    """Get or load the primary storage embedder.

    Used for vectorizing text for storage in LanceDB.
    """
    global _primary_embedder
    if _primary_embedder is None:
        with _primary_lock:
            if _primary_embedder is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading primary embedder (nomic-ai/nomic-embed-text-v1.5) ...",
                )
                t0 = time.time()
                from sentence_transformers import SentenceTransformer

                # Attempt 1: load from local cache.
                # IMPORTANT: _force_offline_env() keeps HF_HUB_OFFLINE=1 and
                # TRANSFORMERS_OFFLINE=1 set for the *entire* duration of
                # SentenceTransformer.__init__, including the
                # modeling_hf_nomic_bert.state_dict_from_pretrained() call that
                # happens deep inside transformers' lazy sub-module loading.
                # The old pattern (set/finally/restore wrapping only the top call)
                # was stripping the vars before the custom weight loader ran.
                try:
                    with _force_offline_env():
                        _primary_embedder = SentenceTransformer(
                            "nomic-ai/nomic-embed-text-v1.5",
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                except Exception as e:
                    logger.debug("Failed to load primary embedder from cache: %s", e)
                    if is_offline:
                        raise
                    # Attempt 2: online download (first-run path).
                    try:
                        _primary_embedder = SentenceTransformer(
                            "nomic-ai/nomic-embed-text-v1.5",
                            trust_remote_code=True,
                            local_files_only=False,
                        )
                    except Exception as e2:
                        _handle_load_error(e2, "nomic-ai/nomic-embed-text-v1.5")
                logger.info("Primary embedder loaded in %.2fs", time.time() - t0)
    return _primary_embedder


def get_primary_inference_lock() -> threading.RLock:
    """Return the inference lock that must be held during all encode() calls
    on the primary embedder.  Using an RLock allows the same thread to
    re-enter (e.g., warmup then sweep on the same thread).
    """
    return _primary_inference_lock


def get_reranker() -> Any:
    """Get or load the cross-encoder reranker.

    Used to re-score and filter the top-K retrieval results.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) ...",
                )
                t0 = time.time()
                from sentence_transformers import CrossEncoder

                # Attempt 1: load from local cache.
                try:
                    with _force_offline_env():
                        _reranker = CrossEncoder(
                            "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True
                        )
                except Exception as e:
                    logger.debug("Failed to load reranker from cache: %s", e)
                    if is_offline:
                        raise
                    # Attempt 2: online download (first-run path).
                    try:
                        _reranker = CrossEncoder(
                            "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=False
                        )
                    except Exception as e2:
                        _handle_load_error(e2, "cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker loaded in %.2fs", time.time() - t0)
    return _reranker


_colbert_reranker: Any = None
_colbert_lock = threading.Lock()


def get_colbert_reranker() -> Any:
    """Get or load the AnswerAI ColBERT late-interaction reranker."""
    global _colbert_reranker
    if _colbert_reranker is None:
        with _colbert_lock:
            if _colbert_reranker is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading ColBERT reranker (answerdotai/answerai-colbert-small-v1) [offline=%s]...",
                    is_offline,
                )
                t0 = time.time()
                try:
                    from lancedb.rerankers import ColbertReranker
                    from rerankers.models.colbert_ranker import ColBERTModel

                    # Monkey-patch for transformers v5.x compatibility
                    original_init = ColBERTModel.__init__

                    def patched_init(self, config, verbose: int):
                        original_init(self, config, verbose)
                        if hasattr(self, "post_init"):
                            self.post_init()
                        elif not hasattr(self, "all_tied_weights_keys"):
                            self.all_tied_weights_keys = {}

                    ColBERTModel.__init__ = patched_init

                    # Attempt 1: load from local cache.
                    try:
                        with _force_offline_env():
                            _colbert_reranker = ColbertReranker(
                                model_name="answerdotai/answerai-colbert-small-v1", column="content"
                            )
                    except Exception as e:
                        logger.debug("Failed to load ColBERT reranker from cache: %s", e)
                        if is_offline:
                            raise
                        # Attempt 2: online download (first-run path).
                        _colbert_reranker = ColbertReranker(
                            model_name="answerdotai/answerai-colbert-small-v1", column="content"
                        )

                    logger.info("ColBERT reranker loaded in %.2fs", time.time() - t0)
                except (ImportError, ModuleNotFoundError) as e:
                    # rerankers package not installed — ColBERT is optional
                    logger.warning(
                        "ColBERT reranker unavailable (%s). "
                        "Install with: pip install rerankers[transformers]. "
                        "Falling back to cross-encoder only.",
                        e,
                    )
                    _colbert_reranker = None  # sentinel: skip ColBERT in search
                except Exception as e:
                    _handle_load_error(e, "answerdotai/answerai-colbert-small-v1")
    return _colbert_reranker


def get_gemini_circuit_breaker() -> Any:
    """Get the circuit breaker for Gemini API calls only.

    This breaker is intentionally **not** shared with Ollama — a Gemini DNS
    failure must never prevent local Ollama from being called.
    """
    global _gemini_breaker
    if _gemini_breaker is None:
        with _gemini_breaker_lock:
            if _gemini_breaker is None:
                from audiobench.memory.llm_caller import CircuitBreaker
                # Trip after 3 consecutive Gemini failures; probe again after 60 s.
                _gemini_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    return _gemini_breaker


def get_llm_circuit_breaker() -> Any:
    """Backward-compatible alias — returns the Gemini-only circuit breaker."""
    return get_gemini_circuit_breaker()


def reset_gemini_circuit_breaker() -> None:
    """Manually reset the Gemini circuit breaker to CLOSED.

    Call this when the user switches model mid-session (e.g. /set model default)
    so a previous network outage does not keep blocking future synthesis calls.
    """
    breaker = get_gemini_circuit_breaker()
    breaker.reset()


def pre_warm_retrieval_pipeline() -> None:
    """Warm up all models sequentially to avoid load penalties during requests.

    Called at daemon startup and FastAPI lifespan.
    """
    logger.info("Pre-warming retrieval pipeline models...")
    t0 = time.time()

    get_boundary_embedder()
    get_primary_embedder()
    get_reranker()
    get_colbert_reranker()  # optional — won't crash if rerankers pkg is absent

    logger.info("Retrieval pipeline fully warmed in %.2fs", time.time() - t0)


def get_memory_store() -> Any:
    """Get the thread-safe singleton MemoryStore instance."""
    global _memory_store
    if _memory_store is None:
        with _memory_store_lock:
            if _memory_store is None:
                from audiobench.memory.memory_store import MemoryStore
                _memory_store = MemoryStore()
    return _memory_store


def get_segment_store() -> Any:
    """Get the thread-safe singleton SegmentVectorStore instance."""
    global _segment_store
    if _segment_store is None:
        with _segment_store_lock:
            if _segment_store is None:
                from audiobench.memory.memory_store import SegmentVectorStore
                _segment_store = SegmentVectorStore()
    return _segment_store

