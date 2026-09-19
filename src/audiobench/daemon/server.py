"""Daemon server — asyncio Unix socket server for the memory layer.

Holds all ML models resident in memory and serves requests from CLI commands
via newline-delimited JSON over a Unix domain socket.

Protocol: {"cmd": str, "args": {...}, "request_id": str}  →  {"success": bool, "data": {...}, "request_id": str}
"""

from __future__ import annotations

import asyncio
import ctypes
import fcntl
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from audiobench.playback.service import PlaybackService

# Best-effort glibc page release after large embedding batches.  Only
# available on Linux; silently skipped on other platforms.
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None


def _release_pages() -> None:
    """Return glibc heap pages to the OS after a large embedding batch.

    gc.collect() alone does nothing for PyTorch's CPU caching allocator.
    malloc_trim(0) asks glibc to release free arenas above the current
    high-water mark.  Confirmed to return ~75 % of batch-allocated pages
    on this machine.  No-op if libc is unavailable.
    """
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(ctypes.c_size_t(0))

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import DaemonRequest, DaemonResponse, DaemonError
from audiobench.memory.chunking import content_aware_router
from audiobench.memory.memory_store import MemoryStore, SegmentVectorStore
from audiobench.memory.singletons import (
    get_primary_embedder,
    get_reranker,
    pre_warm_retrieval_pipeline,
)
from audiobench.storage.models import TranscriptionRecord
from audiobench.supervisor.registry import upsert_process

logger = get_logger("daemon.server")

# Module-level state (lives for the lifetime of the daemon process)
_start_time: float = 0.0
_last_request_time: float = 0.0
_active_clients: int = 0
_query_cache: Any = None
_memory_store: MemoryStore | None = None
_segment_store: SegmentVectorStore | None = None
import threading

from audiobench.daemon.sweep_state import SweepState  # type hinting

_sweep_state: SweepState | None = None  # loaded lazily

_optimize_lock = threading.Lock()
_optimize_in_progress: bool = False


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _get_store() -> MemoryStore:
    """Get the module-level MemoryStore instance (must have been initialised)."""
    if _memory_store is None:
        raise RuntimeError("MemoryStore not initialised — was pre_warm called?")
    return _memory_store


def _get_segment_store() -> SegmentVectorStore:
    """Get the module-level SegmentVectorStore instance (must have been initialised)."""
    if _segment_store is None:
        raise RuntimeError("SegmentVectorStore not initialised — was pre_warm called?")
    return _segment_store


def _handle_ping(args: dict[str, Any]) -> dict[str, Any]:
    """Return daemon health and stats."""
    import psutil

    from audiobench.daemon.intelligence.calibration import get_calibration_tracker
    from audiobench.daemon.intelligence.timing_model import get_timing_model
    from audiobench.daemon.sweep_state import get_sweep_state

    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    sweep_consecutive_failures = 0
    try:
        state = get_sweep_state()
        queue_depth = state.unindexed_transcript_count() + state.pending_segment_count()
        sweep_consecutive_failures = state.consecutive_sweep_failures
    except Exception:
        queue_depth = 0

    store_version = _get_store().model_version if _memory_store else "unknown"

    tracker = get_calibration_tracker()
    timing = get_timing_model()

    return {
        "alive": True,
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 2),
        "embedding_model_version": store_version,
        "memory_mb": round(memory_mb, 2),
        "queue_depth": queue_depth,
        "sweep_consecutive_failures": sweep_consecutive_failures,
        "models": {
            "embedding": store_version,
        },
        "calibration": tracker.get_summary(),
        "timing": timing.get_summary()
    }


def _handle_embed(args: dict[str, Any]) -> dict[str, Any]:
    """Embed and write a single expression to LanceDB."""
    _get_store().write_node(
        expression_id=int(args["expression_id"]),
        content=str(args["content"]),
        source_type=str(args["source_type"]),
        speaker=args.get("speaker"),
    )
    return {"embedded": True}


def _handle_search(args: dict[str, Any]) -> dict[str, Any]:
    """Hybrid search over memory store."""
    query = str(args["query"])
    top_k = int(args.get("top_k", 5))
    speaker_filter = args.get("speaker_filter")
    audio_file_id = args.get("audio_file_id")
    work_id = args.get("work_id")
    hyde_document = args.get("hyde_document")
    use_bm25 = args.get("use_bm25", True)
    use_dense = args.get("use_dense", True)
    use_colbert = args.get("use_colbert", True)

    results = _get_store().search(
        query=query,
        top_k=top_k,
        speaker_filter=speaker_filter,
        audio_file_id=audio_file_id,
        work_id=work_id,
        hyde_document=hyde_document,
        use_bm25=use_bm25,
        use_dense=use_dense,
        use_colbert=use_colbert,
    )

    return {"results": results}


def _handle_delete(args: dict[str, Any]) -> dict[str, Any]:
    """Delete an expression node from LanceDB."""
    _get_store().delete_node(int(args["expression_id"]))
    return {"deleted": True}


def _handle_delete_batch(args: dict[str, Any]) -> dict[str, Any]:
    """Delete a batch of expression nodes from LanceDB in a single operation."""
    expression_ids = [int(eid) for eid in args.get("expression_ids", [])]
    if expression_ids:
        _get_store().delete_node_batch(expression_ids)
    return {"deleted": len(expression_ids)}


def _handle_status(args: dict[str, Any]) -> dict[str, Any]:
    """Return daemon and store statistics."""
    from audiobench.daemon.intelligence.timing_model import get_timing_model
    store = _get_store()
    node_count = store.count_nodes()
    timing_summary = get_timing_model().get_summary()
    pb = _get_playback_service().status() if _playback_service is not None else {"playing": False}
    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "embedding_model_version": store.model_version,
        "total_nodes": node_count,
        "timing_model": timing_summary,
        "playback": pb,
    }


def _handle_register_process(args: dict[str, Any]) -> dict[str, Any]:
    """Register a CLI subprocess in the managed_processes table."""
    name = str(args.get("name", ""))
    pid = args.get("pid")
    upsert_process(name, "running", pid=pid)
    return {"ok": True}


def _handle_unregister_process(args: dict[str, Any]) -> dict[str, Any]:
    """Mark a CLI subprocess as stopped in the managed_processes table."""
    name = str(args.get("name", ""))
    exit_code = args.get("exit_code", 0)
    upsert_process(name, "stopped", last_exit_code=exit_code)
    return {"ok": True}


def _handle_log_events(args: dict[str, Any]) -> dict[str, Any]:
    """Receive a batch of event payloads from a CLI process and record them
    via the daemon's in-process ObservabilitySubscriber.

    This lets CLI subprocesses forward observability events to the daemon's
    already-open journal.db writer thread instead of each opening their own
    connection."""
    from audiobench.observatory.subscriber import get_subscriber

    payloads: list[dict] = args.get("payloads", [])
    if not isinstance(payloads, list):
        return {"ok": False, "error": "payloads must be a list"}
    sub = get_subscriber()
    for payload in payloads:
        if isinstance(payload, dict):
            sub.record(**payload)
    return {"ok": True, "count": len(payloads)}


def _handle_chunk(args: dict[str, Any]) -> dict[str, Any]:
    """Run text through the chunking pipeline and return chunks."""
    text = str(args.get("text", ""))
    # audio_file_id is available but chunking is purely semantic/text-based right now
    diarized = bool(args.get("diarized", False))

    # We don't have the diarized segments here. The protocol currently sends text and a diarized bool.
    # To fully support diarization in the daemon, we should accept a segments list in args.
    # For now, we pass None to segments since the protocol doesn't send them yet.
    # We will update protocol later if needed.
    segments = args.get("segments")

    chunks = content_aware_router(text, diarized_segments=segments if diarized else None)

    results = []
    for c in chunks:
        res = {"content": c.content, "uuid": c.uuid, "tier": c.tier}
        if c.speaker:
            res["speaker"] = c.speaker
        results.append(res)

    return {"chunks": results}


def _handle_rerank(args: dict[str, Any]) -> dict[str, Any]:
    """Rerank search results using the CrossEncoder.

    Inputs are truncated before forming pairs to prevent tensor size mismatches.
    CrossEncoders (ms-marco-MiniLM series) have a 512-token limit for the
    concatenated [CLS] query [SEP] doc [SEP] input. Long HyDE documents or
    long segment texts can exceed this, causing a RuntimeError.

    Conservative limits: query ≤ 512 chars, doc ≤ 1024 chars — well within
    the 512-token window even at average token density of ~4 chars/token.
    """
    query = str(args["query"])[:512]
    docs = args.get("docs", [])

    if not docs:
        return {"scores": []}

    # Truncate each document to prevent tensor mismatch
    truncated_docs = [str(d)[:1024] for d in docs]

    reranker = get_reranker()
    pairs = [(query, doc) for doc in truncated_docs]

    # predict returns a numpy array, must convert to list
    scores = reranker.predict(pairs).tolist()

    return {"scores": scores}


def _handle_embed_query(args: dict[str, Any]) -> dict[str, Any]:
    """Embed a query string using the daemon's warm Nomic model.

    Returns a 768-dim vector. The caller (DenseStream/ColBERTStream) uses this
    vector for ANN search rather than cold-loading the model in the CLI process.
    """
    from audiobench.memory.singletons import get_primary_inference_lock

    text = str(args["text"])[:12_000]  # bound token count
    prefixed = f"search_query: {text}"
    model = get_primary_embedder()
    with get_primary_inference_lock():
        vector = model.encode(prefixed).tolist()
    return {"vector": vector}


def _handle_embed_segment(args: dict[str, Any]) -> dict[str, Any]:
    """Embed a single audio segment and write it to the segment_vectors table.

    Generates the vector using the warm Nomic model, then delegates the
    upsert to SegmentVectorStore.upsert_segment_with_vector so the table
    write and the embedding happen in the same daemon process.
    """
    from audiobench.memory.singletons import get_primary_inference_lock

    segment_id = int(args["segment_id"])
    text = str(args["text"])[:12_000]  # bound token count
    start_time = float(args["start_time"])
    end_time = float(args["end_time"])
    source_file = str(args["source_file"])

    prefixed = f"search_document: {text}"
    model = get_primary_embedder()
    with get_primary_inference_lock():
        vector = model.encode(prefixed).tolist()

    _get_segment_store().upsert_segment_with_vector(
        segment_id=segment_id,
        text=text,
        start_time=start_time,
        end_time=end_time,
        source_file=source_file,
        vector=vector,
    )
    return {"embedded": True}


def _handle_check_cache(args: dict[str, Any]) -> dict[str, Any]:
    """Check semantic cache."""
    from audiobench.memory.memory_store import QueryCacheStore

    global _query_cache
    if "_query_cache" not in globals() or _query_cache is None:
        _query_cache = QueryCacheStore()

    query = str(args["query"])
    dist = float(args.get("distance_threshold", 0.05))
    result = _query_cache.check_cache(query, dist)
    return {"result": result}


def _handle_write_cache(args: dict[str, Any]) -> dict[str, Any]:
    """Write to semantic cache."""
    from audiobench.memory.memory_store import QueryCacheStore

    global _query_cache
    if "_query_cache" not in globals() or _query_cache is None:
        _query_cache = QueryCacheStore()

    query = str(args["query"])
    answer = str(args["answer"])
    hyde_document = args.get("hyde_document")
    _query_cache.write_cache(query, answer, hyde_document)
    return {"cached": True}


async def _handle_pipeline(args: dict[str, Any]) -> Any:
    """Execute a multi-step pipeline."""
    from audiobench.daemon.pipeline import PipelineExecutor
    executor = PipelineExecutor()
    async for frame in executor.run(args):
        yield frame


async def _handle_operate(args: dict[str, Any]) -> Any:
    """Execute a semantic operation via operators.py."""
    from audiobench.daemon.operators import operate_on
    target = args["target"]
    verb = str(args["verb"])
    context = args.get("context", {})
    store = _get_store()
    async for frame in operate_on(target, verb, context, store):
        yield frame


def _handle_autocomplete(args: dict[str, Any]) -> dict[str, Any]:
    """Semantic autocomplete fast-path."""
    from audiobench.daemon.autocomplete import get_autocomplete_index
    from audiobench.memory.singletons import get_primary_embedder, get_primary_inference_lock

    prefix = str(args.get("prefix", ""))
    if not prefix:
        return {"results": []}

    index = get_autocomplete_index()
    if not index.ready:
        return {"error": "INDEX_NOT_READY"}

    model = get_primary_embedder()
    with get_primary_inference_lock():
        vector = model.encode(f"search_query: {prefix}").tolist()

    results = index.lookup(vector, k=int(args.get("top_k", 5)))

    expression_ids = [r["expression_id"] for r in results if "expression_id" in r]
    if expression_ids:
        from audiobench.storage.expression_repository import ExpressionRepository
        repo = ExpressionRepository()
        expr_map = repo.get_by_ids(expression_ids)

        enriched_results = []
        for r in results:
            expr_id = r.get("expression_id")
            if expr_id in expr_map:
                expr = expr_map[expr_id]
                r["speaker"] = expr.speaker
                r["source_type"] = expr.source_type
                enriched_results.append(r)
        results = enriched_results

    return {"results": results}


def _handle_confirm_inference(args: dict[str, Any]) -> dict[str, Any]:
    """Confirm a proposed system inference."""
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session
    from audiobench.daemon.intelligence.calibration import get_calibration_tracker

    expression_id = int(args["expression_id"])
    with get_session() as session:
        session.execute(
            sql_text("UPDATE expressions SET inference_status = 'confirmed' WHERE id = :eid"),
            {"eid": expression_id}
        )
        session.commit()

    get_calibration_tracker().record_confirm(expression_id)
    return {"status": "ok", "expression_id": expression_id, "action": "confirmed"}


def _handle_reject_inference(args: dict[str, Any]) -> dict[str, Any]:
    """Reject a proposed system inference."""
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session
    from audiobench.daemon.intelligence.calibration import get_calibration_tracker

    expression_id = int(args["expression_id"])
    with get_session() as session:
        session.execute(
            sql_text("UPDATE expressions SET inference_status = 'rejected' WHERE id = :eid"),
            {"eid": expression_id}
        )
        session.commit()

    get_calibration_tracker().record_reject(expression_id)
    return {"status": "ok", "expression_id": expression_id, "action": "rejected"}


def _handle_authorize_proposal(args: dict[str, Any]) -> dict[str, Any]:
    """Authorize a daemon_proposal: write confirmed to DB and hot-register the operator."""
    from audiobench.daemon.intelligence.operator_registry import get_operator_registry

    expression_id = int(args["expression_id"])
    get_operator_registry().authorize(expression_id)
    return {"status": "ok", "expression_id": expression_id, "action": "authorized"}


def _handle_get_inferences(args: dict[str, Any]) -> dict[str, Any]:
    """Return proposed system inferences for the InferencesFeed panel."""
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session

    with get_session() as session:
        rows = session.execute(
            sql_text("""
            SELECT id, source_type, content, created_at
            FROM expressions
            WHERE source_type IN ('system_inference', 'drift_observation', 'potential_relation')
              AND inference_status = 'proposed'
            ORDER BY created_at DESC
            LIMIT 200
            """)
        ).fetchall()

    inferences = [
        {"id": r[0], "source_type": r[1], "content": r[2], "created_at": str(r[3])}
        for r in rows
    ]
    return {"inferences": inferences}


def _handle_get_proposals(args: dict[str, Any]) -> dict[str, Any]:
    """Return pending daemon proposals for the ProposalsFeed panel."""
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session

    with get_session() as session:
        rows = session.execute(
            sql_text("""
            SELECT id, source_type, content, created_at
            FROM expressions
            WHERE source_type = 'daemon_proposal'
              AND inference_status IN ('proposed', 'deferred')
            ORDER BY created_at DESC
            LIMIT 100
            """)
        ).fetchall()

    proposals = [
        {"id": r[0], "source_type": r[1], "content": r[2], "created_at": str(r[3])}
        for r in rows
    ]
    return {"proposals": proposals}



def _handle_optimize(args: dict[str, Any]) -> dict[str, Any]:
    """Run LanceDB optimize on all tables (on-demand via CLI)."""
    from audiobench.daemon.lancedb_optimizer import _do_optimize_all_tables
    global _optimize_in_progress

    with _optimize_lock:
        if _optimize_in_progress:
            return {"error": "Optimization already in progress"}
        _optimize_in_progress = True

    try:
        result = _do_optimize_all_tables(triggered_by="cli_command")
        return result
    finally:
        with _optimize_lock:
            _optimize_in_progress = False


def _handle_reload_settings(args: dict[str, Any]) -> dict[str, Any]:
    """Invalidate the in-memory settings cache so the daemon reloads on next access."""
    from audiobench.core.settings import invalidate_settings_cache
    invalidate_settings_cache()
    return {"status": "ok"}


_playback_service: PlaybackService | None = None


def _get_playback_service() -> PlaybackService:
    global _playback_service
    if _playback_service is None:
        from audiobench.playback.service import PlaybackService

        _playback_service = PlaybackService()
    return _playback_service


def _handle_playback_play(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.play(
        args["file_path"],
        start_pos=float(args.get("start_pos", 0.0)),
        speed=float(args.get("speed", 1.0)),
        audio_file_id=args.get("audio_file_id"),
        transcription_id=args.get("transcription_id"),
    )


def _handle_playback_sync_session(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.sync_session(
        args["file_path"],
        transcription_id=args.get("transcription_id"),
        position=float(args.get("position", 0.0)),
        audio_file_id=args.get("audio_file_id"),
    )


def _handle_playback_seek(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.seek(float(args["position"]))


def _handle_playback_seek_relative(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.seek_relative(float(args.get("offset", 0.0)))


def _handle_playback_pause(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.pause()


def _handle_playback_resume(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.resume()


def _handle_playback_toggle(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.toggle()


def _handle_playback_speed(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.set_speed(float(args["speed"]))


def _handle_playback_stop(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    return svc.stop()


def _handle_playback_status(args: dict[str, Any]) -> dict[str, Any]:
    if _playback_service is None:
        return {
            "playing": False,
            "paused": False,
            "position": 0.0,
            "duration": 0.0,
            "speed": 1.0,
            "file": None,
            "audio_file_id": None,
            "transcription_id": None,
        }
    return _playback_service.status()


def _handle_playback_context(args: dict[str, Any]) -> dict[str, Any]:
    svc = _get_playback_service()
    pos = float(args["position"]) if "position" in args and args["position"] is not None else None
    window = int(args.get("window", 3))
    return svc.get_context_window(position=pos, window=window)


_HANDLERS: dict[str, Any] = {
    "reload_settings": _handle_reload_settings,
    "ping": _handle_ping,
    "embed": _handle_embed,
    "search": _handle_search,
    "delete": _handle_delete,
    "delete_batch": _handle_delete_batch,
    "status": _handle_status,
    "chunk": _handle_chunk,
    "rerank": _handle_rerank,
    "check_cache": _handle_check_cache,
    "write_cache": _handle_write_cache,
    "embed_query": _handle_embed_query,
    "embed_segment": _handle_embed_segment,
    "pipeline": _handle_pipeline,
    "operate": _handle_operate,
    "autocomplete": _handle_autocomplete,
    "confirm_inference": _handle_confirm_inference,
    "reject_inference": _handle_reject_inference,
    "authorize_proposal": _handle_authorize_proposal,
    "get_inferences": _handle_get_inferences,
    "get_proposals": _handle_get_proposals,
    "register_process": _handle_register_process,
    "unregister_process": _handle_unregister_process,
    "log_events": _handle_log_events,
    "optimize": _handle_optimize,
    "playback_play": _handle_playback_play,
    "playback_sync_session": _handle_playback_sync_session,
    "playback_seek": _handle_playback_seek,
    "playback_seek_relative": _handle_playback_seek_relative,
    "playback_pause": _handle_playback_pause,
    "playback_resume": _handle_playback_resume,
    "playback_toggle": _handle_playback_toggle,
    "playback_speed": _handle_playback_speed,
    "playback_stop": _handle_playback_stop,
    "playback_status": _handle_playback_status,
    "playback_context": _handle_playback_context,
}


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------


import inspect
from collections.abc import AsyncGenerator


async def _dispatch(raw: str) -> AsyncGenerator[str, None]:
    """Parse a JSON request line, call handler, yield JSON response lines."""
    global _last_request_time
    _last_request_time = time.time()

    request_id = "unknown"
    try:
        req: DaemonRequest = json.loads(raw)
        request_id = req.get("request_id", "unknown")  # type: ignore[assignment]
        cmd = req.get("cmd", "")
        args = req.get("args", {})

        handler = _HANDLERS.get(cmd)
        if handler is None:
            raise ValueError(f"Unknown command: {cmd!r}")

        if inspect.isasyncgenfunction(handler):
            async for frame in handler(args):
                frame["request_id"] = request_id
                yield json.dumps(frame)
        else:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, handler, args)
            response: DaemonResponse = {"status": "ok", "success": True, "data": data, "request_id": request_id}
            yield json.dumps(response)
    except Exception as exc:
        logger.exception("Error handling request %s", request_id)
        error_payload: DaemonError = {
            "code": "OPERATION_FAILED",
            "message": str(exc),
            "request_id": str(request_id)
        }
        response: DaemonResponse = {"status": "error", "success": False, "error": error_payload, "request_id": str(request_id)}
        yield json.dumps(response)


# ---------------------------------------------------------------------------
# asyncio connection handler
# ---------------------------------------------------------------------------


async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a single client connection — reads one request, yields multiple response frames."""
    peer = writer.get_extra_info("peername", "<unknown>")
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not raw:
            return
        raw_str = raw.decode("utf-8").strip()

        # Before logging, check if it's autocomplete (high-frequency, low-signal)
        try:
            parsed_cmd = json.loads(raw_str).get("cmd", "") if raw_str else ""
        except json.JSONDecodeError:
            parsed_cmd = ""

        if parsed_cmd != "autocomplete":
            logger.debug("Received from %s: %s", peer, raw_str[:120])

        async for response_str in _dispatch(raw_str):
            writer.write((response_str + "\n").encode("utf-8"))
            await writer.drain()

    except TimeoutError:
        logger.warning("Connection from %s timed out", peer)
        try:
            err = {"code": "TIMEOUT", "message": "Connection timed out", "request_id": "unknown"}
            res = {"status": "error", "success": False, "error": err, "request_id": "unknown"}
            writer.write((json.dumps(res) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    except (ConnectionResetError, BrokenPipeError):
        logger.warning("Client %s disconnected before response could be sent", peer)
    except Exception:
        logger.exception("Unexpected error for connection from %s", peer)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

# Module-level lock FD — held open for the entire daemon lifetime.
# fcntl advisory locks are tied to the FD's inode, not the path, so this
# must stay open or the exclusive lock is released immediately.
_LOCK_FD: int | None = None


def _check_existing_daemon(pid_path: Path) -> int | None:
    """Informational PID check — fast path for a clear user-facing error message.

    Returns the running PID if another daemon instance is verifiably alive,
    or None if the PID file is absent, corrupt, or stale (process is dead).

    This is a UX layer only, not the authoritative guard.  The fcntl exclusive
    lock in _acquire_exclusive_lock() is the real mutex.  If the two ever
    disagree, the flock result is authoritative.
    """
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None  # corrupt file — treat as stale

    # os.kill(pid, 0) sends no signal but probes whether the PID is alive.
    try:
        os.kill(pid, 0)
        return pid  # PID exists and we can signal it — another daemon is running
    except ProcessLookupError:
        return None  # PID is dead — stale file from a previous crash
    except PermissionError:
        # PID exists but is owned by a different user.  Still alive.
        return pid


def _acquire_exclusive_lock(lock_path: Path) -> bool:
    """Acquire an exclusive advisory lock on lock_path.

    This is the authoritative single-instance guard.  The kernel enforces
    mutual exclusion atomically — there is no TOCTOU race.

    Returns True if the lock was acquired (this process is the sole owner).
    Returns False if another process already holds the lock.

    The acquired FD is stored in _LOCK_FD and must remain open for the entire
    daemon lifetime.  Closing it releases the lock immediately.  The kernel
    releases it automatically and atomically when the process dies, which
    occurs before the supervisor's Popen() call for any replacement process
    (because proc.poll() returning non-None proves waitpid() already fired).
    """
    global _LOCK_FD
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # non-blocking
        _LOCK_FD = fd  # keep open — releasing closes the lock
        return True
    except OSError:
        os.close(fd)
        return False  # another process holds the lock


def _write_pid_file(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_path)


def _cleanup(socket_path: Path, pid_path: Path) -> None:
    # Unlink runtime artifacts first so the path is free for any replacement.
    for p in (socket_path, pid_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    # Release the exclusive lock last — this is the moment another daemon
    # instance is allowed to acquire it.  The FD's inode lock is independent
    # of the filename, so unlinking the lock file above does not release it.
    global _LOCK_FD
    if _LOCK_FD is not None:
        try:
            os.close(_LOCK_FD)
        except OSError:
            pass
        _LOCK_FD = None
    logger.info("Cleanup complete")


def _do_sweep_once() -> None:
    """Perform one pass of the RAG consistency sweep (called from the loop)."""
    from sqlalchemy import text as sql_text

    from audiobench.daemon.sweep_state import get_sweep_state
    from audiobench.memory.enums import SourceType
    from audiobench.storage.expression_repository import ExpressionRepository

    state = get_sweep_state()

    # ── Pick up any new work added since last tick ─────────────────────────
    _refresh_sweep_state_incremental(state)

    # ── Expression sweep (transcript deque → expression chunks → LanceDB) ──
    tx_ids = state.pop_transcript_batch(size=20)
    if not tx_ids:
        logger.info("RAG sweep: all transcripts indexed (deque empty).")
    else:
        logger.info("RAG sweep: indexing %d transcripts from deque...", len(tx_ids))
        expr_repo = ExpressionRepository()
        success_ids: list[int] = []
        failed_ids: list[int] = []

        with get_session() as session:
            # Eager load the required fields to avoid lazy-loading outside session.
            # SQLite text() doesn't accept tuple bindings for IN — expand to
            # individual named placeholders (:id0, :id1, ...) like the segments loop.
            tx_named = {f"id{i}": tid for i, tid in enumerate(tx_ids)}
            tx_placeholders = ", ".join(f":id{i}" for i in range(len(tx_ids)))
            rows = session.execute(
                sql_text(
                    f"SELECT t.id, t.full_text, t.audio_file_id, t.language_probability, t.language, af.work_id "
                    f"FROM transcriptions t "
                    f"LEFT JOIN audio_files af ON t.audio_file_id = af.id "
                    f"WHERE t.id IN ({tx_placeholders})"
                ),
                tx_named,
            ).mappings().all()
            tx_data = list(rows)

        for tx in tx_data:
            try:
                text = tx["full_text"] or ""
                if not text.strip():
                    success_ids.append(tx["id"])
                    continue

                chunks = content_aware_router(text)
                if chunks:
                    from audiobench.memory.chunking import _clean_text, parent_child_grouper
                    from audiobench.memory.enums import RelationType

                    parent_groups = parent_child_grouper(chunks)
                    nodes = []

                    # Tier 1: full-text document node (not embedded in LanceDB)
                    cleaned_text = _clean_text(text)
                    t1_expr = expr_repo.register(
                        content=cleaned_text,
                        source_type=SourceType.AUDIO_TRANSCRIPT.value,
                        source_id=tx["id"],
                        work_id=tx["work_id"],
                        graph_role="sweep_document",
                    )

                    for pg in parent_groups:
                        # Tier 2: passage/context group node (not embedded in LanceDB)
                        t2_expr = expr_repo.register(
                            content=pg.parent_text,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value,
                            source_id=tx["id"],
                            work_id=tx["work_id"],
                            graph_role="sweep_passage",
                        )
                        expr_repo.link(t2_expr.id, t1_expr.id, RelationType.SOURCE.value)

                        # Tier 3: embedded leaf node (written to LanceDB)
                        for child in pg.children:
                            t3_expr = expr_repo.register(
                                content=child.content,
                                source_type=SourceType.AUDIO_TRANSCRIPT.value,
                                source_id=tx["id"],
                                speaker=child.speaker,
                                work_id=tx["work_id"],
                                graph_role="sweep_chunk",
                            )
                            expr_repo.link(t3_expr.id, t2_expr.id, RelationType.SOURCE.value)

                            nodes.append({
                                "expression_id": t3_expr.id,
                                "content": child.content,
                                "source_type": SourceType.AUDIO_TRANSCRIPT.value,
                                "speaker": child.speaker,
                                "audio_file_id": tx["audio_file_id"],
                                "confidence": tx["language_probability"],
                                "original_language": tx["language"],
                                "work_id": t3_expr.work_id,
                            })

                    if nodes:
                        _get_store().batch_write_nodes(nodes)
                        from audiobench.daemon.lancedb_optimizer import increment_unoptimized_writes
                        increment_unoptimized_writes(len(nodes))
                        _maybe_trigger_threshold_optimize()

                success_ids.append(tx["id"])
            except Exception as e:
                logger.error(
                    "RAG sweep: failed to index transcript %d: %s",
                    tx["id"], e,
                )
                failed_ids.append(tx["id"])

        if success_ids:
            with get_session() as session:
                sid_named = {f"id{i}": sid for i, sid in enumerate(success_ids)}
                sid_placeholders = ", ".join(f":id{i}" for i in range(len(success_ids)))
                session.execute(
                    sql_text(f"UPDATE transcriptions SET is_indexed = 1 WHERE id IN ({sid_placeholders})"),
                    sid_named,
                )
                session.commit()

        logger.info(
            "RAG sweep: indexed %d transcripts (%d failed).",
            len(success_ids), len(failed_ids),
        )
        if failed_ids:
            state.requeue_transcripts(failed_ids)

    # ── Search query sweep (query deque → expressions → LanceDB) ────────────
    # Ingests completed search queries as SEARCH_QUERY + SEARCH_SYNTHESIS
    # expressions, builds ELABORATES / THEMATIC / TEMPORAL relations, and
    # embeds both into LanceDB.  First pass retroactively covers all existing
    # sessions; subsequent passes handle new queries within 5 minutes.
    query_ids = state.pop_query_batch(size=50)
    if not query_ids:
        logger.debug("RAG sweep: all search queries ingested (deque empty).")
    else:
        logger.info(
            "RAG sweep: ingesting %d search queries as expressions...", len(query_ids)
        )
        from audiobench.memory.search_ingester import SearchIngester
        _ingester = SearchIngester()
        q_success_ids: list[int] = []
        q_failed_ids: list[int] = []
        for qid in query_ids:
            try:
                _ingester.ingest_query(qid)
                q_success_ids.append(qid)
            except Exception as exc:
                logger.error(
                    "RAG sweep: failed to ingest search query %d: %s", qid, exc
                )
                q_failed_ids.append(qid)
        if q_failed_ids:
            state.requeue_queries(q_failed_ids)
        logger.info(
            "RAG sweep: ingested %d search queries as expressions (%d failed).",
            len(q_success_ids), len(q_failed_ids),
        )

    # ── Search summary sweep (session deque → SEARCH_SESSION_SUMMARY exprs) ─
    # Ingests AI-generated session summaries as expressions with INSPIRED_BY
    # links to all constituent query expressions.
    summary_session_ids = state.pop_summary_session_batch(size=20)
    if summary_session_ids:
        logger.info(
            "RAG sweep: ingesting %d session summaries as expressions...",
            len(summary_session_ids),
        )
        from audiobench.memory.search_ingester import SearchIngester as _SI
        _sum_ingester = _SI()
        s_failed_ids: list[int] = []
        for sid in summary_session_ids:
            try:
                _sum_ingester.ingest_session_summary(sid)
            except Exception as exc:
                logger.error(
                    "RAG sweep: failed to ingest session summary %d: %s", sid, exc
                )
                s_failed_ids.append(sid)
        if s_failed_ids:
            state.requeue_summary_sessions(s_failed_ids)
        logger.info(
            "RAG sweep: ingested %d session summaries (%d failed).",
            len(summary_session_ids) - len(s_failed_ids), len(s_failed_ids),
        )

    # ── Segment sweep (segment deque → batch embed → LanceDB + SQLite) ─────

    #
    # Sub-batch at 64 segments per iteration rather than pulling 256 at once.
    #
    # Rationale from measurement (2026-07-20): a 256-segment monolithic batch
    # peaked at ~8 GB RSS (model.encode accumulates all 4×64 sub-batches before
    # returning, LanceDB builds one Arrow table for all 256 rows simultaneously).
    # Processing 64 segments per pass limits each iteration to one model forward
    # pass + one 64-row LanceDB write, then malloc_trim(0) returns the freed
    # glibc arena to the OS before the next pass.  Four passes instead of one,
    # but each stays well under the 4 GB intelligence-task gate threshold.
    _SEG_SUB_BATCH = 64
    seg_ids = state.pop_segment_batch(size=_SEG_SUB_BATCH)
    if not seg_ids:
        logger.info("RAG sweep: all segments vectorized (deque empty).")
    else:
        logger.info(
            "RAG sweep: vectorizing %d segments from deque (sub-batch %d)...",
            len(seg_ids), _SEG_SUB_BATCH,
        )
        with get_session() as session:
            named_params = {f"id{i}": sid for i, sid in enumerate(seg_ids)}
            placeholders = ", ".join(f":id{i}" for i in range(len(seg_ids)))
            rows = session.execute(
                sql_text(
                    f"SELECT s.id, s.text, s.start_time, s.end_time, af.file_path "
                    f"FROM segments s "
                    f"JOIN transcriptions t ON s.transcription_id = t.id "
                    # LEFT JOIN so segments whose transcription has audio_file_id=NULL
                    # are returned (with af.file_path=NULL) rather than silently
                    # dropped by an INNER JOIN, which would leave them stuck in the
                    # deque forever re-queuing on every sweep tick.
                    f"LEFT JOIN audio_files af ON t.audio_file_id = af.id "
                    f"WHERE s.id IN ({placeholders})"
                ),
                named_params,
            ).mappings().all()
            rows = list(rows)

        # Segments whose transcription has no linked audio file cannot be
        # vectorized — they have no audio content to embed against.  Drain
        # them permanently so they stop re-queuing on every sweep tick.
        # NOTE: vector_indexed=1 here means "permanently skipped — no audio
        # file, never embedded," NOT "successfully embedded and searchable."
        # This deliberately overloads the flag to avoid a schema change for
        # what should be a rare edge case (test fixtures leaking into the
        # production DB).  See also: scripts/cleanup_test_fixtures.py.
        ineligible_ids = [int(r["id"]) for r in rows if r["file_path"] is None]
        if ineligible_ids:
            logger.warning(
                "RAG sweep: %d segment(s) have no linked audio file "
                "(audio_file_id=NULL on their transcription). Marking as "
                "permanently skipped (vector_indexed=1, never embedded): %s",
                len(ineligible_ids),
                ineligible_ids,
            )
            with get_session() as skip_upd:
                skip_params = {f"id{i}": sid for i, sid in enumerate(ineligible_ids)}
                skip_placeholders = ", ".join(f":id{i}" for i in range(len(ineligible_ids)))
                skip_upd.execute(
                    sql_text(
                        f"UPDATE segments SET vector_indexed=1 WHERE id IN ({skip_placeholders})"
                    ),
                    skip_params,
                )
                skip_upd.commit()

        rows = [r for r in rows if r["file_path"] is not None]

        if rows:
            from audiobench.memory.singletons import get_primary_inference_lock
            seg_store = _get_segment_store()
            model = get_primary_embedder()
            try:
                import psutil as _psutil
                _proc = _psutil.Process()
                state.seg_batch_count += 1
                _batch_num = state.seg_batch_count
                texts = [f"search_document: {r['text'][:2_000]}" for r in rows]
                with get_primary_inference_lock():
                    vectors = model.encode(
                        texts, batch_size=64, show_progress_bar=False,
                    ).tolist()
                _rss_post_embed = _proc.memory_info().rss / (1024 * 1024)
                logger.debug(
                    "RAG sweep sub-batch #%d RSS after encode (%d segs): %.1f MB",
                    _batch_num, len(rows), _rss_post_embed,
                )
                mapped_rows = [
                    dict(
                        segment_id=r["id"],
                        text=r["text"],
                        start_time=r["start_time"],
                        end_time=r["end_time"],
                        source_file=r["file_path"],
                    )
                    for r in rows
                ]
                seg_store.batch_upsert_segments(
                    rows=mapped_rows, vectors=vectors
                )
                _rss_post_write = _proc.memory_info().rss / (1024 * 1024)
                logger.debug(
                    "RAG sweep sub-batch #%d RSS after LanceDB write (%d segs): %.1f MB  (delta from encode: %+.1f MB)",
                    _batch_num, len(rows), _rss_post_write, _rss_post_write - _rss_post_embed,
                )
                from audiobench.daemon.lancedb_optimizer import increment_unoptimized_writes
                increment_unoptimized_writes(len(mapped_rows))
                _maybe_trigger_threshold_optimize()

                done_ids = [int(r["id"]) for r in rows]
                with get_session() as upd:
                    upd_params = {f"id{i}": sid for i, sid in enumerate(done_ids)}
                    id_placeholders = ", ".join(f":id{i}" for i in range(len(done_ids)))
                    upd.execute(
                        sql_text(
                            f"UPDATE segments SET vector_indexed=1 WHERE id IN ({id_placeholders})"
                        ),
                        upd_params,
                    )
                    upd.commit()
                logger.info(
                    "RAG sweep: vectorized %d segments in sub-batch.", len(rows)
                )
                # ── Success: reset the segment-level failure counter ────────
                state.consecutive_sweep_failures = 0
                # ── Release pages back to OS before next sweep tick ─────────
                # Frees glibc arena growth from the embedding batch.  Does not
                # affect PyTorch's internal caching allocator pool, but returns
                # ~75 % of the batch-allocated glibc pages (measured).
                _release_pages()
            except Exception as seg_exc:
                logger.error(
                    "RAG sweep: failed to batch vectorize segments: %s", seg_exc
                )
                state.requeue_segments(seg_ids)
                # Track on SweepState immediately — visible in `daemon status`
                # even before the threshold is reached.
                state.consecutive_sweep_failures += 1
                if state.consecutive_sweep_failures >= _SWEEP_MAX_CONSECUTIVE_FAILURES:
                    # Re-raise so the outer circuit breaker fires its exponential
                    # backoff. Without this, the outer loop sees no exception from
                    # the segment block and resets its own counter — making it
                    # blind to exactly this failure shape.
                    raise RuntimeError(
                        f"Segment vectorization failed {state.consecutive_sweep_failures} "
                        f"consecutive time(s) — outer circuit breaker engaging. "
                        f"Last error: {seg_exc}"
                    ) from seg_exc

    reconciled = _reconcile_metadata()
    if reconciled > 0:
        logger.info("RAG sweep: reconciled metadata for %d expressions.", reconciled)

    _maybe_trigger_threshold_optimize()


def _reconcile_metadata() -> int:
    """Drain the reconciliation_queue and update LanceDB metadata."""
    import sqlite3

    from audiobench.observatory.db import get_journal_db_path

    conn = sqlite3.connect(str(get_journal_db_path()), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, expression_id, work_id FROM reconciliation_queue LIMIT 1000"
        ).fetchall()
        if not rows:
            return 0

        store = _get_store()
        for row in rows:
            store.update_expression_work_id(
                expression_id=row["expression_id"],
                work_id=row["work_id"]
            )

        ids = [row["id"] for row in rows]
        conn.execute(
            f"DELETE FROM reconciliation_queue WHERE id IN ({','.join('?' * len(ids))})",
            ids
        )
        conn.commit()
        return len(rows)
    except Exception as e:
        logger.error("RAG sweep: metadata reconciliation failed: %s", e)
        return 0
    finally:
        conn.close()


def _maybe_trigger_threshold_optimize() -> None:
    """Check write threshold and run optimize in background if exceeded."""
    from audiobench.core.settings import get_settings
    from audiobench.daemon.lancedb_optimizer import read_optimize_state

    settings = get_settings()
    threshold = settings.lancedb_optimize_write_threshold
    if threshold <= 0:
        return

    state = read_optimize_state()
    current_writes = state["unoptimized_writes"]

    if current_writes >= threshold:
        global _optimize_in_progress
        with _optimize_lock:
            if _optimize_in_progress:
                return
            _optimize_in_progress = True

        logger.info("Write threshold reached (%d >= %d). Triggering background optimize.", current_writes, threshold)

        def run_bg() -> None:
            from audiobench.daemon.lancedb_optimizer import _do_optimize_all_tables
            global _optimize_in_progress

            try:
                _do_optimize_all_tables(triggered_by="write_threshold")
            except Exception as e:
                logger.error("Background optimize failed: %s", e)
            finally:
                with _optimize_lock:
                    _optimize_in_progress = False

        threading.Thread(target=run_bg, name="lancedb-optimize-bg", daemon=True).start()


def _refresh_sweep_state_incremental(state) -> None:
    """Pick up any transcripts or segments added since the last sweep tick.

    Only queries for IDs **not already in the deques** by comparing the
    current DB un-indexed set against the tail of the deque.  This avoids
    re-queuing duplicates after restarts while still catching new work.
    """
    from sqlalchemy import text as sql_text

    from audiobench.core.db_session import get_session as _gs

    try:
        with _gs() as session:
            # New un-indexed transcripts
            new_tx = session.execute(
                sql_text("SELECT id FROM transcriptions WHERE is_indexed=0 ORDER BY id")
            ).scalars().all()
            queued_tx = set(state.unindexed_transcript_ids)
            fresh_tx = [i for i in new_tx if i not in queued_tx]
            if fresh_tx:
                state.push_transcripts(fresh_tx)
                logger.debug(
                    "SweepState refresh: +%d new transcript IDs", len(fresh_tx)
                )

            # New un-vectorized segments
            new_seg = session.execute(
                sql_text(
                    "SELECT id FROM segments WHERE vector_indexed=0 ORDER BY id"
                )
            ).scalars().all()
            queued_seg = set(state.pending_segment_ids)
            fresh_seg = [i for i in new_seg if i not in queued_seg]
            if fresh_seg:
                state.push_segments(fresh_seg)
                logger.debug(
                    "SweepState refresh: +%d new segment IDs", len(fresh_seg)
                )

            # New uningested search queries
            new_queries = session.execute(
                sql_text(
                    "SELECT sq.id FROM search_queries sq "
                    "WHERE NOT EXISTS ( "
                    "    SELECT 1 FROM expressions e "
                    "    WHERE e.source_type = 'search_query' "
                    "      AND e.source_id = sq.id "
                    ") ORDER BY sq.id"
                )
            ).scalars().all()
            queued_q = set(state.uningested_query_ids)
            fresh_q = [i for i in new_queries if i not in queued_q]
            if fresh_q:
                state.push_queries(fresh_q)
                logger.debug(
                    "SweepState refresh: +%d new search query IDs", len(fresh_q)
                )

            # New uningested session summaries
            new_summaries = session.execute(
                sql_text(
                    "SELECT ss.id FROM search_sessions ss "
                    "WHERE ss.session_summary IS NOT NULL "
                    "  AND NOT EXISTS ( "
                    "      SELECT 1 FROM expressions e "
                    "      WHERE e.source_type = 'search_session_summary' "
                    "        AND e.source_id = ss.id "
                    "  ) ORDER BY ss.id"
                )
            ).scalars().all()
            queued_s = set(state.uningested_summary_session_ids)
            fresh_s = [i for i in new_summaries if i not in queued_s]
            if fresh_s:
                for _sid in fresh_s:
                    state.push_summary_session(_sid)
                logger.debug(
                    "SweepState refresh: +%d new session summary IDs", len(fresh_s)
                )
    except Exception as exc:
        logger.warning("SweepState incremental refresh failed: %s", exc)



# Circuit breaker threshold — how many consecutive sweep failures before the loop
# backs off exponentially and logs at ERROR level.
# Starting guess: 5. Not derived from data — watch real failure cadence once live
# and adjust. Named here so it's findable and honest about being a starting point.
_SWEEP_MAX_CONSECUTIVE_FAILURES = 5


def _rag_consistency_sweep_sync() -> None:
    """Periodic background worker — runs forever, sweeping every 5 minutes.

    The first pass starts after a 10-second warm-up delay so the daemon can
    finish initialisation and start accepting client connections before any
    heavy embedding work begins.

    Circuit breaker: after _SWEEP_MAX_CONSECUTIVE_FAILURES consecutive failures
    the loop backs off exponentially (capped at 1 hour) and logs at ERROR so
    the failure is visible in `audiobench daemon status`, not just buried in
    DEBUG output.
    """
    SWEEP_INTERVAL_SECONDS = 300  # 5 minutes

    # Initial delay — let the daemon become ready first.
    time.sleep(10)

    settings = get_settings()
    from audiobench.daemon.lancedb_optimizer import _should_optimize_on_startup
    if _should_optimize_on_startup(settings.lancedb_optimize_interval_days):
        logger.info("Startup policy: running LanceDB optimize (stale or first run).")
        def run_bg_startup() -> None:
            from audiobench.daemon.lancedb_optimizer import _do_optimize_all_tables
            global _optimize_in_progress
            with _optimize_lock:
                if _optimize_in_progress:
                    return
                _optimize_in_progress = True
            try:
                _do_optimize_all_tables(triggered_by="startup_check")
            except Exception as e:
                logger.error("Startup optimization failed: %s", e)
            finally:
                with _optimize_lock:
                    _optimize_in_progress = False
        threading.Thread(target=run_bg_startup, name="lancedb-optimize-startup", daemon=True).start()

    consecutive_failures = 0

    while True:
        try:
            _do_sweep_once()
            # Outer-level success: reset the outer counter only.
            # _sweep_state.consecutive_sweep_failures is owned by the inner
            # segment handler — it resets itself on a successful segment batch.
            if consecutive_failures > 0:
                logger.info(
                    "RAG sweep recovered after %d consecutive outer failure(s).",
                    consecutive_failures,
                )
            consecutive_failures = 0

        except Exception as exc:
            consecutive_failures += 1
            if _sweep_state is not None:
                _sweep_state.consecutive_sweep_failures = consecutive_failures

            if consecutive_failures >= _SWEEP_MAX_CONSECUTIVE_FAILURES:
                # Exponential backoff, capped at 1 hour.
                backoff = min(SWEEP_INTERVAL_SECONDS * consecutive_failures, 3600)
                logger.error(
                    "RAG sweep has failed %d consecutive time(s) — backing off %ds. "
                    "Run `audiobench daemon status` to inspect. Last error: %s",
                    consecutive_failures, backoff, exc,
                )
                time.sleep(backoff)
                continue
            else:
                logger.error("RAG sweep loop: unexpected error: %s", exc)

        logger.info("RAG sweep: sleeping %ds until next pass.", SWEEP_INTERVAL_SECONDS)
        time.sleep(SWEEP_INTERVAL_SECONDS)


async def _serve(socket_path: Path, pid_path: Path) -> None:
    """Start the Unix socket server and serve until SIGTERM/SIGINT."""
    global _start_time, _memory_store, _segment_store, _sweep_state

    # Remove stale socket from previous crash
    socket_path.unlink(missing_ok=True)

    # --- warm models BEFORE writing PID file so the client waits properly ---
    logger.info("Pre-warming retrieval pipeline...")
    await asyncio.get_running_loop().run_in_executor(None, pre_warm_retrieval_pipeline)
    logger.info("Models warm. Initialising stores...")
    _memory_store = await asyncio.get_running_loop().run_in_executor(None, MemoryStore)
    _segment_store = await asyncio.get_running_loop().run_in_executor(None, SegmentVectorStore)
    logger.info("SegmentVectorStore ready (%d segments embedded).", _segment_store.count_embedded())

    # --- Initialise the in-memory O(1) sweep state -------------------------
    from audiobench.daemon.sweep_state import init_sweep_state

    _sweep_state = init_sweep_state()
    await asyncio.get_running_loop().run_in_executor(None, _sweep_state.load_from_db)
    await asyncio.get_running_loop().run_in_executor(
        None, _sweep_state.sync_indexed_expression_ids
    )
    logger.info(
        "SweepState ready — %d known hashes | %d LanceDB expr IDs | "
        "%d pending segments | %d unindexed transcripts",
        len(_sweep_state.known_hashes),
        len(_sweep_state.indexed_expression_ids),
        _sweep_state.pending_segment_count(),
        _sweep_state.unindexed_transcript_count(),
    )

    from audiobench.daemon.recovery import get_startup_recovery
    recovery = get_startup_recovery()
    await asyncio.get_running_loop().run_in_executor(None, recovery.run)

    from audiobench.jobs.runner import start_pid_watcher
    start_pid_watcher()

    from audiobench.daemon.autocomplete import get_autocomplete_index
    await asyncio.get_running_loop().run_in_executor(None, get_autocomplete_index().build)

    from audiobench.daemon.intelligence import IntelligenceScheduler
    from audiobench.daemon.intelligence.blind_spot_detector import BlindSpotDetector
    from audiobench.daemon.intelligence.connection_surfer import ConnectionSurfer
    from audiobench.daemon.intelligence.drift_detector import DriftDetector
    from audiobench.daemon.intelligence.operator_registry import get_operator_registry
    from audiobench.daemon.intelligence.pattern_detector import PatternDetector
    from audiobench.daemon.intelligence.proposal_generator import ProposalGenerator
    from audiobench.daemon.intelligence.search_session_sweep import SearchSessionSweep

    # Load dynamic operators
    get_operator_registry().load_from_db(_get_store() if _memory_store else None)

    scheduler = IntelligenceScheduler()
    scheduler.register(PatternDetector())
    scheduler.register(DriftDetector())
    scheduler.register(ConnectionSurfer())
    scheduler.register(BlindSpotDetector())
    scheduler.register(ProposalGenerator())
    scheduler.register(SearchSessionSweep())   # search knowledge layer
    asyncio.create_task(scheduler.run_loop())


    _start_time = time.time()
    _write_pid_file(pid_path)
    server = await asyncio.start_unix_server(
        _handle_connection,
        path=str(socket_path),
        limit=104857600  # 100MB limit for massive transcript JSON payloads
    )
    logger.info("Daemon listening on %s (pid=%d)", socket_path, os.getpid())

    # Start background RAG sync loop — runs in a daemon thread, sweeping every 5 min.
    import threading
    sweep_thread = threading.Thread(
        target=_rag_consistency_sweep_sync,
        name="rag-sweep",
        daemon=True,  # killed automatically when the main process exits
    )
    sweep_thread.start()
    logger.info("RAG sweep thread started (interval=300s).")

    from audiobench.daemon.pipeline_recovery import pipeline_recovery_sweep_loop
    pipeline_sweep_thread = threading.Thread(
        target=pipeline_recovery_sweep_loop,
        name="pipeline-sweep",
        daemon=True,
    )
    pipeline_sweep_thread.start()

    # Set permissions so any user can connect (adjust to 0o600 for single-user)
    socket_path.chmod(0o666)

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _on_signal)
    loop.add_signal_handler(signal.SIGINT, _on_signal)

    async with server:
        await stop_event.wait()
        server.close()
        await server.wait_closed()

    _cleanup(socket_path, pid_path)


def _handle_work_assigned(audio_file_id: int, work_id: int, **kwargs: Any) -> None:
    """Handle a work_assigned event (W3 stub / W4 queue insert)."""
    logger.info("Received work_assigned event for audio_file_id=%d -> work_id=%d", audio_file_id, work_id)

    import sqlite3

    from audiobench.core.db_session import get_session
    from audiobench.memory.enums import SourceType
    from audiobench.observatory.db import get_journal_db_path
    from audiobench.storage.models import ExpressionRecord

    try:
        with get_session() as session:
            exprs = session.query(ExpressionRecord.id).filter(
                ExpressionRecord.source_id.in_(
                    session.query(TranscriptionRecord.id)
                    .filter_by(audio_file_id=audio_file_id)
                ),
                ExpressionRecord.source_type == SourceType.AUDIO_TRANSCRIPT.value
            ).all()
            expr_ids = [e.id for e in exprs]

        if not expr_ids:
            return

        conn = sqlite3.connect(str(get_journal_db_path()), timeout=5.0)
        try:
            tuples = [(eid, work_id) for eid in expr_ids]
            conn.executemany(
                "INSERT INTO reconciliation_queue (expression_id, work_id) VALUES (?, ?)",
                tuples
            )
            conn.commit()
            logger.info("Queued %d expressions for work_id=%d reconciliation", len(expr_ids), work_id)
        finally:
            conn.close()
    except Exception as e:
        logger.error("Failed to queue expressions for reconciliation: %s", e)


def run() -> None:
    """Entry point — start the daemon (blocks until SIGTERM/SIGINT)."""
    from audiobench.core.logger_factory import setup_logging
    setup_logging("DEBUG")

    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    settings = get_settings()
    socket_path = Path(settings.daemon_socket_path)
    pid_path = Path(settings.daemon_pid_path)
    lock_path = pid_path.with_suffix(".lock")  # e.g. /tmp/audiobench-daemon.lock

    # ── Layer 1: PID check (UX fast path) ─────────────────────────────────
    # Informational only — gives a clear, actionable error message early,
    # before any models are loaded.  Not the authoritative guard; see Layer 2.
    existing_pid = _check_existing_daemon(pid_path)
    if existing_pid is not None:
        logger.error(
            "Daemon already running (PID %d). "
            "Run `audiobench daemon stop` first, or `audiobench daemon status` to inspect.",
            existing_pid,
        )
        sys.exit(1)

    # ── Layer 2: Exclusive flock (authoritative mutex) ─────────────────────
    # Kernel-enforced, TOCTOU-proof.  If the PID check above and this check
    # ever disagree, this result is authoritative — the flock cannot be tricked
    # by stale files or a race between two processes reading the same PID file.
    if not _acquire_exclusive_lock(lock_path):
        logger.error(
            "Could not acquire exclusive lock at %s. "
            "Another daemon instance may be starting simultaneously. "
            "Run `audiobench daemon stop` and retry.",
            lock_path,
        )
        sys.exit(1)

    logger.info("Single-instance lock acquired (pid=%d, lock=%s).", os.getpid(), lock_path)

    # ── All clear — proceed with startup ──────────────────────────────────
    from audiobench.events import get_bus
    from audiobench.observatory.db import init_journal_db
    from audiobench.observatory.subscriber import get_subscriber

    init_journal_db()
    get_bus().on("*", get_subscriber().record)
    get_bus().on("work_assigned", _handle_work_assigned)

    asyncio.run(_serve(socket_path, pid_path))
