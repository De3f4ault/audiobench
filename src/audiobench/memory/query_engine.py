"""Semantic memory query engine.

Orchestrates vector search, cross-encoder reranking, SQLite graph
traversal, and LLM synthesis to answer queries using the expression graph.

Engineering standards applied (EQ-1 through EQ-4):
- All expression lookups are batched via get_by_ids() / get_parents_batch().
  No per-item get_by_id() calls inside loops.
- LLM synthesis uses Ok/Err result types — no nested try/except chains.
- Every exception path logs a warning and returns a structured result.
  No bare ``except: pass`` blocks.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from audiobench.cli.repl.session import ReplSession

from audiobench.chat.chat_store import ChatRepository
from audiobench.chat.providers.provider_outcome import LlmOutcome, OutcomeType
from audiobench.chat.providers.router import FallbackPolicy, ProviderRouter, router_for_settings
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.factory import get_daemon_client
from audiobench.memory.enums import SourceType
from audiobench.memory.retrieval_streams import ColBERTStream, DenseStream, FTS5Stream
from audiobench.memory.rrf_fusion import FusedResult, _temporal_dedup, filter_micro_fragments, rrf_merge
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import BookmarkRecord

logger = get_logger("memory.query")

_search_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="search_stream")

# ── Result types (EQ-3) ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ok:
    """Successful LLM call result."""
    value: str
    model: str = "unknown"
    # Populated only when Ollama failed and Gemini succeeded — carries the primary
    # error so synthesis_engine.py can surface it in the footer.
    fallback_from: str | None = None   # "ollama" if primary failed and secondary succeeded
    fallback_reason: str | None = None # primary error string on fallback success


@dataclass(frozen=True)
class Err:
    """Failed LLM call result — carries the reason without raising."""
    error: str


LLMResult = Ok | Err


# ── Public result dataclass ───────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    answer: str | None
    sources: list[dict[str, Any]]
    query_time_seconds: float = 0.0
    hyde_document: str | None = None
    synthesis_failed: bool = False
    synthesis_error: str | None = None


# ── Engine ────────────────────────────────────────────────────────────────────

class MemoryQueryEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.daemon = get_daemon_client()
        self.expr_repo = ExpressionRepository()
        self.chat_repo = ChatRepository()
        self.router = router_for_settings(self.settings)

    def query(
        self,
        text: str,
        top_k: int = 20,
        speaker_filter: str | None = None,
        preset: str = "balanced",
        enable_hyde: bool | None = None,
        enable_cross_encoder: bool | None = None,
        use_bm25: bool | None = None,
        use_dense: bool | None = None,
        use_colbert: bool | None = None,
        use_cache: bool = True,
        session: "ReplSession | None" = None,
    ) -> QueryResult:
        t0 = time.time()
        logger.info("Starting memory query for: %s (Preset: %s)", text, preset)

        # Resolve preset flags
        if preset == "fast":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, False
        elif preset == "deep":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = True, True, True, True, True
        else:  # balanced
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, True

        final_hyde   = enable_hyde           if enable_hyde           is not None else p_hyde
        final_cross  = enable_cross_encoder  if enable_cross_encoder  is not None else p_cross
        final_bm25   = use_bm25              if use_bm25              is not None else p_bm25
        final_dense  = use_dense             if use_dense             is not None else p_dense
        final_colbert = use_colbert          if use_colbert           is not None else p_colbert

        # ── Step 0: Semantic Cache ────────────────────────────────────────────
        if use_cache:
            cached = self.daemon.check_cache(text)
            if cached:
                return QueryResult(
                    query=text,
                    answer=cached["answer"],
                    sources=[],
                    query_time_seconds=time.time() - t0,
                    hyde_document=cached.get("hyde_document"),
                )

        # ── Step 1: HyDE Generation ───────────────────────────────────────────
        hyde_document: str | None = None
        if final_hyde:
            hyde_prompt = (
                "Please write a realistic hypothetical excerpt from a spoken recording, such as an audiobook, "
                "podcast, interview, or personal reflection, that directly answers or provides context for the "
                "following query. Limit your response to approximately 100 to 150 words. Do not exceed 150 words. "
                "Do not include any preambles, greetings, facts you are unsure about, or meta-commentary. Just "
                f"produce the raw hypothetical text as if it were a direct transcript.\n\nQuery: {text}"
            )
            outcome = self.router.generate(
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.7,
            )
            if outcome.type == OutcomeType.OK and outcome.content:
                hyde_document = outcome.content
                logger.debug("HyDE document generated (%d chars)", len(hyde_document))
            else:
                reason = str(outcome.error) if outcome.error else (outcome.fallback_reason or "HyDE failed")
                logger.warning("HyDE generation failed: %s — proceeding without HyDE", reason)

        # ── Step 2: Vector Search via Daemon ─────────────────────────────────
        try:
            results = self.daemon.search(
                query=text,
                top_k=top_k,
                speaker_filter=speaker_filter,
                hyde_document=hyde_document,
                use_bm25=final_bm25,
                use_dense=final_dense,
                use_colbert=final_colbert,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Daemon search failed: %s", exc)
            results = []

        if not results:
            return QueryResult(query=text, answer="No relevant memory found.", sources=[])

        # ── Batch-fetch all candidate expressions (EQ-1: eliminates N+1) ─────
        expr_ids = [
            r["expression_id"]
            for r in results
            if r.get("expression_id") is not None
        ]
        expr_map = self.expr_repo.get_by_ids(expr_ids, session=session)   # 1 query for all IDs

        candidates: list[tuple[Any, float]] = [
            (expr_map[r["expression_id"]], r.get("score", 0.0))
            for r in results
            if r.get("expression_id") in expr_map
        ]

        if not candidates:
            return QueryResult(
                query=text, answer="No relevant memory found in database.", sources=[]
            )

        # ── Step 2.5: Parent expansion via batch (EQ-2: eliminates N+1) ──────
        child_ids = [expr.id for expr, _ in candidates]
        parents = self.expr_repo.get_parents_batch(child_ids, session=session)  # 1 JOIN query

        expanded: dict[int, tuple[Any, float]] = {}
        for expr, score in candidates:
            parent = parents.get(expr.id)
            if parent is not None:
                if parent.id not in expanded or score > expanded[parent.id][1]:
                    expanded[parent.id] = (parent, score)
            else:
                if expr.id not in expanded:
                    expanded[expr.id] = (expr, score)

        expanded_candidates = sorted(expanded.values(), key=lambda x: x[1], reverse=True)

        # ── Step 3: CrossEncoder Reranking ────────────────────────────────────
        if final_cross:
            try:
                docs = [c[0].content for c in expanded_candidates]
                scores = self.daemon.rerank(text, docs)
                scored = sorted(
                    zip(expanded_candidates, scores), key=lambda x: x[1], reverse=True
                )
                top_candidates = [(c[0], float(sc)) for c, sc in scored[:5]]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranking via daemon failed: %s — using vector scores", exc)
                top_candidates = list(expanded_candidates[:5])
        else:
            top_candidates = list(expanded_candidates[:5])

        # ── Step 4: Graph Traversal for Enriched Context ─────────────────────
        # Batch-fetch parents again for the smaller top_candidates set (EQ-2)
        top_ids = [expr.id for expr, _ in top_candidates]
        top_parents = self.expr_repo.get_parents_batch(top_ids, session=session)

        # Batch-fetch all inference relations in one pass (EQ-1)
        all_in_rel_source_ids: list[int] = []
        top_in_rels: list[list[Any]] = []
        for expr, _ in top_candidates:
            in_rels = self.expr_repo.get_relations(expr.id, direction="in")
            top_in_rels.append(in_rels)
            all_in_rel_source_ids.extend(
                rel.from_expression_id for rel in in_rels
            )
        inference_map = self.expr_repo.get_by_ids(all_in_rel_source_ids, session=session)  # 1 query

        # Batch-fetch bookmarks for parent source_ids
        parent_source_ids = [
            top_parents[eid].source_id
            for eid in top_ids
            if eid in top_parents and top_parents[eid].source_id is not None
            and top_parents[eid].source_type in (
                SourceType.TRANSCRIPT_SEGMENT.value,
                SourceType.AUDIO_TRANSCRIPT.value,
            )
        ]
        bookmark_map: dict[int, list[BookmarkRecord]] = {}
        if parent_source_ids:
            with get_session() as session:
                from sqlalchemy import or_
                from audiobench.storage.models import TranscriptionRecord

                tx_rows = (
                    session.query(TranscriptionRecord.id, TranscriptionRecord.audio_file_id)
                    .filter(TranscriptionRecord.id.in_(parent_source_ids))
                    .all()
                )
                tx_to_af = {tx_id: af_id for tx_id, af_id in tx_rows if af_id is not None}
                all_af_ids = list(set(tx_to_af.values()))

                bm_records = (
                    session.query(BookmarkRecord)
                    .filter(
                        or_(
                            BookmarkRecord.transcription_id.in_(parent_source_ids),
                            BookmarkRecord.audio_file_id.in_(all_af_ids) if all_af_ids else False,
                        )
                    )
                    .all()
                )
                for bm in bm_records:
                    session.expunge(bm)
                    if bm.transcription_id in parent_source_ids:
                        bookmark_map.setdefault(bm.transcription_id, []).append(bm)
                    for tx_id, af_id in tx_to_af.items():
                        if bm.audio_file_id == af_id:
                            existing = bookmark_map.setdefault(tx_id, [])
                            if bm not in existing:
                                existing.append(bm)

        context_blocks: list[str] = []
        sources: list[dict[str, Any]] = []

        for idx, ((expr, vec_score), in_rels) in enumerate(
            zip(top_candidates, top_in_rels)
        ):
            sources.append({
                "id": expr.id,
                "type": expr.source_type,
                "content": expr.content,
                "score": vec_score,
            })

            block = f"--- Memory Fragment {idx + 1} ---\n"
            block += f"Type: {expr.source_type}\n"
            if expr.speaker:
                block += f"Speaker: {expr.speaker}\n"
            block += f"Content: {expr.content}\n"

            parent = top_parents.get(expr.id)
            if parent:
                if parent.source_type == SourceType.TRANSCRIPT_SEGMENT.value:
                    block += f"Broader Topic/Context: {parent.content}\n"
                elif parent.source_type == SourceType.AUDIO_TRANSCRIPT.value:
                    block += "From Main Transcript.\n"

                if parent.source_id and parent.source_type in (
                    SourceType.TRANSCRIPT_SEGMENT.value,
                    SourceType.AUDIO_TRANSCRIPT.value,
                ):
                    bookmarks = bookmark_map.get(parent.source_id, [])
                    if bookmarks:
                        b_texts = [
                            f"Bookmark at {b.timestamp}s: {b.name} - {b.notes}"
                            for b in bookmarks[:3]
                        ]
                        block += "Related Bookmarks:\n- " + "\n- ".join(b_texts) + "\n"

            # Use pre-fetched inference map (EQ-1: no per-rel get_by_id)
            inferences = [
                inference_map[rel.from_expression_id].content
                for rel in in_rels
                if rel.from_expression_id in inference_map
                and inference_map[rel.from_expression_id].source_type
                == SourceType.SYSTEM_INFERENCE.value
            ]
            if inferences:
                block += "System Inferences:\n- " + "\n- ".join(inferences) + "\n"

            context_blocks.append(block)

        # ── Step 5: LLM Synthesis (EQ-3: Ok/Err, no nested try/except) ───────
        context_text = "\n\n".join(context_blocks)
        prompt = (
            "You are a research synthesis assistant working with audio transcript fragments.\n"
            "Answer the query grounded in the fragments below. Reason WITH them — don't just extract.\n\n"
            "CITATION RULES:\n"
            "  • Cite inline with bracketed numbers immediately after the claim: [1] or [2][4].\n"
            "  • One citation per distinct claim. Do not stack citations on every sentence.\n"
            "  • Never write the word 'Fragment'. Use only the bracket number.\n\n"
            "If the fragments do not fully answer the query, state clearly what they DO illuminate\n"
            "and what they leave open. Be direct — no hedging, no filler.\n\n"
            f"QUERY: {text}\n\n"
            f"FRAGMENTS:\n{context_text}\n\n"
            "Provide a clear, structured answer with inline citations."
        )

        outcome = self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        if outcome.type == OutcomeType.OK and outcome.content:
            answer = outcome.content
            self.daemon.write_cache(text, answer, hyde_document=hyde_document)
            return QueryResult(
                query=text,
                answer=answer,
                sources=sources,
                query_time_seconds=time.time() - t0,
                hyde_document=hyde_document,
            )
        else:
            reason = str(outcome.error) if outcome.error else (outcome.fallback_reason or "Synthesis failed")
            logger.error("Synthesis failed completely: %s", reason)
            return QueryResult(
                query=text,
                answer=None,
                sources=sources,
                query_time_seconds=time.time() - t0,
                hyde_document=hyde_document,
                synthesis_failed=True,
                synthesis_error=reason,
            )



# ── ResearchEngine ────────────────────────────────────────────────────────────


@dataclass
class ResearchResult:
    """Result returned by ResearchEngine.search()."""

    query: str
    sources: list[FusedResult] = field(default_factory=list)
    prior_synthesis_hits: list = field(default_factory=list)
    query_time_seconds: float = 0.0
    retrieval_time_seconds: float = 0.0   # time from start → RRF fusion complete
    synthesis_time_seconds: float = 0.0   # time for LLM synthesis call only
    answer: str | None = None
    synthesis_failed: bool = False        # True = hard failure, no answer at all
    synthesis_is_fallback: bool = False   # True = extractive fallback (LLM unavailable)
    synthesis_error: str | None = None
    # Names of streams that returned 0 results (used by display for ✗ badges)
    streams_skipped: list[tuple[str, str]] = field(default_factory=list)
    # True when HyDE was requested but fell back to direct query embedding
    hyde_fallback: bool = False
    hyde_document: str | None = None


class ResearchEngine:
    """Parallel retrieval orchestrator with observatory telemetry.

    Calls the three retrieval streams concurrently via a thread pool, fuses
    results with RRF, synthesises via LLM, and emits structured events to the
    observatory for every stage so latency can be inspected live.

    Thread safety: each stream is instantiated fresh per call; no shared mutable
    state between threads.
    """

    _STREAM_TIMEOUT: float = 15.0  # seconds; streams that exceed this are skipped
    _SYNTHESIS_RRF_FLOOR: float = 0.020  # Minimum RRF score to reach LLM synthesis
    _MAX_SYNTHESIS_FRAGMENTS: int = 8

    @staticmethod
    def _extractive_fallback(frags: list["FusedResult"], query: str) -> str:  # type: ignore[name-defined]
        """Build a readable answer from the top fragments when no LLM is available.

        This is intentionally minimal: just the fragment text with source/timestamp
        headers, in relevance order.  It is NOT a synthesis — it is a presentation
        of the raw evidence so the session stays useful even when offline.
        """
        lines = [f"**No LLM available — top retrieved fragments for:** {query}\n"]
        for i, fr in enumerate(frags[:5], 1):
            source = Path(fr.source_file).stem if fr.source_file else "unknown"
            ts = f"{fr.start_time:.1f}s–{fr.end_time:.1f}s"
            lines.append(f"**[{i}] {source} \u00b7 {ts}**")
            lines.append(fr.text.strip())
            if i < len(frags[:5]):
                lines.append("")
        return "\n".join(lines)

    def search(
        self,
        query: str,
        top_k: int = 10,
        preset: str = "balanced",
        mmr_lambda: float = 0.5,
        focus_source: str | None = None,
        model: str | None = None,
        diversity_weight: float = 0.4,
        pinned_fragments: list[FusedResult] | None = None,
        prior_synthesis: str | None = None,
        session_id: int | None = None,
        query_id: int | None = None,
        skip_synthesis: bool = False,
    ) -> ResearchResult:
        """Run all three streams in parallel, fuse via RRF, synthesise, return results.

        Preset flags:
          fast     → FTS5 + Dense, no HyDE, no ColBERT reranker
          balanced → FTS5 + Dense + ColBERT, no HyDE
          deep     → FTS5 + Dense + ColBERT + HyDE (LLM generates hypothetical doc)

        Args:
            prior_synthesis: Optional synthesis text from the previous search in this
                session. When provided, injected BEFORE the fragment context block so
                it reads as background the model may draw on or ignore. The LLM handles
                relevance judgment — no heuristic decision-making here.

                Prompt order: [PRIOR CONTEXT] → [NEW FRAGMENTS] → [INSTRUCTION]
                Rationale: fragments are the primary evidence for the current question
                and should be the most proximate input before the instruction. Prior
                synthesis framed first is clearly background to set aside if irrelevant;
                framed last it risks anchoring the LLM on the old answer.

                NOTE: Default ordering chosen with mechanistic reasoning, not yet
                empirically confirmed. Before treating as settled — test both orderings
                on a real multi-search thread and observe continuation quality.
                Same discipline as diversity_weight=0.4 and the 45-minute alignment
                cutoff: unmeasured default, named as such.

        Each stream is called in its own thread.  If a stream raises or times
        out it is silently skipped — partial results are still returned.
        All stages emit ``log_event()`` calls under subsystem='memory.search' so
        the user can inspect timing with ``audiobench obs tail --subsystem memory.search``.
        """
        from audiobench.observatory.context import log_event, start_trace

        trace_id = start_trace()  # noqa: F841  — sets ContextVar for log_event
        t0 = time.perf_counter()

        # ── Preset → stream flags ──────────────────────────────────────────────
        use_mmr = False
        mmr_lam = 0.5
        if preset == "fast":
            use_colbert = False
            use_hyde = False
        elif preset == "deep":
            use_colbert = True
            use_hyde = True
        elif preset == "synthesis":
            # MMR diversifies the tail. We run ColBERT but restrict it internally
            # to only return top 5, anchoring the highly relevant head in RRF
            # without destroying MMR diversity in the tail.
            use_colbert = True
            use_hyde = False
            use_mmr = True
            mmr_lam = 0.5
        else:  # balanced (default)
            use_colbert = True
            use_hyde = False

        # ── Stage 1: Query reformulation ─────────────────────────────────────
        # When skip_synthesis=True (e.g. /search in chat), skip the LLM reformulation
        # entirely — use the raw query as keywords. This makes /search fully offline-capable.
        #
        # Shared router — built ONCE and reused for Stage 1 (reformulation), Stage 1b
        # (HyDE), and Stage 4 (synthesis).  Using three independent router_for_settings()
        # calls caused Flaw D: each had its own circuit-breaker state so a 402 on Stage 1
        # triggered a fresh local swap on Stage 1b and again on Stage 4, multiplying the
        # freeze.  A single instance shares breaker state across all stages.
        settings = get_settings()
        _search_router = router_for_settings(settings)
        if model:
            _search_router.set_primary_model(model)

        t_reform = time.perf_counter()
        if skip_synthesis:
            from audiobench.memory.query_types import ReformulatedQuery
            rq = ReformulatedQuery(
                original=query,
                bm25_keywords=query,
                semantic_query=query,
                hyde_anchor=f"Hypothetical answer to: {query}",
                dense_query=query,
            )
        else:
            from audiobench.memory.query_reformulator import QueryReformulator
            reformulator = QueryReformulator(model=model, router=_search_router)
            rq = reformulator.reformulate(query)
        reform_ms = (time.perf_counter() - t_reform) * 1000
        log_event(
            subsystem="memory.search",
            event_type="query.reformulated",
            message=f"Query reformulated: '{query[:80]}'",
            duration_ms=reform_ms,
            metadata={"bm25_keywords": rq.bm25_keywords[:80], "preset": preset},
        )

        # ── Stage 1b: HyDE generation (deep preset only) ──────────────────────
        hyde_fallback = False
        if use_hyde:
            t_hyde = time.perf_counter()
            hyde_prompt = (
                "Write a realistic hypothetical excerpt from a spoken recording "
                "(audiobook, podcast, or interview) that directly answers the query. "
                "Limit to 100-150 words. Raw transcript only — no preamble.\n\n"
                f"Query: {query}"
            )
            outcome = _search_router.generate(
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.7,
                fallback_policy=FallbackPolicy.NO_LOCAL_CPU,
            )
            if outcome.type == OutcomeType.OK and outcome.content:
                doc = outcome.content
                import dataclasses
                rq = dataclasses.replace(rq, hyde_document=doc)
                hyde_ms = (time.perf_counter() - t_hyde) * 1000
                log_event(
                    subsystem="memory.search",
                    event_type="hyde.ok",
                    message="HyDE document generated",
                    duration_ms=hyde_ms,
                    metadata={"doc_chars": len(doc)},
                )
            else:
                reason = str(outcome.error) if outcome.error else (outcome.fallback_reason or "HyDE failed")
                hyde_fallback = True
                logger.warning(
                    "HyDE generation failed (%s) — falling back to direct query embedding", reason
                )
                log_event(
                    subsystem="memory.search",
                    event_type="hyde.failed",
                    message=f"HyDE failed: {reason[:80]}",
                    level="WARNING",
                )

        fts_hits: list = []
        dense_hits: list = []
        colbert_hits: list = []
        synthesis_hits: list = []

        # ── Stage 2: Parallel stream retrieval ───────────────────────────────
        dense_extra: dict = {"preset": preset, "mmr_lambda": mmr_lambda, "focus_source": focus_source} if use_mmr else {"focus_source": focus_source}
        fts_extra: dict = {"focus_source": focus_source}

        from audiobench.memory.retrieval_streams import SynthesisStream

        # Synthesis stream has its own small cap (2) so prior-knowledge hits never
        # crowd out actual audio fragment results.  This is independent of top_k.
        _SYNTHESIS_TOP_K = 2

        stream_tasks: dict[str, tuple] = {
            "fts5":  (FTS5Stream(),    fts_hits,    fts_extra),
            "dense": (DenseStream(),   dense_hits,  dense_extra),
            # Named 'recap' (not 'synthesis') to avoid collision with LLM synthesis
            # status markers in the UI.  0-hit results here are normal — no prior
            # session context simply means this is an early search or a new topic.
            "recap": (SynthesisStream(), synthesis_hits, {"top_k": _SYNTHESIS_TOP_K, "session_id": session_id}),
        }
        if use_colbert:
            stream_tasks["colbert"] = (ColBERTStream(), colbert_hits, {"preset": preset})

        stream_timings: dict[str, float] = {}
        streams_skipped: list[tuple[str, str]] = []

        def _timed_retrieve(name: str, stream: object, bucket: list, extra: dict) -> None:
            t_s = time.perf_counter()
            if "top_k" in extra:
                # Stream supplies its own top_k (e.g. SynthesisStream).
                k = extra.pop("top_k")
                hits = stream.retrieve(rq, k, **extra) if extra else stream.retrieve(rq, k)  # type: ignore[attr-defined]
            elif extra:
                hits = stream.retrieve(rq, top_k, **extra)  # type: ignore[attr-defined]
            else:
                hits = stream.retrieve(rq, top_k)  # type: ignore[attr-defined]
            elapsed_ms = (time.perf_counter() - t_s) * 1000
            bucket.extend(hits)
            stream_timings[name] = elapsed_ms

        import contextvars

        future_to_name = {
            _search_pool.submit(
                contextvars.copy_context().run,
                _timed_retrieve, name, stream, bucket, extra,
            ): name
            for name, (stream, bucket, extra) in stream_tasks.items()
        }

        done, not_done = wait(future_to_name.keys(), timeout=self._STREAM_TIMEOUT * 2)

        for future in done:
            name = future_to_name[future]
            try:
                future.result()  # Should return immediately since it's in `done`
                bucket = stream_tasks[name][1]
                hit_count = len(bucket)
                if hit_count == 0 and name != "recap":
                    # 'recap' returning 0 hits is normal (no prior session context);
                    # don't surface it as a skipped-stream warning.
                    streams_skipped.append((name, "0 hits"))
                log_event(
                    subsystem="memory.search",
                    event_type=f"stream.{name}",
                    message=f"Stream '{name}' returned {hit_count} hits",
                    duration_ms=stream_timings.get(name),
                    metadata={"hit_count": hit_count},
                )
            except Exception as exc:  # noqa: BLE001
                streams_skipped.append((name, "failed"))
                logger.warning("Stream '%s' failed: %s — skipping", name, exc)
                log_event(
                    subsystem="memory.search",
                    event_type=f"stream.{name}",
                    message=f"Stream '{name}' failed: {exc}",
                    level="WARNING",
                    metadata={"error": str(exc)},
                )

        for future in not_done:
            name = future_to_name[future]
            streams_skipped.append((name, "timeout - inference lock busy or ingestion in progress"))
            logger.warning("Stream '%s' timed out after %.1fs — skipping", name, self._STREAM_TIMEOUT * 2)
            log_event(
                subsystem="memory.search",
                event_type=f"stream.{name}.timeout",
                message=f"Stream '{name}' timed out",
                level="WARNING",
                metadata={"timeout": self._STREAM_TIMEOUT * 2},
            )

        # ── Stage 3: RRF fusion ──────────────────────────────────────────────
        t_fuse = time.perf_counter()
        fused = rrf_merge(fts_hits, dense_hits, colbert_hits, top_n=top_k, diversity_weight=diversity_weight)
        fuse_ms = (time.perf_counter() - t_fuse) * 1000
        log_event(
            subsystem="memory.search",
            event_type="rrf.fused",
            message=f"RRF produced {len(fused)} fused results",
            duration_ms=fuse_ms,
            metadata={"fused_count": len(fused)},
        )

        # ── Stage 3.5: Temporal deduplication (always on) ─────────────────────
        pre_dedup_count = len(fused)
        fused = _temporal_dedup(fused)
        dedup_removed = pre_dedup_count - len(fused)
        if dedup_removed:
            log_event(
                subsystem="memory.search",
                event_type="dedup.temporal",
                message=f"Temporal dedup removed {dedup_removed} overlapping fragment(s)",
                metadata={"removed": dedup_removed, "kept": len(fused)},
            )

        # ── Stage 3.6: Micro-fragment filtering ───────────────────────────────
        # Drops fragments that are too short to contribute meaningful signal to
        # synthesis (e.g. 1-second transcription artifacts, single proper nouns).
        # Removed segments are merged onto the nearest surviving neighbour from
        # the same source so Fragment Reader H/L navigation has no gaps.
        pre_micro_count = len(fused)
        fused = filter_micro_fragments(fused)
        micro_removed = pre_micro_count - len(fused)
        if micro_removed:
            log_event(
                subsystem="memory.search",
                event_type="dedup.micro_fragments",
                message=f"Micro-fragment filter removed {micro_removed} noise fragment(s)",
                metadata={"removed": micro_removed, "kept": len(fused)},
            )

        # Prepend any pinned fragments that aren't already in fused
        if pinned_fragments:
            existing_sids = {fr.segment_id for fr in fused}
            to_prepend = [fr for fr in pinned_fragments if fr.segment_id not in existing_sids]
            fused = to_prepend + fused

        # Snapshot retrieval wall-clock: everything above is retrieval,
        # everything below is synthesis.  Both are reported separately in the UI.
        t_retrieval_done = time.perf_counter()

        # Persist synthesis hits used as context for this search
        if query_id is not None and query_id >= 0 and synthesis_hits:
            from audiobench.memory.session_store import persist_synthesis_context
            try:
                persist_synthesis_context(query_id, synthesis_hits)
            except Exception as e:
                logger.warning("Failed to persist synthesis context: %s", e)

        if not fused:
            return ResearchResult(
                query=query,
                sources=fused,
                prior_synthesis_hits=synthesis_hits,
                query_time_seconds=t_retrieval_done - t0,
                retrieval_time_seconds=t_retrieval_done - t0,
                synthesis_time_seconds=0.0,
                answer="No relevant segments found.",
                streams_skipped=streams_skipped,
                hyde_fallback=hyde_fallback,
                hyde_document=rq.hyde_document,
            )

        # ── Stage 4: LLM synthesis ───────────────────────────────────────────
        pinned_sids = {fr.segment_id for fr in (pinned_fragments or [])}
        # Track (display_num, fragment) pairs so the LLM citation numbers [N]
        # are identical to what the user sees on screen (1-based fused rank).
        synthesis_pairs: list[tuple[int, FusedResult]] = []
        for fused_idx, fr in enumerate(fused, 1):
            if len(synthesis_pairs) >= self._MAX_SYNTHESIS_FRAGMENTS:
                break
            if fr.segment_id in pinned_sids or fr.rrf_score >= self._SYNTHESIS_RRF_FLOOR:
                synthesis_pairs.append((fused_idx, fr))

        synthesis_frags = [fr for _, fr in synthesis_pairs]

        if not synthesis_pairs and fused:
            # Fallback to top retrieved fragments if nothing met the floor score
            synthesis_pairs = list(enumerate(fused[:min(3, len(fused))], 1))
            synthesis_frags = [fr for _, fr in synthesis_pairs]

        def _fmt_header(display_num: int, fr: FusedResult) -> str:
            """Build the per-fragment header sent to the LLM.

            display_num matches the number shown in the search results UI,
            so when the LLM writes [N] the user can immediately map it back
            to the correct fragment on screen.
            """
            source_str = f" | {Path(fr.source_file).stem}" if fr.source_file else ""
            return f"[{display_num}{source_str} | {fr.start_time:.1f}s–{fr.end_time:.1f}s]"

        context_text = "\n\n".join(
            f"{_fmt_header(display_num, fr)}\n{fr.text}"
            for display_num, fr in synthesis_pairs
        )
        # ── Synthesis carryforward: inject prior session context ──────────────
        # Prior synthesis (from the previous search in this session) is injected
        # BEFORE the fragment block, framing it as optional background context.
        # The LLM decides whether it's relevant — no heuristic connect/don't-connect
        # decision that can be wrong. Degrades gracefully to nothing when offline
        # (prior_synthesis=None is passed when LLM is unavailable).
        _MAX_PRIOR_WORDS = 1500  # unmeasured default — adjust if sessions hit this ceiling
        prior_context_block = ""
        if prior_synthesis:
            prior_words = prior_synthesis.split()
            if len(prior_words) > _MAX_PRIOR_WORDS:
                truncated = " ".join(prior_words[:_MAX_PRIOR_WORDS])
                prior_synthesis = truncated + " [...truncated]"
            prior_context_block = (
                "PRIOR CONTEXT (from your previous search in this session — "
                "use if relevant, ignore entirely if not):\n"
                f"{prior_synthesis}\n\n"
                "---\n\n"
            )

        synthesis_prompt = (
            "You are a research synthesis assistant working with audio transcript fragments\n"
            "from the user's personal listening library (audiobooks, podcasts, interviews,\n"
            "lectures, and personal recordings).\n\n"
            "Your goal is a genuinely insightful answer — reason WITH the fragments, not just extract from them.\n"
            "Identify patterns, tensions, and implications the user may not have noticed.\n\n"
            "OUTPUT STRUCTURE (follow this order):\n"
            "  1. A direct 1-2 sentence answer to the query.\n"
            "  2. Analysis — develop your thinking in clearly titled sections using **bold headers**.\n"
            "  3. Close with exactly two sections:\n"
            "       **What the fragments illuminate** — the key insight the sources collectively reveal.\n"
            "       **○ What the fragments leave open** — genuine gaps, unresolved angles, or worth pursuing further.\n\n"
            "CITATION RULES (strictly follow):\n"
            "  • Cite inline with bracketed numbers immediately after the claim: [1] or [3][5].\n"
            "  • One citation per distinct claim — do not stack citations on every sentence.\n"
            "  • Never write the word 'Fragment'. Use only the bracket number.\n"
            "  • Only cite numbers present in the FRAGMENTS block.\n\n"
            "TONE AND DEPTH:\n"
            "  • Match depth to query complexity: a factual question gets a focused answer;\n"
            "    a philosophical or open-ended question gets fuller exploration.\n"
            "  • Be direct — no hedging phrases like 'it seems that' or 'one could argue'.\n"
            "  • If fragments conflict with each other, name the tension explicitly.\n\n"
            f"{prior_context_block}"
            f"QUERY: {query}\n\n"
            f"FRAGMENTS:\n{context_text}\n\n"
            "Synthesize your answer now."
        )

        if skip_synthesis:
            return ResearchResult(
                query=query,
                sources=fused,
                prior_synthesis_hits=synthesis_frags,
                query_time_seconds=t_retrieval_done - t0,
                retrieval_time_seconds=t_retrieval_done - t0,
                synthesis_time_seconds=0.0,
                answer=None,
                synthesis_failed=False,
                streams_skipped=streams_skipped,
                hyde_fallback=hyde_fallback,
                hyde_document=rq.hyde_document,
            )

        settings = get_settings()
        model_name = model or settings.ollama_model

        # "ollama" as a literal model name is a common /set model mistake —
        # treat it as "use the configured default" so we don't send a 404-causing
        # model name to the /api/chat endpoint.
        if model_name.lower() == "ollama":
            model_name = settings.ollama_model

        t_synth = time.perf_counter()

        # Reuse the shared _search_router (created at Stage 1) so circuit-breaker
        # state accumulated during reformulation and HyDE carries into synthesis.
        # This prevents Flaw D: a fresh router on a 402 would rediscover and
        # attempt the failed cloud model, triggering a redundant local swap.
        outcome = _search_router.generate(
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.2,
            fallback_policy=FallbackPolicy.NO_LOCAL_CPU,
        )

        if outcome.type == OutcomeType.OK and outcome.content:
            answer = outcome.content
            t_done = time.perf_counter()
            synth_s = t_done - t_synth
            synth_ms = synth_s * 1000
            log_event(
                subsystem="memory.search",
                event_type="synthesis.ok",
                message="Synthesis completed",
                duration_ms=synth_ms,
                metadata={"answer_chars": len(answer)},
            )
            return ResearchResult(
                query=query,
                sources=fused,
                prior_synthesis_hits=synthesis_hits,
                query_time_seconds=t_done - t0,
                retrieval_time_seconds=t_retrieval_done - t0,
                synthesis_time_seconds=synth_s,
                answer=answer,
                streams_skipped=streams_skipped,
                hyde_fallback=hyde_fallback,
                hyde_document=rq.hyde_document,
            )
        else:
            reason = str(outcome.error) if outcome.error else (outcome.fallback_reason or "synthesis failed")
            t_done = time.perf_counter()
            synth_s = t_done - t_synth
            synth_ms = synth_s * 1000
            logger.error("ResearchEngine synthesis failed: %s", reason)
            log_event(
                subsystem="memory.search",
                event_type="synthesis.failed",
                message=f"Synthesis failed: {reason[:120]}",
                level="ERROR",
                duration_ms=synth_ms,
                metadata={"error": reason[:200]},
            )
            # ── Extractive fallback ───────────────────────────────────────
            # Fragments are already retrieved and in memory. Rather than
            # returning nothing when the LLM is unavailable, surface the top
            # fragments as a readable answer so the session stays useful.
            # synthesis_is_fallback=True lets the UI label it distinctly.
            fallback_answer = self._extractive_fallback(synthesis_frags, query) if synthesis_frags else None
            return ResearchResult(
                query=query,
                sources=fused,
                prior_synthesis_hits=synthesis_hits,
                query_time_seconds=t_done - t0,
                retrieval_time_seconds=t_retrieval_done - t0,
                synthesis_time_seconds=synth_s,
                answer=fallback_answer,
                synthesis_failed=fallback_answer is None,   # only truly failed if no frags
                synthesis_is_fallback=fallback_answer is not None,
                synthesis_error=reason,
                streams_skipped=streams_skipped,
                hyde_fallback=hyde_fallback,
                hyde_document=rq.hyde_document,
            )

