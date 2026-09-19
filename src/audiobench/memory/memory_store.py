"""MemoryStore — LanceDB vector storage for Semantic Expressions.

Provides direct access to LanceDB for writing, deleting, and hybrid searching
of semantic memory nodes.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import SearchResult
from audiobench.memory.embedding_engine import EmbeddingEngine

logger = get_logger("memory.memory_store")


class ExpressionNode(LanceModel):
    """LanceDB schema for stored semantic expressions."""

    expression_id: int
    vector: Vector(768)  # type: ignore[valid-type]
    content: str
    embedding_model_version: str
    embedded_at: str
    source_type: str
    speaker: str | None = None
    audio_file_id: int | None = None
    confidence: float | None = None
    original_language: str | None = None
    work_id: int | None = None


class QueryCacheNode(LanceModel):
    """LanceDB schema for cached semantic queries."""

    query: str
    vector: Vector(768)  # type: ignore[valid-type]
    answer: str
    hyde_document: str | None = None
    created_at: str


class SpeakerProfileNode(LanceModel):
    """LanceDB schema for persistent speaker voice prints (ECAPA-TDNN)."""

    profile_id: str
    name: str
    vector: Vector(192)  # type: ignore[valid-type]
    created_at: str


class MemoryStore:
    """LanceDB adapter for audiobench expressions."""

    def __init__(self) -> None:
        """Initialize the connection to the LanceDB instance."""
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "expressions"
        self._engine = EmbeddingEngine()

        # We assume the primary embedder model version is standard
        self.model_version = "nomic-embed-text-v1.5"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB table '%s'", self.table_name)
            self.db.create_table(self.table_name, schema=ExpressionNode)
            # NOTE: No FTS index created. The FTS (inverted) index triggered a
            # Rust panic in lance-index 7.0.0 (builder.rs out-of-bounds) when
            # optimize() ran on a fragmented table, causing silent data loss.
            # All retrieval uses vector similarity search — FTS is not needed.

        # Scalar index on expression_id is required for merge_insert performance.
        # Without it merge_insert does a full table scan and runs ~1.7x slower
        # than the old delete+add (measured: 22.7ms vs 12.9ms on 10k rows).
        # NOTE: In lance-0.38.2, create_scalar_index is NOT idempotent — it always
        # submits a CreateIndex transaction and bumps table version. Calling it
        # unconditionally races concurrent Rewrite transactions (e.g. optimizer)
        # leading to "Retryable commit conflict". We check existing indices first.
        try:
            tbl = self.db.open_table(self.table_name)
            existing_indices = tbl.list_indices()
            has_expr_idx = any(
                getattr(idx, "name", "") == "expression_id_idx"
                or "expression_id" in getattr(idx, "columns", [])
                for idx in existing_indices
            )
            if not has_expr_idx:
                tbl.create_scalar_index("expression_id")
        except Exception as _idx_exc:
            # Non-fatal: merge_insert still works without the index, just slower.
            logger.warning("Could not create scalar index on expression_id: %s", _idx_exc)

    @property
    def table(self) -> Any:
        """Dynamically fetch the latest table manifest to avoid stale fragment crashes."""
        return self.db.open_table(self.table_name)

    def add_expression(
        self,
        content: str,
        source_type: str,
        *,
        inference_status: str | None = None,  # accepted for compat; stored in SQLite if column exists
        source_id: int | None = None,
        speaker: str | None = None,
    ) -> None:
        """Register a new expression in SQLite and immediately embed it into LanceDB.

        This is the convenience entry-point used by all intelligence tasks
        (PatternDetector, ConnectionSurfer, BlindSpotDetector, etc.).  It:
        1. Delegates to ExpressionRepository.register() for SQLite persistence
           and content-hash deduplication.
        2. Calls write_node() to compute the Nomic vector and add it to LanceDB.

        Args:
            content:          The text of the expression.
            source_type:      Category label (e.g. 'system_inference', 'daemon_calibration').
            inference_status: Optional status tag; stored on the SQLite record when
                              the ExpressionRecord has an inference_status column.
            source_id:        Optional FK to the originating row.
            speaker:          Optional speaker label.
        """
        from audiobench.storage.expression_repository import ExpressionRepository

        repo = ExpressionRepository()
        record = repo.register(
            content=content,
            source_type=source_type,
            source_id=source_id,
            speaker=speaker,
        )

        # Persist the inference_status flag if the column exists on the model
        if inference_status is not None:
            try:
                from sqlalchemy import text as _t

                from audiobench.core.db_session import get_session as _gs
                with _gs() as _s:
                    _s.execute(
                        _t("UPDATE expressions SET inference_status = :status WHERE id = :id"),
                        {"status": inference_status, "id": record.id},
                    )
                    _s.commit()
            except Exception as exc:
                # Column may not exist in older migrations — non-fatal
                logger.debug("add_expression: could not set inference_status: %s", exc)

        # Write vector to LanceDB so the expression is immediately searchable
        try:
            self.write_node(
                expression_id=record.id,
                content=content,
                source_type=source_type,
                speaker=speaker,
            )
        except Exception as exc:
            logger.error("add_expression: LanceDB write failed for expression #%d: %s", record.id, exc)

    def write_node(
        self,
        expression_id: int,
        content: str,
        source_type: str,
        speaker: str | None = None,
    ) -> None:
        """Embed and write a single expression to LanceDB via atomic merge_insert.

        Uses merge_insert (a single LanceDB transaction) instead of the previous
        delete()+add() two-step, which had a race window where a concurrent
        optimize() call could compact-out the deleted row before the add() commit,
        permanently losing the expression.
        """
        vector = self._engine.embed_for_storage(content).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = ExpressionNode(
            expression_id=expression_id,
            vector=vector,
            content=content,
            embedding_model_version=self.model_version,
            embedded_at=now_str,
            source_type=source_type,
            speaker=speaker,
        )
        tbl = self.table
        # IMPORTANT: always pass schema=tbl.schema here.
        # pa.Table.from_pylist() produces nullable=True fields by default.
        # The LanceModel-defined table has nullable=False for expression_id,
        # content, and source_type. LanceDB rejects the append with a schema
        # mismatch error at runtime if the nullability doesn't match exactly.
        data = pa.Table.from_pylist([record.model_dump()], schema=tbl.schema)
        tbl.merge_insert("expression_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(data)

    def batch_write_nodes(
        self,
        nodes: list[dict],
        batch_size: int = 64,
    ) -> None:
        """Embed and write a batch of expressions in one model forward-pass.

        Uses merge_insert (a single LanceDB transaction) so updates are atomic:
        there is no window between delete and re-add where a concurrent optimize()
        call could compact-out expressions before they are re-inserted.  The old
        delete()+add() two-step accumulated ~1,944 permanently-lost expressions
        across multiple optimize() cycles (confirmed by forensic ID-set analysis).

        Performance note: merge_insert requires a scalar index on expression_id
        to match the old delete+add latency (12.9ms vs 22.7ms on 10k rows at
        batch_size=64). The index is created idempotently in MemoryStore.__init__.

        Args:
            nodes:      List of dicts with keys: expression_id, content,
                        source_type, speaker (optional), audio_file_id (optional),
                        confidence (optional), original_language (optional),
                        work_id (optional).
            batch_size: sentence-transformers sub-batch size (64 is safe).
        """
        if not nodes:
            return

        texts = [n["content"] for n in nodes]
        vectors = self._engine.embed_batch_for_storage(texts, batch_size=batch_size)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        records = [
            ExpressionNode(
                expression_id=int(n["expression_id"]),
                vector=v,
                content=n["content"],
                embedding_model_version=self.model_version,
                embedded_at=now_str,
                source_type=n["source_type"],
                speaker=n.get("speaker"),
                audio_file_id=n.get("audio_file_id"),
                confidence=n.get("confidence"),
                original_language=n.get("original_language"),
                work_id=n.get("work_id"),
            )
            for n, v in zip(nodes, vectors)
        ]

        tbl = self.table
        data = pa.Table.from_pylist([r.model_dump() for r in records], schema=tbl.schema)
        tbl.merge_insert("expression_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(data)

        logger.info("batch_write_nodes: wrote %d expressions to LanceDB", len(records))

    def update_expression_work_id(self, expression_id: int, work_id: int) -> None:
        """Update the work_id of an existing expression in LanceDB."""
        try:
            self.table.update(where=f"expression_id = {expression_id}", values={"work_id": work_id})
        except Exception as e:
            logger.error("Failed to update work_id for expression_id=%d: %s", expression_id, e)

    def search(
        self,
        query: str,
        top_k: int = 5,
        speaker_filter: str | None = None,
        audio_file_id: int | None = None,
        work_id: int | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]:
        """Perform a dynamic search over expressions based on enabled strategies."""
        if not use_bm25 and not use_dense:
            logger.warning("Both BM25 and Dense search disabled. Defaulting to Dense.")
            use_dense = True

        query_vector = self._engine.embed_for_query(query).tolist()

        if hyde_document and use_dense:
            hyde_vector = self._engine.embed_for_query(hyde_document).tolist()
            import numpy as np

            query_vector = ((np.array(query_vector) + np.array(hyde_vector)) / 2.0).tolist()

        # Determine base search strategy
        if use_bm25 and use_dense:
            # LanceDB native RRF handles the hybrid merge
            search_query = (
                self.table.search(query_type="hybrid", fts_columns="content")
                .vector(query_vector)
                .text(query)
            )
        elif use_bm25:
            search_query = self.table.search(query, query_type="fts")
        else:
            search_query = self.table.search(query_vector, query_type="vector")

        search_query = search_query.limit(top_k * 3)

        where_clauses = []
        if speaker_filter:
            where_clauses.append(f"speaker = '{speaker_filter}'")
        if audio_file_id is not None:
            where_clauses.append(f"audio_file_id = {audio_file_id}")
        if work_id is not None:
            where_clauses.append(f"work_id = {work_id}")

        prefilter = " AND ".join(where_clauses) if where_clauses else None
        if prefilter:
            search_query = search_query.where(prefilter)

        # Apply ColBERT reranking if enabled
        if use_colbert:
            from audiobench.memory.singletons import get_colbert_reranker

            reranker = get_colbert_reranker()
            search_query = search_query.rerank(reranker=reranker)
        elif use_bm25 and use_dense:
            # Explicitly use RRF if ColBERT is off but Hybrid is on
            from lancedb.rerankers import RRFReranker

            search_query = search_query.rerank(reranker=RRFReranker())

        results = search_query.to_list()

        final_results = []
        for r in results:
            # In LanceDB, the score key differs based on the reranker or query type.
            # _distance (vector), score (fts), or _relevance_score (rerankers)
            score = r.get("_relevance_score", r.get("score", r.get("_distance", 0.0)))

            final_results.append(
                {
                    "expression_id": r["expression_id"],
                    "content": r["content"],
                    "source_type": r["source_type"],
                    "speaker": r.get("speaker"),
                    "score": float(score),
                }
            )

            if len(final_results) >= top_k:
                break

        return final_results

    def delete_node(self, expression_id: int) -> None:
        """Delete an expression node from LanceDB."""
        self.table.delete(f"expression_id = {expression_id}")

    def delete_node_batch(self, expression_ids: list[int]) -> None:
        """Delete multiple expression nodes from LanceDB in a single operation."""
        if not expression_ids:
            return
        ids_str = ", ".join(str(int(i)) for i in expression_ids)
        self.table.delete(f"expression_id IN ({ids_str})")

    def get_vectors(self, expression_ids: list[int]) -> dict[int, list[float]]:
        """Retrieve vectors for a list of expression IDs."""
        if not expression_ids:
            return {}
        ids_str = ", ".join(str(i) for i in expression_ids)
        rows = self.table.search().where(f"expression_id IN ({ids_str})").to_list()
        return {int(r["expression_id"]): r["vector"] for r in rows}

    def count_nodes(self) -> int:
        """Get the total number of nodes in the store."""
        return self.table.count_rows()


class QueryCacheStore:
    """LanceDB adapter for caching semantic queries."""

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "query_cache"
        self._engine = EmbeddingEngine()

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB cache table '%s'", self.table_name)
            self.db.create_table(self.table_name, schema=QueryCacheNode)

    @property
    def table(self) -> Any:
        """Dynamically fetch the latest table manifest to avoid stale fragment crashes."""
        return self.db.open_table(self.table_name)

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check if a semantically identical query exists in the cache."""
        query_vector = self._engine.embed_for_query(query).tolist()

        # We only need the top 1 result
        results = self.table.search(query_vector).limit(1).to_list()

        if results:
            best_match = results[0]
            distance = float(best_match.get("_distance", 1.0))
            if distance <= distance_threshold:
                logger.info("Cache hit for query '%s' (distance: %.4f)", query, distance)
                return {
                    "answer": best_match["answer"],
                    "hyde_document": best_match.get("hyde_document"),
                    "distance": distance,
                }

        return None

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write a synthesized answer to the cache."""
        query_vector = self._engine.embed_for_query(query).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = QueryCacheNode(
            query=query,
            vector=query_vector,
            answer=answer,
            hyde_document=hyde_document,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Cached answer for query '%s'", query)


class SpeakerProfileStore:
    """LanceDB adapter for persistent speaker voice prints (ECAPA-TDNN)."""

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "speaker_profiles"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB speaker profiles table '%s'", self.table_name)
            self.db.create_table(self.table_name, schema=SpeakerProfileNode)

    @property
    def table(self) -> Any:
        """Dynamically fetch the latest table manifest to avoid stale fragment crashes."""
        return self.db.open_table(self.table_name)

    def identify_speaker(self, voice_print: list[float], threshold: float = 0.82) -> str | None:
        """Find the closest known speaker for a given voice print.
        
        Args:
            voice_print: 192-D list of floats from SpeechBrain.
            threshold: Minimum cosine similarity required to confirm match.
                       (LanceDB uses distance, so distance <= 1 - threshold)
        """
        if self.table.count_rows() == 0:
            return None

        results = self.table.search(voice_print).limit(1).to_list()

        if results:
            best_match = results[0]
            # LanceDB distance is typically 1 - cosine_similarity for vectors
            # So a cosine similarity of 0.82 means distance of 0.18
            distance = float(best_match.get("_distance", 1.0))
            max_distance = 1.0 - threshold

            if distance <= max_distance:
                logger.info(
                    "Voice matched! '%s' (dist: %.4f < %.4f)",
                    best_match["name"], distance, max_distance
                )
                return best_match["name"]
            else:
                logger.debug(
                    "Voice NOT matched. Closest was '%s' (dist: %.4f > %.4f)",
                    best_match["name"], distance, max_distance
                )

        return None

    def save_speaker(self, profile_id: str, name: str, voice_print: list[float]) -> None:
        """Save or update a speaker's voice print in the database."""
        # Delete if it exists to allow updates.
        # LanceDB raises when the filter matches zero rows on some versions — log, never swallow.
        try:
            self.table.delete(f"profile_id = '{profile_id}'")
        except Exception as exc:
            logger.debug("LanceDB delete failed during upsert (expected if new profile): %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SpeakerProfileNode(
            profile_id=profile_id,
            name=name,
            vector=voice_print,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Saved speaker profile for '%s' (ID: %s)", name, profile_id)


# ── Segment Vector Store ──────────────────────────────────────────────────────


class SegmentVectorNode(LanceModel):
    """LanceDB schema for audio segment embeddings.

    One record per segment row in SQLite. Carries timestamps and source file
    path so the display layer can show provenance without additional DB lookups.
    """

    segment_id: int           # FK → segments.id in SQLite
    vector: Vector(768)       # type: ignore[valid-type]  # Nomic nomic-embed-text-v1.5
    text: str                 # raw transcript text of the segment
    start_time: float         # seconds from audio start
    end_time: float           # seconds from audio end
    source_file: str          # absolute path to the audio file
    embedded_at: str          # ISO-8601 timestamp of when this was embedded


class SegmentVectorStore:
    """LanceDB adapter for audio segment vector embeddings.

    Mirrors the structure of MemoryStore but targets the 'segment_vectors'
    table and is keyed by segment_id (not expression_id).

    The daemon's _rag_consistency_sweep_sync populates this table
    automatically for every new segment. The CLI backfill command
    'audiobench db embed-segments' handles segments that existed before
    this feature was introduced.
    """

    table_name: str = "segment_vectors"

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self._engine = EmbeddingEngine()
        self.model_version = "nomic-embed-text-v1.5"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB segment vectors table '%s'", self.table_name)
            self.db.create_table(self.table_name, schema=SegmentVectorNode)

    @property
    def table(self) -> Any:
        """Dynamically fetch the latest table manifest to avoid stale fragment crashes."""
        return self.db.open_table(self.table_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_segment(
        self,
        segment_id: int,
        text: str,
        start_time: float,
        end_time: float,
        source_file: str,
    ) -> None:
        """Embed text and write (or overwrite) a segment into LanceDB.

        Idempotent: safe to call multiple times for the same segment_id.
        Deletes the old record first to allow content updates.
        """
        # Delete stale record if it exists
        try:
            self.table.delete(f"segment_id = {segment_id}")
        except Exception as exc:
            logger.debug("Segment delete (upsert) for id=%d: %s", segment_id, exc)

        vector = self._engine.embed_for_storage(text).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = SegmentVectorNode(
            segment_id=segment_id,
            vector=vector,
            text=text,
            start_time=start_time,
            end_time=end_time,
            source_file=source_file,
            embedded_at=now_str,
        )
        self.table.add([record])
        logger.debug("Upserted segment_id=%d into segment_vectors", segment_id)

    def upsert_segment_with_vector(
        self,
        segment_id: int,
        text: str,
        start_time: float,
        end_time: float,
        source_file: str,
        vector: list[float],
    ) -> None:
        """Write a segment using a pre-computed vector (from the daemon's warm model).

        Used by the daemon handlers so the warm Nomic model in the daemon
        process is reused rather than cold-loading it in the CLI process.
        """
        try:
            self.table.delete(f"segment_id = {segment_id}")
        except Exception as exc:
            logger.debug("Segment delete (upsert_with_vector) for id=%d: %s", segment_id, exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SegmentVectorNode(
            segment_id=segment_id,
            vector=vector,
            text=text,
            start_time=start_time,
            end_time=end_time,
            source_file=source_file,
            embedded_at=now_str,
        )
        self.table.add([record])
        logger.debug("Upserted segment_id=%d (pre-computed vector) into segment_vectors", segment_id)

    def batch_upsert_segments(
        self,
        rows: list[dict],
        vectors: list[list[float]],
    ) -> None:
        """Write a batch of pre-computed segment vectors to LanceDB in one operation."""
        if not rows:
            return
        ids = [r["segment_id"] for r in rows]
        id_list = ", ".join(str(i) for i in ids)
        try:
            self.table.delete(f"segment_id IN ({id_list})")
        except Exception as exc:
            logger.debug("Batch segment delete: %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        records = [
            SegmentVectorNode(
                segment_id=int(r["segment_id"]),
                vector=v,
                text=r["text"],
                start_time=float(r["start_time"]),
                end_time=float(r["end_time"]),
                source_file=r["source_file"] or "",
                embedded_at=now_str,
            )
            for r, v in zip(rows, vectors)
        ]
        self.table.add(records)
        logger.info("Batch upserted %d segments into segment_vectors", len(records))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        return_vectors: bool = False,
    ) -> list[dict]:
        """ANN search over segment embeddings.

        Returns a list of plain dicts with keys:
            segment_id, text, start_time, end_time, source_file, _distance
        When ``return_vectors=True``, each dict also contains a ``vector``
        key (list[float]) for downstream MMR cosine computation.
        """
        results = (
            self.table.search(query_vector, query_type="vector")
            .limit(top_k)
            .to_list()
        )
        out: list[dict] = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "vector"}
            if return_vectors:
                raw_vec = r.get("vector")
                if raw_vec is not None:
                    row["vector"] = raw_vec.tolist() if hasattr(raw_vec, "tolist") else list(raw_vec)
            out.append(row)
        return out


    def count_embedded(self) -> int:
        """Total number of segments currently embedded."""
        return self.table.count_rows()

    def get_embedded_ids(self) -> set[int]:
        """Return the set of segment_ids already in this table.

        Used by the daemon sweep and backfill command to find which
        segments still need embedding without redundant work.
        """
        rows = self.table.search().select(["segment_id"]).limit(100_000).to_list()
        return {int(r["segment_id"]) for r in rows}


    def identify_speaker(self, voice_print: list[float], threshold: float = 0.82) -> str | None:
        """Find the closest known speaker for a given voice print.
        
        Args:
            voice_print: 192-D list of floats from SpeechBrain.
            threshold: Minimum cosine similarity required to confirm match.
                       (LanceDB uses distance, so distance <= 1 - threshold)
        """
        if self.table.count_rows() == 0:
            return None

        results = self.table.search(voice_print).limit(1).to_list()

        if results:
            best_match = results[0]
            # LanceDB distance is typically 1 - cosine_similarity for vectors
            # So a cosine similarity of 0.82 means distance of 0.18
            distance = float(best_match.get("_distance", 1.0))
            max_distance = 1.0 - threshold

            if distance <= max_distance:
                logger.info(
                    "Voice matched! '%s' (dist: %.4f < %.4f)",
                    best_match["name"], distance, max_distance
                )
                return best_match["name"]
            else:
                logger.debug(
                    "Voice NOT matched. Closest was '%s' (dist: %.4f > %.4f)",
                    best_match["name"], distance, max_distance
                )

        return None

    def save_speaker(self, profile_id: str, name: str, voice_print: list[float]) -> None:
        """Save or update a speaker's voice print in the database."""
        # Delete if it exists to allow updates.
        # LanceDB raises when the filter matches zero rows on some versions — log, never swallow.
        try:
            self.table.delete(f"profile_id = '{profile_id}'")
        except Exception as exc:
            logger.debug("LanceDB delete failed during upsert (expected if new profile): %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SpeakerProfileNode(
            profile_id=profile_id,
            name=name,
            vector=voice_print,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Saved speaker profile for '%s' (ID: %s)", name, profile_id)

