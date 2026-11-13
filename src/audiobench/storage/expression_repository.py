"""Repository — CRUD operations for Expression memory data.

Provides a clean interface for managing semantic memory expressions:
- Registering new expressions
- Linking expressions via directed relations
- Querying by ID and retrieving relations
- Navigating expression hierarchies (walking to parents)

Batch methods (EQ-1 / EQ-2):
- get_by_ids(): single WHERE IN query for N expressions — replaces N individual get_by_id() calls.
- get_parents_batch(): one JOIN to resolve all source-relation parents — replaces N walk_to_parent() calls.

Privacy-tier enforcement (Stage 2):
- register() accepts privacy_tier, stored in expressions.privacy_tier.
- get_by_id / get_by_ids / get_parents_batch / walk_to_parent accept session=None.
  When session is None or session.effective_tier() < record.privacy_tier, the
  returned record's content is replaced with "[REDACTED]" in-memory only.
  The underlying SQLite row is never mutated (session.expunge() is called before
  any redaction assignment, closing the write path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sqlalchemy import asc
from sqlalchemy.orm import Session

from audiobench.core import db_session


def get_session():
    """Get database session dynamically from db_session module to prevent mock freeze across tests."""
    return db_session.get_session()
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import RelationType
from audiobench.storage.models import ExpressionRecord, ExpressionRelation, ExpressionSegmentMap

if TYPE_CHECKING:
    from audiobench.cli.repl.session import ReplSession

logger = get_logger("storage.expression_repository")


def _redact_expression(record: ExpressionRecord, session: "ReplSession | None") -> None:
    """Redact record.content in-place if session tier < record.privacy_tier.

    The record MUST already be expunged from SQLAlchemy before this is called
    so the assignment cannot be flushed back to the database.
    """
    effective_tier: int = session.effective_tier() if session is not None else 0
    if record.privacy_tier > 0 and record.privacy_tier > effective_tier:
        record.content = "[REDACTED]"
        # Attach a sentinel so callers can detect redaction without re-checking tiers.
        object.__setattr__(record, "redacted", True)
    else:
        object.__setattr__(record, "redacted", False)


class ExpressionRepository:
    """CRUD and traversal operations for semantic memory expressions."""

    def register(
        self,
        content: str,
        source_type: str,
        *,
        source_id: int | None = None,
        session_type: str | None = None,
        session_id: int | None = None,
        speaker: str | None = None,
        work_id: int | None = None,
        graph_role: str | None = None,
        privacy_tier: int = 0,
        chapter_id: int | None = None,
        calibration_key: str | None = None,
    ) -> ExpressionRecord:
        """Register a new semantic expression.

        Args:
            privacy_tier: Security classification to persist on the row.
                0 = public (default). 2 = intimate (owner-voice or derived from it).
                The caller is responsible for computing the correct tier via
                TranscriptionRepository.get_max_privacy_tier_for_transcriptions()
                or get_max_privacy_tier_for_segments() before calling register().
            chapter_id: Optional ID of the chapter containing this expression.

        Returns:
            The created ExpressionRecord.
        """
        import hashlib

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        with get_session() as session:
            # Check for existing semantic content within this source_type to prevent duplication.
            # graph_role is included in the composite key so that T1/T2/T3 nodes for the
            # same transcript are never collapsed into one row even when their text is identical
            # (e.g. a single-chunk transcript where parent_text == full cleaned text).
            existing = session.query(ExpressionRecord).filter_by(
                content_hash=content_hash,
                source_type=source_type,
                source_id=source_id,
                graph_role=graph_role,
            ).first()
            if existing:
                logger.debug("Deduplicated expression #%d via content_hash", existing.id)
                mutated = False
                # Upgrade tier if the new call claims a higher tier for the same content.
                if privacy_tier > existing.privacy_tier:
                    existing.privacy_tier = privacy_tier
                    mutated = True
                    logger.debug(
                        "Upgraded expression #%d privacy_tier to %d",
                        existing.id, privacy_tier,
                    )
                if not existing.chapter_id and chapter_id:
                    existing.chapter_id = chapter_id
                    mutated = True
                    logger.debug(
                        "Linked expression #%d to chapter %d",
                        existing.id, chapter_id,
                    )
                if not existing.work_id and work_id:
                    existing.work_id = work_id
                    mutated = True
                    logger.debug(
                        "Assigned work_id %d to existing expression #%d",
                        work_id, existing.id,
                    )
                if mutated:
                    session.commit()
                session.expunge(existing)
                return existing

            record = ExpressionRecord(
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_id=source_id,
                session_type=session_type,
                session_id=session_id,
                speaker=speaker,
                work_id=work_id,
                graph_role=graph_role,
                privacy_tier=privacy_tier,
                chapter_id=chapter_id,
                calibration_key=calibration_key,
            )
            session.add(record)
            session.commit()

            self._resolve_pending_relations(session, record)

            session.refresh(record)
            logger.info(
                "Registered %s expression #%d (privacy_tier=%d)",
                source_type, record.id, privacy_tier,
            )
            # Create a detached instance to return outside session
            session.expunge(record)
            return record

    def _resolve_pending_relations(self, session, record: ExpressionRecord) -> None:
        """Resolve any pending forward references to this newly created expression."""
        if not record.source_id:
            return

        from audiobench.storage.models import PendingRelation
        pending = session.query(PendingRelation).filter_by(
            to_expression_id_hint=record.source_id,
            to_source_type=record.source_type
        ).all()

        if pending:
            for p in pending:
                rel = ExpressionRelation(
                    from_expression_id=p.from_expression_id,
                    to_expression_id=record.id,
                    relation_type=p.relation_type,
                    created_by="system"
                )
                session.add(rel)
                session.delete(p)
            session.commit()
            logger.info("Resolved %d pending relations for expression #%d", len(pending), record.id)

    def register_batch(
        self,
        items: list[dict],
        known_hashes: set[str] | None = None,
    ) -> list[ExpressionRecord]:
        """Register a batch of semantic expressions.

        Args:
            items:        List of dicts with keys: content, source_type,
                          source_id (optional), speaker (optional),
                          privacy_tier (optional, default 0).
            known_hashes: The in-memory O(1) hash set from SweepState.
                          When provided, deduplication is a pure RAM lookup —
                          no SQL query fires for already-known content.
                          New hashes are added to the set in-place.

        Returns:
            List of ExpressionRecords (newly created + deduplicated).
        """
        import hashlib

        if not items:
            return []

        # ── Step 1: dedup within the incoming batch (pure Python) ──────────
        unique_items: dict[str, dict] = {}
        for item in items:
            h = hashlib.sha256(item["content"].encode("utf-8")).hexdigest()
            if h not in unique_items:
                item["_hash"] = h
                unique_items[h] = item

        items_to_process = list(unique_items.values())

        # ── Step 2: separate already-known from genuinely new ───────────────
        # We NO LONGER drop `already_known` items early.
        # Even if we know an expression exists in SQLite via `known_hashes`,
        # we MUST query the DB for its ID so we can return the `ExpressionRecord`.
        # If we drop it here, the caller (RAG sweep) will never pass it to LanceDB
        # for embedding, breaking startup recovery!
        need_db_lookup = items_to_process
        already_known: list = []

        # ── Step 3: for items still uncertain, query DB once ────────────────
        deduped: list[ExpressionRecord] = []
        new_records: list[ExpressionRecord] = []

        if need_db_lookup:
            source_type = need_db_lookup[0]["source_type"]
            db_hashes = [item["_hash"] for item in need_db_lookup]

            with get_session() as session:
                existing = (
                    session.query(ExpressionRecord)
                    .filter(
                        ExpressionRecord.content_hash.in_(db_hashes),
                        ExpressionRecord.source_type == source_type,
                    )
                    .all()
                )
                existing_by_key = {(r.content_hash, r.source_id): r for r in existing}

                for item in need_db_lookup:
                    key = (item["_hash"], item.get("source_id"))
                    item_tier = int(item.get("privacy_tier", 0))
                    if key in existing_by_key:
                        rec = existing_by_key[key]
                        logger.debug(
                            "Deduplicated expression #%d via content_hash and source_id",
                            rec.id,
                        )
                        # Upgrade tier if the new batch claims a higher tier.
                        if item_tier > rec.privacy_tier:
                            rec.privacy_tier = item_tier
                        deduped.append(rec)
                        # Teach the set so next batch hits RAM only
                        if known_hashes is not None:
                            known_hashes.add(item["_hash"])
                    else:
                        rec = ExpressionRecord(
                            content=item["content"],
                            content_hash=item["_hash"],
                            source_type=item["source_type"],
                            source_id=item.get("source_id"),
                            speaker=item.get("speaker"),
                            work_id=item.get("work_id"),
                            privacy_tier=item_tier,
                        )
                        session.add(rec)
                        new_records.append(rec)

                if new_records:
                    session.flush()
                    for rec in new_records:
                        self._resolve_pending_relations(session, rec)
                    session.commit()
                    if known_hashes is not None:
                        for rec in new_records:
                            if rec.content_hash:
                                known_hashes.add(rec.content_hash)
                    logger.info(
                        "Registered %d new + %d deduplicated expressions",
                        len(new_records),
                        len(deduped) + len(already_known),
                    )

                for rec in list(existing) + new_records:
                    try:
                        session.expunge(rec)
                    except Exception:
                        pass

        if already_known:
            logger.debug(
                "Skipped %d expressions via O(1) in-memory hash set",
                len(already_known),
            )

        return deduped + new_records

    def get_by_id(
        self,
        expression_id: int,
        session: "ReplSession | None" = None,
    ) -> ExpressionRecord | None:
        """Get a single expression by ID.

        Args:
            session: The active ReplSession, or None.  When None (all standalone
                CLI commands, daemon workers, and background tasks), the caller is
                treated as locked at Tier 0.  Expressions whose privacy_tier >= 2
                have their content replaced with "[REDACTED]" in memory only —
                the underlying SQLite row is never written.
        """
        with get_session() as db_session:
            record = db_session.query(ExpressionRecord).filter_by(id=expression_id).first()
            if record:
                db_session.expunge(record)
                _redact_expression(record, session)
            return record

    def get_by_ids(
        self,
        ids: list[int],
        session: "ReplSession | None" = None,
    ) -> dict[int, ExpressionRecord]:
        """Batch-fetch expressions by ID in a single WHERE IN query.

        Returns a dict keyed by expression ID.  Returns ``{}`` immediately when
        *ids* is empty — no query is issued (SQLite raises on empty IN clause).

        Args:
            ids:     List of expression IDs to fetch.
            session: The active ReplSession, or None.  Protected records are
                     redacted in-memory before being placed in the result dict.

        Returns:
            Dict mapping expression_id → ExpressionRecord for every ID found.
            IDs that do not exist in the database are absent from the result.
        """
        if not ids:
            return {}
        with get_session() as db_session:
            records = (
                db_session.query(ExpressionRecord)
                .filter(ExpressionRecord.id.in_(ids))
                .all()
            )
            result: dict[int, ExpressionRecord] = {}
            for r in records:
                db_session.expunge(r)
                _redact_expression(r, session)
                result[r.id] = r
            return result

    def get_parents_batch(
        self,
        child_ids: list[int],
        session: "ReplSession | None" = None,
    ) -> dict[int, ExpressionRecord]:
        """Resolve source-relation parents for a batch of child expression IDs.

        Issues a single JOIN query instead of N individual walk_to_parent() calls,
        reducing complexity from O(N×Q) to O(1) queries.

        Only follows the first outbound ``source`` relation per child.
        Children that have no source relation are absent from the result.

        Args:
            child_ids: List of expression IDs whose parents we want.
            session:   The active ReplSession, or None.  Protected parent records
                       are redacted in-memory before being placed in the result dict.

        Returns:
            Dict mapping child_expression_id → parent ExpressionRecord.
        """
        if not child_ids:
            return {}
        with get_session() as db_session:
            rows = (
                db_session.query(
                    ExpressionRelation.from_expression_id,
                    ExpressionRecord,
                )
                .join(
                    ExpressionRecord,
                    ExpressionRecord.id == ExpressionRelation.to_expression_id,
                )
                .filter(
                    ExpressionRelation.from_expression_id.in_(child_ids),
                    ExpressionRelation.relation_type
                    == RelationType.SOURCE.value,
                )
                .all()
            )
            result: dict[int, ExpressionRecord] = {}
            for child_id, parent_rec in rows:
                if child_id not in result:  # keep first source relation only
                    db_session.expunge(parent_rec)
                    _redact_expression(parent_rec, session)
                    result[child_id] = parent_rec
            return result

    def link(
        self,
        from_id: int,
        to_id: int,
        relation_type: str,
        *,
        weight: float = 1.0,
        created_by: str = "system",
    ) -> ExpressionRelation:
        """Create a directed relation between two expressions.

        Idempotent: if an identical edge (from_id, to_id, relation_type) already
        exists, returns the original relation rather than inserting a duplicate.
        The returned relation's created_at reflects the original insertion time,
        not the current call time.

        Raises:
            ValueError: if from_id == to_id (self-link). This indicates a
                graph_role dedup collision — check that graph_role is being
                passed correctly at the call site in _do_sweep_once().

        Returns:
            The created or pre-existing ExpressionRelation.
        """
        if from_id == to_id:
            raise ValueError(
                f"Self-link rejected: expression #{from_id} cannot link to itself "
                f"(relation_type={relation_type!r}). This indicates a graph_role "
                f"dedup collision — verify graph_role is passed at all three tier "
                f"register() call sites in _do_sweep_once()."
            )
        with get_session() as session:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = (
                sqlite_insert(ExpressionRelation)
                .values(
                    from_expression_id=from_id,
                    to_expression_id=to_id,
                    relation_type=relation_type,
                    weight=weight,
                    created_by=created_by,
                )
                .on_conflict_do_nothing(
                    index_elements=["from_expression_id", "to_expression_id", "relation_type"]
                )
            )
            session.execute(stmt)
            session.commit()
            relation = session.query(ExpressionRelation).filter_by(
                from_expression_id=from_id,
                to_expression_id=to_id,
                relation_type=relation_type,
            ).first()
            logger.info("Linked %d -> %d (type: %s)", from_id, to_id, relation_type)
            session.expunge(relation)
            return relation

    def link_segments_batch(self, links: list[tuple[int, int]]) -> None:
        """Batch link expressions to segments in expression_segment_map.

        Idempotent: ignores already-existing expression-segment pairs.

        Args:
            links: list of (expression_id, segment_id) tuples.
        """
        if not links:
            return
        with get_session() as session:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            unique_links = [{"expression_id": eid, "segment_id": sid} for eid, sid in set(links)]
            stmt = (
                sqlite_insert(ExpressionSegmentMap)
                .values(unique_links)
                .on_conflict_do_nothing(
                    index_elements=["expression_id", "segment_id"]
                )
            )
            session.execute(stmt)
            session.commit()
            logger.info("Linked %d expression-segment entries in junction table", len(unique_links))

    def get_relations(
        self,
        expression_id: int,
        direction: Literal["out", "in", "both"] = "both",
        relation_type: str | None = None,
    ) -> list[ExpressionRelation]:
        """Get relations involving this expression.

        Args:
            expression_id: The ID of the expression.
            direction: 'out' (where this is source), 'in' (where this is target), or 'both'.
            relation_type: Optional filter by relation type.

        Returns:
            List of ExpressionRelation records.
        """
        with get_session() as session:
            query = session.query(ExpressionRelation)

            if direction == "out":
                query = query.filter_by(from_expression_id=expression_id)
            elif direction == "in":
                query = query.filter_by(to_expression_id=expression_id)
            else:
                query = query.filter(
                    (ExpressionRelation.from_expression_id == expression_id)
                    | (ExpressionRelation.to_expression_id == expression_id)
                )

            if relation_type:
                query = query.filter_by(relation_type=relation_type)

            records = query.order_by(asc(ExpressionRelation.created_at)).all()
            for r in records:
                session.expunge(r)
            return records

    def update_inference_status(
        self,
        expression_id: int,
        status: str,
    ) -> bool:
        """Update the inference status of an expression.

        Args:
            expression_id: The ID of the expression.
            status: Must match InferenceStatus values.

        Returns:
            True if updated, False if not found.
        """
        with get_session() as session:
            record = session.query(ExpressionRecord).filter_by(id=expression_id).first()
            if not record:
                return False

            record.inference_status = status
            session.commit()
            logger.info("Updated expression #%d inference_status to '%s'", expression_id, status)
            return True

    def walk_to_parent(
        self,
        expression_id: int,
        session: "ReplSession | None" = None,
    ) -> ExpressionRecord | None:
        """Follow a 'source' relation upward to find the parent expression.

        Returns the target expression of the first outbound 'source' relation found,
        or None if no such relation exists.

        Args:
            session: The active ReplSession, or None.  Propagated to get_by_id()
                     so the parent's content is redacted if its tier exceeds the
                     caller's effective tier.
        """
        out_relations = self.get_relations(
            expression_id, direction="out", relation_type=RelationType.SOURCE.value
        )

        if not out_relations:
            return None

        # Follow the first source relation
        parent_id = out_relations[0].to_expression_id
        return self.get_by_id(parent_id, session=session)

    def delete_by_source(
        self,
        source_type: str,
        source_id: int,
        session: Session | None = None,
    ) -> int:
        """Delete all expressions for a given source entity from SQLite and LanceDB.

        Returns the number of expressions deleted.
        """
        if not source_id:
            return 0
        if session is not None:
            return self._delete_by_source_impl(session, source_type, source_id, should_commit=False)
        else:
            with get_session() as s:
                return self._delete_by_source_impl(s, source_type, source_id, should_commit=True)

    def _delete_by_source_impl(
        self,
        session: Session,
        source_type: str,
        source_id: int,
        should_commit: bool,
    ) -> int:
        records = (
            session.query(ExpressionRecord)
            .filter_by(source_type=source_type, source_id=source_id)
            .all()
        )
        if not records:
            return 0

        expr_ids = [r.id for r in records]

        from audiobench.core.settings import get_settings
        if not get_settings().disable_memory:
            try:
                from audiobench.daemon.factory import get_daemon_client
                client = get_daemon_client()
                try:
                    # Single batch IPC call — O(1) round-trips regardless of size
                    client.delete_batch(expr_ids)
                except AttributeError:
                    # Graceful fallback for older daemon builds during rolling restart
                    for expr_id in expr_ids:
                        try:
                            client.delete(expr_id)
                        except Exception as e:
                            logger.warning("Failed to delete vector %d from daemon: %s", expr_id, e)
            except Exception as e:
                logger.warning("Could not connect to daemon for vector deletion: %s", e)

        for r in records:
            session.delete(r)

        if should_commit:
            session.commit()
        else:
            session.flush()

        logger.info(
            "Deleted %d expression(s) for source_type='%s' source_id=%d",
            len(records),
            source_type,
            source_id,
        )
        return len(records)
