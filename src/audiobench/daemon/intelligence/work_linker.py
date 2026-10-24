"""Work linker daemon intelligence task.

Populates the ``works`` table and links ``audio_files.work_id`` for all files
that currently have no work assignment.

Algorithm:
1. Query all audio files with work_id IS NULL.
2. For each file, call entity_extractor.extract_from_filename().
3. If confidence >= medium: look up or create a WorkRecord matching
   (normalised_title, normalised_author).  Assign work_id.
4. If confidence == low: fall back to extract_from_transcript() using the
   first 200 words of the earliest transcription for that file.
5. Commit in batches of 50.

Deduplication: works are keyed on (normalised_title, author).  Normalisation
strips leading articles ("The ", "A ", "An "), collapses whitespace, and
lower-cases.  This ensures that "The Way of Zen" and "way of zen" resolve to
the same work.

Run-once friendly: files already assigned a work_id are skipped entirely, so
the linker can be added to the daemon scheduler and run on every cycle without
redundant work.
"""

from __future__ import annotations

import re

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.daemon.intelligence.entity_extractor import (
    _known_author_in,
    detect_prayer_or_personal,
    detect_scripture_author,
    extract_from_transcript,
)
from audiobench.daemon.intelligence.work_identity import resolve_work_identity
from audiobench.storage.models import AudioFileRecord, ExpressionRecord, TranscriptionRecord, WorkRecord

logger = get_logger("daemon.intelligence.work_linker")

_LEADING_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _normalise_title(title: str) -> str:
    """Lowercase, strip leading articles, collapse whitespace."""
    title = _LEADING_ARTICLES.sub("", title.strip()).strip()
    return re.sub(r"\s+", " ", title).lower()


def _get_first_transcript_text(audio_file_id: int, session) -> str | None:  # type: ignore[type-arg]
    """Return the first 200 words from the earliest transcription for this file."""
    try:
        row = (
            session.query(TranscriptionRecord)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(TranscriptionRecord.created_at)
            .first()
        )
        if row:
            text = row.full_text or row.raw_text
            if text:
                words = text.split()
                return " ".join(words[:200])
    except Exception as e:
        logger.debug("Could not fetch transcript for file #%d: %s", audio_file_id, e)
    return None


class WorkLinker:
    """Daemon intelligence task that assigns work_id to all unlinked audio files."""

    INTERVAL_SECONDS: int = 3600  # 1 hour
    BATCH_SIZE = 50
    IS_INFERENCE_TASK: bool = False

    def run(self) -> None:
        """Main entry point — called by the intelligence scheduler."""
        with get_session() as session:
            unlinked = (
                session.query(AudioFileRecord)
                .filter(AudioFileRecord.work_id.is_(None))
                .all()
            )

            if unlinked:
                self._link_unlinked(session, unlinked)
            else:
                logger.debug("WorkLinker: all audio files already linked to works")

            self._reconcile_unlinked_expressions(session)
            self._refine_missing_authors(session)

    def _link_unlinked(self, session, unlinked: list[AudioFileRecord]) -> None:
        logger.info("WorkLinker: %d unlinked audio files to process", len(unlinked))
        linked = 0
        skipped = 0

        # Cache: (normalised_title, author_lower) -> work_id
        work_cache: dict[tuple[str, str | None], int] = {}

        for i, af in enumerate(unlinked):
            title, author, resolution_source = resolve_work_identity(
                file_path=af.file_path,
                file_name=af.file_name,
                sequence_key=af.sequence_key,
            )

            # Optional fallback to domain heuristics and transcript if author is missing
            if not author:
                transcript_text = _get_first_transcript_text(af.id, session) or ""
                author = detect_prayer_or_personal(
                    title=title or "",
                    file_name=af.file_name or "",
                    text_sample=transcript_text,
                )
                if not author:
                    author = detect_scripture_author(
                        title=title or "",
                        text_sample=transcript_text,
                    )
                if not author and transcript_text:
                    transcript_entity = extract_from_transcript(transcript_text)
                    if transcript_entity and transcript_entity.author and transcript_entity.confidence in ("high", "medium"):
                        author = transcript_entity.author

            if not title:
                logger.debug("WorkLinker: could not extract title for %s — skipping", af.file_name)
                skipped += 1
                continue

            norm_title = _normalise_title(title)
            author_lower = author.lower() if author else None
            cache_key = (norm_title, author_lower)

            if cache_key in work_cache:
                work_id = work_cache[cache_key]
            else:
                existing = None
                if author_lower:
                    # Exact (title, author) match first
                    existing = (
                        session.query(WorkRecord)
                        .filter(
                            WorkRecord.author.ilike(f"%{author}%"),  # type: ignore[arg-type]
                            WorkRecord.title.ilike(f"%{title}%"),  # type: ignore[arg-type]
                        )
                        .first()
                    )
                if not existing:
                    # Title-only match (author may have been unextractable earlier)
                    existing = (
                        session.query(WorkRecord)
                        .filter(WorkRecord.title.ilike(f"%{title}%"))  # type: ignore[arg-type]
                        .first()
                    )

                if existing:
                    work_id = existing.id
                    if (not existing.author or existing.author == "Unknown Author") and author:
                        existing.author = author
                        session.commit()
                else:
                    new_work = WorkRecord(
                        title=title,
                        author=author,
                    )
                    session.add(new_work)
                    session.flush()
                    work_id = new_work.id
                    session.commit()
                    logger.info(
                        "WorkLinker: created Work #%d: '%s' by '%s'",
                        work_id,
                        title[:60],
                        author or "unknown",
                    )

                work_cache[cache_key] = work_id

            # Assign work_id on the audio file and its sweep_document expressions
            file_row = session.query(AudioFileRecord).filter_by(id=af.id).first()
            if file_row:
                file_row.work_id = work_id
                # Also backfill work_id on all sweep_document expressions
                # that belong to this file (via transcription -> expression chain)
                expr_subq = (
                    session.query(ExpressionRecord.id)
                    .join(
                        TranscriptionRecord,
                        ExpressionRecord.source_id == TranscriptionRecord.id,
                    )
                    .filter(
                        TranscriptionRecord.audio_file_id == af.id,
                    )
                )
                session.query(ExpressionRecord).filter(
                    ExpressionRecord.work_id.is_(None),
                    ExpressionRecord.source_type == "audio_transcript",
                ).filter(
                    ExpressionRecord.id.in_(expr_subq)
                ).update({"work_id": work_id}, synchronize_session=False)
                session.commit()

                # Synchronize LanceDB metadata
                try:
                    from audiobench.memory.memory_store import MemoryStore
                    store = MemoryStore()
                    store.table.update(where=f"audio_file_id = {af.id}", values={"work_id": work_id})
                except Exception as ldb_err:
                    logger.debug("WorkLinker: could not update LanceDB work_id for audio_file %d: %s", af.id, ldb_err)

                # Event-driven ingest linking: enqueue chunks for graph_linker
                chunk_rows = (
                    session.query(ExpressionRecord.id)
                    .filter(
                        ExpressionRecord.work_id == work_id,
                        ExpressionRecord.source_type == "audio_transcript",
                        ExpressionRecord.graph_role == "sweep_chunk",
                    )
                    .filter(ExpressionRecord.id.in_(expr_subq))
                    .all()
                )
                chunk_ids = [r[0] for r in chunk_rows]
                if chunk_ids:
                    try:
                        from audiobench.memory.graph_linker import enqueue_for_linking

                        enqueued = enqueue_for_linking(chunk_ids)
                        logger.info(
                            "WorkLinker: enqueued %d chunks for work #%d to graph_linker",
                            enqueued,
                            work_id,
                        )
                    except Exception as eq_err:
                        logger.warning(
                            "WorkLinker: failed to enqueue chunks for linking: %s",
                            eq_err,
                        )

            linked += 1
            if (i + 1) % 50 == 0:
                logger.info("WorkLinker: processed %d / %d", i + 1, len(unlinked))

        logger.info(
            "WorkLinker: complete — linked=%d, skipped=%d (no extractable title)",
            linked,
            skipped,
        )

    def _refine_missing_authors(self, session) -> int:
        """Attempt to fill in author for works where author is missing or unknown."""
        missing = (
            session.query(WorkRecord)
            .filter(
                WorkRecord.author.is_(None)
                | (WorkRecord.author == "")
                | (WorkRecord.author == "Unknown Author")
            )
            .all()
        )
        if not missing:
            return 0

        refined = 0
        for work in missing:
            files = session.query(AudioFileRecord).filter_by(work_id=work.id).all()
            found_author = None

            # 1. Try resolving from audio file metadata / filename / tags
            for af in files:
                t, a, _ = resolve_work_identity(
                    file_path=af.file_path,
                    file_name=af.file_name,
                    sequence_key=af.sequence_key,
                )
                if a and a.lower() != "unknown author":
                    found_author = a
                    break

            # 2. Try specialized domain rules (prayers, personal logs, series, lectures)
            if not found_author:
                for af in files:
                    transcript_text = _get_first_transcript_text(af.id, session) or ""
                    found_author = detect_prayer_or_personal(
                        title=work.title or "",
                        file_name=af.file_name or "",
                        text_sample=transcript_text,
                    )
                    if found_author:
                        break

            # 3. Try scripture detection
            if not found_author:
                for af in files:
                    transcript_text = _get_first_transcript_text(af.id, session) or ""
                    found_author = detect_scripture_author(
                        title=work.title or "",
                        text_sample=transcript_text,
                    )
                    if found_author:
                        break

            # 4. Fallback to title-only domain heuristics if files/transcripts were empty
            if not found_author:
                found_author = (
                    detect_prayer_or_personal(
                        title=work.title or "",
                        file_name="",
                        text_sample="",
                    )
                    or detect_scripture_author(
                        title=work.title or "",
                        text_sample="",
                    )
                    or _known_author_in(work.title or "")
                )

            # 5. Try resolving from transcript announcement or known authors
            if not found_author:
                for af in files:
                    transcript_text = _get_first_transcript_text(af.id, session)
                    if transcript_text:
                        transcript_entity = extract_from_transcript(transcript_text)
                        if transcript_entity and transcript_entity.author and transcript_entity.confidence in ("high", "medium"):
                            found_author = transcript_entity.author
                            break

            if found_author:
                work.author = found_author
                refined += 1
                logger.info(
                    "WorkLinker: refined author for Work #%d ('%s') -> '%s'",
                    work.id,
                    work.title[:40],
                    found_author,
                )

        if refined > 0:
            session.commit()
            logger.info("WorkLinker: refined authors for %d works", refined)
        return refined

    def _reconcile_unlinked_expressions(self, session) -> int:
        """Backfill work_id on any expressions whose audio_file is linked but expressions are NULL.

        Handles cases where RAG sweep created expressions before WorkLinker assigned
        a work_id to the audio file, or where work_id assignments occurred out of order.
        Also verifies and synchronizes LanceDB work_id metadata.
        """
        # Find all audio files that have a work_id assigned, but have expressions with work_id IS NULL
        unlinked_pairs = (
            session.query(AudioFileRecord.id, AudioFileRecord.work_id)
            .join(TranscriptionRecord, TranscriptionRecord.audio_file_id == AudioFileRecord.id)
            .join(ExpressionRecord, ExpressionRecord.source_id == TranscriptionRecord.id)
            .filter(
                AudioFileRecord.work_id.is_not(None),
                ExpressionRecord.source_type == "audio_transcript",
                ExpressionRecord.work_id.is_(None),
            )
            .distinct()
            .all()
        )

        total_reconciled = 0
        if unlinked_pairs:
            logger.info("WorkLinker: found %d audio files with unlinked expressions to reconcile", len(unlinked_pairs))
            for af_id, work_id in unlinked_pairs:
                expr_subq = (
                    session.query(ExpressionRecord.id)
                    .join(TranscriptionRecord, ExpressionRecord.source_id == TranscriptionRecord.id)
                    .filter(TranscriptionRecord.audio_file_id == af_id)
                )
                updated = (
                    session.query(ExpressionRecord)
                    .filter(
                        ExpressionRecord.work_id.is_(None),
                        ExpressionRecord.source_type == "audio_transcript",
                        ExpressionRecord.id.in_(expr_subq),
                    )
                    .update({"work_id": work_id}, synchronize_session=False)
                )
                session.commit()
                total_reconciled += updated

                # Synchronize LanceDB metadata
                try:
                    from audiobench.memory.memory_store import MemoryStore
                    store = MemoryStore()
                    store.table.update(where=f"audio_file_id = {af_id}", values={"work_id": float(work_id)})
                except Exception as ldb_err:
                    logger.debug("WorkLinker: could not update LanceDB work_id for audio_file %d: %s", af_id, ldb_err)

                # Event-driven ingest linking: enqueue chunks for graph_linker
                chunk_rows = (
                    session.query(ExpressionRecord.id)
                    .filter(
                        ExpressionRecord.work_id == work_id,
                        ExpressionRecord.source_type == "audio_transcript",
                        ExpressionRecord.graph_role == "sweep_chunk",
                        ExpressionRecord.id.in_(expr_subq),
                    )
                    .all()
                )
                chunk_ids = [r[0] for r in chunk_rows]
                if chunk_ids:
                    try:
                        from audiobench.memory.graph_linker import enqueue_for_linking
                        enqueued = enqueue_for_linking(chunk_ids)
                        logger.info(
                            "WorkLinker: reconciled and enqueued %d chunks for work #%d to graph_linker",
                            enqueued,
                            work_id,
                        )
                    except Exception as eq_err:
                        logger.warning("WorkLinker: failed to enqueue reconciled chunks for linking: %s", eq_err)

            logger.info("WorkLinker: reconciled %d expressions across %d audio files", total_reconciled, len(unlinked_pairs))

        # Check LanceDB for any audio files with work_id IS NULL in LanceDB
        try:
            from audiobench.memory.memory_store import MemoryStore
            store = MemoryStore()
            tbl = store.table
            ldb_nulls = tbl.search().where("work_id IS NULL").select(["audio_file_id"]).limit(1000).to_pandas()
            if not ldb_nulls.empty and "audio_file_id" in ldb_nulls:
                null_af_ids = [int(x) for x in ldb_nulls["audio_file_id"].dropna().unique()]
                for af_id in null_af_ids:
                    af_row = session.query(AudioFileRecord).filter_by(id=af_id).first()
                    if af_row and af_row.work_id is not None:
                        store.table.update(where=f"audio_file_id = {af_id}", values={"work_id": float(af_row.work_id)})
                        logger.info("WorkLinker: backfilled LanceDB work_id=%d for audio_file #%d", af_row.work_id, af_id)
        except Exception as ldb_err:
            logger.debug("WorkLinker: LanceDB work_id reconciliation check skipped: %s", ldb_err)

        return total_reconciled
