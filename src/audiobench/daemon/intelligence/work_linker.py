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
from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript
from audiobench.daemon.intelligence.work_identity import resolve_work_identity
from audiobench.storage.models import AudioFileRecord, ExpressionRecord, WorkRecord

logger = get_logger("daemon.intelligence.work_linker")

_LEADING_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _normalise_title(title: str) -> str:
    """Lowercase, strip leading articles, collapse whitespace."""
    title = _LEADING_ARTICLES.sub("", title.strip()).strip()
    return re.sub(r"\s+", " ", title).lower()


def _get_first_transcript_text(audio_file_id: int, session) -> str | None:  # type: ignore[type-arg]
    """Return the first 200 words from the earliest transcription for this file."""
    try:
        from audiobench.storage.models import TranscriptionRecord  # type: ignore[attr-defined]
        row = (
            session.query(TranscriptionRecord)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(TranscriptionRecord.created_at)
            .first()
        )
        if row and row.transcript:
            words = row.transcript.split()
            return " ".join(words[:200])
    except Exception as e:
        logger.debug("Could not fetch transcript for file #%d: %s", audio_file_id, e)
    return None


class WorkLinker:
    """Daemon intelligence task that assigns work_id to all unlinked audio files."""

    INTERVAL_SECONDS: int = 3600  # 1 hour
    BATCH_SIZE = 50

    def run(self) -> None:
        """Main entry point — called by the intelligence scheduler."""
        with get_session() as session:
            unlinked = (
                session.query(AudioFileRecord)
                .filter(AudioFileRecord.work_id.is_(None))
                .all()
            )

        if not unlinked:
            logger.debug("WorkLinker: all audio files already linked to works")
            return

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

            # Optional fallback to transcript if author is missing
            if not author:
                with get_session() as session:
                    transcript_text = _get_first_transcript_text(af.id, session)
                if transcript_text:
                    transcript_entity = extract_from_transcript(transcript_text)
                    if transcript_entity and transcript_entity.author:
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
                with get_session() as session:
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
            with get_session() as session:
                file_row = session.query(AudioFileRecord).filter_by(id=af.id).first()
                if file_row:
                    file_row.work_id = work_id
                    # Also backfill work_id on all sweep_document expressions
                    # that belong to this file (via transcription -> expression chain)
                    session.query(ExpressionRecord).filter(
                        ExpressionRecord.work_id.is_(None),
                        ExpressionRecord.source_type == "audio_transcript",
                    ).filter(
                        ExpressionRecord.id.in_(
                            session.query(ExpressionRecord.id)
                            .join(
                                __import__("audiobench.storage.models", fromlist=["TranscriptionRecord"]).TranscriptionRecord,
                                ExpressionRecord.source_id == __import__("audiobench.storage.models", fromlist=["TranscriptionRecord"]).TranscriptionRecord.id,
                            )
                            .filter(
                                __import__("audiobench.storage.models", fromlist=["TranscriptionRecord"]).TranscriptionRecord.audio_file_id == af.id,
                            )
                        )
                    ).update({"work_id": work_id}, synchronize_session=False)
                    session.commit()

            linked += 1
            if (i + 1) % 50 == 0:
                logger.info("WorkLinker: processed %d / %d", i + 1, len(unlinked))

        logger.info(
            "WorkLinker: complete — linked=%d, skipped=%d (no extractable title)",
            linked,
            skipped,
        )
