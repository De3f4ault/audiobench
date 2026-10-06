"""ChapterEmbedder daemon intelligence task.

Periodically discovers chapters that need vector embedding (Tier 2 Passage / Chapter level)
and embeds them using contextual prefixing (Author, Work, Chapter).

Enforces the 'last_embedded_at' dirty-flag discipline:
  - Discovers chapters where is_ghost == 0 and last_embedded_at IS NULL.
  - Aggregates segment text for each chapter.
  - Formats with build_context_prefix().
  - Writes a 'sweep_passage' expression into SQLite and LanceDB.
  - Updates chapter.last_embedded_at timestamp.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.daemon.intelligence.scheduler import IntelligenceTask
from audiobench.memory.context_prefix import build_context_prefix

if TYPE_CHECKING:
    from audiobench.memory.memory_store import MemoryStore

logger = get_logger("daemon.intelligence.chapter_embedder")


class ChapterEmbedder(IntelligenceTask):
    """Idle-time background embedder for Tier 2 Chapter / Passage nodes."""

    INTERVAL_SECONDS: int = 300  # Run every 5 minutes during idle window
    BATCH_SIZE: int = 20         # Process up to 20 chapters per cycle to stay within budget

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self._store = memory_store

    def _get_store(self) -> MemoryStore | None:
        if self._store is None:
            try:
                from audiobench.memory.singletons import get_memory_store
                self._store = get_memory_store()
            except Exception:
                try:
                    from audiobench.daemon.server import _get_store
                    self._store = _get_store()
                except Exception as e:
                    logger.debug("ChapterEmbedder: could not obtain MemoryStore: %s", e)
        return self._store

    async def run(self) -> None:
        """Run chapter embedding cycle."""
        await asyncio.to_thread(self._run_sync)

    def _run_sync(self) -> None:
        store = self._get_store()
        if store is None:
            logger.debug("ChapterEmbedder: MemoryStore unavailable, skipping run")
            return

        with get_session() as session:
            # Query chapters needing embedding
            unembedded = session.execute(
                sql_text("""
                    SELECT c.id, c.audio_file_id, c.title, c.summary,
                           af.work_id, w.title as work_title, w.author
                    FROM chapters c
                    JOIN audio_files af ON c.audio_file_id = af.id
                    LEFT JOIN works w ON af.work_id = w.id
                    WHERE c.is_ghost = 0
                      AND c.last_embedded_at IS NULL
                    LIMIT :limit
                """),
                {"limit": self.BATCH_SIZE},
            ).fetchall()

            if not unembedded:
                logger.debug("ChapterEmbedder: no unembedded chapters found")
                return

            logger.info("ChapterEmbedder: processing %d unembedded chapters", len(unembedded))

            for cid, afid, title, summary, work_id, work_title, author in unembedded:
                # 1. Gather text content: prefer summary if available, else first ~3000 chars of segments
                text_content = summary
                if not text_content:
                    seg_rows = session.execute(
                        sql_text("""
                            SELECT text FROM segments
                            WHERE chapter_id = :cid
                            ORDER BY segment_index ASC
                        """),
                        {"cid": cid},
                    ).fetchall()
                    combined = " ".join(r[0].strip() for r in seg_rows if r[0])
                    if combined:
                        text_content = combined[:3000]

                if not text_content or len(text_content.strip()) < 20:
                    # Not enough text to embed; mark embedded to avoid infinite loop
                    session.execute(
                        sql_text("UPDATE chapters SET last_embedded_at = :now WHERE id = :cid"),
                        {"now": datetime.now(UTC), "cid": cid},
                    )
                    continue

                # 2. Build contextual prefix
                prefix = build_context_prefix(
                    author=author,
                    work_title=work_title,
                    chapter_title=title,
                )
                prefixed_content = f"{prefix}{text_content.strip()}"

                # 3. Create or update expression in SQLite
                # Check if an expression for this chapter already exists
                existing_expr_id = session.execute(
                    sql_text("""
                        SELECT id FROM expressions
                        WHERE source_type = 'chapter_summary' AND source_id = :cid
                        LIMIT 1
                    """),
                    {"cid": cid},
                ).scalar()

                if existing_expr_id is None:
                    session.execute(
                        sql_text("""
                            INSERT INTO expressions (
                                content, source_type, source_id, work_id, chapter_id, graph_role, privacy_tier, created_at
                            ) VALUES (
                                :content, 'chapter_summary', :cid, :work_id, :cid, 'sweep_passage', 0, :now
                            )
                        """),
                        {
                            "content": text_content.strip(),
                            "cid": cid,
                            "work_id": work_id,
                            "now": datetime.now(UTC),
                        },
                    )
                    expr_id = session.execute(sql_text("SELECT last_insert_rowid()")).scalar()
                else:
                    expr_id = existing_expr_id
                    session.execute(
                        sql_text("""
                            UPDATE expressions
                            SET content = :content, work_id = :work_id, chapter_id = :cid
                            WHERE id = :eid
                        """),
                        {
                            "content": text_content.strip(),
                            "work_id": work_id,
                            "cid": cid,
                            "eid": expr_id,
                        },
                    )

                # 4. Embed and write into LanceDB
                try:
                    store.write_node(
                        expression_id=expr_id,
                        content=prefixed_content,
                        source_type="chapter_summary",
                        chapter_id=cid,
                        chapter_title=title,
                        author=author,
                        work_id=work_id,
                        audio_file_id=afid,
                    )
                except Exception as e:
                    logger.error("ChapterEmbedder: failed to write node for chapter %d: %s", cid, e)
                    continue

                # 5. Mark chapter dirty flag as clean
                session.execute(
                    sql_text("UPDATE chapters SET last_embedded_at = :now WHERE id = :cid"),
                    {"now": datetime.now(UTC), "cid": cid},
                )

            session.commit()
            logger.info("ChapterEmbedder: batch complete")
