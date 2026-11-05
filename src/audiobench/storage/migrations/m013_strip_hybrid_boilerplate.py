"""One-time migration: strip publisher preambles from hybrid sweep_chunk expressions.

A hybrid chunk is one that starts with a publisher disclaimer but also contains
substantive literary or philosophical content after it.  Example:

    "This isa LibriVox recording. Al libriVox recordings are in the public
     domain. For more information visit LibriVox.org. chapter two of the great
     gatsby by f scott fitzgerald about half way between west egg and new york..."

After running this script the chunk becomes:

    "chapter two of the great gatsby by f scott fitzgerald about half way
     between west egg and new york..."

The vector in LanceDB is then regenerated from the clean text so that cosine
similarity during knowledge graph edge detection reflects the actual content,
not the publisher stamp.

Safe to run multiple times (idempotent via content_hash deduplication in
ExpressionRepository.register).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Allow running from project root: python src/audiobench/storage/migrations/m013_strip_hybrid_boilerplate.py
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


def migrate() -> None:
    from audiobench.core.db_session import get_session
    from audiobench.core.logger_factory import get_logger
    from audiobench.memory.boilerplate import clean_audio_disclaimers, is_pure_boilerplate
    from audiobench.memory.memory_store import MemoryStore
    from audiobench.storage.models import ExpressionRecord

    logger = get_logger("migration.m013")

    BOILERPLATE_SIGNATURES = [
        "librivox",
        "liebervox",
        "librefox",
        "libravox",
        "library vox",
        "this is audible",
        "sounds true presents",
        "tantor audio",
        "brilliance audio",
        "blackstone audio",
    ]
    MIN_HYBRID_LEN = 300  # chars — shorter than this is pure and handled by SQL migration 039

    store = MemoryStore()

    with get_session() as session:
        all_chunks = session.query(ExpressionRecord).filter(
            ExpressionRecord.graph_role == "sweep_chunk",
        ).all()

        hybrid_ids = []
        for expr in all_chunks:
            text_lower = expr.content.lower()
            has_sig = any(sig in text_lower for sig in BOILERPLATE_SIGNATURES)
            if has_sig and len(expr.content) >= MIN_HYBRID_LEN:
                hybrid_ids.append(expr.id)

    logger.info("Found %d hybrid chunks to strip", len(hybrid_ids))

    stripped_count = 0
    skipped_count = 0

    for expr_id in hybrid_ids:
        with get_session() as session:
            expr = session.query(ExpressionRecord).filter_by(id=expr_id).first()
            if not expr:
                continue

            original = expr.content
            cleaned = clean_audio_disclaimers(original)

            if cleaned == original:
                skipped_count += 1
                continue  # nothing to strip

            if is_pure_boilerplate(cleaned):
                # After stripping the preamble there is still nothing left —
                # this should have been caught by the SQL migration; skip.
                logger.warning(
                    "Chunk #%d became pure boilerplate after stripping — skipping (should be purged by 039 SQL migration)",
                    expr_id,
                )
                skipped_count += 1
                continue

            new_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

            # Update SQLite
            expr.content = cleaned
            expr.content_hash = new_hash
            session.commit()
            logger.debug("Stripped chunk #%d: %d -> %d chars", expr_id, len(original), len(cleaned))

            # Drop old vector from LanceDB and re-embed
            try:
                store.delete_node(expr_id)
            except Exception as e:
                logger.warning("Could not delete old vector for #%d: %s", expr_id, e)

            try:
                store.write_node(
                    expression_id=expr_id,
                    content=cleaned,
                    source_type=expr.source_type,
                    speaker=expr.speaker,
                )
            except Exception as e:
                logger.warning("Could not re-embed chunk #%d: %s", expr_id, e)

            stripped_count += 1

    logger.info(
        "m013 complete: %d chunks stripped and re-embedded, %d skipped (no change needed)",
        stripped_count,
        skipped_count,
    )
    print(f"m013: stripped={stripped_count}, skipped={skipped_count}")


if __name__ == "__main__":
    migrate()
