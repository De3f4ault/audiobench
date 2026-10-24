"""Migration m014: Consolidate fragmented works and synchronize LanceDB.

Reads the frozen manifest generated from the library audit, consolidates 810 works
down to 235 target works, prunes 575 orphaned works, and backfills work_id and author
into LanceDB's expressions table via merge_insert.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import sys
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import bindparam, text

from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.storage.models import AudioFileRecord, WorkRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration.m014")

EXPECTED_MANIFEST_SHA256 = "d1a822e634edee89cff97b10508d7c77c573ebcbf4fbc9d20110eca8b286798c"
EXPECTED_TARGET_WORKS = 235
EXPECTED_TOTAL_FILES = 824
EXPECTED_LANCEDB_EXPRESSIONS = 21684


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_migration(manifest_path_str: str, dry_run: bool = False) -> None:
    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    settings = get_settings()

    # 1. Validate manifest integrity
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_mtime = datetime.datetime.fromtimestamp(manifest_path.stat().st_mtime).isoformat()
    manifest_size = len(manifest_bytes)

    logger.info("Manifest Path: %s", manifest_path)
    logger.info("Manifest Size: %d bytes", manifest_size)
    logger.info("Manifest MTime: %s", manifest_mtime)
    logger.info("Manifest SHA256: %s", manifest_sha256)

    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA256 mismatch! Expected {EXPECTED_MANIFEST_SHA256}, got {manifest_sha256}"
        )

    # 2. Validate pre-migration SQLite backup exists
    backup_path = Path(settings.data_dir) / "transcriptions.db.pre_work_consolidation_20260930"
    if not backup_path.exists() or backup_path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"Pre-migration backup missing or invalid: {backup_path}")
    logger.info("Pre-migration backup verified: %s (%d bytes)", backup_path, backup_path.stat().st_size)

    clusters = json.loads(manifest_bytes.decode("utf-8"))
    logger.info("Loaded %d cluster definitions from manifest", len(clusters))

    if len(clusters) != EXPECTED_TARGET_WORKS:
        raise ValueError(f"Expected {EXPECTED_TARGET_WORKS} clusters in manifest, got {len(clusters)}")

    total_files = sum(c["file_count"] for c in clusters)
    if total_files != EXPECTED_TOTAL_FILES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_FILES} files in manifest, got {total_files}")

    # 3. Apply SQLite consolidation
    with get_session() as session:
        initial_works_count = session.query(WorkRecord).count()
        logger.info("Initial SQLite works count: %d", initial_works_count)

        claimed_work_ids: set[int] = set()
        multi_file_reassigned = 0

        for c_idx, cluster in enumerate(clusters, 1):
            c_title = cluster["title"]
            c_author = cluster["author"]
            file_ids = [f["file_id"] for f in cluster["files"]]

            # Query existing work_ids for these files
            af_rows = (
                session.query(AudioFileRecord.id, AudioFileRecord.work_id)
                .filter(AudioFileRecord.id.in_(file_ids))
                .all()
            )
            existing_work_ids = [af[1] for af in af_rows if af[1] is not None]
            available_work_ids = [wid for wid in existing_work_ids if wid not in claimed_work_ids]

            if available_work_ids:
                target_work_id = min(available_work_ids)
            else:
                new_work = WorkRecord(title=c_title, author=c_author)
                session.add(new_work)
                session.flush()
                target_work_id = new_work.id

            claimed_work_ids.add(target_work_id)

            # Update canonical WorkRecord
            target_work = session.query(WorkRecord).filter_by(id=target_work_id).one()
            target_work.title = c_title
            target_work.author = c_author

            # Repoint all files in this cluster
            session.execute(
                text("UPDATE audio_files SET work_id = :target_work_id WHERE id IN :file_ids").bindparams(
                    bindparam("file_ids", expanding=True)
                ),
                {"target_work_id": target_work_id, "file_ids": file_ids},
            )

            # Repoint all expressions attached to these files
            session.execute(
                text(
                    """
                    UPDATE expressions
                    SET work_id = :target_work_id
                    WHERE id IN (
                        SELECT e.id
                        FROM expressions e
                        JOIN transcriptions tx ON e.source_id = tx.id
                        WHERE e.source_type = 'audio_transcript'
                          AND tx.audio_file_id IN :file_ids
                    )
                    """
                ).bindparams(bindparam("file_ids", expanding=True)),
                {"target_work_id": target_work_id, "file_ids": file_ids},
            )

            if len(file_ids) > 1:
                multi_file_reassigned += len(file_ids)

        session.flush()

        # Identify orphaned works
        orphaned_works = session.execute(
            text(
                """
                SELECT id FROM works
                WHERE id NOT IN (
                    SELECT DISTINCT work_id
                    FROM audio_files
                    WHERE work_id IS NOT NULL
                )
                """
            )
        ).scalars().all()

        orphaned_count = len(orphaned_works)
        logger.info("Identified %d orphaned works to prune", orphaned_count)

        if orphaned_count > 0:
            session.execute(
                text("DELETE FROM works WHERE id IN :orphaned_ids").bindparams(
                    bindparam("orphaned_ids", expanding=True)
                ),
                {"orphaned_ids": list(orphaned_works)},
            )
            session.flush()

        final_works_count = session.query(WorkRecord).count()
        unlinked_count = session.query(AudioFileRecord).filter(AudioFileRecord.work_id.is_(None)).count()

        logger.info("Final SQLite works count: %d (Expected: %d)", final_works_count, EXPECTED_TARGET_WORKS)
        logger.info("Final SQLite unlinked files: %d (Expected: 0)", unlinked_count)

        if final_works_count != EXPECTED_TARGET_WORKS:
            raise RuntimeError(f"Expected {EXPECTED_TARGET_WORKS} works after pruning, found {final_works_count}")
        if unlinked_count != 0:
            raise RuntimeError(f"Expected 0 unlinked files after migration, found {unlinked_count}")

        if dry_run:
            logger.info("Dry run requested; rolling back SQLite transaction")
            session.rollback()
            return

        session.commit()
        logger.info("SQLite work consolidation committed successfully")

    # 4. Synchronize LanceDB expressions table
    logger.info("Synchronizing LanceDB expressions table...")
    db = lancedb.connect(settings.lancedb_path)
    tbl = db.open_table("expressions")
    lancedb_initial_count = tbl.count_rows()
    logger.info("LanceDB expressions initial row count: %d", lancedb_initial_count)

    with get_session() as session:
        # Fetch expression_id, work_id, and canonical author from SQLite
        # graph_role = 'sweep_chunk' covers all rows in LanceDB
        rows = session.execute(
            text(
                """
                SELECT e.id AS expression_id, e.work_id, w.author
                FROM expressions e
                JOIN works w ON e.work_id = w.id
                WHERE e.work_id IS NOT NULL
                  AND (e.graph_role = 'sweep_chunk' OR e.source_type = 'audio_transcript')
                """
            )
        ).fetchall()

    logger.info("Retrieved %d expression mappings from SQLite", len(rows))

    # Build PyArrow Table for merge_insert
    expr_ids = [r[0] for r in rows]
    work_ids = [float(r[1]) if r[1] is not None else None for r in rows]
    authors = [r[2] if r[2] is not None else "" for r in rows]

    source_arrow = pa.Table.from_arrays(
        [
            pa.array(expr_ids, type=pa.int64()),
            pa.array(work_ids, type=pa.float64()),
            pa.array(authors, type=pa.string()),
        ],
        names=["expression_id", "work_id", "author"],
    )

    logger.info("Executing merge_insert on LanceDB expressions...")
    tbl.merge_insert("expression_id").when_matched_update_all().execute(source_arrow)
    logger.info("LanceDB merge_insert completed")

    logger.info("Running LanceDB cleanup_old_versions()...")
    tbl.cleanup_old_versions()
    logger.info("LanceDB cleanup completed")

    # 5. Verify LanceDB Post-Migration Invariants
    lancedb_final_count = tbl.count_rows()
    arrow_tbl = tbl.to_arrow()
    valid_work_ids = pc.count(arrow_tbl["work_id"], mode="only_valid").as_py()
    valid_authors = pc.count(arrow_tbl["author"], mode="only_valid").as_py()

    logger.info("LanceDB final row count: %d (Expected: %d)", lancedb_final_count, EXPECTED_LANCEDB_EXPRESSIONS)
    logger.info("LanceDB non-null work_id count: %d (Audio expressions expected: >= 20148)", valid_work_ids)
    logger.info("LanceDB non-null author count: %d", valid_authors)

    if lancedb_final_count != EXPECTED_LANCEDB_EXPRESSIONS:
        raise RuntimeError(f"LanceDB row count drifted: {lancedb_final_count} != {EXPECTED_LANCEDB_EXPRESSIONS}")
    if valid_work_ids < 20148:
        raise RuntimeError(f"LanceDB valid work_id count incomplete: {valid_work_ids} < 20148")

    logger.info("Migration m014 completed successfully with 0 errors")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    manifest_arg = args[0] if args else (
        "/home/de3f4ault/.gemini/antigravity/brain/9b832915-1b8e-43a9-9bae-8fa2b1f5ab7a/scratch/work_consolidation_audit_report.json"
    )
    is_dry_run = "--dry-run" in sys.argv
    run_migration(manifest_arg, dry_run=is_dry_run)
