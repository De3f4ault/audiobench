"""Migration script m017: Backfill audio_files.tags from audited manifest.

Backfills audio_files.tags with format metadata dictionary from the frozen audit manifest
(work_consolidation_audit_report.json). Preserves any existing label tags (e.g. ['youtube'])
under the '_labels' key.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.storage.models import AudioFileRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration.m017")

EXPECTED_MANIFEST_SHA256 = "d1a822e634edee89cff97b10508d7c77c573ebcbf4fbc9d20110eca8b286798c"
EXPECTED_TOTAL_FILES = 824


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

    # 2. Validate pre-migration backup
    backup_path = Path(settings.data_dir) / "transcriptions.db.pre_tags_backfill_20260930"
    if not backup_path.exists() or backup_path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"Pre-migration backup missing or invalid: {backup_path}")
    logger.info("Pre-migration backup verified: %s (%d bytes)", backup_path, backup_path.stat().st_size)

    clusters = json.loads(manifest_bytes.decode("utf-8"))
    manifest_files: dict[int, dict] = {}
    for c in clusters:
        for f in c["files"]:
            manifest_files[f["file_id"]] = f

    if len(manifest_files) != EXPECTED_TOTAL_FILES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_FILES} files in manifest, got {len(manifest_files)}")

    with get_session() as session:
        db_files = session.query(AudioFileRecord).order_by(AudioFileRecord.id).all()
        logger.info("Total audio_files in database: %d", len(db_files))

        if len(db_files) != EXPECTED_TOTAL_FILES:
            raise RuntimeError(f"Database audio_files count {len(db_files)} != {EXPECTED_TOTAL_FILES}")

        updated_count = 0
        non_empty_count = 0
        empty_count = 0
        preserved_labels_count = 0

        # Spot-check collection for logging
        spot_check_ids = [716, 251, 636, 297, 133, 83, 87, 288, 305, 740]
        spot_checks = []

        for rec in db_files:
            m_file = manifest_files.get(rec.id)
            if not m_file:
                raise RuntimeError(f"File #{rec.id} not found in manifest!")

            raw_manifest_tags = dict(m_file.get("tags") or {})

            # Check existing tags for any labels
            existing_labels = []
            if rec.tags:
                try:
                    existing_parsed = json.loads(rec.tags)
                    if isinstance(existing_parsed, list):
                        existing_labels = [item for item in existing_parsed if item]
                    elif isinstance(existing_parsed, dict) and "_labels" in existing_parsed:
                        existing_labels = list(existing_parsed["_labels"])
                except Exception:
                    pass

            if existing_labels:
                raw_manifest_tags["_labels"] = existing_labels
                preserved_labels_count += 1

            new_tags_json = json.dumps(raw_manifest_tags, ensure_ascii=False)
            rec.tags = new_tags_json
            updated_count += 1

            if raw_manifest_tags and any(k != "_labels" for k in raw_manifest_tags):
                non_empty_count += 1
            else:
                empty_count += 1

            if rec.id in spot_check_ids:
                spot_checks.append((rec.id, rec.file_name, new_tags_json))

        logger.info("\n--- Spot-Check Preview (10 files) ---")
        for fid, fname, t_json in spot_checks:
            preview = t_json[:120] + ("..." if len(t_json) > 120 else "")
            logger.info("  File #%d: '%s' -> %s", fid, fname[:35], preview)

        logger.info("\nMigration Statistics:")
        logger.info("  Total files updated: %d", updated_count)
        logger.info("  Files with format metadata: %d", non_empty_count)
        logger.info("  Files with empty format tags: %d", empty_count)
        logger.info("  Files with preserved labels: %d", preserved_labels_count)

        if dry_run:
            logger.info("Dry-run requested; rolling back SQLite transaction")
            session.rollback()
            return

        session.commit()
        logger.info("Successfully committed tags backfill to audio_files table")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    manifest_arg = args[0] if args else (
        "/home/de3f4ault/.gemini/antigravity/brain/9b832915-1b8e-43a9-9bae-8fa2b1f5ab7a/scratch/work_consolidation_audit_report.json"
    )
    is_dry_run = "--dry-run" in sys.argv
    run_migration(manifest_arg, dry_run=is_dry_run)
