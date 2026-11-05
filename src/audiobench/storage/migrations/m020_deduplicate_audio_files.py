"""Migration script m020: Deduplicate redundant audio files, transcriptions, and expressions.

Eliminates verified duplicate redownloads from the library:
1. Pair 1: AF 310 ('Biblical Teachings on Wisdom Faith and Righteousness_1.mp3') -> Duplicate of AF 299.
2. Pair 2: AF 345 ('Overcoming Ego Through Accountability and Love_1.mp3') -> Duplicate of AF 315.
Note: Pair 3 (AF 296 vs AF 313) is retained because content verification proved they are
two distinct recordings with different psalm compositions.

Execution steps:
- Validates pre-migration SQLite and LanceDB backups.
- Cascading bottom-up deletion in SQLite with PRAGMA foreign_keys = ON:
  1. expression_segment_map (125 rows)
  2. expression_relations (56 rows)
  3. expressions (57 rows: 2 documents, 10 passages, 45 chunks)
  4. segments (87 rows)
  5. chapters (2 rows)
  6. transcriptions (2 rows: Tx 728, Tx 774)
  7. audio_files (2 rows: AF 310, AF 345)
  8. works (2 rows: Work 180, Work 215)
- LanceDB vector deletion of the 45 leaf chunk expressions + table optimization.
- Filesystem deletion of the redundant _1.mp3 files.
- Complete invariant verification across SQLite and LanceDB.
"""

from __future__ import annotations

from pathlib import Path

import lancedb
from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("migrations.m020_deduplicate_audio_files")

TARGET_AF_IDS = (310, 345)
TARGET_TX_IDS = (728, 774)
TARGET_WORK_IDS = (180, 215)
SURVIVING_AF_IDS = (299, 315)


def run_migration() -> None:
    print("=" * 70)
    print("MIGRATION m020: AUDIO FILE DEDUPLICATION & VECTOR CLEANUP")
    print("=" * 70)

    settings = get_settings()

    # 1. Verify backups exist
    db_backup = Path("data/transcriptions.db.pre_m020_dedup_20261002")
    lance_backup = Path("data/lancedb.pre_m020_dedup_20261002")

    if not db_backup.exists() or db_backup.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"Required SQLite backup {db_backup} missing or invalid. Aborting.")
    if not lance_backup.exists():
        raise RuntimeError(f"Required LanceDB backup {lance_backup} missing. Aborting.")

    print(f"Verified SQLite backup:  {db_backup} ({db_backup.stat().st_size} bytes)")
    print(f"Verified LanceDB backup: {lance_backup}")

    # 2. SQLite Cascading Deletion
    with get_session() as session:
        # Enforce foreign key constraints
        session.execute(text("PRAGMA foreign_keys = ON"))

        # Pre-check surviving records exist and are healthy
        surviving = session.execute(
            text(f"SELECT id, file_name, work_id FROM audio_files WHERE id IN ({','.join(map(str, SURVIVING_AF_IDS))})")
        ).fetchall()
        if len(surviving) != len(SURVIVING_AF_IDS):
            raise RuntimeError(f"Surviving audio files missing! Expected {SURVIVING_AF_IDS}, found {surviving}")
        print(f"Verified surviving canonical audio files: {surviving}")

        # Check if targets already deleted or still present
        in_afs = ",".join(map(str, TARGET_AF_IDS))
        target_afs_present = session.execute(
            text(f"SELECT id, file_name, file_path FROM audio_files WHERE id IN ({in_afs})")
        ).fetchall()

        if target_afs_present:
            print(f"Found {len(target_afs_present)} target audio files in SQLite. Executing deletion...")
            in_txs = ",".join(map(str, TARGET_TX_IDS))
            in_works = ",".join(map(str, TARGET_WORK_IDS))

            expr_rows = session.execute(
                text(f"SELECT id, graph_role FROM expressions WHERE source_id IN ({in_txs})")
            ).fetchall()
            target_eids = [r[0] for r in expr_rows]
            chunk_eids = [r[0] for r in expr_rows if r[1] == "sweep_chunk"]

            print(f"Target expressions identified: {len(target_eids)} total ({len(chunk_eids)} sweep_chunk leaves)")
            in_eids = ",".join(map(str, target_eids))

            # Bottom-up deletion sequence
            if target_eids:
                del_esm = session.execute(
                    text(f"DELETE FROM expression_segment_map WHERE expression_id IN ({in_eids})")
                ).rowcount
                print(f"  1. expression_segment_map deleted: {del_esm} rows")

                del_rel = session.execute(
                    text(f"DELETE FROM expression_relations WHERE from_expression_id IN ({in_eids}) OR to_expression_id IN ({in_eids})")
                ).rowcount
                print(f"  2. expression_relations deleted:   {del_rel} rows")

                del_expr = session.execute(
                    text(f"DELETE FROM expressions WHERE id IN ({in_eids})")
                ).rowcount
                print(f"  3. expressions deleted:            {del_expr} rows")

            del_seg = session.execute(
                text(f"DELETE FROM segments WHERE transcription_id IN ({in_txs})")
            ).rowcount
            print(f"  4. segments deleted:               {del_seg} rows")

            del_chap = session.execute(
                text(f"DELETE FROM chapters WHERE audio_file_id IN ({in_afs})")
            ).rowcount
            print(f"  5. chapters deleted:               {del_chap} rows")

            del_tx = session.execute(
                text(f"DELETE FROM transcriptions WHERE id IN ({in_txs})")
            ).rowcount
            print(f"  6. transcriptions deleted:         {del_tx} rows")

            del_af = session.execute(
                text(f"DELETE FROM audio_files WHERE id IN ({in_afs})")
            ).rowcount
            print(f"  7. audio_files deleted:            {del_af} rows")

            del_work = session.execute(
                text(f"DELETE FROM works WHERE id IN ({in_works})")
            ).rowcount
            print(f"  8. works deleted:                  {del_work} rows")

            session.commit()
            print("SQLite transaction committed successfully.")
        else:
            print("Target audio files already deleted from SQLite. Skipping SQL deletion.")

        # Flush WAL to main database file
        session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))

    # 3. LanceDB Vector Cleanup
    print("\nExecuting LanceDB vector deletion...")
    db = lancedb.connect(settings.lancedb_path)
    tbl = db.open_table("expressions")
    lance_initial = tbl.count_rows()
    print(f"LanceDB current row count: {lance_initial}")

    # Query LanceDB directly for remaining work_id 180 and 215 expressions
    res_180 = tbl.search().where("work_id = 180").limit(100).to_arrow()
    res_215 = tbl.search().where("work_id = 215").limit(100).to_arrow()
    lance_chunk_eids = res_180["expression_id"].to_pylist() + res_215["expression_id"].to_pylist()

    if lance_chunk_eids:
        print(f"Found {len(lance_chunk_eids)} target leaf chunk expressions in LanceDB. Deleting...")
        lance_filter = f"expression_id IN ({','.join(map(str, lance_chunk_eids))})"
        tbl.delete(lance_filter)
        lance_pre_opt = tbl.count_rows()
        print(f"LanceDB row count after delete: {lance_pre_opt} (Expected: {lance_initial - len(lance_chunk_eids)})")
    else:
        print("No target expressions found in LanceDB. Already deleted.")

    # Verify filtered searches return 0 matches
    search_180 = tbl.search().where("work_id = 180").limit(10).to_arrow()
    search_215 = tbl.search().where("work_id = 215").limit(10).to_arrow()
    if len(search_180) > 0 or len(search_215) > 0:
        raise RuntimeError("LanceDB search returned deleted work IDs post-deletion!")
    print("Verified predicate search for Work 180 and Work 215 returns 0 rows.")

    print("Running LanceDB optimize()...")
    tbl.optimize()
    lance_final = tbl.count_rows()
    print(f"LanceDB final row count after optimize: {lance_final} (Expected: 21639)")

    # 4. On-Disk File Cleanup
    print("\nCleaning up redundant audio files on disk...")
    files_to_remove = [
        Path("data/library/Biblical Teachings on Wisdom Faith and Righteousness_1.mp3"),
        Path("data/library/Overcoming Ego Through Accountability and Love_1.mp3"),
    ]
    for p in files_to_remove:
        if p.exists():
            p.unlink()
            print(f"  Deleted redundant file from disk: {p}")
        else:
            print(f"  File not found on disk (already removed?): {p}")

    # 5. Final Invariant Verification
    print("\n" + "=" * 70)
    print("FINAL INVARIANT VERIFICATION")
    print("=" * 70)
    with get_session() as session:
        post_af = session.execute(text("SELECT COUNT(*) FROM audio_files")).scalar()
        post_works = session.execute(text("SELECT COUNT(*) FROM works")).scalar()
        post_tx = session.execute(text("SELECT COUNT(*) FROM transcriptions")).scalar()
        post_expr = session.execute(text("SELECT COUNT(*) FROM expressions")).scalar()
        post_chunks = session.execute(text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk'")).scalar()
        post_esm = session.execute(text("SELECT COUNT(*) FROM expression_segment_map")).scalar()
        unlinked_af = session.execute(text("SELECT COUNT(*) FROM audio_files WHERE work_id IS NULL")).scalar()
        unlinked_chunks = session.execute(text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NULL")).scalar()

        print(f"SQLite audio_files:            {post_af} (Expected: 819)")
        print(f"SQLite works:                  {post_works} (Expected: 233)")
        print(f"SQLite transcriptions:         {post_tx} (Expected: 808)")
        print(f"SQLite total expressions:      {post_expr} (Expected: 27123)")
        print(f"SQLite sweep_chunk leaves:     {post_chunks} (Expected: 20103)")
        print(f"SQLite expression_segment_map: {post_esm} (Expected: 74448)")
        print(f"Unlinked audio files:          {unlinked_af} (Expected: 0)")
        print(f"Unlinked sweep_chunks:         {unlinked_chunks} (Expected: 0)")
        print(f"LanceDB expressions:           {lance_final} (Expected: 21639)")

        assert post_af == 819, f"audio_files count mismatch: {post_af} != 819"
        assert post_works == 233, f"works count mismatch: {post_works} != 233"
        assert post_tx == 808, f"transcriptions count mismatch: {post_tx} != 808"
        assert post_expr == 27123, f"expressions count mismatch: {post_expr} != 27123"
        assert post_chunks == 20103, f"sweep_chunks count mismatch: {post_chunks} != 20103"
        assert post_esm == 74448, f"expression_segment_map mismatch: {post_esm} != 74448"
        assert unlinked_af == 0, f"unlinked audio files found: {unlinked_af}"
        assert unlinked_chunks == 0, f"unlinked sweep_chunks found: {unlinked_chunks}"
        assert lance_final == 21639, f"LanceDB count mismatch: {lance_final} != 21639"

    print("\nMigration m020 completed successfully with 0 errors.")


if __name__ == "__main__":
    run_migration()
