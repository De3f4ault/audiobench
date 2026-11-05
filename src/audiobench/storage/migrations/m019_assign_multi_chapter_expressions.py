"""Migration script m019: Assign multi-chapter expressions chapter_id and sync LanceDB.

Milestone 2 of expression grounding:
1. Backfills expressions.chapter_id for 4,751 multi-chapter sweep_chunk expressions
   using temporal interval join on (start_time + end_time) / 2.0 via expression_segment_map.
2. Excludes the 5 degenerate transcriptions (598, 613, 614, 615, 616).
3. Syncs chapter_id and chapter_title to LanceDB expressions table using read-modify-write.
"""

from __future__ import annotations

import time
from pathlib import Path

import lancedb
import pandas as pd
from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("migrations.m019_assign_multi_chapter_expressions")

DEGENERATE_TX_IDS = (598, 613, 614, 615, 616)


def run_sqlite_chapter_assignment() -> dict[str, int]:
    """Assign chapter_id in SQLite expressions table for multi-chapter sweep chunks."""
    print("=" * 70)
    print("MILESTONE 2 - STEP 1: SQLITE MULTI-CHAPTER EXPRESSIONS CHAPTER_ID BACKFILL")
    print("=" * 70)

    backup_path = Path("data/transcriptions.db.pre_m019_chapters_20261002")
    if not backup_path.exists():
        raise RuntimeError(f"Required backup {backup_path} does not exist. Aborting.")

    print(f"Verified pre-migration backup at {backup_path} ({backup_path.stat().st_size} bytes)")

    with get_session() as session:
        # Pre-check initial null chapter_id sweep chunks
        total_sweep_chunks = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk'")
        ).scalar()
        initial_null = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NULL")
        ).scalar()
        initial_linked = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NOT NULL")
        ).scalar()

        print(f"Total sweep_chunks:        {total_sweep_chunks}")
        print(f"Initially linked:          {initial_linked} ({initial_linked/total_sweep_chunks*100:.2f}%)")
        print(f"Initially unlinked (NULL): {initial_null} ({initial_null/total_sweep_chunks*100:.2f}%)")

        if initial_null != 4751:
            print(f"WARNING: Expected exactly 4751 unlinked chunks, found {initial_null}")

        # Compute midpoint join
        print("\nComputing chunk temporal bounds and matching chapters...")
        t0 = time.perf_counter()

        query = text(f"""
            WITH chunk_bounds AS (
                SELECT 
                    e.id AS expr_id,
                    t.audio_file_id,
                    MIN(s.start_time) AS c_start,
                    MAX(s.end_time) AS c_end,
                    (MIN(s.start_time) + MAX(s.end_time)) / 2.0 AS c_midpoint
                FROM expressions e
                JOIN transcriptions t ON e.source_id = t.id
                JOIN expression_segment_map esm ON e.id = esm.expression_id
                JOIN segments s ON esm.segment_id = s.id
                WHERE e.graph_role = 'sweep_chunk'
                  AND e.chapter_id IS NULL
                  AND e.source_id NOT IN {DEGENERATE_TX_IDS}
                GROUP BY e.id, t.audio_file_id
            )
            SELECT 
                cb.expr_id,
                ch.id AS matched_chapter_id
            FROM chunk_bounds cb
            JOIN chapters ch 
                ON cb.audio_file_id = ch.audio_file_id 
               AND cb.c_midpoint >= ch.start_time 
               AND (
                   cb.c_midpoint < ch.end_time
                   OR ch.chapter_index = (
                       SELECT MAX(ch2.chapter_index)
                       FROM chapters ch2
                       WHERE ch2.audio_file_id = cb.audio_file_id AND ch2.is_ghost = 0
                   )
               )
        """)

        matched_rows = session.execute(query).fetchall()
        t1 = time.perf_counter()
        print(f"Computed {len(matched_rows)} matches in {t1 - t0:.2f}s.")

        if len(matched_rows) != initial_null:
            print(f"Discrepancy: {len(matched_rows)} matches vs {initial_null} target rows.")

        # Batch update SQLite expressions
        print("\nExecuting batch update in SQLite...")
        t2 = time.perf_counter()
        update_stmt = text("UPDATE expressions SET chapter_id = :matched_chapter_id WHERE id = :expr_id")
        updates = [{"expr_id": r[0], "matched_chapter_id": r[1]} for r in matched_rows]
        session.execute(update_stmt, updates)
        session.commit()
        t3 = time.perf_counter()
        print(f"Updated {len(updates)} expressions in {t3 - t2:.2f}s.")

        # Post-check invariants
        final_linked = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NOT NULL")
        ).scalar()
        final_null = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NULL")
        ).scalar()

        print(f"\nFinal linked sweep_chunks:   {final_linked} ({final_linked/total_sweep_chunks*100:.2f}%)")
        print(f"Final unlinked sweep_chunks: {final_null} ({final_null/total_sweep_chunks*100:.2f}%)")

        assert final_null == 0, f"Expected 0 unlinked sweep chunks, found {final_null}"
        assert final_linked == total_sweep_chunks, f"Expected {total_sweep_chunks} linked sweep chunks, found {final_linked}"

        return {
            "updated": len(updates),
            "total_sweep_chunks": total_sweep_chunks,
            "final_linked": final_linked,
            "final_null": final_null,
        }


def run_lancedb_chapter_sync() -> None:
    """Sync updated chapter_id and chapter_title from SQLite to LanceDB expressions table."""
    print("\n" + "=" * 70)
    print("MILESTONE 2 - STEP 2: LANCEDB EXPRESSIONS CHAPTER_ID & CHAPTER_TITLE SYNC")
    print("=" * 70)

    settings = get_settings()
    lancedb_dir = Path(settings.data_dir) / "lancedb"
    db = lancedb.connect(str(lancedb_dir))
    tbl = db.open_table("expressions")

    n_initial = len(tbl)
    print(f"Initial LanceDB expressions rows: {n_initial}")

    # Fetch SQLite enrichment data
    print("\nFetching updated chapter metadata from SQLite...")
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    e.id AS expression_id,
                    e.chapter_id AS chapter_id,
                    c.title AS chapter_title
                FROM expressions e
                LEFT JOIN chapters c ON e.chapter_id = c.id
                WHERE e.chapter_id IS NOT NULL
            """)
        ).fetchall()

    update_df = pd.DataFrame(
        rows, columns=["expression_id", "chapter_id_new", "chapter_title_new"]
    )
    print(f"Fetched {len(update_df)} non-null chapter rows from SQLite.")

    # Read existing LanceDB table into DataFrame
    print("\nReading LanceDB table into memory...")
    t0 = time.perf_counter()
    existing_df = tbl.to_lance().to_table().to_pandas()
    t1 = time.perf_counter()
    print(f"Read {len(existing_df)} rows in {t1 - t0:.2f}s.")

    initial_lance_null_chaps = existing_df["chapter_id"].isna().sum()
    print(f"Initial null chapter_id in LanceDB: {initial_lance_null_chaps}")

    # Merge updated chapter data
    print("\nMerging chapter updates into LanceDB DataFrame...")
    merged_df = existing_df.merge(
        update_df,
        on="expression_id",
        how="left",
    )

    # Use new values where available, otherwise fall back to existing
    merged_df["chapter_id"] = merged_df["chapter_id_new"].combine_first(merged_df["chapter_id"])
    merged_df["chapter_title"] = merged_df["chapter_title_new"].combine_first(merged_df["chapter_title"])

    merged_df.drop(columns=["chapter_id_new", "chapter_title_new"], inplace=True)

    # Ensure dtypes match Arrow / LanceDB expectations
    merged_df["expression_id"] = merged_df["expression_id"].astype("int64")
    merged_df["chapter_id"] = merged_df["chapter_id"].astype(pd.Int64Dtype())
    merged_df["chapter_title"] = merged_df["chapter_title"].where(
        merged_df["chapter_title"].notna(), other=None
    )

    expected_cols = [
        "expression_id",
        "vector",
        "content",
        "embedding_model_version",
        "embedded_at",
        "source_type",
        "speaker",
        "audio_file_id",
        "confidence",
        "original_language",
        "work_id",
        "chapter_id",
        "chapter_title",
        "author",
    ]
    merged_df = merged_df[expected_cols]

    # Overwrite LanceDB table
    print(f"\nWriting {len(merged_df)} rows to LanceDB (mode='overwrite')...")
    t0 = time.perf_counter()
    db.create_table("expressions", data=merged_df, mode="overwrite")
    t1 = time.perf_counter()
    print(f"Overwrite complete in {t1 - t0:.2f}s.")

    # Cleanup old versions and optimize
    tbl_new = db.open_table("expressions")
    print("\nCleaning up old table versions and optimizing...")
    try:
        tbl_new.cleanup_old_versions()
        print("cleanup_old_versions: OK")
    except Exception as e:
        print(f"cleanup_old_versions notice: {e}")

    try:
        tbl_new.optimize()
        print("optimize: OK")
    except Exception as e:
        print(f"optimize notice: {e}")

    # Post-migration verification
    print("\n" + "=" * 70)
    print("POST-MIGRATION LANCEDB VERIFICATION")
    print("=" * 70)
    res_df = tbl_new.to_pandas()
    n_final = len(res_df)
    n_chapter = res_df["chapter_id"].notna().sum()
    n_null_chapter = res_df["chapter_id"].isna().sum()

    print(f"Final LanceDB rows:         {n_final} (was {n_initial})")
    print(f"chapter_id IS NOT NULL:     {n_chapter} ({n_chapter/n_final*100:.2f}%)")
    print(f"chapter_id IS NULL:         {n_null_chapter} (was {initial_lance_null_chaps})")

    assert n_final == n_initial, f"Row count mismatch! {n_final} != {n_initial}"
    # Verify that no sweep_chunk has null chapter_id in LanceDB
    with get_session() as session:
        sweep_expr_ids = set(
            session.execute(
                text("SELECT id FROM expressions WHERE graph_role = 'sweep_chunk'")
            ).scalars().all()
        )

    null_sweep_in_lance = res_df[res_df["expression_id"].isin(sweep_expr_ids)]["chapter_id"].isna().sum()
    print(f"Null sweep_chunks in LanceDB: {null_sweep_in_lance}")
    assert null_sweep_in_lance == 0, f"Expected 0 null sweep chunks in LanceDB, found {null_sweep_in_lance}"
    print("LanceDB sync verified successfully.")


def run_migration() -> None:
    t_start = time.perf_counter()
    res_sqlite = run_sqlite_chapter_assignment()
    run_lancedb_chapter_sync()
    t_end = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"MIGRATION M019 COMPLETE in {t_end - t_start:.2f}s")
    print(f"Total expressions updated: {res_sqlite['updated']}")
    print(f"Final unlinked sweep chunks in SQLite: {res_sqlite['final_null']}")
    print("=" * 70)


if __name__ == "__main__":
    run_migration()
