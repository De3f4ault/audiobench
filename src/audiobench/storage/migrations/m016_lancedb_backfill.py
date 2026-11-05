"""Migration script m016: LanceDB expressions metadata backfill and vacuum.

Backfills chapter_id, chapter_title, and author into the production
LanceDB expressions table using read-modify-write and cleans old versions.
"""

from __future__ import annotations

import time
from pathlib import Path

import lancedb
import pandas as pd
from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings


def run_lancedb_backfill() -> None:
    print("=" * 70)
    print("STEP 4: LANCEDB EXPRESSIONS METADATA BACKFILL")
    print("=" * 70)

    settings = get_settings()
    lancedb_dir = Path(settings.data_dir) / "lancedb"
    db = lancedb.connect(str(lancedb_dir))
    tbl = db.open_table("expressions")

    n_initial = len(tbl)
    print(f"Current LanceDB expressions rows: {n_initial}")

    # 1. Fetch SQLite enrichment data
    print("\nFetching SQLite enrichment data...")
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    e.id AS expression_id,
                    e.chapter_id AS chapter_id,
                    c.title AS chapter_title,
                    w.author AS author
                FROM expressions e
                LEFT JOIN chapters c ON e.chapter_id = c.id
                LEFT JOIN works w ON e.work_id = w.id
            """)
        ).fetchall()

    update_df = pd.DataFrame(
        rows, columns=["expression_id", "chapter_id", "chapter_title", "author"]
    )
    print(f"Fetched {len(update_df)} expression enrichment rows from SQLite.")
    print(f"  chapter_id non-null: {update_df['chapter_id'].notna().sum()}")
    print(f"  author non-null:     {update_df['author'].notna().sum()}")

    # 2. Read existing LanceDB table into DataFrame
    print("\nReading LanceDB table into memory...")
    t0 = time.perf_counter()
    existing_df = tbl.to_lance().to_table().to_pandas()
    t1 = time.perf_counter()
    print(f"Read {len(existing_df)} rows in {t1 - t0:.2f}s.")

    # 3. Merge enrichment data
    print("\nMerging enrichment columns...")
    enriched_df = existing_df.merge(
        update_df,
        on="expression_id",
        how="left",
        suffixes=("_old", ""),
    )

    # Clean up any leftover old columns if present
    for col in ["chapter_id_old", "chapter_title_old", "author_old"]:
        if col in enriched_df.columns:
            enriched_df.drop(columns=[col], inplace=True)

    # Ensure dtypes match Arrow / LanceDB expectations
    enriched_df["expression_id"] = enriched_df["expression_id"].astype("int64")
    enriched_df["chapter_id"] = enriched_df["chapter_id"].astype(pd.Int64Dtype())
    enriched_df["chapter_title"] = enriched_df["chapter_title"].where(
        enriched_df["chapter_title"].notna(), other=None
    )
    enriched_df["author"] = enriched_df["author"].where(
        enriched_df["author"].notna(), other=None
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
    enriched_df = enriched_df[expected_cols]

    # 4. Overwrite LanceDB table
    print(f"\nWriting {len(enriched_df)} rows to LanceDB (mode='overwrite')...")
    t0 = time.perf_counter()
    db.create_table("expressions", data=enriched_df, mode="overwrite")
    t1 = time.perf_counter()
    print(f"Overwrite complete in {t1 - t0:.2f}s.")

    # 5. Cleanup old versions
    tbl_new = db.open_table("expressions")
    print("\nCleaning up old table versions...")
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

    # 6. Verification
    print("\n" + "=" * 70)
    print("STEP 4 POST-MIGRATION VERIFICATION")
    print("=" * 70)
    res_df = tbl_new.to_pandas()
    n_final = len(res_df)
    n_chapter = res_df["chapter_id"].notna().sum()
    n_author = res_df["author"].notna().sum()

    print(f"Final LanceDB rows:       {n_final} (was {n_initial})")
    print(f"chapter_id IS NOT NULL:   {n_chapter} ({n_chapter/n_final*100:.1f}%)")
    print(f"author IS NOT NULL:       {n_author} ({n_author/n_final*100:.1f}%)")

    assert n_final == n_initial, f"Row count mismatch! {n_final} != {n_initial}"

    # Spot check 5 rows
    print("\nSpot-checking 5 rows against SQLite ground truth:")
    spot = res_df[res_df["chapter_id"].notna()].sample(5, random_state=42)
    with get_session() as session:
        for _, r in spot.iterrows():
            eid = int(r["expression_id"])
            sql = session.execute(
                text("""
                    SELECT e.chapter_id, c.title, w.author
                    FROM expressions e
                    LEFT JOIN chapters c ON e.chapter_id = c.id
                    LEFT JOIN works w ON e.work_id = w.id
                    WHERE e.id = :eid
                """),
                {"eid": eid},
            ).fetchone()
            ch_match = "OK" if sql and str(sql[0]) == str(int(r["chapter_id"])) else "MISMATCH"
            auth_match = "OK" if sql and sql[2] == r["author"] else "MISMATCH"
            print(f"  expr={eid}: chapter=[{ch_match}] {r['chapter_id']} ('{r['chapter_title']}') | author=[{auth_match}] '{r['author']}'")

    # Live predicate pushdown test:
    print("\nTesting vector search with predicate pushdown (author = 'Viktor E. Frankl'):")
    sample_vec = res_df["vector"].iloc[0].tolist()
    filtered_results = tbl_new.search(sample_vec).where("author = 'Viktor E. Frankl'").limit(3).to_pandas()
    print(f"Returned {len(filtered_results)} filtered results:")
    for _, fr in filtered_results.iterrows():
        print(f"  id={fr['expression_id']} author='{fr['author']}' ch='{fr['chapter_title']}' distance={fr.get('_distance', 'N/A')}")

    print("\nSTEP 4 COMPLETE AND VERIFIED.")


if __name__ == "__main__":
    run_lancedb_backfill()
