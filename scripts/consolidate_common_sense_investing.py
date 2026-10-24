"""Consolidation script for The Little Book of Common Sense Investing.

Safely merges fragmented works #906-#968 into canonical work #906,
re-derives the author dynamically from Track 01 narration using the fixed
entity extractor, prunes invalid intra-work semantic shortcuts, updates LanceDB
vector metadata, and verifies database integrity with pre/post assertions.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


def run_consolidation(db_path: str, dry_run: bool = True) -> None:
    p = Path(db_path)
    if not p.exists():
        print(f"Error: Database {db_path} does not exist.")
        sys.exit(1)

    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA foreign_keys = ON;")
    c = conn.cursor()

    print(f"=== Common Sense Investing Consolidation (dry_run={dry_run}) ===")

    # 1. Dynamically Derive Author & Title from Track 01 via extract_from_transcript
    c.execute(
        "SELECT full_text FROM transcriptions WHERE audio_file_id = 1131"
    )
    row = c.fetchone()
    if not row or not row[0]:
        print("Error: Could not retrieve Track 01 transcript for Audio File #1131.")
        sys.exit(1)

    from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript

    track01_entity = extract_from_transcript(row[0])
    print(f"Dynamic Track 01 extraction:")
    print(f"   Author:     {track01_entity.author}")
    print(f"   Title:      {track01_entity.title}")
    print(f"   Confidence: {track01_entity.confidence}")

    if track01_entity.author != "John C. Bogle" or track01_entity.confidence != "high":
        print(f"Assertion failed: Derived author is {track01_entity.author!r} (conf: {track01_entity.confidence!r}), expected 'John C. Bogle' (high).")
        sys.exit(1)

    canonical_title = "The Little Book of Common Sense Investing"
    canonical_author = track01_entity.author

    # 2. Identify Target Works & Audio Files
    c.execute(
        "SELECT id, title, author FROM works WHERE id BETWEEN 906 AND 968 ORDER BY id"
    )
    target_works = c.fetchall()
    print(f"Found {len(target_works)} works to consolidate (#906 to #968).")
    if len(target_works) != 63:
        print(f"Assertion failed: Expected 63 works, found {len(target_works)}.")
        sys.exit(1)

    c.execute(
        "SELECT id, file_name, work_id FROM audio_files WHERE work_id BETWEEN 906 AND 968 ORDER BY id"
    )
    target_files = c.fetchall()
    print(f"Found {len(target_files)} audio files currently mapped to these works.")
    if len(target_files) != 63:
        print(f"Assertion failed: Expected 63 audio files, found {len(target_files)}.")
        sys.exit(1)

    # 3. Identify Target Expressions
    c.execute(
        "SELECT id, work_id FROM expressions WHERE work_id BETWEEN 906 AND 968 ORDER BY id"
    )
    target_exprs = c.fetchall()
    expr_ids = [r[0] for r in target_exprs]
    print(f"Found {len(expr_ids)} expressions currently mapped to these works.")
    if len(expr_ids) != 472:
        print(f"Assertion failed: Expected 472 expressions, found {len(expr_ids)}.")
        sys.exit(1)

    placeholders = ",".join(map(str, expr_ids))

    # 4. Dynamically Calculate Intra-Work Semantic Relations
    c.execute(f"""
        SELECT id, from_expression_id, to_expression_id, created_by, weight
        FROM expression_relations
        WHERE from_expression_id IN ({placeholders})
          AND to_expression_id IN ({placeholders})
          AND relation_type = 'semantic'
    """)
    internal_semantic_edges = c.fetchall()
    print(
        f"Dynamically identified {len(internal_semantic_edges)} intra-cluster semantic edges to prune."
    )
    if len(internal_semantic_edges) != 218:
        print(f"Assertion failed: Expected 218 internal semantic edges, found {len(internal_semantic_edges)}.")
        sys.exit(1)

    # Count internal source edges
    c.execute(f"""
        SELECT count(*)
        FROM expression_relations
        WHERE from_expression_id IN ({placeholders})
          AND to_expression_id IN ({placeholders})
          AND relation_type != 'semantic'
    """)
    internal_source_count = c.fetchone()[0]
    print(f"Internal hierarchical (source) edges to preserve: {internal_source_count}")
    if internal_source_count != 346:
        print(f"Assertion failed: Expected 346 internal source edges, found {internal_source_count}.")
        sys.exit(1)

    # Count external edges
    c.execute(f"""
        SELECT count(*)
        FROM expression_relations
        WHERE (from_expression_id IN ({placeholders}) AND to_expression_id NOT IN ({placeholders}))
           OR (from_expression_id NOT IN ({placeholders}) AND to_expression_id IN ({placeholders}))
    """)
    external_count = c.fetchone()[0]
    print(f"External semantic edges to preserve: {external_count}")
    if external_count != 432:
        print(f"Assertion failed: Expected 432 external edges, found {external_count}.")
        sys.exit(1)

    # 5. Check LanceDB
    import lancedb

    ldb_dir = p.parent / "lancedb"
    db_lancedb = lancedb.connect(str(ldb_dir))
    ldb_table = db_lancedb.open_table("expressions")
    ldb_matching = len(
        ldb_table.search()
        .where("work_id >= 906 AND work_id <= 968", prefilter=True)
        .limit(1000)
        .to_pandas()
    )
    print(
        f"LanceDB vector table has {ldb_matching} rows with work_id between 906 and 968."
    )
    if ldb_matching != 342:
        print(f"Assertion failed: Expected 342 LanceDB rows, found {ldb_matching}.")
        sys.exit(1)

    if dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        print(f"Work #906 will be renamed to:")
        print(f"   Title:  {canonical_title}")
        print(f"   Author: {canonical_author}")
        print(f"Audio files to reassign to #906:       {len(target_files)}")
        print(f"Expressions to reassign to #906:       {len(expr_ids)}")
        print(f"Intra-work semantic edges to delete:   {len(internal_semantic_edges)}")
        print(f"Internal source edges preserved:       {internal_source_count}")
        print(f"External semantic edges preserved:     {external_count}")
        print(f"Redundant works to delete (#907-#968): {len(target_works) - 1}")
        print(f"LanceDB rows to update (work_id=906, author='{canonical_author}'): {ldb_matching}")
        print("\nDry run complete. No database or LanceDB modifications were made.")
        return

    # --- EXECUTION PATH ---
    backup_path = p.with_name(f"{p.name}.bak_pre_common_sense_investing_consolidation")
    print(f"\nCreating pre-flight backup at: {backup_path}")
    shutil.copy2(p, backup_path)
    print("Backup verified.")

    try:
        conn.execute("BEGIN TRANSACTION;")

        # A. Rename Work #906
        conn.execute(
            """
            UPDATE works
            SET title = ?,
                author = ?
            WHERE id = 906
        """,
            (canonical_title, canonical_author),
        )

        # B. Reassign Audio Files
        conn.execute(
            """
            UPDATE audio_files
            SET work_id = 906
            WHERE id BETWEEN 1131 AND 1193
        """
        )

        # C. Reassign Expressions
        conn.execute(
            f"""
            UPDATE expressions
            SET work_id = 906
            WHERE id IN ({placeholders})
        """
        )

        # D. Prune Dynamic Intra-Cluster Semantic Edges
        if internal_semantic_edges:
            edge_ids = ",".join(str(r[0]) for r in internal_semantic_edges)
            conn.execute(f"""
                DELETE FROM expression_relations
                WHERE id IN ({edge_ids})
            """)

        # E. Delete Redundant Works (#907 to #968)
        conn.execute("""
            DELETE FROM works
            WHERE id BETWEEN 907 AND 968
        """)

        # F. Verify Invariants before Commit
        c.execute("SELECT count(*) FROM works WHERE id BETWEEN 906 AND 968")
        remaining_works = c.fetchone()[0]
        assert remaining_works == 1, f"Expected 1 work remaining, found {remaining_works}"

        c.execute("SELECT count(*) FROM audio_files WHERE work_id = 906")
        reassigned_files = c.fetchone()[0]
        assert reassigned_files == 63, f"Expected 63 audio files on #906, found {reassigned_files}"

        c.execute("SELECT count(*) FROM expressions WHERE work_id = 906")
        reassigned_exprs = c.fetchone()[0]
        assert reassigned_exprs == 472, f"Expected 472 expressions on #906, found {reassigned_exprs}"

        c.execute(f"""
            SELECT count(*)
            FROM expression_relations
            WHERE from_expression_id IN ({placeholders})
              AND to_expression_id IN ({placeholders})
              AND relation_type = 'semantic'
        """)
        remaining_internal_edges = c.fetchone()[0]
        assert remaining_internal_edges == 0, f"Expected 0 internal semantic edges, found {remaining_internal_edges}"

        c.execute(f"""
            SELECT count(*)
            FROM expression_relations
            WHERE from_expression_id IN ({placeholders})
              AND to_expression_id IN ({placeholders})
              AND relation_type != 'semantic'
        """)
        remaining_source_edges = c.fetchone()[0]
        assert remaining_source_edges == 346, f"Expected 346 internal source edges, found {remaining_source_edges}"

        conn.commit()
        print("SQLite transaction successfully committed.")

    except Exception as e:
        conn.rollback()
        print(f"Error during execution, rolled back: {e}")
        sys.exit(1)

    # G. LanceDB Migration
    print("\nUpdating LanceDB expressions table...")
    ldb_table.update(
        where="work_id >= 906 AND work_id <= 968",
        values={"work_id": 906, "author": canonical_author},
    )

    # Verify LanceDB
    ldb_post = len(
        ldb_table.search()
        .where(f"work_id = 906 AND author = '{canonical_author}'", prefilter=True)
        .limit(1000)
        .to_pandas()
    )
    ldb_stragglers = len(
        ldb_table.search()
        .where("work_id >= 907 AND work_id <= 968", prefilter=True)
        .limit(1000)
        .to_pandas()
    )

    print(f"LanceDB rows with work_id=906 and author='{canonical_author}': {ldb_post}")
    print(f"LanceDB stragglers with work_id between 907 and 968: {ldb_stragglers}")

    assert ldb_post == 342, f"Expected 342 updated rows in LanceDB, found {ldb_post}"
    assert ldb_stragglers == 0, f"Expected 0 stragglers in LanceDB, found {ldb_stragglers}"

    print("\n=== Consolidation complete and independently verified ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate Common Sense Investing audiobook cluster"
    )
    parser.add_argument(
        "--db-path",
        default="data/transcriptions.db",
        help="Path to SQLite transcriptions database",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the consolidation (defaults to dry-run)",
    )
    args = parser.parse_args()

    run_consolidation(args.db_path, dry_run=not args.execute)


if __name__ == "__main__":
    main()
