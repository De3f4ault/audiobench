"""Consolidation script for Think and Grow Rich audiobook cluster.

Safely merges fragmented works #969-#980 into canonical work #969,
dynamically derives the canonical title and author via resolve_work_identity()
using the general Author/Narrator slash-splitting logic, prunes invalid
intra-work semantic shortcuts, updates LanceDB vector metadata, and verifies
database integrity with strict pre/post assertions.
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

    print(f"=== Think and Grow Rich Consolidation (dry_run={dry_run}) ===")

    # 1. Dynamically Derive Author & Title from Audio File #1195 via resolve_work_identity
    c.execute(
        "SELECT file_path, file_name, tags FROM audio_files WHERE id = 1195"
    )
    row = c.fetchone()
    if not row:
        print("Error: Could not retrieve Audio File #1195.")
        sys.exit(1)

    file_path, file_name, raw_tags = row
    from audiobench.daemon.intelligence.work_identity import (
        extract_tags_from_file,
        resolve_work_identity,
    )

    tags = extract_tags_from_file(file_path) if file_path else {}
    canonical_title, canonical_author, resolution_source = resolve_work_identity(
        file_path=file_path,
        file_name=file_name,
        tags=tags,
    )

    print(f"Dynamic resolution via resolve_work_identity:")
    print(f"   Title:      {canonical_title}")
    print(f"   Author:     {canonical_author}")
    print(f"   Source:     {resolution_source}")

    if canonical_title != "Think and Grow Rich":
        print(f"Assertion failed: Derived title is {canonical_title!r}, expected 'Think and Grow Rich'.")
        sys.exit(1)

    if canonical_author != "Napoleon Hill":
        print(f"Assertion failed: Derived author is {canonical_author!r}, expected 'Napoleon Hill'.")
        sys.exit(1)

    # 2. Identify Target Works & Audio Files
    c.execute(
        "SELECT id, title, author FROM works WHERE id BETWEEN 969 AND 980 ORDER BY id"
    )
    target_works = c.fetchall()
    print(f"Found {len(target_works)} works to consolidate (#969 to #980).")
    if len(target_works) != 12:
        print(f"Assertion failed: Expected 12 works, found {len(target_works)}.")
        sys.exit(1)

    c.execute(
        "SELECT id, file_name, work_id FROM audio_files WHERE work_id BETWEEN 969 AND 980 ORDER BY id"
    )
    target_files = c.fetchall()
    print(f"Found {len(target_files)} audio files currently mapped to these works.")
    if len(target_files) != 12:
        print(f"Assertion failed: Expected 12 audio files, found {len(target_files)}.")
        sys.exit(1)

    # 3. Identify Target Expressions
    c.execute(
        "SELECT id, work_id FROM expressions WHERE work_id BETWEEN 969 AND 980 ORDER BY id"
    )
    target_exprs = c.fetchall()
    expr_ids = [r[0] for r in target_exprs]
    print(f"Found {len(expr_ids)} expressions currently mapped to these works.")
    if len(expr_ids) != 548:
        print(f"Assertion failed: Expected 548 expressions, found {len(expr_ids)}.")
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
    if len(internal_semantic_edges) != 165:
        print(f"Assertion failed: Expected 165 internal semantic edges, found {len(internal_semantic_edges)}.")
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
    if internal_source_count != 524:
        print(f"Assertion failed: Expected 524 internal source edges, found {internal_source_count}.")
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
    if external_count < 584:
        print(f"Assertion failed: Expected at least 584 external edges, found {external_count}.")
        sys.exit(1)

    # 5. Check LanceDB
    import lancedb

    ldb_dir = p.parent / "lancedb"
    db_lancedb = lancedb.connect(str(ldb_dir))
    ldb_table = db_lancedb.open_table("expressions")
    ldb_matching = len(
        ldb_table.search()
        .where("work_id >= 969 AND work_id <= 980", prefilter=True)
        .limit(1000)
        .to_pandas()
    )
    print(
        f"LanceDB vector table has {ldb_matching} rows with work_id between 969 and 980."
    )
    if ldb_matching != 444:
        print(f"Assertion failed: Expected 444 LanceDB rows, found {ldb_matching}.")
        sys.exit(1)

    if dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        print(f"Work #969 will be updated to:")
        print(f"   Title:  {canonical_title}")
        print(f"   Author: {canonical_author}")
        print(f"Audio files to reassign to #969:       {len(target_files)}")
        print(f"Expressions to reassign to #969:       {len(expr_ids)}")
        print(f"Intra-work semantic edges to delete:   {len(internal_semantic_edges)}")
        print(f"Internal source edges preserved:       {internal_source_count}")
        print(f"External semantic edges preserved:     {external_count}")
        print(f"Redundant works to delete (#970-#980): {len(target_works) - 1}")
        print(f"LanceDB rows to update (work_id=969, author='{canonical_author}'): {ldb_matching}")
        print("\nDry run complete. No database or LanceDB modifications were made.")
        return

    # --- EXECUTION PATH ---
    backup_path = p.with_name(f"{p.name}.bak_pre_think_and_grow_rich_consolidation")
    print(f"\nCreating pre-flight backup at: {backup_path}")
    shutil.copy2(p, backup_path)
    print("Backup verified.")

    try:
        conn.execute("BEGIN TRANSACTION;")

        # A. Update Work #969
        conn.execute(
            """
            UPDATE works
            SET title = ?,
                author = ?
            WHERE id = 969
        """,
            (canonical_title, canonical_author),
        )

        # B. Reassign Audio Files
        conn.execute(
            """
            UPDATE audio_files
            SET work_id = 969
            WHERE id BETWEEN 1195 AND 1206
        """
        )

        # C. Reassign Expressions
        conn.execute(
            f"""
            UPDATE expressions
            SET work_id = 969
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

        # E. Delete Redundant Works (#970 to #980)
        conn.execute("""
            DELETE FROM works
            WHERE id BETWEEN 970 AND 980
        """)

        # F. Verify Invariants before Commit
        c.execute("SELECT count(*) FROM works WHERE id BETWEEN 969 AND 980")
        remaining_works = c.fetchone()[0]
        assert remaining_works == 1, f"Expected 1 work remaining, found {remaining_works}"

        c.execute("SELECT count(*) FROM audio_files WHERE work_id = 969")
        reassigned_files = c.fetchone()[0]
        assert reassigned_files == 12, f"Expected 12 audio files on #969, found {reassigned_files}"

        c.execute("SELECT count(*) FROM expressions WHERE work_id = 969")
        reassigned_exprs = c.fetchone()[0]
        assert reassigned_exprs == 548, f"Expected 548 expressions on #969, found {reassigned_exprs}"

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
        assert remaining_source_edges == 524, f"Expected 524 internal source edges, found {remaining_source_edges}"

        c.execute(f"""
            SELECT count(*)
            FROM expression_relations
            WHERE (from_expression_id IN ({placeholders}) AND to_expression_id NOT IN ({placeholders}))
               OR (from_expression_id NOT IN ({placeholders}) AND to_expression_id IN ({placeholders}))
        """)
        remaining_external_edges = c.fetchone()[0]
        assert remaining_external_edges == external_count, f"Expected {external_count} external edges preserved, found {remaining_external_edges}"

        conn.commit()
        print("SQLite transaction successfully committed.")

    except Exception as e:
        conn.rollback()
        print(f"Error during execution, rolled back: {e}")
        sys.exit(1)

    # G. LanceDB Migration
    print("\nUpdating LanceDB expressions table...")
    ldb_table.update(
        where="work_id >= 969 AND work_id <= 980",
        values={"work_id": 969, "author": canonical_author},
    )

    # Verify LanceDB
    ldb_post = len(
        ldb_table.search()
        .where(f"work_id = 969 AND author = '{canonical_author}'", prefilter=True)
        .limit(1000)
        .to_pandas()
    )
    ldb_stragglers = len(
        ldb_table.search()
        .where("work_id >= 970 AND work_id <= 980", prefilter=True)
        .limit(1000)
        .to_pandas()
    )

    print(f"LanceDB rows with work_id=969 and author='{canonical_author}': {ldb_post}")
    print(f"LanceDB stragglers with work_id between 970 and 980: {ldb_stragglers}")

    assert ldb_post == 444, f"Expected 444 updated rows in LanceDB, found {ldb_post}"
    assert ldb_stragglers == 0, f"Expected 0 stragglers in LanceDB, found {ldb_stragglers}"

    print("\n=== Consolidation complete and independently verified ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate Think and Grow Rich audiobook cluster"
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
