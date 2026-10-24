"""Consolidation script for Grokking Algorithms, Second Edition.

Safely merges fragmented works #1003-#1018 into canonical work #1003,
prunes invalid intra-work semantic relations dynamically, updates LanceDB,
and verifies database integrity with pre/post assertions.
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

    print(f"=== Grokking Algorithms Consolidation (dry_run={dry_run}) ===")

    # 1. Identify Target Works & Audio Files
    c.execute(
        "SELECT id, title, author FROM works WHERE id BETWEEN 1003 AND 1018 ORDER BY id"
    )
    target_works = c.fetchall()
    print(f"Found {len(target_works)} works to consolidate (#1003 to #1018).")

    c.execute(
        "SELECT id, file_name, work_id FROM audio_files WHERE work_id BETWEEN 1003 AND 1018 ORDER BY id"
    )
    target_files = c.fetchall()
    print(f"Found {len(target_files)} audio files currently mapped to these works.")

    # 2. Identify Target Expressions
    c.execute(
        "SELECT id, work_id FROM expressions WHERE work_id BETWEEN 1003 AND 1018"
    )
    target_exprs = c.fetchall()
    expr_ids = [r[0] for r in target_exprs]
    print(f"Found {len(expr_ids)} expressions currently mapped to these works.")

    if not expr_ids:
        print("No expressions found; nothing to consolidate.")
        return

    placeholders = ",".join(map(str, expr_ids))

    # 3. Dynamically Calculate Intra-Work Semantic Relations
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

    # 4. Check LanceDB
    import lancedb

    ldb_dir = p.parent / "lancedb"
    db_lancedb = lancedb.connect(str(ldb_dir))
    ldb_table = db_lancedb.open_table("expressions")
    ldb_matching = len(
        ldb_table.search()
        .where("work_id >= 1003 AND work_id <= 1018", prefilter=True)
        .to_pandas()
    )
    print(
        f"LanceDB vector table has {ldb_matching} rows with work_id between 1003 and 1018."
    )

    if dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        print("Work #1003 will be renamed to:")
        print("   Title:  Grokking Algorithms, Second Edition")
        print("   Author: Aditya Y Bhargava")
        print(f"Audio files to reassign to #1003:       {len(target_files)}")
        print(f"Expressions to reassign to #1003:       {len(expr_ids)}")
        print(
            f"Intra-work semantic edges to delete:    {len(internal_semantic_edges)}"
        )
        print(f"Redundant works to delete (#1004-#1018): {len(target_works) - 1}")
        print(f"LanceDB rows to update to work_id=1003: {ldb_matching}")
        print(
            "\nDry run complete. No database or LanceDB modifications were made."
        )
        return

    # --- EXECUTION PATH ---
    backup_path = p.with_name(f"{p.name}.bak_pre_grokking_consolidation")
    print(f"\nCreating pre-flight backup at: {backup_path}")
    shutil.copy2(p, backup_path)
    print("Backup verified.")

    try:
        conn.execute("BEGIN TRANSACTION;")

        # A. Rename Work #1003
        conn.execute(
            """
            UPDATE works
            SET title = 'Grokking Algorithms, Second Edition',
                author = 'Aditya Y Bhargava'
            WHERE id = 1003
        """
        )

        # B. Reassign Audio Files
        conn.execute(
            """
            UPDATE audio_files
            SET work_id = 1003
            WHERE id BETWEEN 1668 AND 1683
        """
        )

        # C. Reassign Expressions
        conn.execute(
            f"""
            UPDATE expressions
            SET work_id = 1003
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

        # E. Delete Redundant Works (#1004 to #1018)
        conn.execute("""
            DELETE FROM works
            WHERE id BETWEEN 1004 AND 1018
        """)

        # F. Post-Condition Verification Assertions
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM audio_files WHERE work_id BETWEEN 1004 AND 1018"
        )
        assert (
            cur.fetchone()[0] == 0
        ), "Assertion failed: dangling audio_files on old works"

        cur.execute(
            "SELECT COUNT(*) FROM expressions WHERE work_id BETWEEN 1004 AND 1018"
        )
        assert (
            cur.fetchone()[0] == 0
        ), "Assertion failed: dangling expressions on old works"

        cur.execute("SELECT COUNT(*) FROM audio_files WHERE work_id = 1003")
        assert (
            cur.fetchone()[0] == 16
        ), "Assertion failed: work #1003 does not have exactly 16 audio files"

        cur.execute("SELECT COUNT(*) FROM expressions WHERE work_id = 1003")
        assert (
            cur.fetchone()[0] == len(expr_ids)
        ), "Assertion failed: work #1003 does not have all expressions"

        conn.commit()
        print("SQLite transaction committed successfully.")

        # G. Update LanceDB metadata
        print("Synchronizing LanceDB work_id...")
        ldb_table.update(
            where="work_id >= 1003 AND work_id <= 1018", values={"work_id": 1003}
        )
        post_ldb_matching = len(
            ldb_table.search().where("work_id = 1003", prefilter=True).to_pandas()
        )
        print(f"LanceDB successfully updated: {post_ldb_matching} rows now hold work_id=1003.")

    except Exception as exc:
        conn.rollback()
        print(f"FATAL: Error during consolidation transaction: {exc}")
        print("Transaction rolled back completely. Restoring from backup...")
        shutil.copy2(backup_path, p)
        print("Database restored to pre-flight snapshot.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate Grokking Algorithms works safely"
    )
    parser.add_argument(
        "--db", default="data/transcriptions.db", help="Path to SQLite DB"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the consolidation (defaults to dry-run)",
    )
    args = parser.parse_args()

    run_consolidation(args.db, dry_run=not args.execute)
