"""Consolidation script for The New Testament lecture series cluster.

Safely merges fragmented works #981-#991 into canonical work #981,
dynamically validates title and author from Track 01 spoken introduction,
accounts for Work #981's 24-file distribution (Disk 1 + Disk 8 coalescence),
prunes invalid intra-work semantic shortcuts, updates LanceDB vector metadata,
and verifies database integrity with strict pre/post assertions.
"""

from __future__ import annotations

import argparse
import re
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

    print(f"=== The New Testament Consolidation (dry_run={dry_run}) ===")

    # 1. Dynamically Validate Author & Title from Track 01 Spoken Introduction
    c.execute(
        "SELECT full_text FROM transcriptions WHERE audio_file_id = 1208"
    )
    row = c.fetchone()
    if not row or not row[0]:
        print("Error: Could not retrieve Track 01 transcript for Audio File #1208.")
        sys.exit(1)

    txt = row[0]
    m_title = re.search(r"titled,?\s*(The\s+New\s+Testament)", txt, re.IGNORECASE)
    m_lecturer = re.search(r"The\s+lecturer\s+is\s+Professor\s+(Bart\s+Ehrman)", txt, re.IGNORECASE)

    if not m_title or not m_lecturer:
        print("Error: Could not dynamically extract title or lecturer from Track 01 spoken introduction.")
        print(f"Found title match: {m_title}, lecturer match: {m_lecturer}")
        sys.exit(1)

    canonical_title = "The New Testament"
    canonical_author = "Bart D. Ehrman"

    print("Dynamic verification via Track 01 spoken introduction:")
    print(f"   Spoken Title:     {m_title.group(1)} => Canonical: {canonical_title}")
    print(f"   Spoken Lecturer:  {m_lecturer.group(1)} => Canonical: {canonical_author}")

    # 2. Identify Target Works & Verify Per-Work File Distribution
    c.execute(
        "SELECT id, title, author FROM works WHERE id BETWEEN 981 AND 991 ORDER BY id"
    )
    target_works = c.fetchall()
    print(f"\nFound {len(target_works)} works to consolidate (#981 to #991).")
    if len(target_works) != 11:
        print(f"Assertion failed: Expected 11 works, found {len(target_works)}.")
        sys.exit(1)

    c.execute(
        "SELECT id, file_name, work_id FROM audio_files WHERE work_id BETWEEN 981 AND 991 ORDER BY id"
    )
    target_files = c.fetchall()
    print(f"Found {len(target_files)} audio files currently mapped to these works.")
    if len(target_files) != 144:
        print(f"Assertion failed: Expected 144 audio files, found {len(target_files)}.")
        sys.exit(1)

    # Validate specific distribution: Work #981 holds 24 files (Disk 1 + Disk 8), others hold 12 each
    c.execute(
        "SELECT work_id, count(*) FROM audio_files WHERE work_id BETWEEN 981 AND 991 GROUP BY work_id ORDER BY work_id"
    )
    dist = dict(c.fetchall())
    print(f"Work #981 file count: {dist.get(981)} (expected 24 due to Disk 1 + Disk 8 coalescence).")
    if dist.get(981) != 24:
        print(f"Assertion failed: Expected Work #981 to hold 24 files, found {dist.get(981)}.")
        sys.exit(1)

    for wid in range(982, 992):
        cnt = dist.get(wid, 0)
        if cnt != 12:
            print(f"Assertion failed: Expected Work #{wid} to hold 12 files, found {cnt}.")
            sys.exit(1)

    print("Per-work audio file distribution verified (24 files on #981, 12 files on each of #982-#991).")

    # 3. Identify Target Expressions
    c.execute(
        "SELECT id, work_id FROM expressions WHERE work_id BETWEEN 981 AND 991 ORDER BY id"
    )
    target_exprs = c.fetchall()
    expr_ids = [r[0] for r in target_exprs]
    print(f"\nFound {len(expr_ids)} expressions currently mapped to these works.")
    if len(expr_ids) != 1068:
        print(f"Assertion failed: Expected 1068 expressions, found {len(expr_ids)}.")
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
    if len(internal_semantic_edges) != 744:
        print(f"Assertion failed: Expected 744 internal semantic edges, found {len(internal_semantic_edges)}.")
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
    if internal_source_count != 780:
        print(f"Assertion failed: Expected 780 internal source edges, found {internal_source_count}.")
        sys.exit(1)

    # Count external edges dynamically (tolerant of concurrent daemon ingestion)
    c.execute(f"""
        SELECT count(*)
        FROM expression_relations
        WHERE (from_expression_id IN ({placeholders}) AND to_expression_id NOT IN ({placeholders}))
           OR (from_expression_id NOT IN ({placeholders}) AND to_expression_id IN ({placeholders}))
    """)
    external_count = c.fetchone()[0]
    print(f"External semantic edges to preserve: {external_count}")
    if external_count == 0:
        print("Warning: Expected non-zero external edges connecting this cluster to the library.")
        sys.exit(1)

    # 5. Check LanceDB
    import lancedb

    ldb_dir = p.parent / "lancedb"
    db_lancedb = lancedb.connect(str(ldb_dir))
    ldb_table = db_lancedb.open_table("expressions")
    ldb_matching = len(
        ldb_table.search()
        .where("work_id >= 981 AND work_id <= 991", prefilter=True)
        .limit(5000)
        .to_pandas()
    )
    print(
        f"LanceDB vector table has {ldb_matching} rows with work_id between 981 and 991."
    )
    if ldb_matching != 770:
        print(f"Assertion failed: Expected 770 LanceDB rows, found {ldb_matching}.")
        sys.exit(1)

    if dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        print(f"Work #981 will be updated to:")
        print(f"   Title:  {canonical_title}")
        print(f"   Author: {canonical_author}")
        print(f"Audio files to reassign to #981:       {len(target_files)}")
        print(f"Expressions to reassign to #981:       {len(expr_ids)}")
        print(f"Intra-work semantic edges to delete:   {len(internal_semantic_edges)}")
        print(f"Internal source edges preserved:       {internal_source_count}")
        print(f"External semantic edges preserved:     {external_count}")
        print(f"Redundant works to delete (#982-#991): {len(target_works) - 1}")
        print(f"LanceDB rows to update (work_id=981, author='{canonical_author}'): {ldb_matching}")
        print("\nDry run complete. No database or LanceDB modifications were made.")
        return

    # --- EXECUTION PATH ---
    backup_path = p.with_name(f"{p.name}.bak_pre_new_testament_consolidation")
    print(f"\nCreating pre-flight backup at: {backup_path}")
    shutil.copy2(p, backup_path)
    print("Backup verified.")

    try:
        conn.execute("BEGIN TRANSACTION;")

        # A. Update Work #981
        conn.execute(
            """
            UPDATE works
            SET title = ?,
                author = ?
            WHERE id = 981
        """,
            (canonical_title, canonical_author),
        )

        # B. Reassign Audio Files
        conn.execute(
            """
            UPDATE audio_files
            SET work_id = 981
            WHERE id BETWEEN 1208 AND 1351
        """
        )

        # C. Reassign Expressions
        conn.execute(
            f"""
            UPDATE expressions
            SET work_id = 981
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

        # E. Delete Redundant Works (#982 to #991)
        conn.execute("""
            DELETE FROM works
            WHERE id BETWEEN 982 AND 991
        """)

        # F. Verify Invariants before Commit
        c.execute("SELECT count(*) FROM works WHERE id BETWEEN 981 AND 991")
        remaining_works = c.fetchone()[0]
        assert remaining_works == 1, f"Expected 1 work remaining, found {remaining_works}"

        c.execute("SELECT count(*) FROM audio_files WHERE work_id = 981")
        reassigned_files = c.fetchone()[0]
        assert reassigned_files == 144, f"Expected 144 audio files on #981, found {reassigned_files}"

        c.execute("SELECT count(*) FROM expressions WHERE work_id = 981")
        reassigned_exprs = c.fetchone()[0]
        assert reassigned_exprs == 1068, f"Expected 1068 expressions on #981, found {reassigned_exprs}"

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
        assert remaining_source_edges == 780, f"Expected 780 internal source edges, found {remaining_source_edges}"

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
        where="work_id >= 981 AND work_id <= 991",
        values={"work_id": 981, "author": canonical_author},
    )

    # Verify LanceDB
    ldb_post = len(
        ldb_table.search()
        .where(f"work_id = 981 AND author = '{canonical_author}'", prefilter=True)
        .limit(5000)
        .to_pandas()
    )
    ldb_stragglers = len(
        ldb_table.search()
        .where("work_id >= 982 AND work_id <= 991", prefilter=True)
        .limit(5000)
        .to_pandas()
    )

    print(f"LanceDB rows with work_id=981 and author='{canonical_author}': {ldb_post}")
    print(f"LanceDB stragglers with work_id between 982 and 991: {ldb_stragglers}")

    assert ldb_post == 770, f"Expected 770 updated rows in LanceDB, found {ldb_post}"
    assert ldb_stragglers == 0, f"Expected 0 stragglers in LanceDB, found {ldb_stragglers}"

    print("\n=== Consolidation complete and independently verified ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate The New Testament lecture series cluster"
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
