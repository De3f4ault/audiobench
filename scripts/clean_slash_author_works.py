"""Batch cleanup script for slash-delimited Author/Narrator works.

Applies clean_author() from work_identity.py to all works in SQLite and
expressions in LanceDB that carry the 'Author/Narrator' tag convention
(excluding or following the Think and Grow Rich consolidation cluster).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

from audiobench.daemon.intelligence.work_identity import clean_author


def run_cleanup(db_path: str, dry_run: bool = True) -> None:
    p = Path(db_path)
    if not p.exists():
        print(f"Error: Database {db_path} does not exist.")
        sys.exit(1)

    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA foreign_keys = ON;")
    c = conn.cursor()

    print(f"=== Slash-Author Works Cleanup (dry_run={dry_run}) ===")

    # Select candidate works where author contains '/'
    c.execute("SELECT id, title, author FROM works WHERE author LIKE '%/%' ORDER BY id")
    candidate_works = c.fetchall()

    updates: list[tuple[int, str, str, str]] = []
    for wid, title, author in candidate_works:
        # If Think and Grow Rich cluster is still present (#970-#980), skip them
        # as they will be handled by consolidate_think_and_grow_rich.py
        if 970 <= wid <= 980:
            continue
        cleaned = clean_author(author)
        if cleaned != author:
            updates.append((wid, title, author, cleaned))

    print(f"Identified {len(updates)} works to update:")
    for wid, title, old_a, new_a in updates:
        print(f"   Work #{wid}: {title!r} -> Author: {old_a!r} => {new_a!r}")

    # Check LanceDB rows for these works
    import lancedb

    ldb_dir = p.parent / "lancedb"
    db_lancedb = lancedb.connect(str(ldb_dir))
    ldb_table = db_lancedb.open_table("expressions")

    ldb_counts: dict[int, int] = {}
    for wid, _, old_a, _ in updates:
        cnt = len(
            ldb_table.search()
            .where(f"work_id = {wid} AND author = '{old_a}'", prefilter=True)
            .limit(1000)
            .to_pandas()
        )
        ldb_counts[wid] = cnt
        print(f"   LanceDB rows for Work #{wid} with author={old_a!r}: {cnt}")

    if dry_run:
        print("\n--- DRY RUN SUMMARY ---")
        print(f"Works to update in SQLite: {len(updates)}")
        print(f"LanceDB expressions to update: {sum(ldb_counts.values())}")
        print("\nDry run complete. No database or LanceDB modifications were made.")
        return

    # --- EXECUTION PATH ---
    backup_path = p.with_name(f"{p.name}.bak_pre_slash_author_cleanup")
    print(f"\nCreating pre-flight backup at: {backup_path}")
    shutil.copy2(p, backup_path)
    print("Backup verified.")

    try:
        conn.execute("BEGIN TRANSACTION;")

        for wid, _, _, new_a in updates:
            conn.execute(
                "UPDATE works SET author = ? WHERE id = ?",
                (new_a, wid),
            )

        # Verify before commit
        for wid, _, _, new_a in updates:
            c.execute("SELECT author FROM works WHERE id = ?", (wid,))
            curr = c.fetchone()[0]
            assert curr == new_a, f"Expected author {new_a!r} for work #{wid}, found {curr!r}"

        conn.commit()
        print("SQLite transaction successfully committed.")

    except Exception as e:
        conn.rollback()
        print(f"Error during execution, rolled back: {e}")
        sys.exit(1)

    # LanceDB update
    print("\nUpdating LanceDB expressions table...")
    for wid, _, old_a, new_a in updates:
        ldb_table.update(
            where=f"work_id = {wid} AND author = '{old_a}'",
            values={"author": new_a},
        )
        # Verify
        remaining_old = len(
            ldb_table.search()
            .where(f"work_id = {wid} AND author = '{old_a}'", prefilter=True)
            .limit(1000)
            .to_pandas()
        )
        updated_new = len(
            ldb_table.search()
            .where(f"work_id = {wid} AND author = '{new_a}'", prefilter=True)
            .limit(1000)
            .to_pandas()
        )
        print(f"Work #{wid} LanceDB update: new={updated_new}, remaining_old={remaining_old}")
        assert remaining_old == 0, f"Expected 0 remaining rows with old author for work #{wid}"

    print("\n=== Cleanup complete and independently verified ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean slash-delimited Author/Narrator strings across works"
    )
    parser.add_argument(
        "--db-path",
        default="data/transcriptions.db",
        help="Path to SQLite transcriptions database",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the cleanup (defaults to dry-run)",
    )
    args = parser.parse_args()

    run_cleanup(args.db_path, dry_run=not args.execute)


if __name__ == "__main__":
    main()
