#!/usr/bin/env python3
"""
Promotes legacy knowledge gap proposals to schema_version 2 with LLM naming and populated stranded_queries.

Operates on the 3 historical proposal clusters in data/transcriptions.db:
- Cluster 1 (Late Aug 2026: #33196, #33199, #33202, #33204, #33205, #33208, #33216)
- Cluster 2 (Mid Sept 2026: #35636, #36632, #36836, #37152, #37441)
- Cluster 3 (Oct 2026: #86710)

Issues exactly 1 LLM naming call per cluster to avoid redundant token costs and ensure semantic consistency.
Preserves existing inference_status (#33216 confirmed, others proposed).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from audiobench.daemon.intelligence.gap_naming import get_queries_in_window, name_knowledge_gap


CLUSTERS = [
    {
        "name": "Cluster 1: Late August 2026",
        "row_ids": [33196, 33199, 33202, 33204, 33205, 33208, 33216],
        "anchor_ts": "2026-08-26 04:33:42",
    },
    {
        "name": "Cluster 2: Mid September 2026",
        "row_ids": [35636, 36632, 36836, 37152, 37441],
        "anchor_ts": "2026-09-19 10:36:14",
    },
    {
        "name": "Cluster 3: October 2026",
        "row_ids": [86710],
        "anchor_ts": "2026-10-10 17:09:50",
    },
]


def promote_proposals(db_path: Path, dry_run: bool = False) -> None:
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"Connecting to {db_path}")
    print(f"Mode: {'DRY RUN (no database writes)' if dry_run else 'LIVE EXECUTION'}")
    print("=" * 80)

    total_updated = 0

    for cluster in CLUSTERS:
        c_name = cluster["name"]
        row_ids = cluster["row_ids"]
        anchor_ts = cluster["anchor_ts"]

        print(f"\nProcessing {c_name} ({len(row_ids)} proposals: {row_ids})")

        # 1. Fetch queries within 14-day lookback
        queries = get_queries_in_window(conn, anchor_ts, window_days=14)
        print(f"  Discovered {len(queries)} correlated search queries in 14-day lookback window.")
        if not queries:
            print("  Warning: No queries found, falling back to default topic title.")
            medoid = "Audio Library Coverage Gap"
        else:
            medoid = sorted(queries, key=len, reverse=True)[0]
            print(f"  Representative medoid query: \"{medoid[:80]}...\"")

        # 2. Call LLM naming engine once for the cluster
        print("  Querying naming engine...")
        title, diagnostic, rec, name_source = name_knowledge_gap(queries, medoid)
        print(f"  Generated Title:    \"{title}\"")
        print(f"  Diagnostic:         \"{diagnostic}\"")
        print(f"  Recommended:        \"{rec}\"")
        print(f"  Naming Source:      {name_source}")

        # 3. Apply to all proposals in the cluster
        for row_id in row_ids:
            row = cur.execute(
                "SELECT content, inference_status FROM expressions WHERE id = ?",
                (row_id,),
            ).fetchone()
            if not row:
                print(f"  Warning: Row #{row_id} not found in DB, skipping.")
                continue

            orig_content = row[0]
            status = row[1]

            # Parse original distance if present
            dist = 0.25
            thresh = 0.20
            try:
                orig_data = json.loads(orig_content)
                payload = orig_data.get("payload", {})
                dist = payload.get("distance", payload.get("centroid_distance", 0.25))
                thresh = payload.get("threshold", 0.20)
            except Exception:
                pass

            envelope = {
                "proposal_type": "knowledge_gap",
                "schema_version": 2,
                "title": title,
                "summary": f"Query cluster diverges from audio corpus (distance {dist:.3f} > {thresh:.2f})",
                "evidence": f"Centroid distance {dist:.3f} vs threshold {thresh:.2f} over 14 days",
                "payload": {
                    "distance": dist,
                    "threshold": thresh,
                    "window_days": 14,
                    "diagnostic": diagnostic,
                    "recommended_ingestion": rec,
                    "stranded_queries": queries,
                    "name_source": name_source,
                    "medoid_query": medoid,
                    "cluster_name": c_name,
                },
            }

            envelope_json = json.dumps(envelope)

            if not dry_run:
                cur.execute(
                    "UPDATE expressions SET content = ? WHERE id = ?",
                    (envelope_json, row_id),
                )
            print(f"  -> Proposal #{row_id} [status={status}]: ready with {len(queries)} queries")
            total_updated += 1

    if not dry_run:
        conn.commit()
        print("\n" + "=" * 80)
        print(f"Database commit successful. Updated {total_updated} proposal rows.")
    else:
        print("\n" + "=" * 80)
        print(f"Dry run complete. {total_updated} proposal rows would be updated.")

    conn.close()


def verify_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT id, inference_status, content
        FROM expressions
        WHERE source_type = 'daemon_proposal'
        ORDER BY id ASC
        """
    ).fetchall()

    print("\n--- Database Verification ---")
    print(f"Total daemon proposals: {len(rows)}")
    assert len(rows) == 13, f"Expected 13 rows, found {len(rows)}"

    for r in rows:
        row_id, status, content = r[0], r[1], r[2]
        data = json.loads(content)
        assert data.get("schema_version") == 2, f"Row #{row_id} schema_version != 2"
        assert data.get("proposal_type") == "knowledge_gap", f"Row #{row_id} type != knowledge_gap"
        payload = data.get("payload", {})
        queries = payload.get("stranded_queries", [])
        assert len(queries) > 0, f"Row #{row_id} has empty stranded_queries"
        assert payload.get("name_source") in ("llm_gemini", "llm_ollama", "vector_medoid")

        if row_id in (33196, 33199, 33202, 33204, 33205, 33208, 33216):
            assert len(queries) == 48, f"Row #{row_id} expected 48 queries, got {len(queries)}"
        elif row_id in (35636, 36632, 36836, 37152, 37441):
            assert len(queries) == 19, f"Row #{row_id} expected 19 queries, got {len(queries)}"
        elif row_id == 86710:
            assert len(queries) == 14, f"Row #{row_id} expected 14 queries, got {len(queries)}"

        if row_id == 33216:
            assert status == "confirmed", f"Row #33216 lost confirmed status: {status}"
        else:
            assert status == "proposed", f"Row #{row_id} status modified unexpectedly: {status}"
        print(f"  Verified #{row_id}: title=\"{data.get('title')}\" [status={status}, source={payload.get('name_source')}, queries={len(queries)}]")

    conn.close()
    print("All 13 proposals successfully verified against schema invariants.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote legacy knowledge gap proposals to schema_version 2.")
    parser.add_argument("--db", type=Path, default=Path("data/transcriptions.db"), help="Path to transcriptions.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview promotion without modifying database")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing database rows")
    args = parser.parse_args()

    if args.verify_only:
        verify_database(args.db)
    else:
        promote_proposals(args.db, dry_run=args.dry_run)
        if not args.dry_run:
            verify_database(args.db)


if __name__ == "__main__":
    main()
