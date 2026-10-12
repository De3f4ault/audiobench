#!/usr/bin/env python3
"""
Backfill historical plain-text daemon_proposal expressions into the canonical
DaemonProposalEnvelope JSON schema (schema_version = 2).

Converts all 13 legacy plain-text rows in data/transcriptions.db into structured
knowledge_gap proposals with name_source="legacy_unnamed", preserving their
existing inference_status (e.g. #33216 remains 'confirmed', others 'proposed').
"""

import json
import re
import sqlite3
import sys
from pathlib import Path


def backfill_proposals(db_path: Path) -> int:
    if not db_path.exists():
        print(f"Database not found at {db_path}", file=sys.stderr)
        return 1

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        SELECT id, content, inference_status, created_at
        FROM expressions
        WHERE source_type = 'daemon_proposal'
        ORDER BY id ASC
        """
    )
    rows = cur.fetchall()

    backfilled_count = 0
    for eid, content, status, created_at in rows:
        content_str = (content or "").strip()
        if content_str.startswith("{"):
            continue

        dist_match = re.search(r"distance=([\d\.]+)", content_str)
        thresh_match = re.search(r"threshold=([\d\.]+)", content_str)

        dist = float(dist_match.group(1)) if dist_match else 0.25
        thresh = float(thresh_match.group(1)) if thresh_match else 0.20

        envelope = {
            "proposal_type": "knowledge_gap",
            "schema_version": 2,
            "title": "Legacy Library Knowledge Gap",
            "summary": f"Search queries diverge from audio corpus (distance {dist:.3f} > {thresh:.2f})",
            "evidence": f"Centroid distance {dist:.3f} vs threshold {thresh:.2f} over 14 days",
            "payload": {
                "distance": dist,
                "threshold": thresh,
                "window_days": 14,
                "diagnostic": "Your search queries over the last 14 days cluster around topics not well covered by your current audio library.",
                "recommended_ingestion": "Consider ingesting more content aligned with your recent research focus.",
                "stranded_queries": [],
                "name_source": "legacy_unnamed",
                "legacy_backfilled": True,
            },
        }

        json_content = json.dumps(envelope)

        cur.execute(
            """
            UPDATE expressions
            SET content = :content
            WHERE id = :eid
            """,
            {"content": json_content, "eid": eid},
        )
        backfilled_count += 1
        print(f"Backfilled expression #{eid} (status={status}, distance={dist:.3f})")

    con.commit()
    con.close()
    print(f"Successfully backfilled {backfilled_count} legacy proposal rows.")
    return 0


if __name__ == "__main__":
    db = Path("data/transcriptions.db")
    sys.exit(backfill_proposals(db))
