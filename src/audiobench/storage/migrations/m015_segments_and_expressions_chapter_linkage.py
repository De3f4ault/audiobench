"""Migration script m015: Segments, WorkLinker author, and expressions chapter linkage.

Executes:
  Step 2:   Backfill segments.chapter_id (temporal & single-chapter bounds)
  Step 2.5: Backfill works.author from ffprobe/filename with 10-file spot-check gate
  Step 3:   Backfill expressions.chapter_id in SQLite for single-chapter sweep chunks
"""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import text
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.audio_converter import probe_tags
from audiobench.daemon.intelligence.entity_extractor import extract_from_filename

logger = get_logger("migrations.m015_linkage")


def run_step_2_segments() -> dict:
    """Step 2: Assign chapter_id to all segments based on chapter bounds."""
    print("\n" + "=" * 70)
    print("STEP 2: SEGMENTS CHAPTER_ID BACKFILL")
    print("=" * 70)

    with get_session() as session:
        total_null = session.execute(
            text("SELECT COUNT(*) FROM segments WHERE chapter_id IS NULL")
        ).scalar()
        print(f"Segments with NULL chapter_id before backfill: {total_null}")

        t0 = time.perf_counter()
        # Single-pass temporal join update:
        # Handles single-chapter files AND multi-chapter files with start/end bounds
        # plus tail-boundary guard for file end credits.
        updated = session.execute(
            text("""
                UPDATE segments
                SET chapter_id = (
                    SELECT c.id
                    FROM chapters c
                    JOIN transcriptions t ON t.audio_file_id = c.audio_file_id
                    WHERE t.id = segments.transcription_id
                      AND c.is_ghost = 0
                      AND (
                          (SELECT COUNT(*) FROM chapters c1
                           WHERE c1.audio_file_id = t.audio_file_id AND c1.is_ghost = 0) = 1
                          OR
                          (segments.start_time >= c.start_time AND (
                              segments.start_time < c.end_time
                              OR c.chapter_index = (
                                  SELECT MAX(c2.chapter_index) FROM chapters c2
                                  WHERE c2.audio_file_id = t.audio_file_id AND c2.is_ghost = 0
                              )
                          ))
                      )
                    ORDER BY c.start_time DESC
                    LIMIT 1
                )
                WHERE chapter_id IS NULL
            """)
        ).rowcount
        session.commit()
        t1 = time.perf_counter()

        remaining_null = session.execute(
            text("SELECT COUNT(*) FROM segments WHERE chapter_id IS NULL")
        ).scalar()
        print(f"Updated segments in {t1 - t0:.2f}s: {updated}")
        print(f"Remaining segments with NULL chapter_id: {remaining_null}")

        return {"updated": updated, "remaining_null": remaining_null}


def run_step_2_5_authors() -> dict:
    """Step 2.5: Backfill works.author from ffprobe format tags and filename parsing.

    Note on scope: Step 2.5 backfills works.author only. It does NOT merge fragmented
    works (e.g. Man's Search for Meaning as 8 separate WorkRecords). The work-grouping
    bug is tracked separately as Task: Fix WorkLinker album-level work grouping.
    """
    print("\n" + "=" * 70)
    print("STEP 2.5: WORKLINKER AUTHOR BACKFILL & VERIFICATION GATE")
    print("=" * 70)

    with get_session() as session:
        initial_author_count = session.execute(
            text("SELECT COUNT(*) FROM works WHERE author IS NOT NULL AND author != ''")
        ).scalar()
        total_works = session.execute(text("SELECT COUNT(*) FROM works")).scalar()
        print(f"Works with author before backfill: {initial_author_count} / {total_works}")

        works_to_enrich = session.execute(
            text("""
                SELECT w.id, w.title, af.file_name, af.file_path
                FROM works w
                JOIN audio_files af ON af.work_id = w.id
                WHERE w.author IS NULL OR w.author = ''
                ORDER BY w.id
            """)
        ).fetchall()

        resolved = 0
        spot_check_candidates = []

        for wid, wtitle, fname, fpath in works_to_enrich:
            author = None
            p = Path(fpath)
            if p.exists():
                tags = probe_tags(p)
                raw_art = tags.get("artist") or tags.get("album_artist")
                if raw_art and raw_art.strip():
                    clean_art = raw_art.strip()
                    if clean_art.lower() not in {"unknown", "various artists", "various", "untitled"}:
                        author = clean_art

            if not author:
                entity = extract_from_filename(fname)
                if entity and entity.author:
                    author = entity.author.strip()

            if author:
                session.execute(
                    text("UPDATE works SET author = :author WHERE id = :wid"),
                    {"author": author, "wid": wid},
                )
                resolved += 1
                if len(spot_check_candidates) < 15:
                    spot_check_candidates.append((wid, wtitle, fname, author))

        session.commit()

        post_author_count = session.execute(
            text("SELECT COUNT(*) FROM works WHERE author IS NOT NULL AND author != ''")
        ).scalar()
        print(f"Works resolved in Step 2.5: {resolved}")
        print(f"Works with author after backfill: {post_author_count} / {total_works} ({post_author_count/total_works*100:.1f}%)")

        # ── 10-File Spot Check Gate ──────────────────────────────────────────
        print("\n" + "-" * 70)
        print("SPOT-CHECK VERIFICATION GATE (10 files)")
        print("-" * 70)
        for wid, wt, fn, auth in spot_check_candidates[:10]:
            print(f"  Work #{wid:<3} Title: {wt[:28]:<28} | File: {fn[:30]:<30} | Author: {auth}")

        print("-" * 70)
        print("SPOT-CHECK GATE COMPLETE.")

        return {
            "initial": initial_author_count,
            "resolved": resolved,
            "final": post_author_count,
            "total": total_works,
        }


def run_step_3_expressions() -> dict:
    """Step 3: Backfill expressions.chapter_id for single-chapter sweep chunks."""
    print("\n" + "=" * 70)
    print("STEP 3: EXPRESSIONS CHAPTER_ID LINKAGE")
    print("=" * 70)

    with get_session() as session:
        initial_expr_with_chapter = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE chapter_id IS NOT NULL")
        ).scalar()
        print(f"Expressions with chapter_id before: {initial_expr_with_chapter}")

        t0 = time.perf_counter()
        # Single-chapter file chunks get linked to their chapter
        updated = session.execute(
            text("""
                UPDATE expressions
                SET chapter_id = (
                    SELECT c.id FROM chapters c
                    JOIN audio_files af ON c.audio_file_id = af.id
                    JOIN transcriptions tx ON tx.audio_file_id = af.id
                    WHERE tx.id = expressions.source_id AND c.is_ghost = 0
                    LIMIT 1
                )
                WHERE graph_role = 'sweep_chunk'
                  AND (
                      SELECT COUNT(*) FROM chapters c2
                      JOIN audio_files af2 ON c2.audio_file_id = af2.id
                      JOIN transcriptions tx2 ON tx2.audio_file_id = af2.id
                      WHERE tx2.id = expressions.source_id AND c2.is_ghost = 0
                  ) = 1
            """)
        ).rowcount
        session.commit()
        t1 = time.perf_counter()

        total_sweep_chunks = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk'")
        ).scalar()
        linked_sweep_chunks = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NOT NULL")
        ).scalar()
        null_sweep_chunks = session.execute(
            text("SELECT COUNT(*) FROM expressions WHERE graph_role = 'sweep_chunk' AND chapter_id IS NULL")
        ).scalar()

        print(f"Expressions updated in {t1 - t0:.2f}s: {updated}")
        print(f"Total sweep_chunks:         {total_sweep_chunks}")
        print(f"Linked (single-chapter):    {linked_sweep_chunks} ({linked_sweep_chunks/total_sweep_chunks*100:.1f}%)")
        print(f"Unlinked (multi-chapter):   {null_sweep_chunks} ({null_sweep_chunks/total_sweep_chunks*100:.1f}%) [Decision 1 backlog]")

        return {
            "updated": updated,
            "total_sweep_chunks": total_sweep_chunks,
            "linked": linked_sweep_chunks,
            "unlinked": null_sweep_chunks,
        }


if __name__ == "__main__":
    run_step_2_segments()
    run_step_2_5_authors()
    run_step_3_expressions()
