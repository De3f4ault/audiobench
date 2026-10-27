"""Benchmark utility for ChapterEmbedder lock-hold vs embedding timing.

Measures the wall-clock duration of:
  1. Discovery query (read-only SQLite transaction)
  2. Per-chapter SQLite expression upsert lock hold (write transaction)
  3. LanceDB vector embedding and merge-insert (zero SQLite lock)
  4. Per-chapter SQLite mark clean lock hold (write transaction)

Demonstrates the reduction of continuous SQLite write-lock hold times
from tens of seconds down to low-millisecond pulses (~1-4ms).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.chapter_embedder import build_context_prefix
from audiobench.memory.singletons import get_memory_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ChapterEmbedder lock-hold vs neural embedding timing."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of unembedded chapters to benchmark (default: 5).",
    )
    return parser.parse_args()


def run_benchmark(limit: int = 5) -> None:
    print("=" * 80)
    print("CHAPTER EMBEDDER LOCK-HOLD VS EMBEDDING LATENCY BENCHMARK")
    print("=" * 80)

    store = get_memory_store()
    if store is None:
        print("ERROR: MemoryStore singleton unavailable.")
        sys.exit(1)

    t_fetch_start = time.perf_counter()
    with get_session() as session:
        unembedded = session.execute(
            sql_text("""
                SELECT c.id, c.audio_file_id, c.title, c.summary,
                       c.start_time, c.end_time,
                       af.work_id, w.title as work_title, w.author
                FROM chapters c
                JOIN audio_files af ON c.audio_file_id = af.id
                LEFT JOIN works w ON af.work_id = w.id
                WHERE c.is_ghost = 0
                  AND c.last_embedded_at IS NULL
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    fetch_elapsed_ms = (time.perf_counter() - t_fetch_start) * 1000.0

    print(f"Discovery Query: {len(unembedded)} unembedded chapters found in {fetch_elapsed_ms:.2f}ms")
    if not unembedded:
        print("No unembedded chapters available to benchmark.")
        return

    print("\n" + "-" * 80)
    print(f"{'Chap ID':<10} | {'SQLite Expr Lock':<18} | {'LanceDB Embed':<16} | {'SQLite Clean Lock':<18} | {'Total Time':<12}")
    print("-" * 80)

    sql_expr_times: list[float] = []
    lancedb_times: list[float] = []
    sql_clean_times: list[float] = []
    total_times: list[float] = []

    for cid, afid, title, summary, start_time, end_time, work_id, work_title, author in unembedded:
        t_chap_start = time.perf_counter()

        # 1. Text gathering
        text_content = summary
        if not text_content:
            with get_session() as session:
                seg_rows = session.execute(
                    sql_text("""
                        SELECT text FROM segments
                        WHERE chapter_id = :cid
                        ORDER BY segment_index ASC
                    """),
                    {"cid": cid},
                ).fetchall()
                combined = " ".join(r[0].strip() for r in seg_rows if r[0])
                if not combined:
                    time_seg_rows = session.execute(
                        sql_text("""
                            SELECT s.text FROM segments s
                            JOIN transcriptions t ON s.transcription_id = t.id
                            WHERE t.audio_file_id = :afid
                              AND s.start_time >= :start_time
                              AND s.start_time < :end_time
                            ORDER BY s.segment_index ASC
                        """),
                        {
                            "afid": afid,
                            "start_time": start_time or 0.0,
                            "end_time": end_time if (end_time and end_time > (start_time or 0.0)) else 1e9,
                        },
                    ).fetchall()
                    combined = " ".join(r[0].strip() for r in time_seg_rows if r[0])
                if combined:
                    text_content = combined[:3000]


        if not text_content or len(text_content.strip()) < 20:
            with get_session() as session:
                session.execute(
                    sql_text("UPDATE chapters SET last_embedded_at = :now WHERE id = :cid"),
                    {"now": datetime.now(UTC), "cid": cid},
                )
                session.commit()
            continue

        prefix = build_context_prefix(
            author=author,
            work_title=work_title,
            chapter_title=title,
        )
        prefixed_content = f"{prefix}{text_content.strip()}"
        clean_content = text_content.strip()
        content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()

        # 2. SQLite expression write lock timing
        t_sql_expr_0 = time.perf_counter()
        with get_session() as session:
            existing_expr_id = session.execute(
                sql_text("""
                    SELECT id FROM expressions
                    WHERE source_type = 'chapter_summary' AND source_id = :cid
                    LIMIT 1
                """),
                {"cid": cid},
            ).scalar()

            if existing_expr_id is None:
                session.execute(
                    sql_text("""
                        INSERT INTO expressions (
                            content, content_hash, source_type, source_id, work_id, chapter_id, graph_role, privacy_tier, created_at
                        ) VALUES (
                            :content, :content_hash, 'chapter_summary', :cid, :work_id, :cid, 'sweep_passage', 0, :now
                        )
                    """),
                    {
                        "content": clean_content,
                        "content_hash": content_hash,
                        "cid": cid,
                        "work_id": work_id,
                        "now": datetime.now(UTC),
                    },
                )
                expr_id = session.execute(sql_text("SELECT last_insert_rowid()")).scalar()
            else:
                expr_id = existing_expr_id
                session.execute(
                    sql_text("""
                        UPDATE expressions
                        SET content = :content, content_hash = :content_hash, work_id = :work_id, chapter_id = :cid
                        WHERE id = :eid
                    """),
                    {
                        "content": clean_content,
                        "content_hash": content_hash,
                        "work_id": work_id,
                        "cid": cid,
                        "eid": expr_id,
                    },
                )
            session.commit()
        t_sql_expr_1 = time.perf_counter()
        sql_expr_ms = (t_sql_expr_1 - t_sql_expr_0) * 1000.0

        # 3. LanceDB embedding (outside SQLite session)
        t_embed_0 = time.perf_counter()
        store.write_node(
            expression_id=expr_id,
            content=prefixed_content,
            source_type="chapter_summary",
            chapter_id=cid,
            chapter_title=title,
            author=author,
            work_id=work_id,
            audio_file_id=afid,
        )
        t_embed_1 = time.perf_counter()
        lancedb_ms = (t_embed_1 - t_embed_0) * 1000.0

        # 4. SQLite mark clean lock timing
        t_clean_0 = time.perf_counter()
        with get_session() as session:
            session.execute(
                sql_text("UPDATE chapters SET last_embedded_at = :now WHERE id = :cid"),
                {"now": datetime.now(UTC), "cid": cid},
            )
            session.commit()
        t_clean_1 = time.perf_counter()
        sql_clean_ms = (t_clean_1 - t_clean_0) * 1000.0

        total_chap_ms = (time.perf_counter() - t_chap_start) * 1000.0

        sql_expr_times.append(sql_expr_ms)
        lancedb_times.append(lancedb_ms)
        sql_clean_times.append(sql_clean_ms)
        total_times.append(total_chap_ms)

        print(
            f"{cid:<10} | "
            f"{sql_expr_ms:>15.2f} ms | "
            f"{lancedb_ms:>13.2f} ms | "
            f"{sql_clean_ms:>15.2f} ms | "
            f"{total_chap_ms:>9.2f} ms"
        )

    print("-" * 80)
    if sql_expr_times:
        mean_sql_lock = (sum(sql_expr_times) + sum(sql_clean_times)) / len(sql_expr_times)
        mean_lancedb = sum(lancedb_times) / len(lancedb_times)
        mean_total = sum(total_times) / len(total_times)
        lock_pct = (mean_sql_lock / mean_total) * 100.0 if mean_total > 0 else 0.0

        print(f"Summary over {len(sql_expr_times)} processed chapters:")
        print(f"  Mean Combined SQLite Lock-Hold Time: {mean_sql_lock:.2f} ms / chapter")
        print(f"  Mean LanceDB Embedding Duration:     {mean_lancedb:.2f} ms / chapter")
        print(f"  Mean Total Processing Duration:      {mean_total:.2f} ms / chapter")
        print(f"  SQLite Lock-Hold Fraction:           {lock_pct:.2f}% of task duration")
        print(f"  Free SQLite Concurrency Window:      {100.0 - lock_pct:.2f}% of task duration")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(limit=args.limit)
