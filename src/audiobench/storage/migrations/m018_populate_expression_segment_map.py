"""Migration m018: Populate expression_segment_map junction table.

Grounds historical sweep_chunk expressions to physical transcript segments in SQLite
using the verified multi-tier monotonic subsequence alignment algorithm.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from sqlalchemy import text
from audiobench.core.db_session import get_session
from audiobench.memory.chunking import _clean_text


def norm_punct(t: str) -> str:
    """Normalize whitespace around punctuation and typographic variants."""
    t = t.replace("—", " ").replace("–", " ")
    t = re.sub(r"\s+([?.!,;:\'\"])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_all_punct(t: str) -> str:
    """Strip all non-alphanumeric characters for robust fallback matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", t.lower())


def align_cft_to_ft(cft: str, ft: str) -> list[int]:
    """Map each character index in cleaned_ft back to raw full_text."""
    cft_to_ft = [0] * len(cft)
    ft_ptr = 0
    cft_ptr = 0
    
    while cft_ptr < len(cft) and ft_ptr < len(ft):
        c = cft[cft_ptr]
        idx = ft.find(c, ft_ptr)
        if idx != -1:
            ft_ptr = idx
            cft_to_ft[cft_ptr] = ft_ptr
            cft_ptr += 1
            ft_ptr += 1
        else:
            cft_ptr += 1
    return cft_to_ft


def run_migration() -> None:
    db_path = Path("data/transcriptions.db")
    backup_path = Path("data/transcriptions.db.pre_esm_backfill_20261002")

    if not backup_path.exists():
        raise RuntimeError(f"Required backup {backup_path} does not exist. Aborting migration.")

    print(f"Verified pre-migration backup at {backup_path} ({backup_path.stat().st_size} bytes)")
    print("Beginning migration m018: populating expression_segment_map...")
    t0 = time.time()

    with get_session() as session:
        # Pre-check existing row count
        pre_count = session.execute(text("SELECT COUNT(*) FROM expression_segment_map")).scalar()
        print(f"Pre-migration expression_segment_map count: {pre_count}")

        # Fetch all transcriptions
        tx_rows = session.execute(text("""
            SELECT t.id, t.full_text, t.duration_seconds, t.engine, af.file_path, w.title
            FROM transcriptions t
            JOIN audio_files af ON t.audio_file_id = af.id
            LEFT JOIN works w ON af.work_id = w.id
            ORDER BY t.id
        """)).fetchall()

        all_links: list[dict[str, int]] = []
        total_chunks = 0
        mapped_chunks = 0
        unmapped_chunks = 0
        degenerate_chunks = 0

        for tx_idx, (tx_id, ft, dur, engine, fpath, title) in enumerate(tx_rows):
            ft = ft or ""

            chunks = session.execute(text("""
                SELECT id, content 
                FROM expressions 
                WHERE source_id = :tx_id AND graph_role = 'sweep_chunk'
                ORDER BY id
            """), {"tx_id": tx_id}).fetchall()

            if not chunks:
                continue

            total_chunks += len(chunks)

            segs = session.execute(text("""
                SELECT id, segment_index, text, start_time, end_time 
                FROM segments 
                WHERE transcription_id = :tx_id 
                ORDER BY segment_index
            """), {"tx_id": tx_id}).fetchall()

            if not segs:
                unmapped_chunks += len(chunks)
                continue

            # Check if this transcription has 0.0/0.0 degenerate segments
            is_unaligned_tx = all(s[3] == 0.0 and s[4] == 0.0 for s in segs)

            # 1. Compute segment spans in raw full_text
            seg_spans = []
            curr_offset = 0
            for sid, sidx, stext, s_start, s_end in segs:
                st_clean = stext.strip()
                if not st_clean:
                    continue
                idx = ft.find(st_clean, curr_offset)
                if idx == -1:
                    idx = ft.find(st_clean)
                if idx != -1:
                    seg_spans.append({
                        "id": sid,
                        "index": sidx,
                        "char_start": idx,
                        "char_end": idx + len(st_clean),
                        "start_time": s_start,
                        "end_time": s_end,
                    })
                    curr_offset = idx + len(st_clean)

            # 2. Clean full_text and align to raw full_text
            cft = _clean_text(ft)
            cft_to_ft = align_cft_to_ft(cft, ft)

            cft_norm = norm_punct(cft)
            ft_alphanum = strip_all_punct(ft)
            ft_alphanum_to_raw = [raw_i for raw_i, ch in enumerate(ft) if ch.isalnum()]

            # 3. Match each chunk
            cft_cursor = 0

            for cid, ccontent in chunks:
                c_clean = ccontent.strip()
                c_start_raw = None
                c_end_raw = None

                # Tier 1: exact cft match
                idx = cft.find(c_clean, cft_cursor)
                if idx != -1:
                    cft_cursor = idx
                    c_start_raw = cft_to_ft[idx]
                    end_idx = min(len(cft_to_ft) - 1, idx + len(c_clean) - 1)
                    c_end_raw = cft_to_ft[end_idx] + 1
                else:
                    # Tier 2: prefix cft match
                    prefix = c_clean[:80]
                    idx = cft.find(prefix, cft_cursor)
                    if idx != -1:
                        cft_cursor = idx
                        c_start_raw = cft_to_ft[idx]
                        if idx + len(c_clean) >= len(cft):
                            c_end_raw = len(ft)
                        else:
                            end_idx = min(len(cft_to_ft) - 1, idx + len(c_clean) - 1)
                            c_end_raw = cft_to_ft[end_idx] + 1
                    else:
                        # Tier 3: normalized cft match
                        c_norm = norm_punct(c_clean)
                        idx_norm = cft_norm.find(c_norm)
                        if idx_norm != -1:
                            prefix_words = c_clean.split()[:4]
                            pw = " ".join(prefix_words)
                            idx_pw = cft.find(pw, cft_cursor)
                            if idx_pw == -1:
                                idx_pw = cft.find(pw)
                            if idx_pw != -1:
                                cft_cursor = idx_pw
                                c_start_raw = cft_to_ft[idx_pw]
                                end_idx = min(len(cft_to_ft) - 1, idx_pw + len(c_clean) - 1)
                                c_end_raw = cft_to_ft[end_idx] + 1

                # If not found in cft, search in raw ft
                if c_start_raw is None:
                    # Tier 4: exact raw ft match
                    idx_raw = ft.find(c_clean)
                    if idx_raw != -1:
                        c_start_raw = idx_raw
                        c_end_raw = idx_raw + len(c_clean)
                    else:
                        # Tier 5: normalized raw ft match
                        words = [w for w in c_clean.split() if len(w) > 3]
                        if len(words) >= 4:
                            search_phrase = " ".join(words[:4])
                            idx_sp = ft.find(search_phrase)
                            if idx_sp == -1:
                                idx_sp = ft.lower().find(search_phrase.lower())
                            if idx_sp != -1:
                                c_start_raw = idx_sp
                                c_end_raw = min(len(ft), idx_sp + len(c_clean))

                if c_start_raw is None:
                    # Tier 6: prefix in raw ft fallback
                    idx_pfx = ft.find(c_clean[:40])
                    if idx_pfx == -1:
                        idx_pfx = ft.lower().find(c_clean[:40].lower())
                    if idx_pfx != -1:
                        c_start_raw = idx_pfx
                        c_end_raw = min(len(ft), idx_pfx + len(c_clean))

                if c_start_raw is None:
                    # Tier 7: alphanumeric stripped fallback
                    c_alpha = strip_all_punct(c_clean)
                    pos_alpha = ft_alphanum.find(c_alpha[:50])
                    if pos_alpha != -1 and pos_alpha < len(ft_alphanum_to_raw):
                        c_start_raw = ft_alphanum_to_raw[pos_alpha]
                        end_pos_alpha = min(len(ft_alphanum_to_raw) - 1, pos_alpha + len(c_alpha) - 1)
                        c_end_raw = ft_alphanum_to_raw[end_pos_alpha] + 1

                if c_start_raw is not None and c_end_raw is not None:
                    matched_segs = [
                        s for s in seg_spans
                        if max(s["char_start"], c_start_raw) < min(s["char_end"], c_end_raw)
                    ]
                    if matched_segs:
                        mapped_chunks += 1
                        if is_unaligned_tx:
                            degenerate_chunks += 1
                        for s in matched_segs:
                            all_links.append({"expression_id": cid, "segment_id": s["id"]})
                    else:
                        unmapped_chunks += 1
                else:
                    unmapped_chunks += 1

            if (tx_idx + 1) % 200 == 0 or (tx_idx + 1) == len(tx_rows):
                print(f"Processed {tx_idx + 1}/{len(tx_rows)} transcriptions... (generated {len(all_links)} links)")

        print(f"\nTotal expressions mapped: {mapped_chunks}/{total_chunks} (100.0% = {mapped_chunks == total_chunks})")
        print(f"Total bridge links to insert: {len(all_links)}")
        print(f"Unmapped expressions: {unmapped_chunks}")
        print(f"Degenerate expressions (0.0/0.0 segments): {degenerate_chunks}")

        if unmapped_chunks > 0:
            raise RuntimeError(f"Migration aborted: {unmapped_chunks} expressions failed to map.")

        # Batch insert into expression_segment_map
        print("Writing links to expression_segment_map...")
        batch_size = 5000
        for i in range(0, len(all_links), batch_size):
            batch = all_links[i : i + batch_size]
            session.execute(
                text("""
                    INSERT OR IGNORE INTO expression_segment_map (expression_id, segment_id)
                    VALUES (:expression_id, :segment_id)
                """),
                batch,
            )

        session.commit()
        elapsed = time.time() - t0
        print(f"Migration m018 committed successfully in {elapsed:.2f}s.")

        # Post-migration validation
        post_count = session.execute(text("SELECT COUNT(*) FROM expression_segment_map")).scalar()
        distinct_exprs = session.execute(text("SELECT COUNT(DISTINCT expression_id) FROM expression_segment_map")).scalar()
        unmapped_db = session.execute(text("""
            SELECT COUNT(*) FROM expressions e
            WHERE e.graph_role = 'sweep_chunk'
              AND NOT EXISTS (SELECT 1 FROM expression_segment_map m WHERE m.expression_id = e.id)
        """)).scalar()

        print("\n" + "=" * 50)
        print("POST-MIGRATION VALIDATION RESULTS")
        print("=" * 50)
        print(f"expression_segment_map total rows: {post_count}")
        print(f"Distinct expressions linked:      {distinct_exprs} / {total_chunks}")
        print(f"Unmapped sweep_chunk expressions: {unmapped_db}")

        assert post_count == len(all_links), f"Row count mismatch: expected {len(all_links)}, got {post_count}"
        assert distinct_exprs == total_chunks, f"Expression coverage mismatch: expected {total_chunks}, got {distinct_exprs}"
        assert unmapped_db == 0, f"Unmapped expressions remain in DB: {unmapped_db}"
        print("ALL DATABASE INVARIANTS VERIFIED CLEANLY.")


if __name__ == "__main__":
    run_migration()
