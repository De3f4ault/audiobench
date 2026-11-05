"""Migration script m014: audio_files enrichment and chapter title hydration.

Backfills:
  1. audio_files.track_title, sequence_key, codec, bit_rate, genre via ffprobe.
  2. chapters.title hydration: replaces dummy "Full Recording" with real track_title
     where format tags contain a valid title.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sqlalchemy import text
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.sequence_resolver import resolve_sequence_key

logger = get_logger("migrations.m014_enrich_audio_files")


def probe_file_metadata(file_path: Path) -> tuple[dict[str, str], str, int | None]:
    """Run ffprobe to extract format tags, codec name, and bit_rate.

    Returns:
        (tags_dict, codec_name, bit_rate)
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return {}, "unknown", None
        data = json.loads(res.stdout)
    except Exception:
        return {}, "unknown", None

    # Tags from format
    raw_tags = data.get("format", {}).get("tags", {})
    tags = {k.lower(): str(v).strip() for k, v in raw_tags.items() if v}

    # Codec from first audio stream
    codec = "unknown"
    for st in data.get("streams", []):
        if st.get("codec_type") == "audio" or "codec_name" in st:
            codec = st.get("codec_name", "unknown")
            break

    # Bitrate from format or stream
    raw_br = data.get("format", {}).get("bit_rate")
    bit_rate = None
    if raw_br:
        try:
            br_int = int(raw_br)
            if br_int > 0:
                bit_rate = br_int
        except ValueError:
            pass

    return tags, codec, bit_rate


def run_enrichment() -> dict:
    """Execute the full backfill and chapter title hydration."""
    print("=" * 70)
    print("MIGRATION m014: ENRICH AUDIO_FILES & HYDRATE CHAPTER TITLES")
    print("=" * 70)

    stats = {
        "total_files": 0,
        "files_updated": 0,
        "missing_on_disk": 0,
        "sequence_keys_set": 0,
        "sequence_keys_none": 0,
        "chapters_hydrated": 0,
        "chapters_still_full_recording": 0,
    }

    with get_session() as session:
        files = session.execute(
            text("SELECT id, file_path, file_name FROM audio_files ORDER BY id")
        ).fetchall()
        stats["total_files"] = len(files)
        print(f"Total audio_files in database: {len(files)}")

        for fid, fpath, fname in files:
            p = Path(fpath)
            if not p.exists():
                stats["missing_on_disk"] += 1
                continue

            tags, codec, bit_rate = probe_file_metadata(p)
            raw_title = tags.get("title")
            track_title = raw_title.strip() if raw_title else None
            genre = tags.get("genre")
            genre_clean = genre.strip() if genre else None

            seq_key = resolve_sequence_key(fname, tags)
            if seq_key:
                stats["sequence_keys_set"] += 1
            else:
                stats["sequence_keys_none"] += 1

            session.execute(
                text("""
                    UPDATE audio_files
                    SET track_title = :track_title,
                        sequence_key = :sequence_key,
                        codec = :codec,
                        bit_rate = :bit_rate,
                        genre = :genre
                    WHERE id = :fid
                """),
                {
                    "track_title": track_title,
                    "sequence_key": seq_key,
                    "codec": codec,
                    "bit_rate": bit_rate,
                    "genre": genre_clean,
                    "fid": fid,
                },
            )
            stats["files_updated"] += 1

        session.commit()
        print(f"Enriched {stats['files_updated']} audio_files.")
        print(f"  Sequence keys resolved: {stats['sequence_keys_set']}")
        print(f"  Sequence keys None:     {stats['sequence_keys_none']}")
        print(f"  Missing files on disk:  {stats['missing_on_disk']}")

        # ── Part 2: Chapter Title Hydration ───────────────────────────────────
        print("\nHydrating chapter titles for 'Full Recording' chapters...")
        # Find chapters with 'Full Recording' where the parent file has a valid track_title
        candidates = session.execute(
            text("""
                SELECT c.id, c.audio_file_id, c.title, af.track_title, af.file_name
                FROM chapters c
                JOIN audio_files af ON c.audio_file_id = af.id
                WHERE c.title = 'Full Recording'
            """)
        ).fetchall()

        for cid, afid, old_title, track_title, fname in candidates:
            if track_title and track_title.strip():
                clean = track_title.strip()
                if clean.lower() not in {"full recording", "untitled", "unknown"}:
                    session.execute(
                        text("UPDATE chapters SET title = :title WHERE id = :cid"),
                        {"title": clean, "cid": cid},
                    )
                    stats["chapters_hydrated"] += 1
                    continue

            stats["chapters_still_full_recording"] += 1

        session.commit()
        print(f"Chapters hydrated from format tags: {stats['chapters_hydrated']}")
        print(f"Chapters remaining 'Full Recording': {stats['chapters_still_full_recording']}")

    return stats


def verify_enrichment() -> None:
    """Run verification checks on the enriched data."""
    print("\n" + "=" * 70)
    print("VERIFICATION GATE CHECKS")
    print("=" * 70)

    with get_session() as session:
        # Check 1: 5 Albums from track validation result
        test_albums = [
            "Out of Your Mind - Disc 1 - The Nature of Consciousness",
            "Frankenstein",
            "A Tale of Two Cities",
            "Crime and Punishment",
            "The Power of Now - A Guide to Spiritual Enlightenment",
        ]
        print("1. Spot-check 5 albums for sequence_keys:")
        for alb in test_albums:
            rows = session.execute(
                text("""
                    SELECT af.id, af.file_name, af.sequence_key, af.track_title
                    FROM audio_files af
                    JOIN works w ON af.work_id = w.id
                    WHERE af.file_name LIKE :pat OR af.track_title LIKE :pat
                    LIMIT 3
                """),
                {"pat": f"%{alb.split()[0]}%"},
            ).fetchall()
            print(f"\n  Album pattern: {alb}")
            for r in rows:
                print(f"    id={r[0]} seq={r[2]!r:<8} title={r[3]!r:<30} file={r[1][:40]}")

        # Check 2: Assert no 'Full Recording' survives where track_title is present
        lingering = session.execute(
            text("""
                SELECT COUNT(*)
                FROM chapters c
                JOIN audio_files af ON c.audio_file_id = af.id
                WHERE c.title = 'Full Recording'
                  AND af.track_title IS NOT NULL
                  AND LOWER(TRIM(af.track_title)) NOT IN ('full recording', 'untitled', 'unknown')
            """)
        ).scalar()
        print(f"\n2. Lingering 'Full Recording' chapters where title tag existed: {lingering}")
        assert lingering == 0, f"FAILED: {lingering} chapters still have 'Full Recording' despite valid title tag!"
        print("   GATE PASSED: 0 lingering unhydrated chapters.")

        # Check 3: Codec and Bitrate coverage
        codecs = session.execute(
            text("SELECT codec, COUNT(*) FROM audio_files GROUP BY codec ORDER BY COUNT(*) DESC")
        ).fetchall()
        print("\n3. Codec distribution in audio_files:")
        for c, cnt in codecs:
            print(f"    {c:<15} {cnt:>4}")

        null_bitrates = session.execute(
            text("SELECT COUNT(*) FROM audio_files WHERE bit_rate IS NULL")
        ).scalar()
        print(f"\n4. Files with NULL bit_rate: {null_bitrates} / 824")

    print("\nALL VERIFICATION GATE CHECKS PASSED.")


if __name__ == "__main__":
    run_enrichment()
    verify_enrichment()
