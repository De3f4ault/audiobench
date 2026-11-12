"""Regression & Invariant Tests: Expression Playback Resolution & Segment Synchronization.

Tests that all expression types (audio_transcript, chapter_summary with/without chapter
transcription_id, duplicate-transcription audio files, and non-audio search_synthesis)
resolve their playback metadata cleanly without 0-sentinels, and that the two display
paths (\\lyrics HUD and \\read) strictly agree on segment counts and chosen transcripts.
"""

from __future__ import annotations

import os
from typing import Any
import pytest
from unittest.mock import MagicMock

from sqlalchemy import text as _sql
from audiobench.core.db_session import get_session
from audiobench.cli.repl.backslash_commands import (
    _resolve_expression_playback_info,
    _resolve_thought_segment,
    _play_expression,
    ReplSession,
)
from audiobench.playback.lyrics_hud import _load_segments_for_playback
from audiobench.cli.commands.play import _find_best_transcript_for_file
from audiobench.storage.repository import TranscriptionRepository
from audiobench.storage.models import (
    AudioFileRecord,
    ChapterRecord,
    ExpressionRecord,
    SegmentRecord,
    TranscriptionRecord,
    WorkRecord,
)


@pytest.fixture
def seeded_db():
    """Seed an isolated database with all expression and transcription scenarios."""
    with get_session() as session:
        work = WorkRecord(title="Test Architecture", author="Martin Fowler")
        session.add(work)
        session.flush()

        # 1. Normal audio file + transcription + audio_transcript expression
        af1 = AudioFileRecord(file_path="/tmp/test_clean_arch.mp3", file_name="clean_arch.mp3", duration_seconds=100.0)
        session.add(af1)
        session.flush()

        tx1 = TranscriptionRecord(
            audio_file_id=af1.id, file_name="clean_arch.mp3", segment_count=5, pipeline_phase="complete"
        )
        session.add(tx1)
        session.flush()

        for i in range(5):
            session.add(
                SegmentRecord(
                    transcription_id=tx1.id,
                    segment_index=i,
                    start_time=float(i * 10),
                    end_time=float(i * 10 + 8),
                    text=f"Clean architecture segment {i}",
                )
            )

        expr1 = ExpressionRecord(
            source_type="audio_transcript",
            source_id=tx1.id,
            work_id=work.id,
            content="Architecture should be independent of frameworks.",
        )
        session.add(expr1)

        # 2. Chapter summary with chapter.transcription_id populated
        af2 = AudioFileRecord(file_path="/tmp/test_refactoring.mp3", file_name="refactoring.mp3", duration_seconds=200.0)
        session.add(af2)
        session.flush()

        tx2 = TranscriptionRecord(
            audio_file_id=af2.id, file_name="refactoring.mp3", segment_count=8, pipeline_phase="complete"
        )
        session.add(tx2)
        session.flush()

        for i in range(8):
            session.add(
                SegmentRecord(
                    transcription_id=tx2.id,
                    segment_index=i,
                    start_time=float(i * 10),
                    end_time=float(i * 10 + 8),
                    text=f"Refactoring segment {i}",
                )
            )

        chap2 = ChapterRecord(
            audio_file_id=af2.id,
            chapter_index=1,
            title="Code Smells",
            start_time=10.0,
            end_time=90.0,
            transcription_id=tx2.id,
        )
        session.add(chap2)
        session.flush()

        expr2 = ExpressionRecord(
            source_type="chapter_summary",
            source_id=chap2.id,
            work_id=work.id,
            content="Summary of code smells chapter.",
        )
        session.add(expr2)

        # 3. Chapter summary with chapter.transcription_id IS NULL (Citation [8] Eat That Frog bug case!)
        af3 = AudioFileRecord(file_path="/tmp/test_eat_that_frog.m4b", file_name="Eat That Frog!.m4b", duration_seconds=6000.0)
        session.add(af3)
        session.flush()

        tx3 = TranscriptionRecord(
            audio_file_id=af3.id, file_name="Eat That Frog!.m4b", segment_count=354, pipeline_phase="complete"
        )
        session.add(tx3)
        session.flush()

        for i in range(354):  # Seed all 354 segments to match production Citation [8] count exactly
            session.add(
                SegmentRecord(
                    transcription_id=tx3.id,
                    segment_index=i,
                    start_time=float(5498.0 + i * 2),
                    end_time=float(5498.0 + i * 2 + 1.8),
                    text=f"Eat that frog segment {i}",
                )
            )

        chap3 = ChapterRecord(
            audio_file_id=af3.id,
            chapter_index=14,
            title="Prepare Thoroughly Before You Begin",
            start_time=5498.345,
            end_time=5800.0,
            transcription_id=None,  # Crucial: NULL transcription_id in chapter record
        )
        session.add(chap3)
        session.flush()

        expr3 = ExpressionRecord(
            source_type="chapter_summary",
            source_id=chap3.id,
            work_id=work.id,
            content="Preparation is key to overcoming procrastination and eating that frog.",
        )
        session.add(expr3)

        # 4. Duplicate transcriptions (Audio file with failed tx and completed tx, like af_id=186)
        af4 = AudioFileRecord(file_path="/tmp/test_chapel.m4a", file_name="chapel.m4a", duration_seconds=1200.0)
        session.add(af4)
        session.flush()

        # Failed transcription (older or failed run, but has segments)
        tx4_failed = TranscriptionRecord(
            audio_file_id=af4.id, file_name="chapel.m4a", segment_count=25, pipeline_phase="failed"
        )
        session.add(tx4_failed)
        session.flush()
        for i in range(25):
            session.add(
                SegmentRecord(
                    transcription_id=tx4_failed.id,
                    segment_index=i,
                    start_time=float(i * 5),
                    end_time=float(i * 5 + 4),
                    text=f"Failed run seg {i}",
                )
            )

        # Completed transcription (should win via 3-tier precedence)
        tx4_complete = TranscriptionRecord(
            audio_file_id=af4.id, file_name="chapel.m4a", segment_count=25, pipeline_phase="complete"
        )
        session.add(tx4_complete)
        session.flush()
        for i in range(25):
            session.add(
                SegmentRecord(
                    transcription_id=tx4_complete.id,
                    segment_index=i,
                    start_time=float(i * 5),
                    end_time=float(i * 5 + 4),
                    text=f"Complete run seg {i}",
                )
            )

        chap4 = ChapterRecord(
            audio_file_id=af4.id,
            chapter_index=1,
            title="Sunday Service",
            start_time=0.0,
            end_time=600.0,
            transcription_id=None,  # NULL
        )
        session.add(chap4)
        session.flush()

        expr4 = ExpressionRecord(
            source_type="chapter_summary",
            source_id=chap4.id,
            work_id=work.id,
            content="Sunday sermon summary on discipleship.",
        )
        session.add(expr4)

        # 5. Non-audio search_synthesis expression
        expr5 = ExpressionRecord(
            source_type="search_synthesis",
            source_id=1,
            work_id=None,
            content="Prior research synthesis card comparing models.",
        )
        session.add(expr5)

        session.commit()

        data = {
            "expr_audio": expr1.id,
            "tx_audio": tx1.id,
            "expr_chap_tid": expr2.id,
            "tx_chap_tid": tx2.id,
            "expr_frog": expr3.id,
            "tx_frog": tx3.id,
            "af_frog": af3.id,
            "expr_dup": expr4.id,
            "tx_dup_complete": tx4_complete.id,
            "tx_dup_failed": tx4_failed.id,
            "af_dup_path": af4.file_path,
            "expr_synthesis": expr5.id,
            "_eids": [expr1.id, expr2.id, expr3.id, expr4.id, expr5.id],
            "_cids": [chap2.id, chap3.id, chap4.id],
            "_tids": [tx1.id, tx2.id, tx3.id, tx4_failed.id, tx4_complete.id],
            "_afids": [af1.id, af2.id, af3.id, af4.id],
            "_wid": work.id,
        }

    try:
        yield data
    finally:
        with get_session() as session:
            for eid in data["_eids"]:
                session.query(ExpressionRecord).filter_by(id=eid).delete()
            for tid in data["_tids"]:
                session.query(SegmentRecord).filter_by(transcription_id=tid).delete()
                session.query(TranscriptionRecord).filter_by(id=tid).delete()
            for cid in data["_cids"]:
                session.query(ChapterRecord).filter_by(id=cid).delete()
            for afid in data["_afids"]:
                session.query(AudioFileRecord).filter_by(id=afid).delete()
            session.query(WorkRecord).filter_by(id=data["_wid"]).delete()
            session.commit()


def _simulate_read_segments(tx_id: int | None, af_id: int | None, file_path: str | None) -> list[dict]:
    """Mirror the exact segment resolution logic used by cmd_read in backslash_commands.py."""
    repo = TranscriptionRepository()
    record = None

    if tx_id:
        record = repo.get_by_id(int(tx_id), session=None)
    elif af_id:
        best_id = None
        if file_path:
            best_id = _find_best_transcript_for_file(repo, file_path)
        else:
            with get_session() as sdb:
                row = sdb.execute(
                    _sql("SELECT file_path FROM audio_files WHERE id = :id"),
                    {"id": af_id},
                ).fetchone()
                if row:
                    best_id = _find_best_transcript_for_file(repo, row[0])
        if best_id:
            record = repo.get_by_id(best_id, session=None)

    if record:
        return record.get("segments", []) or []
    return []


class TestExpressionPlaybackResolution:
    """Tests expression playback resolution invariants across all expression source types."""

    def test_citation_8_eat_that_frog_resolves_synchronized_segments(self, seeded_db):
        """Citation [8] (chapter_summary with c.transcription_id IS NULL).

        Must resolve a valid positive transcription_id via the 4th fallback subquery,
        not 0 and not None. Both display paths must agree and extract all synchronized segments.
        """
        expr_id = seeded_db["expr_frog"]
        with get_session() as sdb:
            res = _resolve_expression_playback_info(sdb, expr_id)
            eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res

            assert eid == expr_id
            assert tx_id is not None and tx_id > 0, f"Expected valid tx_id, got {tx_id}"
            assert tx_id == seeded_db["tx_frog"]
            assert af_id == seeded_db["af_frog"]

            # Lyrics HUD path: simulate with resolved playback state
            mock_client = MagicMock()
            st = {
                "playing": True,
                "file": file_path,
                "audio_file_id": af_id,
                "transcription_id": tx_id,
                "position": 5498.345,
                "duration": 6000.0,
                "speed": 1.0,
                "paused": False,
            }
            hud_tid, hud_title, hud_segments = _load_segments_for_playback(st, client=mock_client)
            read_segments = _simulate_read_segments(tx_id, af_id, file_path)

            assert len(hud_segments) == len(read_segments) == 354
            assert hud_tid == tx_id

    def test_lyrics_and_read_agree_on_all_source_types(self, seeded_db):
        """Invariant: For all source types, \\lyrics and \\read must load the exact same segment count."""
        cases = [
            ("audio_transcript_direct", seeded_db["expr_audio"], True),
            ("chapter_summary_with_tid", seeded_db["expr_chap_tid"], True),
            ("chapter_summary_null_tid", seeded_db["expr_frog"], True),
            ("duplicate_tx_audio_file", seeded_db["expr_dup"], True),
            ("non_audio_search_synthesis", seeded_db["expr_synthesis"], False),
        ]

        with get_session() as sdb:
            for case_name, expr_id, expected_playable in cases:
                res = _resolve_expression_playback_info(sdb, expr_id)
                eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res

                if expected_playable:
                    assert tx_id is not None and tx_id > 0, f"Failed on {case_name}: tx_id is None"
                    assert file_path is not None
                    mock_client = MagicMock()
                    st = {
                        "playing": True,
                        "file": file_path,
                        "audio_file_id": af_id,
                        "transcription_id": tx_id,
                        "position": chap_start_time,
                        "duration": 1000.0,
                        "speed": 1.0,
                        "paused": False,
                    }
                    _, _, hud_segs = _load_segments_for_playback(st, client=mock_client)
                    read_segs = _simulate_read_segments(tx_id, af_id, file_path)

                    assert len(hud_segs) == len(read_segs) > 0, (
                        f"Disagreement for {case_name}: hud={len(hud_segs)}, read={len(read_segs)}"
                    )
                else:
                    assert tx_id is None
                    assert af_id is None
                    assert file_path is None
                    mock_client = MagicMock()
                    st = {}
                    _, _, hud_segs = _load_segments_for_playback(st, client=mock_client)
                    read_segs = _simulate_read_segments(None, None, None)
                    assert len(hud_segs) == len(read_segs) == 0

    def test_duplicate_transcription_resolution_polarity(self, seeded_db):
        """Case 5: Audio file has a failed transcription and a completed transcription.

        Both the SQL resolver and play.py's _find_best_transcript_for_file must select
        the completed transcript over the failed one.
        """
        expr_id = seeded_db["expr_dup"]
        with get_session() as sdb:
            res = _resolve_expression_playback_info(sdb, expr_id)
            eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res

            assert tx_id == seeded_db["tx_dup_complete"], (
                f"Resolver chose {tx_id}, but expected completed transcript {seeded_db['tx_dup_complete']}"
            )
            assert tx_id != seeded_db["tx_dup_failed"]

            best_from_read = _find_best_transcript_for_file(TranscriptionRepository(), seeded_db["af_dup_path"])
            assert tx_id == best_from_read, (
                f"Polarity mismatch on duplicate audio file: resolver={tx_id}, read={best_from_read}"
            )

    def test_chapter_with_explicit_transcription_id_precedence(self, seeded_db):
        """When a chapter record has an explicit transcription_id, it takes precedence."""
        expr_id = seeded_db["expr_chap_tid"]
        with get_session() as sdb:
            res = _resolve_expression_playback_info(sdb, expr_id)
            assert res[1] == seeded_db["tx_chap_tid"]

    def test_play_synthesis_expression_returns_gracefully(self, seeded_db):
        """Prior synthesis expressions (source_type='search_synthesis') must not attempt

        SegmentRecord WHERE transcription_id = 0 and must cleanly inform the user without raising.
        """
        expr_id = seeded_db["expr_synthesis"]
        with get_session() as sdb:
            res = _resolve_expression_playback_info(sdb, expr_id)
            eid, tx_id, wid, content, af_id, file_name, file_path, work_title, work_author, chap_start_time = res

            # Must return clean None for tx_id, af_id, and file_path
            assert tx_id is None
            assert af_id is None
            assert file_path is None

            # _resolve_thought_segment with None must return a valid 3-tuple without querying SegmentRecord
            start_t, end_t, spk = _resolve_thought_segment(sdb, eid, tx_id, content)
            assert (start_t, end_t, spk) == (0.0, 0.0, None)

            # _play_expression must cleanly return False without calling playback_play
            mock_session = MagicMock(spec=ReplSession)
            mock_client = MagicMock()
            success = _play_expression(mock_session, mock_client, expr_id, label="citation [1]")
            assert success is False
            mock_client.playback_play.assert_not_called()

    def test_lyrics_hud_recovers_when_passed_zero_sentinel(self, seeded_db):
        """Defense-in-depth: If a caller erroneously passes tid=0, lyrics_hud must treat it

        as absent and recover via audio_file_id instead of querying transcription_id = 0.
        """
        af_id = seeded_db["af_frog"]
        tx_id_invalid = 0
        mock_client = MagicMock()
        st = {
            "playing": True,
            "audio_file_id": af_id,
            "transcription_id": tx_id_invalid,
        }

        # Passing tid=0 in status should trigger the fallback to audio_file_id and find tx with segments
        tid, title, segs = _load_segments_for_playback(st, client=mock_client)
        assert tid == seeded_db["tx_frog"]
        assert len(segs) == 354
