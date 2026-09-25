"""File dossier command — comprehensive 360-degree inspection of an audio file and its relations.

Aggregates:
  - Audio file metadata (duration, format, codec, sample rate, size, hash, path)
  - Transcriptions (engines, models, languages, word counts, dates)
  - Chapters (indices, titles, time ranges, statuses)
  - Bookmarks & Explanations (points, regions, AI notes)
  - Rendered clips (scanned from data/clips/<slug>/ and registered in DB)
  - Associated chat conversations
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import click
from rich.console import Console

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    ERROR,
    SUCCESS,
    WARNING,
    console,
    format_size,
    make_table,
)
from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.playback.controls import fmt_timestamp
from audiobench.storage.models import (
    AudioFileRecord,
    BookmarkRecord,
    ChapterRecord,
    ChatConversation,
    NoteCapture,
    SegmentRecord,
    StudyProject,
    StudySession,
    TranscriptionRecord,
)


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def render_file_dossier(
    target: str | int,
    target_console: Console,
    interactive: bool = False,
    play: bool = False,
) -> bool:
    """Resolve target and render the 360-degree file dossier to target_console.

    Returns True if an audio record was resolved and displayed, False otherwise.
    """
    target_str = str(target).strip()
    if not target_str:
        target_console.print(f"  [{WARNING}]No target specified.[/]")
        return False

    with get_session() as session:
        audio: AudioFileRecord | None = None
        match_source: str = ""

        # 1. "#N" format -> transcript ID
        if target_str.startswith("#") and target_str[1:].isdigit():
            tx = session.query(TranscriptionRecord).filter_by(id=int(target_str[1:])).first()
            if tx and tx.audio_file_id:
                audio = session.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                match_source = f"Transcript #{tx.id}"

        # 2. "c<N>" or "chat:<N>" format -> chat conversation ID
        elif (target_str.lower().startswith("c") and target_str[1:].isdigit()) or (
            target_str.lower().startswith("chat:") and target_str[5:].isdigit()
        ):
            cid = int(target_str[1:] if target_str.lower().startswith("c") else target_str[5:])
            conv = session.query(ChatConversation).filter_by(id=cid).first()
            if conv and conv.transcript_ids:
                try:
                    tids = json.loads(conv.transcript_ids)
                    if tids:
                        tx = session.query(TranscriptionRecord).filter_by(id=tids[0]).first()
                        if tx and tx.audio_file_id:
                            audio = session.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                            match_source = f"Chat #{conv.id} ('{conv.title}')"
                except Exception:
                    pass

        # 3. Pure integer -> try audio file ID, fallback to transcript ID, fallback to chat conversation ID
        elif target_str.isdigit():
            aid = int(target_str)
            audio = session.query(AudioFileRecord).filter_by(id=aid).first()
            if audio:
                match_source = f"Audio file #{audio.id}"
            else:
                tx = session.query(TranscriptionRecord).filter_by(id=aid).first()
                if tx and tx.audio_file_id:
                    audio = session.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                    match_source = f"Transcript #{tx.id}"
            if not audio:
                conv = session.query(ChatConversation).filter_by(id=aid).first()
                if conv and conv.transcript_ids:
                    try:
                        tids = json.loads(conv.transcript_ids)
                        if tids:
                            tx = session.query(TranscriptionRecord).filter_by(id=tids[0]).first()
                            if tx and tx.audio_file_id:
                                audio = session.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                                match_source = f"Chat #{conv.id} ('{conv.title}')"
                    except Exception:
                        pass

        # 4. Path exists on disk
        elif Path(target_str).exists():
            resolved = str(Path(target_str).resolve())
            audio = session.query(AudioFileRecord).filter_by(file_path=resolved).first()
            if audio:
                match_source = f"path '{Path(target_str).name}'"
            if not audio:
                try:
                    fhash = _file_hash(Path(target_str))
                    audio = session.query(AudioFileRecord).filter_by(file_hash=fhash).first()
                    if audio:
                        match_source = f"file hash '{fhash[:8]}'"
                except Exception:
                    pass

        # 5. Fuzzy text match by filename
        if not audio:
            audio = (
                session.query(AudioFileRecord)
                .filter(AudioFileRecord.file_name.ilike(f"%{target_str}%"))
                .first()
            )
            if audio:
                match_source = f"filename matching '{target_str}'"
            else:
                tx = (
                    session.query(TranscriptionRecord)
                    .filter(TranscriptionRecord.file_name.ilike(f"%{target_str}%"))
                    .first()
                )
                if tx and tx.audio_file_id:
                    audio = session.query(AudioFileRecord).filter_by(id=tx.audio_file_id).first()
                    match_source = f"transcript filename matching '{target_str}'"

        if not audio:
            target_console.print(f"  [{WARNING}]No audio file found matching target: {target}[/]")
            return False

        # Query all related records
        tx_rows = (
            session.query(TranscriptionRecord)
            .filter_by(audio_file_id=audio.id)
            .order_by(TranscriptionRecord.id.desc())
            .all()
        )
        chapter_rows = (
            session.query(ChapterRecord)
            .filter_by(audio_file_id=audio.id)
            .order_by(ChapterRecord.chapter_index)
            .all()
        )
        bookmark_rows = (
            session.query(BookmarkRecord)
            .filter_by(audio_file_id=audio.id)
            .order_by(BookmarkRecord.timestamp)
            .all()
        )

        # Query chat conversations referencing any transcription of this audio file
        tx_ids = [t.id for t in tx_rows]
        all_chats = session.query(ChatConversation).order_by(ChatConversation.id.desc()).all()
        related_chats = []
        for c in all_chats:
            if not c.transcript_ids:
                continue
            try:
                c_tids = json.loads(c.transcript_ids)
                if any(tid in c_tids for tid in tx_ids):
                    related_chats.append(c)
            except Exception:
                if any(f"[{tid}]" in c.transcript_ids or f", {tid}" in c.transcript_ids for tid in tx_ids):
                    related_chats.append(c)

        # Query study projects and sessions for this audio file
        study_projects = (
            session.query(StudyProject)
            .filter_by(audio_file_id=audio.id)
            .order_by(StudyProject.id.desc())
            .all()
        )

        # Query note captures associated with any segment of this audio file
        note_rows = []
        if tx_ids:
            note_rows = (
                session.query(NoteCapture)
                .join(SegmentRecord, NoteCapture.segment_id == SegmentRecord.id)
                .filter(SegmentRecord.transcription_id.in_(tx_ids))
                .order_by(NoteCapture.id.desc())
                .all()
            )

        # Scan for rendered clips in managed directory
        settings = get_settings()
        safe_stem = re.sub(r"[^\w\-]", "_", Path(audio.file_path).stem)[:30]
        slug = f"{audio.id}_{safe_stem}"
        clips_dir = settings.data_dir / "clips" / slug
        rendered_clips: list[Path] = []
        if clips_dir.exists():
            rendered_clips = [p for p in clips_dir.glob("*.mp3") if p.is_file()]

        # ── Render Header ──
        target_console.print()
        src_tag = f" [dim]· resolved via {match_source}[/dim]" if match_source else ""
        target_console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — File Dossier: [{audio.id:03d}] {audio.file_name}{src_tag}")
        target_console.print(f"  [{DIM}]{'─' * 70}[/]")

        # ── File Metadata ──
        dur_str = fmt_timestamp(audio.duration_seconds)
        chan_str = f"{audio.channels}ch" if audio.channels else "mono/stereo"
        sr_str = f"{audio.sample_rate}Hz" if audio.sample_rate else ""
        fmt_str = audio.format.upper() if audio.format else "AUDIO"
        size_str = format_size(audio.file_size_bytes) if audio.file_size_bytes else ""
        hash_preview = audio.file_hash[:16] + "..." if audio.file_hash else "None"

        target_console.print(f"  [{BOLD}][FILE][/]")
        target_console.print(f"    Duration:  {dur_str} | {fmt_str} | {sr_str} {chan_str} | {size_str}")
        target_console.print(f"    Path:      [{DIM}]{audio.file_path}[/]")
        target_console.print(f"    Hash:      [{DIM}]{hash_preview}[/]")
        target_console.print()

        # ── Transcriptions ──
        target_console.print(f"  [{BOLD}][TRANSCRIPTIONS][/] ({len(tx_rows)})")
        if tx_rows:
            for t in tx_rows:
                date_str = t.created_at.strftime("%Y-%m-%d") if t.created_at else ""
                target_console.print(
                    f"    [{ACCENT}]#{t.id}[/] {t.language.upper()} · "
                    f"{t.word_count:,} words · {t.model_name} · [{DIM}]{t.pipeline_phase} · {date_str}[/]"
                )
        else:
            target_console.print(f"    [{DIM}]No transcriptions recorded.[/]")
        target_console.print()

        # ── Chapters ──
        target_console.print(f"  [{BOLD}][CHAPTERS][/] ({len(chapter_rows)})")
        if chapter_rows:
            has_completed_tx = any(t.pipeline_phase == "complete" for t in tx_rows)
            for ch in chapter_rows:
                time_range = f"{fmt_timestamp(ch.start_time)} - {fmt_timestamp(ch.end_time)}"
                is_done = bool(ch.transcription_id) or (
                    has_completed_tx and (ch.chapter_index == 0 or len(chapter_rows) == 1)
                )
                status = "[DONE]" if is_done else "[PENDING]"
                target_console.print(
                    f"    [{DIM}]{ch.chapter_index:2d}[/]  {time_range:<15}  {ch.title:<30}  [{DIM}]{status}[/]"
                )
        else:
            target_console.print(f"    [{DIM}]No chapters recorded.[/]")
        target_console.print()

        # ── Bookmarks & Explanations ──
        target_console.print(f"  [{BOLD}][BOOKMARKS & EXPLAINS][/] ({len(bookmark_rows)})")
        if bookmark_rows:
            for b in bookmark_rows:
                if b.end_timestamp is not None:
                    ts_disp = f"{fmt_timestamp(b.timestamp)}-{fmt_timestamp(b.end_timestamp)}"
                else:
                    ts_disp = fmt_timestamp(b.timestamp)
                badge = f"[{b.bookmark_type}]"
                target_console.print(
                    f"    [{ACCENT}]{badge:<11}[/] {ts_disp:<15} {b.name}"
                )
        else:
            target_console.print(f"    [{DIM}]No bookmarks or explanations recorded.[/]")
        target_console.print()

        # ── Rendered Clips ──
        clip_count = max(len(rendered_clips), len([b for b in bookmark_rows if b.bookmark_type == "clip"]))
        target_console.print(f"  [{BOLD}][CLIPS][/] ({clip_count})")
        if rendered_clips:
            for cp in rendered_clips:
                csize = format_size(cp.stat().st_size)
                target_console.print(f"    {cp.name}  [{DIM}]({csize}) · {cp.parent}[/]")
        elif any(b.bookmark_type == "clip" for b in bookmark_rows):
            for b in bookmark_rows:
                if b.bookmark_type == "clip":
                    target_console.print(f"    {b.name}  [{DIM}]({b.notes})[/]")
        else:
            target_console.print(f"    [{DIM}]No rendered clips on disk.[/]")
        target_console.print()

        # ── Study Sessions ──
        total_sessions = sum(len(p.sessions) for p in study_projects)
        target_console.print(f"  [{BOLD}][STUDY SESSIONS][/] ({total_sessions})")
        if study_projects:
            for p in study_projects:
                proj_name = p.name or f"Project #{p.id}"
                for s in p.sessions:
                    s_date = s.created_at.strftime("%Y-%m-%d") if s.created_at else ""
                    st_status = "[CLOSED]" if s.closed_at else "[ACTIVE]"
                    conv_tag = f"Chat #{s.conversation_id}" if s.conversation_id else ""
                    target_console.print(
                        f"    [{ACCENT}]{proj_name}[/] · Session #{s.session_number}  "
                        f"[{DIM}]{st_status} {conv_tag} · {s_date}[/]"
                    )
        else:
            target_console.print(f"    [{DIM}]No study sessions recorded.[/]")
        target_console.print()

        # ── Notes & Captures ──
        target_console.print(f"  [{BOLD}][NOTES & CAPTURES][/] ({len(note_rows)})")
        if note_rows:
            for n in note_rows:
                n_date = n.created_at.strftime("%Y-%m-%d") if n.created_at else ""
                seg_ts = fmt_timestamp(n.segment.start_time) if n.segment else "–"
                snippet = n.body.replace("\n", " ").strip()
                if len(snippet) > 60:
                    snippet = snippet[:57] + "…"
                target_console.print(
                    f"    [{ACCENT}]Note #{n.id}[/] [{DIM}@{seg_ts}][/] {snippet} [{DIM}]({n_date})[/]"
                )
        else:
            target_console.print(f"    [{DIM}]No notes recorded.[/]")
        target_console.print()

        # ── Associated Conversations ──
        target_console.print(f"  [{BOLD}][CONVERSATIONS][/] ({len(related_chats)})")
        if related_chats:
            for c in related_chats:
                cdate = c.created_at.strftime("%Y-%m-%d") if c.created_at else ""
                target_console.print(
                    f"    [{ACCENT}]Chat #{c.id}[/]  {c.title}  [{DIM}]({c.message_count} msgs · {cdate})[/]"
                )
        else:
            target_console.print(f"    [{DIM}]No associated conversations.[/]")
        target_console.print()

        target_console.print(f"  [{DIM}]{'─' * 70}[/]")

        # Immediate playback if requested
        if play:
            try:
                from audiobench.daemon.factory import get_daemon_client

                client = get_daemon_client()
                primary_tx = tx_ids[0] if tx_ids else None
                client.playback_play(
                    audio.file_path,
                    audio_file_id=audio.id,
                    transcription_id=primary_tx,
                )
                target_console.print(f"  [{SUCCESS}][DONE] Started playback: {audio.file_name}[/]\n")
            except Exception as exc:
                target_console.print(f"  [{WARNING}]Playback error: {exc}[/]\n")
            return True

        # Interactive key handling ONLY if explicitly requested and on a TTY
        if interactive and sys.stdin.isatty():
            target_console.print(f"  [{DIM}][P] Play via daemon   [Q] Quit[/]")
            target_console.print()
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if key in ("p", "P"):
                    from audiobench.daemon.factory import get_daemon_client

                    client = get_daemon_client()
                    primary_tx = tx_ids[0] if tx_ids else None
                    client.playback_play(
                        audio.file_path,
                        audio_file_id=audio.id,
                        transcription_id=primary_tx,
                    )
                    target_console.print(f"  [{SUCCESS}][DONE] Started playback: {audio.file_name}[/]\n")
            except Exception:
                pass
        else:
            target_console.print()

        return True


@click.command(name="info")
@click.argument("target", required=False)
@click.option("-i", "--interactive", is_flag=True, help="Wait for interactive key commands ([P]lay/[Q]uit)")
@click.option("-p", "--play", is_flag=True, help="Immediately start audio playback via daemon")
def info(target: str | None, interactive: bool = False, play: bool = False) -> None:
    """Display 360-degree dossier for an audio file/transcript, or system info.

    \b
    Examples:
      audiobench info 42                 By audio file ID
      audiobench info #143               By transcript ID
      audiobench info meeting.mp3        By filename
      audiobench info                    Show system configuration & settings
      audiobench info 1196 --play        Show dossier and immediately play
    """
    if not target:
        from audiobench.cli.commands.system import render_system_info

        render_system_info()
        return

    render_file_dossier(target, console, interactive=interactive, play=play)


@click.command(name="dossier")
@click.argument("target", required=True)
@click.option("-i", "--interactive", is_flag=True, help="Wait for interactive key commands ([P]lay/[Q]uit)")
@click.option("-p", "--play", is_flag=True, help="Immediately start audio playback via daemon")
def dossier(target: str, interactive: bool = False, play: bool = False) -> None:
    """Display 360-degree file relations dossier for an audio file or transcript.

    \b
    Examples:
      audiobench dossier 42              By audio file ID
      audiobench dossier #143            By transcript ID
      audiobench dossier meeting.mp3     By filename
    """
    render_file_dossier(target, console, interactive=interactive, play=play)
