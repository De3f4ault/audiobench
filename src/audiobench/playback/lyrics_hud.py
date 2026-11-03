"""Universal Live Lyrics / Karaoke Follow HUD for AudioBench daemon playback.

Provides:
  - show_daemon_lyrics(client, console, tx_repo=None): interactive terminal HUD
    synchronized with currently playing daemon audio. Exiting with 'q' or Esc
    leaves audio playing in the background.
  - _normalize_segments(raw_segments): smooths and interpolates clumped timestamps.
  - _find_active_segment(segments, effective_pos): accurate timestamp-for-timestamp matching.
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from audiobench.cli.display.theme import ACCENT, DIM, ERROR, SUCCESS, WARNING
from audiobench.playback.controls import fmt_timestamp

# In-memory persistent cache for user calibration offsets per transcription/file
_SYNC_CALIBRATION_CACHE: dict[int | str, float] = {}


def _normalize_segments(raw_segments: list[dict]) -> list[dict]:
    """Normalize and interpolate transcript segments to guarantee smooth sync.

    1. Filters out blank/empty segments.
    2. Identifies clusters of consecutive segments with identical or overlapping start times
       (common in chunked Gemini/Whisper outputs where multiple sentences share a chunk timestamp).
    3. Proportionally distributes the cluster's duration based on text length, giving each
       segment distinct, monotonic [norm_start, norm_end] timestamps.
    4. Ensures norm_start < norm_end and eliminates degenerate 0.0s intervals.
    """
    if not raw_segments:
        return []

    cleaned: list[dict] = []
    for s in raw_segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        cleaned.append(
            {
                "id": s.get("id"),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": txt,
                "speaker": s.get("speaker"),
            }
        )

    if not cleaned:
        return []

    # Sort primarily by start time, maintaining original sequence for ties
    cleaned.sort(key=lambda s: s["start"])

    # Group into clusters of overlapping or identical start times
    clusters: list[list[dict]] = []
    current_cluster: list[dict] = [cleaned[0]]

    for seg in cleaned[1:]:
        prev = current_cluster[-1]
        # Same start time (within 0.2s) or clustered chunk
        if abs(seg["start"] - prev["start"]) < 0.2:
            current_cluster.append(seg)
        else:
            clusters.append(current_cluster)
            current_cluster = [seg]
    clusters.append(current_cluster)

    normalized: list[dict] = []
    for cluster in clusters:
        if len(cluster) == 1:
            seg = cluster[0]
            start = seg["start"]
            end = max(seg["end"], start + 0.5)
            seg["norm_start"] = start
            seg["norm_end"] = end
            normalized.append(seg)
        else:
            # Clustered segments: distribute span proportionally to text length
            cluster_start = cluster[0]["start"]
            cluster_end = max(s["end"] for s in cluster)
            total_chars = sum(max(1, len(s["text"])) for s in cluster)

            # Ensure minimum sensible span for multi-sentence cluster
            min_span = max(1.5, len(cluster) * 0.8)
            if cluster_end <= cluster_start + 0.5:
                cluster_end = cluster_start + min_span
            span = cluster_end - cluster_start

            curr_t = cluster_start
            for s in cluster:
                weight = max(1, len(s["text"])) / total_chars
                dur = max(0.4, span * weight)
                s["norm_start"] = curr_t
                s["norm_end"] = curr_t + dur
                curr_t += dur
                normalized.append(s)

    # Final pass to ensure strictly monotonic start times and smoothed transitions
    for i in range(len(normalized) - 1):
        if normalized[i]["norm_end"] > normalized[i + 1]["norm_start"]:
            mid = (normalized[i]["norm_end"] + normalized[i + 1]["norm_start"]) / 2
            if mid > normalized[i]["norm_start"]:
                normalized[i]["norm_end"] = mid
                normalized[i + 1]["norm_start"] = mid

    return normalized


def _find_active_segment(segments: list[dict], effective_pos: float) -> int:
    """Find the index of the currently active segment for the given playback position.

    Accurately handles:
    - Exact interval matches [norm_start, norm_end].
    - Silence gaps between segments: smoothly transitions ~0.6s before the next segment starts
      to anticipate speech onset.
    - Out-of-bounds positions (before start, past end).
    """
    if not segments:
        return -1

    if effective_pos <= segments[0]["norm_start"]:
        return 0

    if effective_pos >= segments[-1]["norm_end"]:
        return len(segments) - 1

    starts = [s["norm_start"] for s in segments]
    idx = bisect.bisect_right(starts, effective_pos) - 1
    idx = max(0, min(idx, len(segments) - 1))

    cur = segments[idx]
    if cur["norm_start"] <= effective_pos <= cur["norm_end"]:
        # If in the last 0.4s and next segment is imminent, advance
        if idx + 1 < len(segments):
            nxt = segments[idx + 1]
            if effective_pos >= nxt["norm_start"] - 0.4:
                return idx + 1
        return idx

    # In silence gap between cur["norm_end"] and next["norm_start"]
    if idx + 1 < len(segments):
        nxt = segments[idx + 1]
        time_to_next = nxt["norm_start"] - effective_pos
        gap_dur = nxt["norm_start"] - cur["norm_end"]
        if time_to_next <= 0.8 or (gap_dur > 0 and time_to_next <= gap_dur / 2):
            return idx + 1

    return idx


def _load_segments_for_playback(
    st: dict[str, Any], tx_repo: Any = None, client: Any = None
) -> tuple[int | None, str, list[dict]]:
    """Resolve (transcription_id, title, segments) for currently playing audio."""
    tid = st.get("transcription_id")
    af_id = st.get("audio_file_id")
    file_p = st.get("file") or ""
    title = os.path.basename(file_p) if file_p else "Active Playback"

    from audiobench.storage.models import SegmentRecord, TranscriptionRecord
    from audiobench.core.db_session import get_session

    # 1. Try finding transcription_id if missing
    if tid is None and af_id is not None:
        try:
            with get_session() as session:
                rec = (
                    session.query(TranscriptionRecord)
                    .filter_by(audio_file_id=af_id)
                    .order_by(TranscriptionRecord.id.desc())
                    .first()
                )
                if rec:
                    tid = rec.id
                    if rec.file_name:
                        title = rec.file_name
        except Exception:
            pass

    if tid is None and file_p:
        try:
            from audiobench.storage.models import AudioFileRecord

            with get_session() as session:
                af = (
                    session.query(AudioFileRecord)
                    .filter_by(file_path=os.path.abspath(file_p))
                    .first()
                )
                if af:
                    rec = (
                        session.query(TranscriptionRecord)
                        .filter_by(audio_file_id=af.id)
                        .order_by(TranscriptionRecord.id.desc())
                        .first()
                    )
                    if rec:
                        tid = rec.id
                        if rec.file_name:
                            title = rec.file_name
        except Exception:
            pass

    # 2. Load segments if tid is found
    raw_segments: list[dict] = []
    if tid is not None:
        try:
            with get_session() as session:
                rows = (
                    session.query(SegmentRecord)
                    .filter_by(transcription_id=tid)
                    .order_by(SegmentRecord.segment_index)
                    .all()
                )
                for r in rows:
                    raw_segments.append(
                        {
                            "id": r.id,
                            "start": r.start_time,
                            "end": r.end_time,
                            "text": r.text,
                            "speaker": r.speaker,
                        }
                    )
        except Exception:
            pass

    # 3. If no DB segments found, query daemon context for custom in-memory segments (e.g. /speak)
    if not raw_segments and client is not None and hasattr(client, "playback_context"):
        try:
            ctx = client.playback_context(position=0.0, window=10000)
            if ctx and ctx.get("segments"):
                raw_segments = ctx["segments"]
                if "speech" in file_p or "voice" in file_p:
                    title = "Assistant Voice"
        except Exception:
            pass

    segments = _normalize_segments(raw_segments)
    return tid, title, segments


def _build_lyrics_panel(
    title: str,
    current_time: float,
    total_duration: float,
    paused: bool,
    current_speed: float,
    segments: list[dict],
    current_idx: int,
    console_width: int,
    sync_offset: float = 0.0,
) -> Panel:
    """Build Rich renderable panel with progress bar and scrolling lyrics."""
    # ── Progress bar ──
    pause_icon = "||" if paused else ">"
    bar_width = max(18, min(36, console_width - 40))
    pct = min(current_time / total_duration, 1.0) if total_duration > 0 else 0.0
    filled = int(pct * bar_width)
    if filled < bar_width:
        bar = "━" * filled + "●" + "─" * max(0, bar_width - filled - 1)
    else:
        bar = "━" * bar_width

    time_str = f"{pause_icon}  {fmt_timestamp(current_time)} / {fmt_timestamp(total_duration)}"
    spd_str = f"  {current_speed:.2g}×" if current_speed != 1.0 else ""
    hdr_line = f"{time_str}   {bar}{spd_str}"

    hdr_text = Text(hdr_line)
    hdr_text.stylize("bold cyan" if paused else "white")
    if current_speed != 1.0:
        start = len(hdr_line) - len(spd_str)
        hdr_text.stylize("yellow", start=start)

    # ── Scrolling segments window ──
    lines = Text()
    if not segments:
        lines.append("  [No synchronized transcript segments available]\n", style="dim")
    else:
        win_start = max(0, current_idx - 3)
        win_end = min(len(segments), current_idx + 4)

        if win_start > 0:
            lines.append("⋮\n", style="dim")

        for i in range(win_start, win_end):
            seg = segments[i]
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            ts = fmt_timestamp(seg.get("norm_start", seg.get("start", 0.0)))
            spk = f"[{seg['speaker']}] " if seg.get("speaker") else ""

            if i == current_idx:
                lines.append(f" ▸ {ts}  {spk}{text}\n", style="bold cyan")
            elif i < current_idx:
                lines.append(f"   {ts}  {spk}{text}\n", style="dim")
            else:
                lines.append(f"   {ts}  {spk}{text}\n", style="dim italic")

        if win_end < len(segments):
            lines.append("⋮\n", style="dim")

    # ── Header & Action bar ──
    panel_title = f"[bold]Live Lyrics Follow · {title}[/bold]"
    if sync_offset != 0.0:
        panel_title += f" [bold yellow]({sync_offset:+.1f}s sync)[/bold yellow]"

    footer = Text(
        " [Space] Pause  ·  [←/→] ±5s  ·  [↑/↓] ±30s  ·  [[ / ]] Speed  ·  [< / >] Sync ±0.5s  ·  [0] Reset  ·  [q] Exit",
        style="dim",
    )

    group = Group(
        Align.center(hdr_text),
        Text(""),
        lines,
        Text(""),
        Align.center(footer),
    )

    return Panel(
        group,
        title=panel_title,
        border_style="cyan" if not paused else "yellow",
        padding=(0, 1),
    )


def show_daemon_lyrics(
    client: Any,
    console: Console,
    tx_repo: Any = None,
) -> None:
    """Launch universal interactive follow / lyrics HUD for daemon playback.

    Synchronizes continuously with daemon playback position.
    Exiting via 'q' or Esc restores the prompt while leaving audio playing in background.
    Supports real-time sync offset calibration (< / > or - / +) persisted per track.
    """
    if client is None:
        console.print(f"  [{WARNING}]Daemon client is not connected.[/]")
        return

    st = client.playback_status()
    if not st.get("playing"):
        console.print(
            f"  [{DIM}]No audio currently playing. Start playback first with /play <id|timestamp>.[/]"
        )
        return

    tid, title, segments = _load_segments_for_playback(st, tx_repo, client=client)
    active_file = st.get("file")
    active_tid = tid
    cache_key: int | str = tid if tid is not None else (st.get("file") or "default")
    sync_offset = _SYNC_CALIBRATION_CACHE.get(cache_key, 0.0)

    if not sys.stdin.isatty():
        # Non-interactive fallback (e.g. testing or redirected input)
        pos = st.get("position", 0.0)
        eff_pos = max(0.0, pos + sync_offset)
        idx = _find_active_segment(segments, eff_pos)
        p = _build_lyrics_panel(
            title,
            pos,
            st.get("duration", 0.0),
            st.get("paused", False),
            st.get("speed", 1.0),
            segments,
            max(0, idx),
            console.size.width,
            sync_offset=sync_offset,
        )
        console.print(p)
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        last_sync_time = time.monotonic()
        force_sync = False
        sync_interval = 1.0  # seconds between routine daemon status heartbeats

        with Live(console=console, refresh_per_second=15, transient=True) as live:
            while True:
                # 1. Non-blocking key handling (wait up to 60ms for smooth 15-16Hz loop)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.06)
                now = time.monotonic()
                if rlist:
                    while select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)

                        # Exit HUD: keep audio playing!
                        if ch in ("q", "Q"):
                            return

                        # Space: toggle pause
                        elif ch == " ":
                            st = client.playback_toggle()
                            last_sync_time = time.monotonic()

                        # Speed controls: [ ]
                        elif ch == "[":
                            cur_spd = float(st.get("speed", 1.0))
                            st = client.playback_speed(max(0.25, round(cur_spd * 0.9, 2)))
                            last_sync_time = time.monotonic()
                        elif ch == "]":
                            cur_spd = float(st.get("speed", 1.0))
                            st = client.playback_speed(min(4.0, round(cur_spd * 1.1, 2)))
                            last_sync_time = time.monotonic()

                        # Stop audio explicitly if requested
                        elif ch in ("s", "x"):
                            client.playback_stop()
                            return

                        # Real-time sync offset adjustments:
                        # <, -, , -> delay lyrics by 0.5s
                        # >, +, = -> advance lyrics by 0.5s
                        # 0, r, R -> reset calibration to 0.0s
                        elif ch in ("<", "-", ","):
                            sync_offset = round(sync_offset - 0.5, 2)
                            _SYNC_CALIBRATION_CACHE[cache_key] = sync_offset
                        elif ch in (">", "+", "="):
                            sync_offset = round(sync_offset + 0.5, 2)
                            _SYNC_CALIBRATION_CACHE[cache_key] = sync_offset
                        elif ch in ("0", "r", "R"):
                            sync_offset = 0.0
                            _SYNC_CALIBRATION_CACHE[cache_key] = 0.0

                        # Arrow keys & Escape sequences
                        elif ch == "\x1b":
                            if select.select([sys.stdin], [], [], 0.05)[0]:
                                seq1 = sys.stdin.read(1)
                                if seq1 == "[":
                                    seq2 = sys.stdin.read(1)
                                    if seq2 == "C":  # Right arrow -> +5s
                                        st = client.playback_seek_relative(5.0)
                                    elif seq2 == "D":  # Left arrow -> -5s
                                        st = client.playback_seek_relative(-5.0)
                                    elif seq2 == "A":  # Up arrow -> +30s
                                        st = client.playback_seek_relative(30.0)
                                    elif seq2 == "B":  # Down arrow -> -30s
                                        st = client.playback_seek_relative(-30.0)
                                    last_sync_time = time.monotonic()
                            else:
                                # Bare Escape: exit HUD
                                return

                # 2. Query daemon status only on routine heartbeat or explicit force_sync
                now = time.monotonic()
                if force_sync or (now - last_sync_time >= sync_interval):
                    st = client.playback_status()
                    last_sync_time = now
                    force_sync = False

                    if st.get("playing"):
                        cur_f = st.get("file")
                        cur_t = st.get("transcription_id")
                        if cur_f != active_file or cur_t != active_tid:
                            tid, title, segments = _load_segments_for_playback(st, tx_repo, client=client)
                            active_file = cur_f
                            active_tid = tid
                            cache_key = tid if tid is not None else (active_file or "default")
                            sync_offset = _SYNC_CALIBRATION_CACHE.get(cache_key, 0.0)

                if not st.get("playing"):
                    # Audio naturally ended or stopped; linger briefly before exiting
                    time.sleep(0.5)
                    break

                # 3. Dead-reckoning position interpolation
                base_pos = float(st.get("position", 0.0))
                dur = float(st.get("duration", 0.0))
                paused = bool(st.get("paused", False))
                speed = float(st.get("speed", 1.0))

                if paused:
                    pos = base_pos
                else:
                    elapsed = max(0.0, now - last_sync_time)
                    pos = base_pos + elapsed * speed
                    if dur > 0.0:
                        if pos >= dur:
                            pos = dur
                            force_sync = True  # Natural track end reached: verify on next frame
                        else:
                            pos = min(dur, pos)

                # 4. Find active segment with calibrated offset
                effective_pos = max(0.0, pos + sync_offset)
                current_idx = _find_active_segment(segments, effective_pos)

                # 5. Render frame
                panel = _build_lyrics_panel(
                    title,
                    pos,
                    dur,
                    paused,
                    speed,
                    segments,
                    current_idx,
                    console.size.width,
                    sync_offset=sync_offset,
                )
                live.update(panel)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Check if playback is still active to inform the user
        try:
            cur_st = client.playback_status()
            if cur_st.get("playing"):
                pos_str = fmt_timestamp(cur_st.get("position", 0.0))
                state_str = "Paused" if cur_st.get("paused") else "Playing"
                sync_badge = (
                    f" [{sync_offset:+.1f}s sync]" if sync_offset != 0.0 else ""
                )
                console.print(
                    f"  [{ACCENT}]Lyrics follow closed · audio continuing in background ({state_str} @ {pos_str}{sync_badge})[/]"
                )
        except Exception:
            pass
