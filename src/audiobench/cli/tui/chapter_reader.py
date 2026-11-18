"""ChapterReaderApp — Textual TUI for synchronized audio + transcript reading.

Local mode (Phase 3a): runs in-process with PlaybackService.  The reader
mounts alongside the daemon, reads the hot position cache at 20 Hz (cheap
scalar reads — no IPC), and receives EOF events via a registered callback
pushed from the mpv reader thread.

Architecture:
  - 20 Hz set_interval timer drives position display and SYNC_LOCK auto-scroll
  - PlaybackService.register_eof_listener() + call_from_thread drives EOF push
  - All interactive keypresses use _send_command_nowait (fire-and-forget, ~0ms)
  - SelectionState / Visual Mode: vim-inspired range selection for bookmarking
  - Two scroll modes: SYNC_LOCK (follows playhead) and DECOUPLED (free cursor)
  - Paragraph-boundary jumping: J/K jump to segment boundaries ~silence gaps
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Static
from textual.containers import Vertical, Horizontal
from textual import work

from audiobench.core.logger_factory import get_logger
from audiobench.playback.lyrics_hud import _normalize_segments, _find_active_segment
from audiobench.playback.controls import fmt_timestamp

logger = get_logger("tui.chapter_reader")

# ── Scroll modes ──────────────────────────────────────────────────────────────

class ScrollMode(Enum):
    SYNC_LOCK  = auto()   # cursor follows playhead automatically
    DECOUPLED  = auto()   # cursor moves freely; playhead tracks silently

# ── Visual selection state ────────────────────────────────────────────────────

@dataclass
class SelectionState:
    """Tracks the active visual range selection (vim-style)."""
    active: bool = False
    anchor: int = -1       # segment index where 'v' was pressed
    cursor: int = -1       # current cursor end of selection

    @property
    def lo(self) -> int:
        return min(self.anchor, self.cursor) if self.active else -1

    @property
    def hi(self) -> int:
        return max(self.anchor, self.cursor) if self.active else -1

    def start(self, idx: int) -> None:
        self.active = True
        self.anchor = idx
        self.cursor = idx

    def extend(self, idx: int) -> None:
        if self.active:
            self.cursor = idx

    def clear(self) -> None:
        self.active = False
        self.anchor = -1
        self.cursor = -1

# ── Messages ──────────────────────────────────────────────────────────────────

class PlaybackEnded(Message):
    """Posted from the mpv reader thread (via call_from_thread) on natural EOF."""
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
Screen {
    background: #0e0e0e;
    color: #cecfc9;
}

/* ── Header bar ── */
#header-bar {
    height: 1;
    background: #1a1a1a;
    layout: horizontal;
}
#header-title {
    width: 1fr;
    color: #88aacc;
    text-style: bold;
    content-align: left middle;
    padding: 0 1;
}
#header-status {
    width: auto;
    color: #3d9a50;
    content-align: right middle;
    padding: 0 2;
}
#header-mode {
    width: 14;
    content-align: right middle;
    padding: 0 1;
}

/* ── Progress bar ── */
#progress-bar {
    height: 1;
    background: #141414;
    color: #5a7a9a;
    content-align: left middle;
    padding: 0 1;
}

/* ── Segment scroll area ── */
#scroll-area {
    height: 1fr;
    overflow: hidden;
}
#segments-view {
    height: auto;
    padding: 0 1;
}

/* ── Visual mode badge ── */
#visual-badge {
    height: 1;
    background: #2a3a2a;
    color: #88cc88;
    text-style: bold;
    content-align: center middle;
    display: none;
}
#visual-badge.active {
    display: block;
}

/* ── Help overlay ── */
#help-overlay {
    background: #1a1a1a 90%;
    color: #cecfc9;
    border: solid #444444;
    width: 62;
    height: auto;
    align: center middle;
    display: none;
    padding: 1 2;
}
#help-overlay.visible {
    display: block;
}

Footer {
    background: #1a1a1a;
    color: #606060;
    height: 1;
}
Footer > .footer--key {
    background: #2a2a2a;
    color: #88aacc;
}
"""

# ── Segment renderer ──────────────────────────────────────────────────────────

def _render_segments(
    segments: list[dict],
    cursor: int,
    active_idx: int,
    selection: SelectionState,
    mode: ScrollMode,
    width: int,
) -> str:
    """Render the visible segment window as a Rich markup string.

    Window: cursor ± 12 segments (25-line window).  The active playhead
    segment gets a cyan arrow; prior segments are dim; upcoming are dim+italic.
    Visual selection highlights the selected range in green.
    """
    if not segments:
        return "  [dim]No transcript segments available.[/dim]"

    win_half = 12
    lo = max(0, cursor - win_half)
    hi = min(len(segments), cursor + win_half + 1)
    lines: list[str] = []

    if lo > 0:
        lines.append(f"[dim]  ⋮  ({lo} segments above)[/dim]")

    sel_lo = selection.lo if selection.active else -1
    sel_hi = selection.hi if selection.active else -1

    for i in range(lo, hi):
        seg = segments[i]
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        ts = fmt_timestamp(seg.get("norm_start", seg.get("start", 0.0)))
        spk = f"[{seg['speaker']}] " if seg.get("speaker") else ""

        is_active  = (i == active_idx)
        is_cursor  = (i == cursor)
        in_sel     = selection.active and sel_lo <= i <= sel_hi

        if in_sel:
            # Selected range — green background equivalent via markup
            prefix = " ◈ " if is_cursor else "   "
            lines.append(f"[bold on #1a3a1a]{prefix}{ts}  {spk}{text}[/bold on #1a3a1a]")
        elif is_active and is_cursor:
            lines.append(f"[bold cyan] ▶ {ts}  {spk}{text}[/bold cyan]")
        elif is_active:
            lines.append(f"[bold cyan] ▸ {ts}  {spk}{text}[/bold cyan]")
        elif is_cursor:
            lines.append(f"[bold white] › {ts}  {spk}{text}[/bold white]")
        elif i < active_idx:
            lines.append(f"[dim]   {ts}  {spk}{text}[/dim]")
        else:
            lines.append(f"[dim italic]   {ts}  {spk}{text}[/dim italic]")

    if hi < len(segments):
        lines.append(f"[dim]  ⋮  ({len(segments) - hi} segments below)[/dim]")

    return "\n".join(lines)


def _progress_bar(pos: float, dur: float, speed: float, paused: bool, width: int) -> str:
    """Render a one-line progress bar as Rich markup."""
    icon = "⏸" if paused else "▶"
    bar_w = max(10, min(40, width - 32))
    pct = min(pos / dur, 1.0) if dur > 0 else 0.0
    filled = int(pct * bar_w)
    if filled < bar_w:
        bar = "━" * filled + "●" + "─" * max(0, bar_w - filled - 1)
    else:
        bar = "━" * bar_w
    spd_str = f"  {speed:.2g}×" if speed != 1.0 else ""
    pos_str = fmt_timestamp(pos)
    dur_str = fmt_timestamp(dur)
    return f"{icon}  {pos_str} / {dur_str}   {bar}{spd_str}"


# ── Main app ──────────────────────────────────────────────────────────────────

class ChapterReaderApp(App):
    """Interactive synchronized transcript reader for AudioBench.

    Local mode (Phase 3a): attaches to a live PlaybackService in-process.
    Runs as a blocking ``app.run()`` call; audio continues when the TUI exits.

    Args:
        service:           Live PlaybackService (local mode).
        title:             Display title (book/chapter name).
        segments:          Pre-normalized segment dicts (from lyrics_hud._normalize_segments).
        transcription_id:  DB ID for bookmark writes.
        audio_file_id:     DB ID for chapter lookups.
    """

    CSS = CSS
    TITLE = "AudioBench Reader"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        # ── Playback ──
        Binding("space",      "toggle_pause",    "Pause",       priority=True),
        Binding("left",       "seek_back_5",     "←5s",         priority=True),
        Binding("right",      "seek_fwd_5",      "→5s",         priority=True),
        Binding("shift+left", "seek_back_30",    "←30s",        priority=True, show=False),
        Binding("shift+right","seek_fwd_30",     "→30s",        priority=True, show=False),
        Binding("[",          "speed_down",      "Speed-",      priority=True),
        Binding("]",          "speed_up",        "Speed+",      priority=True),
        # ── Cursor ──
        Binding("j",          "cursor_down",     "↓",           priority=True, show=False),
        Binding("k",          "cursor_up",       "↑",           priority=True, show=False),
        Binding("J",          "para_down",       "Para↓",       priority=True, show=False),
        Binding("K",          "para_up",         "Para↑",       priority=True, show=False),
        Binding("g",          "cursor_top",      "Top",         priority=True, show=False),
        Binding("G",          "cursor_bottom",   "Bottom",      priority=True, show=False),
        # ── Mode ──
        Binding("s",          "toggle_sync",     "Sync",        priority=True),
        Binding("return",     "seek_to_cursor",  "Seek here",   priority=True, show=False),
        # ── Visual selection ──
        Binding("v",          "visual_toggle",   "Visual",      priority=True, show=False),
        Binding("escape",     "visual_cancel",   "Cancel",      priority=True, show=False),
        # ── Loop ──
        Binding("l",          "loop_cursor",     "Loop seg",    priority=True, show=False),
        Binding("L",          "clear_loop",      "Clear loop",  priority=True, show=False),
        # ── Bookmark ──
        Binding("b",          "bookmark_cursor", "Bookmark",    priority=True, show=False),
        # ── UI ──
        Binding("?",          "toggle_help",     "Help",        priority=True),
        Binding("q",          "quit_reader",     "Quit",        priority=True),
    ]

    # Reactive so Textual can auto-invalidate on change
    _cursor: reactive[int] = reactive(0)
    _scroll_mode: reactive[ScrollMode] = reactive(ScrollMode.SYNC_LOCK)

    def __init__(
        self,
        service: Any,
        title: str = "AudioBench Reader",
        segments: list[dict] | None = None,
        transcription_id: int | None = None,
        audio_file_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._service = service
        self._display_title = title
        self._segments: list[dict] = segments or []
        self._transcription_id = transcription_id
        self._audio_file_id = audio_file_id

        # Playback state — updated by 20 Hz tick
        self._pos: float = 0.0
        self._dur: float = 0.0
        self._speed: float = 1.0
        self._paused: bool = False
        self._active_idx: int = 0

        # Visual selection
        self._sel = SelectionState()

        # Help overlay visible
        self._help_visible: bool = False

        # Unregister handle returned by register_eof_listener
        self._eof_unregister: Callable[[], None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # 1-row header
        with Horizontal(id="header-bar"):
            yield Static(f"◈  {self._display_title}", id="header-title")
            yield Static("", id="header-status")
            yield Static("SYNC", id="header-mode")

        # Progress bar
        yield Static("", id="progress-bar")

        # Visual mode badge (hidden until visual mode active)
        yield Static("", id="visual-badge")

        # Segment scroll area
        with Vertical(id="scroll-area"):
            yield Static("", id="segments-view", markup=True)

        # Help overlay (hidden until ?)
        yield Static(self._help_text(), id="help-overlay", markup=True)

        yield Footer()

    def on_mount(self) -> None:
        # Seed initial playback state
        self._sync_from_service()

        # Position the cursor at the active segment
        self._cursor = max(0, self._active_idx)

        # 20 Hz timer — reads hot in-memory cache, no IPC
        self.set_interval(1 / 20, self._tick)

        # Push-based EOF: register with PlaybackService
        def _eof_cb(reason: str) -> None:
            self.call_from_thread(self.post_message, PlaybackEnded(reason))

        if hasattr(self._service, "register_eof_listener"):
            self._eof_unregister = self._service.register_eof_listener(_eof_cb)

        if type(self._service).__name__ == "DaemonPlaybackProxy":
            self.notify("Attached to daemon playback (2 Hz telemetry sync)", timeout=3)

        self._refresh_ui()

    def on_unmount(self) -> None:
        if self._eof_unregister is not None:
            self._eof_unregister()
            self._eof_unregister = None

    # ── 20 Hz tick ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Read cached playback state and repaint. ~0μs — all scalar reads."""
        self._sync_from_service()
        if self._scroll_mode == ScrollMode.SYNC_LOCK and self._segments:
            new_cursor = self._active_idx
            if new_cursor != self._cursor:
                self._cursor = new_cursor
        self._refresh_ui()

    def _sync_from_service(self) -> None:
        """Pull current position from service (zero-cost cache read in local mode)."""
        try:
            st = self._service.status()
            self._pos    = float(st.get("position", 0.0))
            self._dur    = float(st.get("duration",  0.0))
            self._speed  = float(st.get("speed",     1.0))
            self._paused = bool(st.get("paused",     False))
            if self._segments:
                self._active_idx = _find_active_segment(self._segments, self._pos)
        except Exception:
            pass

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        w = self.size.width or 80

        # Progress bar
        self.query_one("#progress-bar", Static).update(
            _progress_bar(self._pos, self._dur, self._speed, self._paused, w)
        )

        # Segments
        self.query_one("#segments-view", Static).update(
            _render_segments(
                self._segments,
                self._cursor,
                self._active_idx,
                self._sel,
                self._scroll_mode,
                w,
            )
        )

        # Mode badge
        mode_label = "SYNC" if self._scroll_mode == ScrollMode.SYNC_LOCK else "FREE"
        mode_color = "green" if self._scroll_mode == ScrollMode.SYNC_LOCK else "yellow"
        self.query_one("#header-mode", Static).update(
            f"[{mode_color}]{mode_label}[/{mode_color}]"
        )

        # Visual badge
        badge = self.query_one("#visual-badge", Static)
        if self._sel.active:
            n = self._sel.hi - self._sel.lo + 1
            badge.update(f"── VISUAL  {n} segment{'s' if n != 1 else ''} selected  [b] bookmark  [Esc] cancel ──")
            badge.add_class("active")
        else:
            badge.remove_class("active")

        # Status
        seg_info = f"{self._cursor + 1}/{len(self._segments)}" if self._segments else "—"
        self.query_one("#header-status", Static).update(f"seg {seg_info}")

    # ── EOF message handler ───────────────────────────────────────────────────

    def on_playback_ended(self, msg: PlaybackEnded) -> None:
        """Received from mpv reader thread via call_from_thread."""
        if msg.reason == "eof":
            self.notify("Track ended — audio stopped.", severity="information")
            # Exit the reader on natural EOF; audio is handled by PlaybackService
            self.call_later(self.action_quit_reader)

    # ── Playback actions ──────────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        try:
            self._service.toggle()
        except Exception:
            pass

    def action_seek_back_5(self) -> None:
        try:
            self._service.seek_relative(-5.0)
        except Exception:
            pass

    def action_seek_fwd_5(self) -> None:
        try:
            self._service.seek_relative(5.0)
        except Exception:
            pass

    def action_seek_back_30(self) -> None:
        try:
            self._service.seek_relative(-30.0)
        except Exception:
            pass

    def action_seek_fwd_30(self) -> None:
        try:
            self._service.seek_relative(30.0)
        except Exception:
            pass

    def action_speed_down(self) -> None:
        try:
            new_spd = max(0.25, round(self._speed * 0.9, 2))
            self._service.set_speed(new_spd)
        except Exception:
            pass

    def action_speed_up(self) -> None:
        try:
            new_spd = min(4.0, round(self._speed * 1.1, 2))
            self._service.set_speed(new_spd)
        except Exception:
            pass

    def action_loop_cursor(self) -> None:
        """Loop the segment under the cursor (or selection range)."""
        try:
            if self._sel.active:
                lo_seg = self._segments[self._sel.lo]
                hi_seg = self._segments[self._sel.hi]
                start = lo_seg.get("norm_start", lo_seg.get("start", 0.0))
                end   = hi_seg.get("norm_end",   hi_seg.get("end",   0.0))
            else:
                seg   = self._segments[self._cursor]
                start = seg.get("norm_start", seg.get("start", 0.0))
                end   = seg.get("norm_end",   seg.get("end",   0.0))
            self._service.loop_segment(start, end)
            self.notify(f"Loop: {fmt_timestamp(start)} → {fmt_timestamp(end)}")
        except Exception as exc:
            self.notify(f"Loop failed: {exc}", severity="error")

    def action_clear_loop(self) -> None:
        try:
            self._service.clear_loop()
            self.notify("Loop cleared.")
        except Exception:
            pass

    def action_seek_to_cursor(self) -> None:
        """Seek to the start of the cursor segment."""
        try:
            seg = self._segments[self._cursor]
            pos = seg.get("norm_start", seg.get("start", 0.0))
            self._service.seek(pos)
        except Exception:
            pass

    # ── Cursor movement ───────────────────────────────────────────────────────

    def _move_cursor(self, delta: int) -> None:
        new = max(0, min(len(self._segments) - 1, self._cursor + delta))
        self._cursor = new
        if self._sel.active:
            self._sel.extend(new)
        if self._scroll_mode == ScrollMode.SYNC_LOCK:
            self._scroll_mode = ScrollMode.DECOUPLED

    def action_cursor_down(self) -> None:
        self._move_cursor(1)
        self._refresh_ui()

    def action_cursor_up(self) -> None:
        self._move_cursor(-1)
        self._refresh_ui()

    def action_cursor_top(self) -> None:
        self._cursor = 0
        if self._sel.active:
            self._sel.extend(0)
        self._scroll_mode = ScrollMode.DECOUPLED
        self._refresh_ui()

    def action_cursor_bottom(self) -> None:
        self._cursor = max(0, len(self._segments) - 1)
        if self._sel.active:
            self._sel.extend(self._cursor)
        self._scroll_mode = ScrollMode.DECOUPLED
        self._refresh_ui()

    def action_para_down(self) -> None:
        """Jump forward to next paragraph boundary (silence gap >= 1.5s or speaker transition)."""
        idx = self._cursor + 1
        segs = self._segments
        while idx < len(segs) - 1:
            gap = segs[idx]["norm_start"] - segs[idx - 1]["norm_end"]
            spk_curr = segs[idx].get("speaker")
            spk_prev = segs[idx - 1].get("speaker")
            spk_changed = bool(spk_curr or spk_prev) and (spk_curr != spk_prev)
            if gap >= 1.5 or spk_changed:
                break
            idx += 1
        self._cursor = min(idx, len(segs) - 1)
        if self._sel.active:
            self._sel.extend(self._cursor)
        self._scroll_mode = ScrollMode.DECOUPLED
        self._refresh_ui()

    def action_para_up(self) -> None:
        """Jump backward to previous paragraph boundary (silence gap >= 1.5s or speaker transition)."""
        idx = self._cursor - 1
        segs = self._segments
        while idx > 0:
            gap = segs[idx]["norm_start"] - segs[idx - 1]["norm_end"]
            spk_curr = segs[idx].get("speaker")
            spk_prev = segs[idx - 1].get("speaker")
            spk_changed = bool(spk_curr or spk_prev) and (spk_curr != spk_prev)
            if gap >= 1.5 or spk_changed:
                break
            idx -= 1
        self._cursor = max(0, idx)
        if self._sel.active:
            self._sel.extend(self._cursor)
        self._scroll_mode = ScrollMode.DECOUPLED
        self._refresh_ui()

    # ── Scroll mode ───────────────────────────────────────────────────────────

    def action_toggle_sync(self) -> None:
        if self._scroll_mode == ScrollMode.SYNC_LOCK:
            self._scroll_mode = ScrollMode.DECOUPLED
        else:
            self._scroll_mode = ScrollMode.SYNC_LOCK
            # Snap cursor to current playhead immediately
            self._cursor = self._active_idx
        self._refresh_ui()

    # ── Visual selection ──────────────────────────────────────────────────────

    def action_visual_toggle(self) -> None:
        if self._sel.active:
            self._sel.clear()
        else:
            self._sel.start(self._cursor)
        self._refresh_ui()

    def action_visual_cancel(self) -> None:
        if self._sel.active:
            self._sel.clear()
            self._refresh_ui()
        elif self._help_visible:
            self.action_toggle_help()

    # ── Bookmarking ───────────────────────────────────────────────────────────

    def action_bookmark_cursor(self) -> None:
        """Bookmark the cursor segment or visual selection range."""
        try:
            af_id = self._audio_file_id
            if not af_id and self._transcription_id:
                from audiobench.core.db_session import get_session
                from audiobench.storage.models import TranscriptionRecord

                with get_session() as session:
                    tx = session.query(TranscriptionRecord).filter_by(id=self._transcription_id).first()
                    if tx and tx.audio_file_id:
                        af_id = tx.audio_file_id

            if not af_id:
                self.notify("Cannot bookmark: no audio file ID associated", severity="error")
                return

            from audiobench.storage.bookmark_repository import BookmarkRepository
            repo = BookmarkRepository()

            if self._sel.active and self._sel.lo != self._sel.hi:
                lo_seg = self._segments[self._sel.lo]
                hi_seg = self._segments[self._sel.hi]
                start_pos = lo_seg.get("norm_start", lo_seg.get("start", 0.0))
                end_pos   = hi_seg.get("norm_end",   hi_seg.get("end",   0.0))
                lo_text   = (lo_seg.get("text") or "").strip()
                hi_text   = (hi_seg.get("text") or "").strip()
                label     = f"{lo_text[:30]} ... {hi_text[:30]}" if lo_text and hi_text else "Selection Region"
                n         = self._sel.hi - self._sel.lo + 1

                repo.add_region(
                    audio_file_id=af_id,
                    start=start_pos,
                    end=end_pos,
                    name=label,
                    bookmark_type="highlight",
                    transcription_id=self._transcription_id,
                )
                self._sel.clear()
                self.notify(f"Bookmarked region ({n} segs) @ {fmt_timestamp(start_pos)} → {fmt_timestamp(end_pos)}")
            else:
                idx = self._sel.lo if self._sel.active else self._cursor
                seg = self._segments[idx]
                pos = seg.get("norm_start", seg.get("start", 0.0))
                label = (seg.get("text") or "")[:60].strip() or f"Bookmark @ {fmt_timestamp(pos)}"
                repo.add(
                    audio_file_id=af_id,
                    timestamp=pos,
                    name=label,
                    bookmark_type="bookmark",
                    transcription_id=self._transcription_id,
                )
                self._sel.clear()
                self.notify(f"Bookmarked segment @ {fmt_timestamp(pos)}")
            self._refresh_ui()
        except Exception as exc:
            self.notify(f"Bookmark failed: {exc}", severity="error")

    # ── Help ──────────────────────────────────────────────────────────────────

    def action_toggle_help(self) -> None:
        self._help_visible = not self._help_visible
        overlay = self.query_one("#help-overlay", Static)
        if self._help_visible:
            overlay.add_class("visible")
        else:
            overlay.remove_class("visible")

    def _help_text(self) -> str:
        return (
            "[bold]AudioBench Reader — Key Bindings[/bold]\n\n"
            "[bold cyan]Playback[/bold cyan]\n"
            "  [bold]Space[/bold]          Pause / Resume\n"
            "  [bold]← / →[/bold]         Seek ±5 seconds\n"
            "  [bold]Shift+← / →[/bold]   Seek ±30 seconds\n"
            "  [bold]Enter[/bold]          Seek to cursor segment\n"
            "  [bold][ / ][/bold]          Speed −/+ 10%\n\n"
            "[bold cyan]Navigation[/bold cyan]\n"
            "  [bold]j / k[/bold]          Cursor down / up\n"
            "  [bold]J / K[/bold]          Jump to next/prev paragraph boundary\n"
            "  [bold]g / G[/bold]          Jump to top / bottom\n"
            "  [bold]s[/bold]              Toggle SYNC_LOCK ↔ DECOUPLED mode\n\n"
            "[bold cyan]Visual Selection (vim-style)[/bold cyan]\n"
            "  [bold]v[/bold]              Start / toggle visual selection\n"
            "  [bold]j / k / J / K[/bold]  Extend selection\n"
            "  [bold]Esc[/bold]            Cancel selection\n\n"
            "[bold cyan]Loop & Bookmark[/bold cyan]\n"
            "  [bold]l[/bold]              Loop cursor segment (or selection)\n"
            "  [bold]L[/bold]              Clear A-B loop\n"
            "  [bold]b[/bold]              Bookmark cursor (or selection)\n\n"
            "[bold cyan]App[/bold cyan]\n"
            "  [bold]?[/bold]              Toggle this help\n"
            "  [bold]q[/bold]              Quit reader (audio keeps playing)\n\n"
            "[dim]Note: {/} keys flag as provisional — see Phase 3 keybinding table.[/dim]\n"
            "[dim]Press [bold]?[/bold] or [bold]Esc[/bold] to close.[/dim]"
        )

    # ── Quit ──────────────────────────────────────────────────────────────────

    def action_quit_reader(self) -> None:
        """Exit the TUI — audio continues playing in the background daemon."""
        self.exit()


# ── Factory function ──────────────────────────────────────────────────────────

def open_reader(
    service: Any,
    *,
    title: str = "AudioBench Reader",
    transcription_id: int | None = None,
    audio_file_id: int | None = None,
    raw_segments: list[dict] | None = None,
) -> None:
    """Launch the ChapterReaderApp for a currently-playing track.

    Loads and normalizes segments from the DB if ``raw_segments`` is not given.
    Blocks until the user quits; audio continues in the background daemon.

    Args:
        service:           Live PlaybackService (local mode).
        title:             Display title (shown in header).
        transcription_id:  If given, segments are loaded from DB.
        audio_file_id:     DB ID for chapter seed lookups.
        raw_segments:      Pre-loaded segment dicts (skips DB load if provided).
    """
    segments: list[dict] = []

    if raw_segments is not None:
        segments = _normalize_segments(raw_segments)
    elif transcription_id:
        try:
            from audiobench.core.db_session import get_session
            from audiobench.storage.models import SegmentRecord

            with get_session() as session:
                rows = (
                    session.query(SegmentRecord)
                    .filter_by(transcription_id=transcription_id)
                    .order_by(SegmentRecord.segment_index)
                    .all()
                )
                raw = [
                    {"id": r.id, "start": r.start_time, "end": r.end_time,
                     "text": r.text, "speaker": r.speaker}
                    for r in rows
                ]
            segments = _normalize_segments(raw)
        except Exception as exc:
            logger.warning("open_reader: could not load segments: %s", exc)

    app = ChapterReaderApp(
        service=service,
        title=title,
        segments=segments,
        transcription_id=transcription_id,
        audio_file_id=audio_file_id,
    )
    app.run()
