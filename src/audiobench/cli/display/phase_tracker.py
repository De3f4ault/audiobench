"""Phase tracker — Rich Live progress display for transcription phases.

Renders a two-mode progress display:
1. Live mode — animated spinners during loading/converting
2. Streaming mode — static phases at top, transcript segments growing below

Usage:
    from audiobench.cli.display.phase_tracker import PhaseTracker

    tracker = PhaseTracker()
    tracker.start()
    tracker.update("loading", "Loading model...", None)
    tracker.on_segment(segment)
    tracker.finalize()
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Detect non-interactive environments that need forced flushing
_FORCE_FLUSH = not sys.stdout.isatty()

from rich.live import Live
from rich.text import Text

from audiobench.cli.display.theme import ACCENT, DIM, SUCCESS, console, format_duration


class PhaseTracker:
    """Renders phased progress using Rich Live display.

    Uses a two-mode approach:
    1. **Live mode** — During loading/converting, a Rich Live display
       shows animated spinners and progress. Uses transient=True so
       the frame vanishes when stopped.
    2. **Streaming mode** — When the first transcript segment arrives,
       Live stops, completed phases print statically at the top, and
       each new segment prints below. Text grows downward in real-time.

    The result: phases stay at the top, transcript builds below,
    summary appears at the very bottom when done.
    """

    PHASES = ["loading", "converting", "uploading", "processing", "transcribing", "aligning", "diarizing", "saving", "embedding"]
    LABELS = {
        "loading": "Loading model",
        "converting": "Converting audio",
        "uploading": "Uploading",
        "processing": "Processing upload",
        "transcribing": "Transcribing",
        "aligning": "Aligning timestamps",
        "diarizing": "Diarizing speakers",
        "saving": "Saving",
        "embedding": "Generating embeddings",
    }
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, quiet: bool = False, events_file: str | Path | None = None) -> None:
        self.quiet = quiet
        self.events_file = Path(events_file) if events_file else None
        self.phase_times: dict[str, float] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0
        self._last_progress: float = 0
        self._spin_idx: int = 0
        # Accumulated segments for live preview + partial save
        self.segments: list = []
        # Rich Live display — handles smooth in-place terminal updates
        self._live: Live | None = None
        # Whether we've switched to streaming mode
        self._streaming: bool = False

    def _emit_event(self, data: dict) -> None:
        if not self.events_file:
            return
        data.setdefault("ts", datetime.now(UTC).isoformat())
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass

    @property
    def _visible_phases(self) -> list[str]:
        """Return phases to display, hiding optional phases if never used."""
        hidden_unless_used = {"uploading", "processing", "aligning", "diarizing"}
        return [
            p
            for p in self.PHASES
            if (p not in hidden_unless_used and (p != "loading" or "loading" in self.phase_times or p == self._current_phase))
            or p in self.phase_times
            or p == self._current_phase
        ]

    def start(self) -> None:
        """Start the Rich Live display. Call before first update."""
        if not self.quiet:
            if sys.stdout.isatty():
                self._live = Live(
                    self,
                    console=console,
                    refresh_per_second=4,
                    transient=True,  # Frame vanishes when stopped
                    redirect_stdout=True,
                    redirect_stderr=True,
                )
                self._live.start()

    def _enter_streaming(self) -> None:
        """Transition from Live mode to streaming mode.

        Stops the Live display (frame vanishes due to transient=True),
        then prints completed phases as static text. After this,
        segments print below via regular console.print().
        """
        if self._streaming:
            return
        self._streaming = True

        # Stop Live — transient=True means the frame disappears cleanly
        if self._live:
            self._live.stop()
            self._live = None

            # Print phases statically at the top ONLY if we had a Live display that was cleared
            for phase in self._visible_phases:
                label = self.LABELS.get(phase, phase)
                if phase in self.phase_times:
                    elapsed_str = format_duration(self.phase_times[phase])
                    console.print(f"  [{SUCCESS}]✓[/]  {label:<24} [{DIM}]{elapsed_str}[/]")
                elif phase == self._current_phase:
                    console.print(f"  [{ACCENT}]◐[/]  {label}...")
                else:
                    console.print(f"  [{DIM}]·  {label}[/]")

        # Blank line separating phases from transcript text
        console.print()

    def on_segment(self, segment: object) -> None:
        """Called after each segment is transcribed.

        On first call, switches to streaming mode (phases at top).
        Then prints each segment below, growing the transcript.
        """
        self._emit_event({
            "t": "segment",
            "text": getattr(segment, "text", ""),
            "start": getattr(segment, "start", 0.0),
            "end": getattr(segment, "end", 0.0),
            "speaker": getattr(segment, "speaker", None),
        })
        self.segments.append(segment)
        if self.quiet:
            return

        # First segment → switch to streaming mode
        if not self._streaming:
            self._enter_streaming()

        self._print_segment(segment)

    def update(self, phase: str, message: str, progress: float | None) -> None:
        """Called by the pipeline on phase transitions."""
        self._emit_event({
            "t": "phase",
            "phase": phase,
            "message": message,
            "progress": progress,
        })
        if self.quiet:
            return

        # Record timing for previous phase
        if self._current_phase and self._current_phase != phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        if phase != self._current_phase:
            self._current_phase = phase
            self._phase_start = time.perf_counter()
            # Static fallback for environments without Live
            if not self._live and not self._streaming:
                label = self.LABELS.get(phase, phase)
                console.print(f"  [{ACCENT}]▶[/]  {label}...")
                if _FORCE_FLUSH:
                    sys.stdout.flush()

        if progress is not None:
            self._last_progress = progress

    def _build_display(self) -> Text:
        """Build the Live display (loading/converting phases only)."""

        self._spin_idx = (self._spin_idx + 1) % len(self.SPINNERS)
        spinner = self.SPINNERS[self._spin_idx]

        display = Text()
        for phase in self._visible_phases:
            label = self.LABELS.get(phase, phase)

            if phase in self.phase_times:
                # ✓ Completed
                elapsed_str = format_duration(self.phase_times[phase])
                display.append("  ✓", style=SUCCESS)
                display.append(f"  {label:<24} ", style="")
                display.append(elapsed_str, style=DIM)
                display.append("\n")
            elif phase == self._current_phase:
                # ⠼ Active with spinner
                display.append(f"  {spinner}", style=ACCENT)
                display.append(f"  {label}", style="")
                display.append("...", style=DIM)
                display.append("\n")
            else:
                # · Pending
                display.append("  ·", style=DIM)
                display.append(f"  {label}", style=DIM)
                display.append("\n")

        return display

    def __rich_console__(self, rconsole, options):
        """Rich renderable protocol — called every refresh cycle."""
        yield self._build_display()

    def abort(self, error: str = "") -> None:
        """Record failure event and stop display without emitting done event."""
        self._emit_event({"t": "error", "message": error})
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def finalize(self) -> None:
        """Record final timing and print completion summary."""
        self._emit_event({"t": "done"})
        if self.quiet:
            return

        if self._current_phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        # Stop Live display if still running
        if self._live:
            self._live.stop()
            self._live = None

        # If we never entered streaming mode AND we had a Live display, print everything now.
        # This covers cases like Gemini where segments arrive all at once.
        # If we didn't have a Live display, phases were already printed statically in update().
        if not self._streaming:
            if self._live:
                # Print phase status lines
                for phase in self._visible_phases:
                    label = self.LABELS.get(phase, phase)
                    if phase in self.phase_times:
                        elapsed_str = format_duration(self.phase_times[phase])
                        console.print(f"  [{SUCCESS}]✓[/]  {label:<24} [{DIM}]{elapsed_str}[/]")
                    else:
                        console.print(f"  [{DIM}]·  {label}[/]")
            else:
                # We just need to mark the final phase as completed if we used static prints
                if self._current_phase in self.phase_times:
                    elapsed_str = format_duration(self.phase_times[self._current_phase])
                    label = self.LABELS.get(self._current_phase, self._current_phase)
                    console.print(f"  [{SUCCESS}]✓[/]  {label} finished in {elapsed_str}")

            # Print transcript text if we have any
            if self.segments:
                console.print()
                for seg in self.segments:
                    self._print_segment(seg)

    def _print_segment(self, segment: object) -> None:
        """Print a single segment with timestamp coloring."""
        text = getattr(segment, "text", "").strip()
        start = getattr(segment, "start", 0.0)
        end = getattr(segment, "end", 0.0)
        if text:
            ts = self._format_ts(start, end)
            console.print(f"  [{DIM}]{ts}[/]  {text}", highlight=False)
            if _FORCE_FLUSH:
                sys.stdout.flush()

    @staticmethod
    def _format_ts(start: float, end: float) -> str:
        """Format start/end as [MM:SS → MM:SS]."""
        s_m, s_s = int(start // 60), int(start % 60)
        e_m, e_s = int(end // 60), int(end % 60)
        return f"[{s_m}:{s_s:02d} → {e_m}:{e_s:02d}]"

    def save_partial(self, input_path: str) -> str | None:
        """Save accumulated segments to a .partial.txt file."""
        if not self.segments:
            return None
        import hashlib

        from audiobench.core.settings import get_settings

        input_p = Path(input_path)
        key = str(input_p.absolute()).encode("utf-8")
        hash_prefix = hashlib.sha256(key).hexdigest()[:12]

        checkpoints_dir = get_settings().data_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        partial_path = str(checkpoints_dir / f"{hash_prefix}.partial.txt")
        lines = []
        for seg in self.segments:
            start = getattr(seg, "start", 0)
            text = getattr(seg, "text", "")
            minutes = int(start // 60)
            seconds = int(start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {text}")
        Path(partial_path).write_text("\n".join(lines), encoding="utf-8")
        return partial_path
