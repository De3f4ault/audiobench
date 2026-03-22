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

import time
from pathlib import Path

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

    PHASES = ["loading", "converting", "uploading", "transcribing", "saving"]
    LABELS = {
        "loading": "Loading model",
        "converting": "Converting audio",
        "uploading": "Uploading",
        "transcribing": "Transcribing",
        "saving": "Saving",
    }
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
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

    @property
    def _visible_phases(self) -> list[str]:
        """Return phases to display, hiding 'uploading' if never used."""
        return [
            p
            for p in self.PHASES
            if p != "uploading" or p in self.phase_times or p == self._current_phase
        ]

    def start(self) -> None:
        """Start the Rich Live display. Call before first update."""
        if not self.quiet:
            self._live = Live(
                self,
                console=console,
                refresh_per_second=10,
                transient=True,  # Frame vanishes when stopped
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

        # Print phases statically at the top
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
        self.segments.append(segment)
        if self.quiet:
            return

        # First segment → switch to streaming mode
        if not self._streaming:
            self._enter_streaming()

        self._print_segment(segment)

    def update(self, phase: str, message: str, progress: float | None) -> None:
        """Called by the pipeline on phase transitions."""
        if self.quiet:
            return

        # Record timing for previous phase
        if self._current_phase and self._current_phase != phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        if phase != self._current_phase:
            self._current_phase = phase
            self._phase_start = time.perf_counter()

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

    def finalize(self) -> None:
        """Record final timing and print completion summary."""
        if self.quiet:
            return

        if self._current_phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        # Stop Live display if still running
        if self._live:
            self._live.stop()
            self._live = None

        # If we never entered streaming mode, print everything now.
        # This covers two cases:
        #   1. No segments at all (e.g. error before transcription)
        #   2. Segments arrived all at once (Gemini) — on_segment added
        #      them to self.segments but _enter_streaming's console.print
        #      got swallowed by the Live display's transient cleanup.
        if not self._streaming:
            # Print phase status lines
            for phase in self._visible_phases:
                label = self.LABELS.get(phase, phase)
                if phase in self.phase_times:
                    elapsed_str = format_duration(self.phase_times[phase])
                    console.print(f"  [{SUCCESS}]✓[/]  {label:<24} [{DIM}]{elapsed_str}[/]")
                else:
                    console.print(f"  [{DIM}]·  {label}[/]")

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
        partial_path = str(Path(input_path).with_suffix(".partial.txt"))
        lines = []
        for seg in self.segments:
            start = getattr(seg, "start", 0)
            text = getattr(seg, "text", "")
            minutes = int(start // 60)
            seconds = int(start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {text}")
        Path(partial_path).write_text("\n".join(lines), encoding="utf-8")
        return partial_path
