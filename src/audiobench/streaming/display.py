"""Rich Live display for real-time streaming transcription.

Shows a dynamic terminal UI with:
- Recording status indicator (listening/recording/processing)
- Running transcript with each utterance
- Stats (word count, segment count, elapsed time)
"""

from __future__ import annotations

import time

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text


class LiveDisplay:
    """Rich Live renderable for streaming transcription display.

    Implements the Rich Console protocol for auto-refresh.
    """

    def __init__(self, quiet: bool = False) -> None:
        self._quiet = quiet
        self._segments: list[str] = []
        self._state: str = "waiting"  # waiting, listening, recording, processing
        self._start_time: float = 0.0
        self._live: Live | None = None

    def start(self) -> None:
        """Start the Live display."""
        self._start_time = time.perf_counter()
        if not self._quiet:
            self._live = Live(self, refresh_per_second=4, console=Console())
            self._live.start()

    def stop(self) -> None:
        """Stop the Live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def set_listening(self) -> None:
        """Show listening state (waiting for speech)."""
        self._state = "listening"

    def set_recording(self) -> None:
        """Show recording state (speech detected)."""
        self._state = "recording"

    def set_processing(self) -> None:
        """Show processing state (transcribing)."""
        self._state = "processing"

    def append_text(self, text: str) -> None:
        """Add a transcribed utterance to the display."""
        self._segments.append(text)
        self._state = "listening"

        if self._quiet:
            # In quiet mode, just print raw text
            print(text, flush=True)

    @property
    def word_count(self) -> int:
        """Total words transcribed so far."""
        return sum(len(seg.split()) for seg in self._segments)

    @property
    def elapsed(self) -> float:
        """Seconds since session started."""
        return time.perf_counter() - self._start_time if self._start_time else 0.0

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Rich Console protocol — called on each refresh cycle."""
        display = Text()

        # Status line
        elapsed_str = self._format_time(self.elapsed)

        if self._state == "listening":
            display.append("  * Listening...   ", style="bold green")
        elif self._state == "recording":
            display.append("  [REC] Recording... ", style="bold red")
        elif self._state == "processing":
            display.append("  [...] Processing...", style="bold yellow")
        else:
            display.append("  [--] Initializing...", style="dim")

        display.append(f"[{elapsed_str}]\n", style="dim")
        display.append("\n")

        # Transcript segments
        if self._segments:
            # Show last 10 segments to avoid terminal overflow
            visible = self._segments[-10:]
            start_idx = max(0, len(self._segments) - 10)

            for i, seg in enumerate(visible):
                idx = start_idx + i
                display.append(f"  {idx + 1:3d}. ", style="dim")
                display.append(f"{seg}\n")

            if len(self._segments) > 10:
                hidden = len(self._segments) - 10
                display.append(f"\n  ... {hidden} earlier segments hidden\n", style="dim")
        else:
            display.append("  Speak into your microphone...\n", style="dim italic")

        # Footer stats
        display.append("\n")
        display.append(
            f"  Words: {self.word_count} │ Segments: {len(self._segments)} │ Ctrl+C to stop\n",
            style="dim",
        )

        yield display

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
