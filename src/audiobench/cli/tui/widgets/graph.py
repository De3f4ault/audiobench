"""BlockGraph — multi-row vertical bar graph, btop-style.

Uses render_line() for efficient per-row rendering.
Each column = one time step.
Each row = a height band drawn bottom-up using ▁▂▃▄▅▆▇█.
Color gradient: grey-green (low) → orange → red (high), matching btop defaults.
"""

from __future__ import annotations

from collections import deque

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widget import Widget

# Eight sub-character block levels (index = eighths filled from bottom)
_BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def _lerp_color(ratio: float) -> str:
    """Interpolate btop's cpu_start → cpu_mid → cpu_end gradient.

    ratio 0.0 = lowest value (grey-green)
    ratio 0.5 = mid  (orange-red)
    ratio 1.0 = peak (bright red)
    """
    # btop default: cpu_start=#9a9d9a  cpu_mid=#d93f37  cpu_end=#da0f0f
    if ratio <= 0.5:
        t = ratio * 2.0
        r = int(0x9a + t * (0xd9 - 0x9a))
        g = int(0x9d + t * (0x3f - 0x9d))
        b = int(0x9a + t * (0x37 - 0x9a))
    else:
        t = (ratio - 0.5) * 2.0
        r = int(0xd9 + t * (0xda - 0xd9))
        g = int(0x3f + t * (0x0f - 0x3f))
        b = int(0x37 + t * (0x0f - 0x37))
    return f"#{r:02x}{g:02x}{b:02x}"


class BlockGraph(Widget):
    """Vertical bar area graph, renders via render_line for efficiency."""

    DEFAULT_CSS = """
    BlockGraph {
        height: 8;
        background: #101010;
    }
    """

    def __init__(
        self,
        max_history: int = 500,
        bg: str = "#101010",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._history: deque[float] = deque([0.0] * max_history, maxlen=max_history)
        self._peak: float = 1.0
        self._bg = bg
        self._style_empty = Style.parse(f"on {bg}")

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, value: float) -> None:
        """Append a new data point and trigger a redraw."""
        self._history.append(value)
        # Slowly decay peak so the graph doesn't stay squashed forever
        self._peak = max(value, self._peak * 0.98, 1.0)
        self.refresh()

    @property
    def current_peak(self) -> float:
        return self._peak

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_line(self, y: int) -> Strip:
        """Called by Textual for each visible row of this widget."""
        width = self.size.width
        height = self.size.height

        if height == 0 or width == 0:
            return Strip.blank(width)

        # Right-justify: most recent value is rightmost column
        history = list(self._history)
        if len(history) < width:
            history = [0.0] * (width - len(history)) + history
        else:
            history = history[-width:]

        peak = max(self._peak, 1.0)

        # y=0 is the TOP row of the widget, y=height-1 is BOTTOM.
        # We want bars to grow upward from the bottom, so:
        row_from_bottom = height - 1 - y  # 0 = bottom row, height-1 = top row

        segments: list[Segment] = []

        for val in history:
            norm = min(val / peak, 1.0)

            # Total sub-character units this value fills (0 to height*8)
            bar_eighths = norm * height * 8.0

            # This row covers eighths [row_from_bottom*8 … (row_from_bottom+1)*8)
            row_start_e = row_from_bottom * 8
            row_end_e   = row_start_e + 8

            if bar_eighths >= row_end_e:
                # Row is fully inside the bar — solid block
                # Color by the normalized FILL RATIO at this row's midpoint
                row_mid_norm = (row_from_bottom + 0.5) / height
                color = _lerp_color(row_mid_norm * norm)
                segments.append(Segment("█", Style.parse(f"{color} on {self._bg}")))

            elif bar_eighths > row_start_e:
                # Row is at the TOP of the bar — partial block
                partial = int(bar_eighths - row_start_e)
                char = _BLOCKS[max(1, min(partial, 8))]
                row_mid_norm = (row_from_bottom + 0.5) / height
                color = _lerp_color(row_mid_norm * norm)
                segments.append(Segment(char, Style.parse(f"{color} on {self._bg}")))

            else:
                # Row is above the bar — empty
                segments.append(Segment(" ", self._style_empty))

        # Ensure we fill exactly `width` cells
        while len(segments) < width:
            segments.append(Segment(" ", self._style_empty))

        return Strip(segments[:width], width)
