"""CLI theme — shared visual constants and helpers.

Provides a consistent look & feel across all CLI commands:
- Color palette
- Panel builders
- Duration / file size formatters
- Status indicators
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# ── Shared console ──────────────────────────────────────────
console = Console(stderr=True)  # Chrome → stderr (keeps stdout clean for piping)
stdout = Console()  # Raw text → stdout

# ── Chat Markdown theme ────────────────────────────────────
CHAT_THEME = Theme(
    {
        "markdown.h1": "bold bright_cyan underline",
        "markdown.h2": "bold bright_cyan",
        "markdown.h3": "bold cyan",
        "markdown.h4": "bold dim cyan",
        "markdown.code": "bright_green on grey11",
        "markdown.code_inline": "bright_green on grey11",
        "markdown.bold": "bold bright_white",
        "markdown.italic": "italic magenta",
        "markdown.block_quote": "italic bright_yellow",
        "markdown.link": "underline bright_blue",
        "markdown.link_url": "dim bright_blue",
        "markdown.item.bullet": "bright_cyan",
        "markdown.hr": "dim cyan",
        "markdown.paragraph": "white",
    }
)
CHAT_CODE_THEME = "dracula"

chat_console = Console(stderr=True, theme=CHAT_THEME)

# ── Color palette ───────────────────────────────────────────
ACCENT = "cyan"
SUCCESS = "green"
PROMPT = "bright_green"
WARNING = "yellow"
ERROR = "red"
DIM = "dim"
BOLD = "bold"

# ── Phase indicators ───────────────────────────────────────
PHASE_DONE = f"[{SUCCESS}]  ✓[/]"
PHASE_ACTIVE = f"[{ACCENT}]  ◐[/]"
PHASE_PENDING = f"[{DIM}]  ○[/]"
PHASE_ERROR = f"[{ERROR}]  ✗[/]"

# ── App branding ───────────────────────────────────────────
APP_NAME = "AudioBench"
APP_VERSION = "0.1.0"


def app_header(subtitle: str = "") -> Panel:
    """Render the app header panel."""
    content = f"[{BOLD} {ACCENT}]{APP_NAME}[/]"
    if subtitle:
        content += f"\n[{DIM}]{subtitle}[/]"
    return Panel(content, border_style=DIM, expand=False)


def summary_panel(lines: list[str], title: str = "Summary") -> Panel:
    """Build a consistent summary panel."""
    content = "\n".join(lines)
    return Panel(content, title=f"[{BOLD}]{title}[/]", border_style=DIM, expand=False)


def error_panel(message: str, detail: str = "") -> Panel:
    """Build an error panel."""
    content = f"[{ERROR}]✗ {message}[/]"
    if detail:
        content += f"\n[{DIM}]{detail}[/]"
    return Panel(content, border_style=ERROR, expand=False)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration.

    Examples:
        0.5    → "0.5s"
        45.2   → "45s"
        125.7  → "2m 5s"
        3661.0 → "1h 1m 1s"
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"

    hours = minutes // 60
    mins = minutes % 60
    parts = [f"{hours}h"]
    if mins:
        parts.append(f"{mins}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts)


def format_size(bytes_: int) -> str:
    """Format byte count into human-readable size.

    Examples:
        1024       → "1.0 KB"
        7002672    → "6.7 MB"
        1073741824 → "1.0 GB"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}" if unit != "B" else f"{bytes_} B"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def make_table(title: str, columns: list[tuple[str, dict]]) -> Table:
    """Create a consistently styled table.

    Args:
        title: Table title.
        columns: List of (name, kwargs) for add_column.
    """
    table = Table(title=title, border_style=DIM, title_style=BOLD)
    for name, kwargs in columns:
        table.add_column(name, **kwargs)
    return table


# ── Format extension mapping ───────────────────────────────
EXT_TO_FORMAT = {
    ".txt": "txt",
    ".srt": "srt",
    ".vtt": "vtt",
    ".json": "json",
}

FORMAT_TO_EXT = {v: k for k, v in EXT_TO_FORMAT.items()}


def detect_format_from_path(path: str) -> str | None:
    """Auto-detect output format from file extension."""
    from pathlib import Path

    ext = Path(path).suffix.lower()
    return EXT_TO_FORMAT.get(ext)
