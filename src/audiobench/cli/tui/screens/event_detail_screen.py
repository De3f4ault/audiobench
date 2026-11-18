"""EventDetailScreen — full-screen event log with search and clipboard copy."""

from __future__ import annotations

import re
import sqlite3
import subprocess

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Input, Label, Static

_LEVEL_COLOR = {
    "INFO":     "#00e676",  # green
    "WARN":     "#ffb300",  # amber
    "ERROR":    "#ef5350",  # red
    "CRITICAL": "#d50000",  # bright red
}


class EventDetailScreen(Screen):
    """Full-screen event viewer with regex filtering and clipboard support."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("slash", "focus_search", "Search", key_display="/"),
        Binding("g", "scroll_top", "Top"),
        Binding("G", "scroll_bottom", "Bottom"),
        Binding("y", "copy_row", "Copy Row"),
    ]

    CSS = """
    EventDetailScreen {
        background: #101010;
    }
    #detail-header {
        height: 1;
        background: #1c1c1c;
        border-bottom: solid #2a2a2a;
    }
    #detail-title {
        width: 1fr;
        color: #cecfc9;
        text-style: bold;
        padding: 0 1;
    }
    #search-box {
        display: none;
        height: 3;
        background: #101010;
        border-bottom: solid #2a2a2a;
    }
    #search-box.-visible {
        display: block;
    }
    #search-input {
        width: 1fr;
        border: none;
        background: #1c1c1c;
        color: #cecfc9;
    }
    #detail-table {
        height: 1fr;
        background: #101010;
    }
    #detail-footer {
        height: 1;
        background: #1c1c1c;
        color: #606060;
    }
    """

    def __init__(self, subsystem: str | None = None, level: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._subsystem = subsystem
        self._level = level
        self._query = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="detail-header"):
            yield Static("◈ Event Detail View", id="detail-title")
            yield Static("[Esc] Close  [/] Search  [y] Copy  [G] Bottom", id="detail-footer")

        with Horizontal(id="search-box"):
            yield Label(" Regex: ", classes="search-label")
            yield Input(placeholder="e.g. error|warn", id="search-input")

        yield DataTable(id="detail-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Time", "Level", "Subsystem", "Event", "Message")
        self._load_data()

    def _load_data(self) -> None:
        from audiobench.observatory.db import get_journal_db_path
        db_path = get_journal_db_path()

        query = "SELECT timestamp, level, subsystem, event_type, message FROM system_events "
        params = []
        conditions = []

        if self._subsystem:
            conditions.append("subsystem = ?")
            params.append(self._subsystem)

        if self._level:
            conditions.append("level = ?")
            params.append(self._level)

        if conditions:
            query += "WHERE " + " AND ".join(conditions) + " "

        # Oldest first so bottom is newest, cap at 5000
        query += "ORDER BY timestamp DESC LIMIT 5000"

        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(query, params).fetchall()
            conn.close()
        except Exception as e:
            self.notify(f"DB Error: {e}", severity="error")
            return

        # Reverse to oldest first
        rows.reverse()

        regex = None
        if self._query:
            try:
                regex = re.compile(self._query, re.IGNORECASE)
            except re.error:
                pass

        table = self.query_one(DataTable)
        table.clear()

        for timestamp, level, subsystem, event_type, message in rows:
            if regex and not regex.search(message) and not regex.search(event_type):
                continue

            lvl_color = _LEVEL_COLOR.get(level, "white")
            table.add_row(
                Text(timestamp[11:19], style="dim"),
                Text(level, style=lvl_color),
                Text(subsystem or "—", style="#3d9a50"),
                Text(event_type or "—", style="#1c6bb0"),
                Text(message or "")
            )

        table.scroll_end(animate=False)

    def action_focus_search(self) -> None:
        box = self.query_one("#search-box")
        inp = self.query_one("#search-input", Input)
        if box.has_class("-visible"):
            box.remove_class("-visible")
            self._query = ""
            self.query_one(DataTable).focus()
            self._load_data()
        else:
            box.add_class("-visible")
            inp.value = self._query
            inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            self._query = event.value
            self._load_data()
            self.query_one(DataTable).focus()

    def action_scroll_top(self) -> None:
        self.query_one(DataTable).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one(DataTable).scroll_end(animate=False)

    def action_copy_row(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is None:
            return

        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            row = table.get_row(row_key)
            msg = row[-1].plain

            # Try Wayland
            try:
                subprocess.run(["wl-copy"], input=msg.encode(), check=True)
                self.notify("Copied to clipboard")
                return
            except Exception:
                pass

            # Try X11
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=msg.encode(), check=True)
                self.notify("Copied to clipboard")
                return
            except Exception:
                pass

            # Try xsel
            try:
                subprocess.run(["xsel", "--clipboard", "--input"], input=msg.encode(), check=True)
                self.notify("Copied to clipboard")
                return
            except Exception:
                pass

            self.notify("Clipboard tools not found (wl-copy/xclip/xsel)", severity="warning")

        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")
