from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from audiobench.daemon.client import DaemonClient


class InferencesFeed(Static):
    """Observatory panel for viewing and actioning system inferences.

    Polling: refreshes every 10s via the Observatory app's tick.

    Key bindings:
        [c]     Confirm selected inference
        [r]     Reject selected inference
        [Enter] Expand — shows full content; for graph_edge_event rows shows
                a structured two-file similarity display

    Columns:
        ID         Expression ID in the database
        Type       source_type (graph_edge_event, system_inference, etc.)
        File A     Source file name (only for graph_edge_event rows)
        File B     Target file name (only for graph_edge_event rows)
        Sim        Cosine similarity (only for graph_edge_event rows)
        Content    Content preview (first 80 chars) for non-edge rows

    The selected row's expression_id is tracked in _selected_id so action
    methods can call the daemon without re-querying the table widget.
    """

    BINDINGS = [
        Binding("c", "confirm_inference", "Confirm Inference", key_display="C"),
        Binding("r", "reject_inference", "Reject Inference", key_display="R"),
        Binding("enter", "expand_content", "View Details", key_display="Enter"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = DaemonClient()
        self._selected_id: int | None = None
        self._rows: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("System Inferences", id="inferences_header"),
            DataTable(id="inferences_table"),
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Type", "File A", "File B", "Sim", "Content")
        table.cursor_type = "row"
        self._refresh_feed()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the selected row's expression_id."""
        if event.row_key is not None:
            try:
                self._selected_id = int(str(event.row_key.value))
            except (ValueError, TypeError):
                self._selected_id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _basename(path: str) -> str:
        """Return just the filename stem, no directory, trimmed to 30 chars."""
        import os
        name = os.path.basename(path)
        # Strip leading hash prefix (e.g. "03956224_Alan Watts...")
        if "_" in name[:9]:
            name = name[name.index("_") + 1:]
        # Remove extension
        name = os.path.splitext(name)[0]
        return name[:30]

    def _build_row(self, inf: dict) -> tuple[str, ...]:
        """Build the DataTable row tuple for one inference row."""
        sid = str(inf["id"])
        stype = inf.get("source_type", "")

        if stype == "graph_edge_event":
            try:
                payload = json.loads(inf.get("content", "{}"))
                file_a = self._basename(payload.get("file_a", ""))
                file_b = self._basename(payload.get("file_b", ""))
                sim = "%.3f" % float(payload.get("sim", 0))
                return (sid, stype, file_a, file_b, sim, "")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Non-edge row: put content in the Content column
        preview = (inf.get("content") or "")[:80].replace("\n", " ")
        return (sid, stype, "", "", "", preview)

    def _refresh_feed(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        try:
            self._rows = self._client.get_inferences()
        except Exception:
            self._rows = []

        for inf in self._rows:
            cols = self._build_row(inf)
            table.add_row(*cols, key=str(inf["id"]))

        if self._rows and self._selected_id is None:
            self._selected_id = self._rows[0]["id"]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_confirm_inference(self) -> None:
        if self._selected_id is None:
            self.notify("No inference selected", severity="warning")
            return
        try:
            self._client.confirm_inference(self._selected_id)
            self.notify("Confirmed inference #%d" % self._selected_id, timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify("Error: %s" % e, severity="error")

    def action_reject_inference(self) -> None:
        if self._selected_id is None:
            self.notify("No inference selected", severity="warning")
            return
        try:
            self._client.reject_inference(self._selected_id)
            self.notify("Rejected inference #%d" % self._selected_id, timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify("Error: %s" % e, severity="error")

    def action_expand_content(self) -> None:
        """Show full details for the selected inference row."""
        if self._selected_id is None:
            return
        for inf in self._rows:
            if inf["id"] == self._selected_id:
                stype = inf.get("source_type", "")
                if stype == "graph_edge_event":
                    try:
                        payload = json.loads(inf.get("content", "{}"))
                        sim = float(payload.get("sim", 0))
                        msg = (
                            "SEMANTIC CONNECTION (sim=%.3f)\n"
                            "From: %s\n"
                            "  ... %s ...\n"
                            "To: %s\n"
                            "  ... %s ...\n"
                            "Expr IDs: #%s <-> #%s"
                        ) % (
                            sim,
                            payload.get("file_a", ""),
                            payload.get("preview_a", "")[:200],
                            payload.get("file_b", ""),
                            payload.get("preview_b", "")[:200],
                            payload.get("from_id", "?"),
                            payload.get("to_id", "?"),
                        )
                        self.notify(msg, timeout=12)
                        return
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                # Fallback: raw content
                self.notify(inf.get("content", ""), timeout=8)
                return
