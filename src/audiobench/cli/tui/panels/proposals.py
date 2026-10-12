import json
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from audiobench.daemon.client import DaemonClient


class ProposalsFeed(Static):
    """Observatory panel for viewing and actioning daemon proposals.

    Polling: refreshes every 10s via the Observatory app's tick.
    Key bindings:
        [a]  Authorize selected proposal → calls DaemonClient.authorize_proposal()
        [d]  Defer proposal              → calls DaemonClient.defer_proposal()
        [r]  Reject selected proposal    → calls DaemonClient.reject_proposal()

    Proposals are rejected via reject_proposal() without debiting
    audio inference calibration stats.
    """

    BINDINGS = [
        Binding("a", "authorize_proposal", "Authorize", key_display="A"),
        Binding("d", "defer_proposal", "Defer (30d)", key_display="D"),
        Binding("r", "reject_proposal", "Reject", key_display="R"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = DaemonClient()
        self._selected_id: int | None = None
        self._rows: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Daemon Proposals", id="proposals_header"),
            DataTable(id="proposals_table")
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Type", "Status", "Title / Summary")
        table.cursor_type = "row"
        self._refresh_feed()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the selected row's expression_id."""
        if event.row_key is not None:
            try:
                self._selected_id = int(str(event.row_key.value))
            except (ValueError, TypeError):
                self._selected_id = None

    def _refresh_feed(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        try:
            self._rows = self._client.get_proposals()
        except Exception:
            self._rows = []

        for prop in self._rows:
            content = (prop.get("content") or "").strip()
            prop_type = "legacy"
            preview = content[:80].replace("\n", " ")
            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        prop_type = data.get("proposal_type", "operator" if "operator_template" in data else "proposal")
                        preview = data.get("title") or data.get("summary") or preview
                except Exception:
                    pass

            table.add_row(
                str(prop["id"]),
                prop_type,
                prop.get("inference_status", "proposed"),
                preview,
                key=str(prop["id"]),
            )

        if self._rows and self._selected_id is None:
            self._selected_id = self._rows[0]["id"]

    def action_authorize_proposal(self) -> None:
        if self._selected_id is None:
            self.notify("No proposal selected", severity="warning")
            return
        try:
            res = self._client.authorize_proposal(self._selected_id)
            action = res.get("action") if isinstance(res, dict) else "authorized"
            if action == "authorized":
                self.notify(f"Authorized proposal #{self._selected_id} — operator hot-registered", timeout=3)
            else:
                self.notify(f"Acknowledged proposal #{self._selected_id}", timeout=3)
            self._refresh_feed()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_defer_proposal(self) -> None:
        if self._selected_id is None:
            self.notify("No proposal selected", severity="warning")
            return
        try:
            self._client.defer_proposal(self._selected_id)
            self.notify(f"Deferred proposal #{self._selected_id} (30d)", timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_reject_proposal(self) -> None:
        if self._selected_id is None:
            self.notify("No proposal selected", severity="warning")
            return
        try:
            self._client.reject_proposal(self._selected_id)
            self.notify(f"Rejected proposal #{self._selected_id}", timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
