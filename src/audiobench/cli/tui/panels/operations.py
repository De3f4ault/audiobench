"""OperationsPanel — active transcription job queue, btop-style."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from rich.text import Text
from textual.binding import Binding
from textual.widgets import DataTable

_STATUS_COLOR = {
    "running":    "#00e676",  # green
    "pending":    "#64b5f6",  # blue
    "failed":     "#ef5350",  # red
    "done":       "#00e676",  # green
    "cancelled":  "#90a4ae",  # grey
}

_STATUS_ICON = {
    "running":    "⟳",
    "pending":    "...",
    "failed":     "X",
    "done":       "OK",
    "cancelled":  "-",
}


def _bar(pct: float, width: int = 8) -> Text:
    """Return a compact progress bar: ▮▮▮░░"""
    filled = int(pct * width)
    empty  = width - filled
    t = Text(no_wrap=True)
    t.append("▮" * filled, style="#00e676")
    t.append("░" * empty,  style="#37474f")
    return t


class OperationsPanel(DataTable):
    """Active + pending transcription jobs. Shows empty state gracefully."""

    COLUMNS = ("", "File", "Engine", "Phase", "Progress", "Started")

    BINDINGS = [
        Binding("k", "kill_job", "Kill"),
    ]

    def on_mount(self) -> None:
        self.show_cursor = True
        self.cursor_type = "row"
        for col in self.COLUMNS:
            self.add_column(col, key=col.lower() or "icon")
        self.set_interval(3.0, self._refresh)
        self._refresh()

    def _get_selected_job(self) -> tuple[str, int | None]:
        if self.cursor_row is None:
            return "", None
        try:
            row_key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
            if not row_key or ":" not in row_key.value:
                return "", None
            jtype, jid_str = row_key.value.split(":", 1)
            return jtype, int(jid_str)
        except Exception:
            return "", None

    def action_kill_job(self) -> None:
        jtype, jid = self._get_selected_job()
        if not jid:
            return

        from audiobench.cli.tui.widgets.confirm_modal import ConfirmModal

        def _on_confirm(confirm: bool) -> None:
            if not confirm:
                return

            if jtype == "unified":
                from audiobench.jobs.scheduler import cancel_job
                cancel_job(jid)
                self.notify(f"Cancelled job #{jid}")

            self._refresh()

        self.app.push_screen(ConfirmModal(f"Kill job #{jid}?"), callback=_on_confirm)

    def _refresh(self) -> None:
        try:
            from audiobench.core.settings import get_settings

            db_path = Path(get_settings().database_url.replace("sqlite:///", ""))
            conn = sqlite3.connect(str(db_path))

            # Query unified_jobs directly — all live job state lives here since 2026-09-17.
            # The old `all_jobs` view (jobs ∪ job_queue) is dropped by migration 022 and
            # never included unified_jobs entries.
            rows = conn.execute(
                "SELECT id, job_type, file_label, slot, status, started_at, command_display "
                "FROM unified_jobs "
                "WHERE status IN ('running', 'pending') "
                "ORDER BY priority ASC, id ASC LIMIT 20"
            ).fetchall()

            conn.close()
        except Exception:
            return

        # Preserve selected row key across refresh
        selected_key = None
        if self.cursor_row is not None:
            try:
                selected_key = self.coordinate_to_cell_key(
                    self.cursor_coordinate
                ).row_key.value
            except Exception:
                pass

        self.clear()

        if not rows:
            return

        for job_id, job_type, file_label, slot, status, started_at, command_display in rows:
            color = _STATUS_COLOR.get(status, "white")
            icon  = _STATUS_ICON.get(status, "?")

            raw_label = file_label or command_display or job_type or "—"
            p = Path(raw_label)
            display_label = p.name if len(str(p)) > 38 else raw_label
            short_ts = (started_at or "")[:16] if started_at else "—"

            # Engine comes from the slot suffix (e.g. "transcription:local" → "local")
            slot_parts = (slot or "").split(":")
            engine_label = slot_parts[-1] if len(slot_parts) > 1 else (slot or "—")

            phase = status
            progress_bar = Text("—", style="dim")

            self.add_row(
                Text(icon, style=f"bold {color}"),
                Text(display_label, style="#cfd8dc"),
                Text(engine_label, style="bold #90a4ae"),
                Text(phase, style=color),
                progress_bar,
                Text(short_ts, style="dim"),
                key=f"unified:{job_id}",
            )

        # Restore selection
        if selected_key:
            for i, row_key in enumerate(self.rows):
                if row_key.value == selected_key:
                    try:
                        self.move_cursor(row=i)
                    except Exception:
                        pass
                    break
