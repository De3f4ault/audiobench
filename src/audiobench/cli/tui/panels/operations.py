"""OperationsPanel — active transcription job queue, btop-style."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from rich.text import Text
from textual.binding import Binding
from textual.widgets import DataTable

_STATUS_COLOR = {
    "processing": "#ffb300",  # amber
    "running":    "#00e676",  # green
    "pending":    "#64b5f6",  # blue
    "failed":     "#ef5350",  # red
    "done":       "#00e676",  # green
}

_STATUS_ICON = {
    "processing": "⟳",
    "running":    "⟳",
    "pending":    "⏳",
    "failed":     "✖",
    "done":       "✔",
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

            if jtype == "cli":
                from audiobench.jobs.repository import JobRepository
                JobRepository().cancel_job(jid)
                self.notify(f"Cancelled CLI background job #{jid}")
            elif jtype == "queue":
                from audiobench.core.settings import get_settings
                db_path = Path(get_settings().database_url.replace("sqlite:///", ""))
                conn = sqlite3.connect(str(db_path))
                conn.execute("UPDATE job_queue SET status = 'failed' WHERE id = ?", (jid,))
                conn.commit()
                conn.close()
                self.notify(f"Cancelled queued job #{jid}")

            self._refresh()

        self.app.push_screen(ConfirmModal(f"Kill {jtype} job #{jid}?"), callback=_on_confirm)

    def _refresh(self) -> None:
        try:
            from audiobench.core.settings import get_settings
            from audiobench.jobs.runner import get_job_phase

            db_path = Path(get_settings().database_url.replace("sqlite:///", ""))
            conn = sqlite3.connect(str(db_path))

            rows = conn.execute(
                "SELECT id, source, label, engine, status, started_at "
                "FROM all_jobs "
                "WHERE status IN ('running', 'pending', 'processing') "
                "ORDER BY started_at ASC LIMIT 20"
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

        for job_id, source, label, engine, status, started_at in rows:
            color = _STATUS_COLOR.get(status, "white")
            icon  = _STATUS_ICON.get(status, "?")

            short_label = label or "—"
            p = Path(short_label)
            display_label = p.name if len(str(p)) > 38 else short_label
            short_ts = (started_at or "")[:16] if started_at else "—"

            phase_raw = get_job_phase(job_id) if source == "jobs" else status
            phase = phase_raw
            progress_bar = Text("—", style="dim")

            if "%" in phase_raw:
                match = re.search(r'(\d+)%', phase_raw)
                if match:
                    pct = float(match.group(1)) / 100.0
                    progress_bar = _bar(pct)
                    phase = phase_raw.split()[0]

            self.add_row(
                Text(icon, style=f"bold {color}"),
                Text(display_label, style="#cfd8dc"),
                Text(engine or ("cli" if source == "jobs" else "—"), style="bold #90a4ae"),
                Text(phase, style=color),
                progress_bar,
                Text(short_ts, style="dim"),
                key=f"{source}:{job_id}",
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
