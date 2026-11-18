"""ProcessesPanel — supervisor process table, btop-style with mini state bars."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.widgets import DataTable

# ── State colour map (truecolor) ───────────────────────────────────────────────

_STATE_COLOR = {
    "running":  "#00e676",   # vivid green
    "stopped":  "#546e7a",   # grey-blue
    "backoff":  "#ffb300",   # amber
    "fatal":    "#ef5350",   # red
    "stale":    "#ef5350",   # red
    "starting": "#40c4ff",   # cyan
    "unknown":  "#78909c",   # muted
}

_STATE_ICON = {
    "running":  "●",
    "stopped":  "○",
    "backoff":  "↺",
    "fatal":    "X",
    "stale":    "!",
    "starting": "◌",
    "unknown":  "?",
}


def _state_cell(state: str) -> Text:
    color = _STATE_COLOR.get(state, "white")
    icon = _STATE_ICON.get(state, "?")
    t = Text(no_wrap=True)
    t.append(f"{icon} ", style=f"bold {color}")
    t.append(state.upper(), style=color)
    return t


def _relative_time(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        import datetime
        then = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.UTC)
        secs = int((now - then).total_seconds())
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        else:
            return f"{secs // 3600}h{(secs % 3600) // 60}m"
    except Exception:
        return ts[:16] if ts else "—"


class ProcessesPanel(DataTable):
    """Live supervisor process table with per-process state icons."""

    COLUMNS = ("●", "Process", "PID", "CPU%", "RAM", "Restarts", "Uptime")

    BINDINGS = [
        Binding("r", "restart_process", "Restart"),
        Binding("s", "stop_process", "Stop"),
        Binding("c", "clear_fatal", "Clear Fatal"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ps_cache: dict[int, psutil.Process] = {}

    def on_mount(self) -> None:
        self.show_cursor = True
        self.cursor_type = "row"
        for col in self.COLUMNS:
            self.add_column(col, key=col.lower().replace("●", "state").replace("%", ""))
        self.set_interval(3.0, self._refresh)
        self._refresh()

    def _get_selected_process(self) -> tuple[str, int | None]:
        if self.cursor_row is None:
            return "", None
        try:
            row_key = self.coordinate_to_cell_key(self.cursor_coordinate)
            row = self.get_row(row_key.row_key)
            name = row[1].plain
            pid_str = row[2].plain
            pid = int(pid_str) if pid_str != "—" else None
            return name, pid
        except Exception:
            return "", None

    def action_restart_process(self) -> None:
        name, pid = self._get_selected_process()
        if not name:
            return

        from audiobench.cli.tui.widgets.confirm_modal import ConfirmModal

        def _on_confirm(confirm: bool) -> None:
            if not confirm:
                return

            self.notify(f"Restarting {name}...")
            import os
            import signal
            import subprocess
            import sys
            import time

            def _do_restart():
                # Direct signal to stop
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    time.sleep(1.0)
                # Spawn a fresh daemon supervisor
                subprocess.Popen([sys.executable, "-m", "audiobench", "daemon", "start"])
                self.app.call_from_thread(self.notify, f"{name} restarted.")

            self.app.run_worker(_do_restart, thread=True)

        self.app.push_screen(ConfirmModal(f"Restart process '{name}'?"), callback=_on_confirm)

    def action_stop_process(self) -> None:
        name, pid = self._get_selected_process()
        if not name or not pid:
            self.notify("Process not running or not selected.", severity="warning")
            return

        from audiobench.cli.tui.widgets.confirm_modal import ConfirmModal

        def _on_confirm(confirm: bool) -> None:
            if not confirm:
                return

            self.notify(f"Stopping {name}...")
            import os
            import signal
            import time

            from audiobench.supervisor.registry import upsert_process

            def _do_stop():
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    upsert_process(name, state="stopped", pid=None, stopped_at=time.time())
                    self.app.call_from_thread(self.notify, f"{name} stopped.")
                except ProcessLookupError:
                    upsert_process(name, state="stopped", pid=None, stopped_at=time.time())
                    self.app.call_from_thread(self.notify, f"{name} already stopped.")
                except Exception as e:
                    self.app.call_from_thread(self.notify, f"Error stopping {name}: {e}", severity="error")

            self.app.run_worker(_do_stop, thread=True)

        self.app.push_screen(ConfirmModal(f"Stop process '{name}'?"), callback=_on_confirm)

    def action_clear_fatal(self) -> None:
        name, _ = self._get_selected_process()
        if not name:
            return

        def _do_clear():
            from audiobench.supervisor.commands import clear_fatal
            clear_fatal(name)
            self.app.call_from_thread(self.notify, f"Cleared fatal state for {name}.")

        self.app.run_worker(_do_clear, thread=True)

    def _refresh(self) -> None:
        try:
            import psutil

            from audiobench.supervisor.registry import get_all
            rows = get_all()
        except Exception:
            return

        self.clear()

        # Cleanup stale processes from cache
        current_pids = {p.get("pid") for p in rows if p.get("pid")}
        self._ps_cache = {pid: proc for pid, proc in self._ps_cache.items() if pid in current_pids}

        for p in rows:
            raw_state = (p.get("state") or "unknown").lower()

            cpu_val = 0.0
            ram_val = 0

            # Validate PID is actually alive
            pid = p.get("pid")
            if raw_state == "running" and pid:
                try:
                    if not psutil.pid_exists(pid):
                        raw_state = "stale"
                    else:
                        if pid not in self._ps_cache:
                            self._ps_cache[pid] = psutil.Process(pid)
                            self._ps_cache[pid].cpu_percent(interval=None) # Seed
                        proc = self._ps_cache[pid]
                        cpu_val = proc.cpu_percent(interval=None)
                        ram_val = proc.memory_info().rss
                except Exception:
                    pass

            pid_str = str(pid) if pid else "—"
            uptime = _relative_time(p.get("updated_at"))
            restarts = p.get("restart_count") or 0

            restart_text = Text(str(restarts))
            if restarts > 3:
                restart_text.stylize("#ef5350")  # red if many restarts
            elif restarts > 0:
                restart_text.stylize("#ffb300")  # amber

            cpu_text = Text(f"{cpu_val:.1f}%" if raw_state == "running" else "—")
            if cpu_val >= 70.0:
                cpu_text.stylize("#ef5350")
            elif cpu_val >= 30.0:
                cpu_text.stylize("#ffb300")
            elif raw_state == "running":
                cpu_text.stylize("#00e676")

            from audiobench.cli.tui.panels.health import _fmt
            ram_text = Text(_fmt(ram_val) if raw_state == "running" else "—", style="dim")

            self.add_row(
                _state_cell(raw_state),
                Text(p["name"], style="bold #cfd8dc"),
                Text(pid_str, style="dim"),
                cpu_text,
                ram_text,
                restart_text,
                Text(uptime, style="dim"),
            )

