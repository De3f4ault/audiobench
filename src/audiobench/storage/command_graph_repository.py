"""Command graph repository — queries over command_events and workflows.

This module provides the intelligence layer reads:
- Pattern detection: what commands follow what commands
- Workflow CRUD: save, list, get, delete named command sequences
- Structural NL parsing: turn a ?-question into a SQLite query fast-path
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta


class CommandGraphRepository:
    """Reads and writes the command_events and workflows tables."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Command Events ────────────────────────────────────────

    def get_recent_commands(self, limit: int = 20) -> list[dict]:
        """Return the N most recent command events, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM command_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_commands_for_file(self, file_id: int, limit: int = 50) -> list[dict]:
        """All commands run while focused on a specific audio file."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM command_events WHERE context_file_id = ? ORDER BY id DESC LIMIT ?",
                (file_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_next_command_suggestions(self, after_command: str, limit: int = 5) -> list[dict]:
        """What commands usually follow `after_command`?

        Uses a simple bigram count over command_events:
        For each event with command=X, look at the next event and count by command.
        Returns list of {"command": str, "count": int} sorted by frequency.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT e2.command, COUNT(*) AS cnt
                FROM command_events e1
                JOIN command_events e2 ON e2.id = e1.id + 1
                WHERE e1.command = ?
                GROUP BY e2.command
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (after_command, limit),
            ).fetchall()
        return [{"command": r["command"], "count": r["cnt"]} for r in rows]

    def get_command_stats(self, days: int = 30) -> list[dict]:
        """Aggregate command frequency over the past N days."""
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT command, COUNT(*) AS cnt, AVG(duration_ms) AS avg_ms
                FROM command_events
                WHERE created_at >= ?
                GROUP BY command
                ORDER BY cnt DESC
                """,
                (since,),
            ).fetchall()
        return [{"command": r["command"], "count": r["cnt"], "avg_ms": r["avg_ms"]} for r in rows]

    def count_transcriptions_since(self, days: int = 7) -> int:
        """Fast-path structural answer: how many transcriptions in the last N days."""
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM command_events WHERE command='transcribe' AND created_at>=?",
                (since,),
            ).fetchone()
        return row[0] if row else 0

    # ── Workflows ─────────────────────────────────────────────

    def save_workflow(self, name: str, steps: list[dict], description: str = "") -> int:
        """Upsert a workflow. Returns the workflow ID."""
        steps_json = json.dumps(steps)
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM workflows WHERE name=?", (name,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE workflows SET steps_json=?, description=?, updated_at=? WHERE name=?",
                    (steps_json, description, now, name),
                )
                return existing["id"]
            else:
                cursor = conn.execute(
                    "INSERT INTO workflows (name, description, steps_json) VALUES (?,?,?)",
                    (name, description, steps_json),
                )
                return cursor.lastrowid

    def list_workflows(self) -> list[dict]:
        """All saved workflows, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, description, steps_json, created_at, updated_at "
                "FROM workflows ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["steps"] = json.loads(d.pop("steps_json", "[]"))
            result.append(d)
        return result

    def get_workflow(self, name: str) -> dict | None:
        """Get a workflow by name, or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM workflows WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["steps"] = json.loads(d.pop("steps_json", "[]"))
        return d

    def delete_workflow(self, name: str) -> bool:
        """Delete a workflow by name. Returns True if deleted."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM workflows WHERE name=?", (name,))
            return cursor.rowcount > 0


def get_command_graph_repo() -> CommandGraphRepository:
    """Return a CommandGraphRepository bound to the live database path."""
    from audiobench.core.settings import get_settings
    settings = get_settings()
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[len("sqlite:///"):]
    else:
        raise RuntimeError(f"CommandGraphRepository only supports SQLite, got: {db_url}")
    return CommandGraphRepository(db_path)
