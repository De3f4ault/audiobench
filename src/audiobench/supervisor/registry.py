"""Supervisor process registry.

Provides read/write access to the managed_processes table in journal.db.
Writes are performed asynchronously via the EventBus and the observatory drain thread.
Reads are synchronous via get_journal_session().
"""

from __future__ import annotations

from typing import Any

from audiobench.observatory.context import log_event
from audiobench.observatory.db import get_journal_session


def upsert_process(
    name: str,
    state: str,
    pid: int | None = None,
    **fields: Any,
) -> None:
    """Upsert process state in the registry.

    This does NOT write directly to the DB to avoid locking.
    Instead, it emits an event which the drain thread picks up
    to call `upsert_process_conn` efficiently.

    Args:
        name: Name of the process (e.g., 'daemon', 'worker')
        state: State ('running', 'stopped', 'fatal', 'backoff')
        pid: Optional OS process ID
        **fields: Additional optional fields (started_at, stopped_at,
                  restart_count, last_exit_code, last_error)
    """
    metadata = {
        "state": state,
        "pid": pid,
        **fields
    }

    # Emit event that the drain thread handles
    log_event(
        subsystem="supervisor",
        event_type="process_state_changed",
        message=f"Process '{name}' transitioned to {state.upper()}",
        level="INFO" if state in ("running", "stopped") else "WARN",
        entity_type="process",
        process=name,
        metadata=metadata,
    )


def get_process(name: str) -> dict[str, Any] | None:
    """Get the current state of a managed process."""
    with get_journal_session() as conn:
        cursor = conn.execute(
            "SELECT name, state, pid, started_at, stopped_at, restart_count, last_exit_code, last_error, updated_at "
            "FROM managed_processes WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "name": row[0],
            "state": row[1],
            "pid": row[2],
            "started_at": row[3],
            "stopped_at": row[4],
            "restart_count": row[5],
            "last_exit_code": row[6],
            "last_error": row[7],
            "updated_at": row[8],
        }


def get_all() -> list[dict[str, Any]]:
    """Get the state of all managed processes."""
    with get_journal_session() as conn:
        cursor = conn.execute(
            "SELECT name, state, pid, started_at, stopped_at, restart_count, last_exit_code, last_error, updated_at "
            "FROM managed_processes ORDER BY name ASC"
        )
        rows = cursor.fetchall()

        return [
            {
                "name": r[0],
                "state": r[1],
                "pid": r[2],
                "started_at": r[3],
                "stopped_at": r[4],
                "restart_count": r[5],
                "last_exit_code": r[6],
                "last_error": r[7],
                "updated_at": r[8],
            }
            for r in rows
        ]
