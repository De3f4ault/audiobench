"""LanceDB lifecycle optimization logic for the audiobench daemon.

This module provides functions to run compaction and cleanup on LanceDB tables,
and to manage state indicating when optimization was last run.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import lancedb

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("daemon.lancedb_optimizer")


def _get_state_file_path() -> Path:
    settings = get_settings()
    return settings.lancedb_path / ".optimize_state.json"


def read_optimize_state() -> dict[str, Any]:
    """Read the optimization state including timestamp and write counter."""
    state_file = _get_state_file_path()
    state = {
        "last_optimized_at": None,
        "unoptimized_writes": 0,
    }
    if not state_file.exists():
        return state
    try:
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)
            last_opt = data.get("last_optimized_at")
            if last_opt:
                # Ensure it's timezone-aware if the string has a Z or +00:00
                if last_opt.endswith("Z"):
                    last_opt = last_opt[:-1] + "+00:00"
                state["last_optimized_at"] = datetime.datetime.fromisoformat(last_opt)
            state["unoptimized_writes"] = data.get("unoptimized_writes", 0)
    except Exception as e:
        logger.warning("Failed to read optimize state: %s", e)
    return state


def increment_unoptimized_writes(count: int) -> int:
    """Atomically increment the unoptimized writes counter and return the new total."""
    state_file = _get_state_file_path()
    temp_file = state_file.with_suffix(".json.tmp")

    # Read existing
    data = {"unoptimized_writes": 0}
    if state_file.exists():
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    new_total = data.get("unoptimized_writes", 0) + count
    data["unoptimized_writes"] = new_total

    # Write back
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, state_file)
    except Exception as e:
        logger.error("Failed to write optimize state: %s", e)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

    return new_total


def write_last_optimized(triggered_by: str) -> None:
    """Write the current time as the last successful optimization and reset writes counter."""
    state_file = _get_state_file_path()
    temp_file = state_file.with_suffix(".json.tmp")
    now_str = datetime.datetime.now(datetime.UTC).isoformat()
    data = {
        "last_optimized_at": now_str,
        "triggered_by": triggered_by,
        "unoptimized_writes": 0, # Reset to 0 on successful optimize
    }
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, state_file)
    except Exception as e:
        logger.error("Failed to write optimize state: %s", e)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


def _should_optimize_on_startup(interval_days: int) -> bool:
    """Determine if optimization is needed based on the startup check policy."""
    if interval_days < 0:
        return False

    state = read_optimize_state()
    last_optimized = state["last_optimized_at"]
    if last_optimized is None:
        return True

    now = datetime.datetime.now(datetime.UTC)
    delta = now - last_optimized
    return delta.total_seconds() >= (interval_days * 86400)


def _get_dir_size(path: Path) -> int:
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception as e:
        logger.warning("Failed to calculate directory size for %s: %s", path, e)
    return total_size


def _optimize_with_retry(table: Any, table_name: str, max_attempts: int = 3) -> None:
    """Optimize a table with exponential backoff on retryable commit conflicts."""
    for attempt in range(1, max_attempts + 1):
        try:
            table.optimize(cleanup_older_than=datetime.timedelta(days=7))
            return
        except Exception as e:
            if "Retryable commit conflict" in str(e) and attempt < max_attempts:
                wait = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                logger.warning(
                    "Optimize '%s' hit retryable conflict (attempt %d/%d), "
                    "retrying in %.2fs...",
                    table_name, attempt, max_attempts, wait,
                )
                time.sleep(wait)
            else:
                raise


def _do_optimize_all_tables(triggered_by: str = "cli_command") -> dict[str, Any]:
    """
    Run LanceDB optimization on all tables in the database.
    
    Returns a dictionary suitable for DaemonResponse.data matching OptimizeResult.
    """
    settings = get_settings()
    lancedb_dir = settings.lancedb_path

    # Read how many writes we are clearing
    state = read_optimize_state()
    cleared_writes = state["unoptimized_writes"]

    if not lancedb_dir.exists():
        return {
            "tables_optimized": [],
            "duration_seconds": 0.0,
            "last_optimized_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "cleared_writes": cleared_writes,
        }

    db = lancedb.connect(str(lancedb_dir))
    tables = db.table_names()

    start_time = time.time()
    optimized_tables = []

    bytes_before = _get_dir_size(lancedb_dir)

    for table_name in tables:
        try:
            logger.info("Optimizing LanceDB table '%s'...", table_name)
            table = db.open_table(table_name)

            # optimize runs compaction and version cleanup.
            # IMPORTANT: cleanup_older_than must NOT be 0 — LanceDB's MVCC
            # needs time to promote new writes to stable versions. Using hours=0
            # (or delete_unverified=True) can vacuum freshly written rows before
            # they are checkpointed, causing silent data loss on the next boot.
            # 7 days matches the LanceDB default and is safe for this workload.
            #
            # NOTE: The 'expressions' table previously had a create_fts_index()
            # call that caused a Rust panic in lance-index 7.0.0 during optimize.
            # That FTS index has been permanently dropped (repair_expressions_table.py)
            # and is no longer created (memory_store.py). Optimize is now safe on
            # all tables.
            _optimize_with_retry(table, table_name)
            optimized_tables.append(table_name)
        except Exception as e:
            logger.error("Failed to optimize table '%s': %s", table_name, e, exc_info=True)


    duration = time.time() - start_time

    bytes_after = _get_dir_size(lancedb_dir)
    bytes_freed = max(0, bytes_before - bytes_after)

    # Update state file
    write_last_optimized(triggered_by)
    now_str = datetime.datetime.now(datetime.UTC).isoformat()

    logger.info("LanceDB optimization complete in %.2fs for %d tables (Freed %.1f MiB, Cleared %d writes).", duration, len(optimized_tables), bytes_freed / 1024 / 1024, cleared_writes)

    return {
        "tables_optimized": optimized_tables,
        "duration_seconds": duration,
        "last_optimized_at": now_str,
        "bytes_freed": bytes_freed,
        "cleared_writes": cleared_writes,
    }
