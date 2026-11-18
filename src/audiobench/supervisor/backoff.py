"""Exponential backoff tracker for supervised processes."""

from __future__ import annotations

import time

MAX_RESTARTS = 3


class ExponentialBackoff:
    """Tracks restart counts and calculates backoff delays."""

    def __init__(self) -> None:
        self.restart_count: int = 0
        self.last_start_time: float = 0.0

    def mark_started(self) -> None:
        """Mark the process as started."""
        self.last_start_time = time.monotonic()

    def mark_failed(self) -> tuple[bool, float]:
        """Mark the process as failed and compute next steps.

        Returns:
            A tuple (is_fatal, delay_seconds).
            is_fatal is True if MAX_RESTARTS is exceeded.
        """
        # If the process ran cleanly for more than 60 seconds, reset backoff
        uptime = time.monotonic() - self.last_start_time
        if uptime > 60.0:
            self.restart_count = 0

        self.restart_count += 1

        if self.restart_count > MAX_RESTARTS:
            return True, 0.0

        # Delay: 1s, 2s, 4s
        delay = 2.0 ** (self.restart_count - 1)
        return False, delay

    def clear(self) -> None:
        """Reset the backoff state."""
        self.restart_count = 0
        self.last_start_time = 0.0
