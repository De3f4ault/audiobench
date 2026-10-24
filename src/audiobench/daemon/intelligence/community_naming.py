"""CommunityNamingTask daemon intelligence task.

Periodically scans active communities marked 'pending' or 'fallback'
and upgrades them to conceptual LLM titles using Gemini -> Ollama -> Fallback.

Enforces:
  - BATCH_SIZE = 5 to bound per-cycle latency and CPU usage.
  - INTERVAL_SECONDS = 600 (10-minute cadence during idle windows).
  - Attempt-state tracking and tiered exponential backoff on repeated failures.
  - Clean scoped database connections.
"""

from __future__ import annotations

import asyncio

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.intelligence.scheduler import IntelligenceTask

logger = get_logger("daemon.intelligence.community_naming")


class CommunityNamingTask(IntelligenceTask):
    """Background task for LLM-based community title enrichment ('Schools of Thought')."""

    INTERVAL_SECONDS: int = 600  # 10 minutes
    BATCH_SIZE: int = 5          # Process up to 5 communities per cycle
    IS_INFERENCE_TASK: bool = True

    async def run(self) -> int:
        """Execute the community naming sweep."""
        return await asyncio.to_thread(self._run_sync)

    def _run_sync(self) -> int:
        from audiobench.community.naming import run_naming_sweep

        upgraded = run_naming_sweep(batch_size=self.BATCH_SIZE)
        if upgraded > 0:
            logger.info("CommunityNamingTask: upgraded %d communities to LLM titles", upgraded)
        else:
            logger.debug("CommunityNamingTask: 0 communities upgraded")
        return upgraded
