import asyncio
from collections import deque
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("audiobench.daemon.intelligence")

@dataclass
class ResourceBudget:
    max_task_overhead_mb: int = 1536  # Max RSS *above baseline* allowed for task work.
    # The absolute ceiling (max_memory_mb) is replaced by an overhead-relative
    # gate measured against the settled resting baseline, guarded by max_absolute_rss_mb.
    # The daemon loads PyTorch models and keeps LanceDB/SQLite caches warm (~3-3.7 GB RSS at rest).
    # Headroom of 1536 MB allows background sweeps and vector queries while staying well within RAM limits.
    max_absolute_rss_mb: int = 6144   # Absolute ceiling guardrail (6 GB on 16 GB system)
    max_cpu_percent: float = 80.0    # Run if overall system CPU is under 80%
    idle_threshold_seconds: int = 120
    max_inferences_per_session: int = 200  # Max task executions per rolling rate-limit window
    rate_limit_window_seconds: int = 3600  # 1-hour sliding rate-limit window
    warning_cooldown_seconds: int = 600  # Minimum seconds between repeated resource abort warnings
    # Circuit breaker: how many consecutive task failures before exponential backoff.
    circuit_breaker_threshold: int = 3

@runtime_checkable
class IntelligenceTask(Protocol):
    INTERVAL_SECONDS: int
    IS_INFERENCE_TASK: bool = False

    def run(self) -> Any:
        """Execute the intelligence task."""
        ...

class IntelligenceScheduler:
    def __init__(self, budget: ResourceBudget | None = None):
        self.budget = budget or ResourceBudget()
        self.tasks: list[IntelligenceTask] = []
        self._last_run_times: dict[type, float] = {}
        self.inferences_performed = 0
        self._task_execution_timestamps: deque[float] = deque()
        self._last_rate_limit_warning_time: float = 0.0
        # Baseline RSS measured after the startup warmup sleep, before any task
        # has run. The gate checks overhead above this value, not absolute RSS.
        self._baseline_rss_mb: float | None = None
        # Per-task consecutive failure counters for the circuit breaker.
        self._task_consecutive_failures: dict[type, int] = {}
        self._task_backoff_until: dict[type, float] = {}
        self._last_resource_warning_time: float = 0.0
        self._last_recalibration_time: float = 0.0

    def register(self, task: IntelligenceTask) -> None:
        self.tasks.append(task)
        logger.info("Registered intelligence task: %s", type(task).__name__)

    def _is_idle(self, last_request_time: float) -> bool:
        if time.time() - last_request_time < self.budget.idle_threshold_seconds:
            return False
        return True

    def _measure_rss_mb(self) -> float | None:
        """Return current process RSS in MB, or None if psutil is unavailable."""
        if psutil is None:
            return None
        if not hasattr(self, "_process"):
            self._process = psutil.Process()
        return float(self._process.memory_info().rss / (1024 * 1024))

    def _prune_execution_window(self, now: float) -> int:
        """Prune timestamps older than rate_limit_window_seconds."""
        cutoff = now - self.budget.rate_limit_window_seconds
        while self._task_execution_timestamps and self._task_execution_timestamps[0] < cutoff:
            self._task_execution_timestamps.popleft()
        return len(self._task_execution_timestamps)

    def _has_resources(
        self,
        candidate_task_name: str | None = None,
        is_inference_task: bool = True,
    ) -> bool:
        if is_inference_task:
            now = time.time()
            recent_count = self._prune_execution_window(now)
            effective_count = max(recent_count, self.inferences_performed)
            if effective_count >= self.budget.max_inferences_per_session:
                if now - self._last_rate_limit_warning_time >= self.budget.warning_cooldown_seconds:
                    task_info = f" for {candidate_task_name}" if candidate_task_name else ""
                    logger.warning(
                        "Intelligence task throttled%s: rate limit reached (%d inferences in rolling %.0fm window, cap=%d)",
                        task_info,
                        effective_count,
                        self.budget.rate_limit_window_seconds / 60,
                        self.budget.max_inferences_per_session,
                    )
                    self._last_rate_limit_warning_time = now
                return False

        if psutil is not None:
            rss_mb = self._measure_rss_mb()
            if rss_mb is not None:
                # Gate on overhead above the settled post-startup baseline.
                if self._baseline_rss_mb is not None:
                    overhead_mb = rss_mb - self._baseline_rss_mb
                    if overhead_mb > self.budget.max_task_overhead_mb:
                        # Invoke garbage collection to release transient structures before aborting
                        gc.collect()
                        refreshed_rss = self._measure_rss_mb()
                        if refreshed_rss is not None:
                            rss_mb = refreshed_rss
                            overhead_mb = rss_mb - self._baseline_rss_mb

                    if overhead_mb > self.budget.max_task_overhead_mb:
                        now = time.time()
                        if now - self._last_resource_warning_time >= self.budget.warning_cooldown_seconds:
                            task_info = f" for {candidate_task_name}" if candidate_task_name else ""
                            logger.warning(
                                "Intelligence task aborted%s: RSS overhead %.1fMB > %.1fMB "
                                "(current=%.1fMB, baseline=%.1fMB)",
                                task_info,
                                overhead_mb, self.budget.max_task_overhead_mb,
                                rss_mb, self._baseline_rss_mb,
                            )
                            self._last_resource_warning_time = now
                        return False
                else:
                    # Fallback: absolute ceiling while baseline is unavailable.
                    if rss_mb > self.budget.max_absolute_rss_mb:
                        now = time.time()
                        if now - self._last_resource_warning_time >= self.budget.warning_cooldown_seconds:
                            task_info = f" for {candidate_task_name}" if candidate_task_name else ""
                            logger.warning(
                                "Intelligence task aborted%s: RSS %.1fMB > %.1fMB (no baseline yet)",
                                task_info,
                                rss_mb, self.budget.max_absolute_rss_mb,
                            )
                            self._last_resource_warning_time = now
                        return False

                # Absolute ceiling guardrail
                if rss_mb > self.budget.max_absolute_rss_mb:
                    now = time.time()
                    if now - self._last_resource_warning_time >= self.budget.warning_cooldown_seconds:
                        task_info = f" for {candidate_task_name}" if candidate_task_name else ""
                        logger.warning(
                            "Intelligence task aborted%s: absolute RSS %.1fMB > %.1fMB ceiling",
                            task_info,
                            rss_mb, self.budget.max_absolute_rss_mb,
                        )
                        self._last_resource_warning_time = now
                    return False

            # Check overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.budget.max_cpu_percent:
                logger.debug(
                    "Intelligence task delayed: CPU usage %.1f%% > %.1f%%",
                    cpu_percent, self.budget.max_cpu_percent,
                )
                return False

        return True

    def _maybe_recalibrate_baseline(self, now: float) -> None:
        """Recalibrate baseline if the daemon has been resting idle and memory has settled."""
        if self._baseline_rss_mb is None or psutil is None:
            return
        # Only recalibrate after at least 5 minutes of no task executions
        last_task_run = max(self._last_run_times.values(), default=0.0)
        if last_task_run > 0.0 and (now - last_task_run < 300):
            return
        # Recalibrate at most once every 10 minutes
        if now - self._last_recalibration_time < 600:
            return

        gc.collect()
        current_rss = self._measure_rss_mb()
        if current_rss is not None:
            max_baseline_mb = min(
                float(self.budget.max_absolute_rss_mb - self.budget.max_task_overhead_mb),
                4608.0,
            )
            if self._baseline_rss_mb < current_rss <= max_baseline_mb:
                old_baseline = self._baseline_rss_mb
                self._baseline_rss_mb = current_rss
                self._last_recalibration_time = now
                logger.info(
                    "Intelligence scheduler recalibrated resting baseline RSS: %.1f MB -> %.1f MB "
                    "(gate fires at baseline + %d MB = %.1f MB)",
                    old_baseline, current_rss,
                    self.budget.max_task_overhead_mb,
                    current_rss + self.budget.max_task_overhead_mb,
                )

    def _is_task_backed_off(self, task_type: type) -> bool:
        """Return True if this task type is in exponential backoff."""
        backoff_until = self._task_backoff_until.get(task_type, 0)
        return time.time() < backoff_until

    def _record_task_success(self, task_type: type) -> None:
        """Reset the circuit breaker for a task after a successful run."""
        self._task_consecutive_failures[task_type] = 0
        self._task_backoff_until.pop(task_type, None)

    def _record_task_failure(self, task_type: type, exc: Exception) -> None:
        """Increment the failure counter and engage exponential backoff if threshold hit."""
        n = self._task_consecutive_failures.get(task_type, 0) + 1
        self._task_consecutive_failures[task_type] = n
        logger.error(
            "Intelligence task %s failed (consecutive failure %d): %s",
            task_type.__name__, n, exc,
        )
        if n >= self.budget.circuit_breaker_threshold:
            # Exponential backoff: 2^(n - threshold) * 30 minutes, capped at 6 hours.
            exponent = n - self.budget.circuit_breaker_threshold
            backoff_seconds = min(2 ** exponent * 1800, 21600)
            self._task_backoff_until[task_type] = time.time() + backoff_seconds
            logger.warning(
                "Intelligence task %s circuit breaker engaged after %d failures — "
                "backing off for %.0f minutes.",
                task_type.__name__, n, backoff_seconds / 60,
            )

    async def run_loop(self) -> None:
        """The main loop for the intelligence scheduler."""
        from audiobench.daemon import server

        # Wait for startup to settle before measuring baseline or running tasks.
        await asyncio.sleep(60)

        # Measure the settled post-startup RSS baseline. This is the model-loaded
        # resident footprint before any task work has been done.
        rss_mb = self._measure_rss_mb()
        if rss_mb is not None:
            self._baseline_rss_mb = rss_mb
            logger.info(
                "Intelligence scheduler baseline RSS: %.1f MB "
                "(gate will fire at baseline + %d MB = %.1f MB)",
                rss_mb, self.budget.max_task_overhead_mb,
                rss_mb + self.budget.max_task_overhead_mb,
            )
        else:
            logger.warning("Intelligence scheduler: psutil unavailable, RSS gate disabled.")

        while True:
            try:
                # Poll server._last_request_time
                last_req = getattr(server, "_last_request_time", time.time())

                if self._is_idle(last_req):
                    now = time.time()
                    candidate_task: IntelligenceTask | None = None
                    for task in self.tasks:
                        task_type = type(task)
                        last_run = self._last_run_times.get(task_type, 0)
                        if now - last_run < task.INTERVAL_SECONDS:
                            continue
                        if self._is_task_backed_off(task_type):
                            backoff_remaining = self._task_backoff_until[task_type] - now
                            logger.debug(
                                "Intelligence task %s skipped (circuit breaker: "
                                "%.0f min remaining).",
                                task_type.__name__, backoff_remaining / 60,
                            )
                            continue
                        is_inf = getattr(task, "IS_INFERENCE_TASK", False)
                        if not self._has_resources(candidate_task_name=task_type.__name__, is_inference_task=is_inf):
                            continue
                        candidate_task = task
                        break

                    if candidate_task is not None:
                        task_type = type(candidate_task)
                        logger.info("Executing intelligence task: %s", task_type.__name__)
                        try:
                            if asyncio.iscoroutinefunction(candidate_task.run):
                                result = await candidate_task.run()
                            else:
                                result = await asyncio.to_thread(candidate_task.run)
                            end_now = time.time()
                            self._last_run_times[task_type] = end_now
                            if getattr(candidate_task, "IS_INFERENCE_TASK", False):
                                inferences_count = result if isinstance(result, int) else 1
                                for _ in range(inferences_count):
                                    self._task_execution_timestamps.append(end_now)
                                self.inferences_performed = self._prune_execution_window(end_now)
                            self._record_task_success(task_type)
                        except Exception as task_exc:
                            self._record_task_failure(task_type, task_exc)
                    else:
                        self._maybe_recalibrate_baseline(now)
            except Exception as e:
                logger.error("Error in intelligence scheduler loop: %s", e)

            await asyncio.sleep(30)
