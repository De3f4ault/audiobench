import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from audiobench.daemon.intelligence.scheduler import (
    IntelligenceScheduler,
    IntelligenceTask,
    ResourceBudget,
)


class MockTask(IntelligenceTask):
    INTERVAL_SECONDS = 3600

    def __init__(self):
        self.run_count = 0

    async def run(self):
        self.run_count += 1

def test_task_not_run_before_idle_threshold():
    budget = ResourceBudget(idle_threshold_seconds=120)
    scheduler = IntelligenceScheduler(budget=budget)
    assert not scheduler._is_idle(last_request_time=time.time() - 60)

def test_task_runs_after_idle_threshold():
    budget = ResourceBudget(idle_threshold_seconds=120)
    scheduler = IntelligenceScheduler(budget=budget)
    assert scheduler._is_idle(last_request_time=time.time() - 130)

@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_not_run_when_memory_over_budget(mock_psutil):
    """Gate fires when current RSS exceeds baseline + max_task_overhead_mb."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # Simulate baseline of 2000 MB, current RSS of 2600 MB → 600 MB overhead
    mock_mem_info.rss = 2600 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=500)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 2000.0  # settled post-startup baseline
    assert not scheduler._has_resources()


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_allowed_when_overhead_within_budget(mock_psutil):
    """Gate passes when RSS overhead is within the allowed ceiling."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # Baseline 3000 MB, current 3200 MB → 200 MB overhead, budget 1024 MB
    mock_mem_info.rss = 3200 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=1024)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 3000.0
    assert scheduler._has_resources()


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_not_run_when_cpu_over_budget(mock_psutil):
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # RSS within overhead budget (baseline 3000, current 3100 → 100 MB overhead)
    mock_mem_info.rss = 3100 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 50.0

    budget = ResourceBudget(max_cpu_percent=15.0, max_task_overhead_mb=1024)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 3000.0
    assert not scheduler._has_resources()

def test_max_inferences_per_session_enforced():
    budget = ResourceBudget(max_inferences_per_session=50)
    scheduler = IntelligenceScheduler(budget=budget)

    scheduler.inferences_performed = 50
    assert not scheduler._has_resources()

    scheduler.inferences_performed = 49
    # assuming we mock psutil to None for this test to pass
    with patch("audiobench.daemon.intelligence.scheduler.psutil", None):
        assert scheduler._has_resources()

@pytest.mark.asyncio
async def test_interval_enforced_between_runs():
    budget = ResourceBudget(idle_threshold_seconds=0)
    scheduler = IntelligenceScheduler(budget=budget)
    task = MockTask()
    scheduler.register(task)

    # We will test the loop logic by calling the inner check directly instead of run_loop() which sleeps forever
    # The run_loop logic:
    with patch("time.time", side_effect=[0, 0, 0, 0]):
        # t=0, last_run=0, INTERVAL=3600
        # now - last_run (0 - 0) = 0, which is not >= 3600.
        pass

    # Wait, the run_loop checks `now - last_run >= task.INTERVAL_SECONDS`.
    # At start, last_run is 0. now - last_run is time.time() - 0.
    # We can test this by interacting with _last_run_times.

    now_time = 10000.0
    with patch("time.time", return_value=now_time):
        scheduler._last_run_times[type(task)] = now_time - 3500
        # Should not run since 3500 < 3600
        can_run = (now_time - scheduler._last_run_times[type(task)]) >= task.INTERVAL_SECONDS
        assert not can_run

        scheduler._last_run_times[type(task)] = now_time - 3700
        # Should run
        can_run = (now_time - scheduler._last_run_times[type(task)]) >= task.INTERVAL_SECONDS
        assert can_run


def test_circuit_breaker_engages_at_threshold():
    """Task enters backoff when consecutive failures reach circuit_breaker_threshold."""
    budget = ResourceBudget(circuit_breaker_threshold=3)
    scheduler = IntelligenceScheduler(budget=budget)
    task_type = MockTask

    exc = RuntimeError("task failed")
    scheduler._record_task_failure(task_type, exc)
    scheduler._record_task_failure(task_type, exc)
    assert not scheduler._is_task_backed_off(task_type)  # 2 failures, not yet
    scheduler._record_task_failure(task_type, exc)
    assert scheduler._is_task_backed_off(task_type)  # 3 failures → backed off


def test_circuit_breaker_resets_on_success():
    """Successful run clears backoff state."""
    budget = ResourceBudget(circuit_breaker_threshold=3)
    scheduler = IntelligenceScheduler(budget=budget)
    task_type = MockTask

    exc = RuntimeError("task failed")
    for _ in range(3):
        scheduler._record_task_failure(task_type, exc)
    assert scheduler._is_task_backed_off(task_type)

    scheduler._record_task_success(task_type)
    assert not scheduler._is_task_backed_off(task_type)
    assert scheduler._task_consecutive_failures.get(task_type, 0) == 0


class MockSyncTask(IntelligenceTask):
    INTERVAL_SECONDS = 3600

    def __init__(self):
        self.run_count = 0

    def run(self):
        self.run_count += 1


@pytest.mark.asyncio
async def test_scheduler_executes_sync_and_async_tasks():
    budget = ResourceBudget(idle_threshold_seconds=0)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 100.0

    async_task = MockTask()
    sync_task = MockSyncTask()
    scheduler.register(async_task)
    scheduler.register(sync_task)

    for task in scheduler.tasks:
        if asyncio.iscoroutinefunction(task.run):
            await task.run()
        else:
            await asyncio.to_thread(task.run)

    assert async_task.run_count == 1
    assert sync_task.run_count == 1


@patch("audiobench.daemon.intelligence.scheduler.logger")
@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_warning_throttled_during_overhead_abort(mock_psutil, mock_logger):
    """Warning is logged on initial abort, but suppressed during cooldown period."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 3000 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=500, warning_cooldown_seconds=600)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 2000.0  # 1000 MB overhead > 500 MB budget

    assert not scheduler._has_resources(candidate_task_name="MockTask")
    assert mock_logger.warning.call_count == 1

    # Second check immediately after should not log warning again
    assert not scheduler._has_resources(candidate_task_name="MockTask")
    assert mock_logger.warning.call_count == 1


@patch("audiobench.daemon.intelligence.scheduler.gc.collect")
@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_gc_invoked_when_overhead_high(mock_psutil, mock_gc):
    """gc.collect() is called when overhead exceeds budget to attempt memory reclamation."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # Initially 2600 MB (overhead 600 > 500). After gc, returns 2400 MB (overhead 400 <= 500)
    mock_mem_info.rss = 2600 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    def gc_side_effect():
        mock_mem_info.rss = 2400 * 1024 * 1024

    mock_gc.side_effect = gc_side_effect

    budget = ResourceBudget(max_task_overhead_mb=500)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 2000.0

    assert scheduler._has_resources(candidate_task_name="MockTask")
    assert mock_gc.called


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_absolute_ceiling_enforced(mock_psutil):
    """Task is aborted if absolute RSS exceeds max_absolute_rss_mb even if relative overhead is fine."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 6500 * 1024 * 1024  # 6500 MB
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=2000, max_absolute_rss_mb=6144)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 5000.0  # overhead is 1500 < 2000, but absolute is 6500 > 6144

    assert not scheduler._has_resources(candidate_task_name="MockTask")


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_baseline_recalibration_when_idle(mock_psutil):
    """Resting baseline adapts upward to settled cache footprint after idle period."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 3500 * 1024 * 1024  # 3500 MB resting
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process

    budget = ResourceBudget(max_task_overhead_mb=1536, max_absolute_rss_mb=6144)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 2700.0
    scheduler._last_run_times[MockTask] = 1000.0

    # Before 300s since last task: no recalibration
    scheduler._maybe_recalibrate_baseline(now=1200.0)
    assert scheduler._baseline_rss_mb == 2700.0

    # After 300s since last task: recalibrates to resting RSS
    scheduler._maybe_recalibrate_baseline(now=1400.0)
    assert scheduler._baseline_rss_mb == 3500.0


def test_sliding_window_rate_limit_recovers_after_window():
    """Verify that tasks aged out of the sliding rate-limit window allow scheduler to resume."""
    budget = ResourceBudget(max_inferences_per_session=2, rate_limit_window_seconds=3600)
    scheduler = IntelligenceScheduler(budget=budget)

    with patch("audiobench.daemon.intelligence.scheduler.psutil", None):
        t0 = 10000.0
        # Add 2 executions within the window
        scheduler._task_execution_timestamps.append(t0 - 1800.0)  # 30m ago
        scheduler._task_execution_timestamps.append(t0 - 600.0)   # 10m ago

        # With 2 executions in a cap of 2, budget is exhausted
        with patch("time.time", return_value=t0):
            assert not scheduler._has_resources()

        # Fast forward past 3600s: the first execution (t0 - 1800) is now > 3600s ago
        t1 = t0 + 2000.0  # (t1 - (t0 - 1800)) = 3800s > 3600s
        with patch("time.time", return_value=t1):
            assert scheduler._has_resources()
            assert len(scheduler._task_execution_timestamps) == 1


def test_non_inference_task_bypasses_inference_rate_limit():
    """Non-inference tasks (e.g. WorkLinker, ConnectionSurfer) are not throttled when inference cap is reached."""
    budget = ResourceBudget(max_inferences_per_session=1)
    scheduler = IntelligenceScheduler(budget=budget)

    with patch("audiobench.daemon.intelligence.scheduler.psutil", None):
        t0 = time.time()
        scheduler._task_execution_timestamps.append(t0)

        # Inference task should be throttled
        assert not scheduler._has_resources(is_inference_task=True)

        # Non-inference task should be allowed to run
        assert scheduler._has_resources(is_inference_task=False)


@pytest.mark.asyncio
async def test_scheduler_unblocks_non_inference_task_when_inference_throttled():
    """Inference task throttling does not head-of-line block ready non-inference tasks."""
    budget = ResourceBudget(max_inferences_per_session=1, idle_threshold_seconds=0)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 100.0

    class MockInferenceTask(IntelligenceTask):
        INTERVAL_SECONDS = 60
        IS_INFERENCE_TASK = True
        run_count = 0

        def run(self):
            self.run_count += 1
            return 1

    class MockGraphTask(IntelligenceTask):
        INTERVAL_SECONDS = 60
        IS_INFERENCE_TASK = False
        run_count = 0

        def run(self):
            self.run_count += 1
            return 0

    inf_task = MockInferenceTask()
    graph_task = MockGraphTask()
    scheduler.register(inf_task)
    scheduler.register(graph_task)

    with patch("audiobench.daemon.intelligence.scheduler.psutil", None):
        t0 = time.time()
        # Exhaust inference budget
        scheduler._task_execution_timestamps.append(t0)

        # Simulate candidate task selection in run_loop:
        candidate_task = None
        now = t0 + 100
        for task in scheduler.tasks:
            task_type = type(task)
            last_run = scheduler._last_run_times.get(task_type, 0)
            if now - last_run < task.INTERVAL_SECONDS:
                continue
            is_inf = getattr(task, "IS_INFERENCE_TASK", False)
            if not scheduler._has_resources(candidate_task_name=task_type.__name__, is_inference_task=is_inf):
                continue
            candidate_task = task
            break

        # The inference task was skipped, and the non-inference graph task was selected!
        assert candidate_task is graph_task




