import json
import logging
from collections import deque

import numpy as np
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.server import _get_store

logger = logging.getLogger("audiobench.daemon.intelligence.timing_model")

class VerbTimer:
    def __init__(self, ema_ms=None, samples=None):
        self.ema_ms = ema_ms
        self.samples = deque(samples or [], maxlen=100)
        self.alpha = 0.1

    def record(self, duration_ms: float):
        if self.ema_ms is None:
            self.ema_ms = duration_ms
        else:
            self.ema_ms = self.alpha * duration_ms + (1 - self.alpha) * self.ema_ms
        self.samples.append(duration_ms)

    def get_p50(self) -> float:
        if not self.samples:
            return 1000.0
        return float(np.percentile(self.samples, 50))

    def get_p95(self) -> float:
        if not self.samples:
            return 1000.0
        return float(np.percentile(self.samples, 95))

class TimingModel:
    def __init__(self):
        self.timers: dict[str, VerbTimer] = {}
        self._unflushed_events = 0
        self._load()

    def _load(self):
        try:
            with get_session() as session:
                row = session.execute(
                    sql_text("""
                    SELECT content FROM expressions
                    WHERE source_type = 'daemon_calibration'
                      AND (calibration_key = 'timing_stats' OR (calibration_key IS NULL AND content LIKE '%"timing_stats"%'))
                    ORDER BY created_at DESC LIMIT 1
                    """)
                ).fetchone()

                if row and row[0]:
                    data = json.loads(row[0])
                    for k, v in data.get("timing_stats", {}).items():
                        self.timers[k] = VerbTimer(
                            ema_ms=v.get("ema_ms"),
                            samples=v.get("samples", [])
                        )
        except Exception as e:
            logger.error(f"Failed to load timing model: {e}")

    def record_latency(self, verb: str, duration_ms: float) -> None:
        if verb not in self.timers:
            self.timers[verb] = VerbTimer()
        self.timers[verb].record(duration_ms)

        self._unflushed_events += 1
        if self._unflushed_events >= 20:
            self._unflushed_events = 0
            self._flush()

    def predict_duration(self, verb: str) -> float:
        if verb not in self.timers or self.timers[verb].ema_ms is None:
            return 1000.0
        return float(self.timers[verb].ema_ms)

    def get_summary(self) -> dict:
        summary = {}
        for k, v in self.timers.items():
            summary[f"{k}_p50_latency_ms"] = v.get_p50()
            summary[f"{k}_p95_latency_ms"] = v.get_p95()
        return summary

    def _flush(self):
        stats_dict = {}
        for k, v in self.timers.items():
            stats_dict[k] = {
                "ema_ms": v.ema_ms,
                "samples": list(v.samples)
            }
        try:
            store = _get_store()
            store.add_expression(
                source_type="daemon_calibration",
                content=json.dumps({"timing_stats": stats_dict}),
                calibration_key="timing_stats",
            )
        except Exception as e:
            logger.error(f"Failed to flush timing model: {e}")

_timing_model = None
def get_timing_model() -> TimingModel:
    global _timing_model
    if _timing_model is None:
        _timing_model = TimingModel()
    return _timing_model
