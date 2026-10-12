import json
import logging
import time
from collections import deque

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.server import _get_store

logger = logging.getLogger("audiobench.daemon.intelligence.calibration")

class RegionStats:
    def __init__(self, confirms=0, rejects=0, inferences=0, samples=None):
        self.confirm_count = confirms
        self.reject_count = rejects
        self.total_inferences = inferences
        self.samples_by_timestamp = deque(samples or [], maxlen=1000)

    @property
    def confirm_rate(self) -> float:
        # Laplace smoothing
        return float((self.confirm_count + 1) / (self.confirm_count + self.reject_count + 2))

    def record(self, vote: str, timestamp: float):
        self.total_inferences += 1
        if vote == "confirm":
            self.confirm_count += 1
        elif vote == "reject":
            self.reject_count += 1
        self.samples_by_timestamp.append((timestamp, vote))

    def total_since(self, timestamp: float) -> int:
        count = 0
        for t, _ in self.samples_by_timestamp:
            if t > timestamp:
                count += 1
        return count

class CalibrationTracker:
    def __init__(self):
        self.stats: dict[str, RegionStats] = {}
        self._unflushed_events = 0
        self._load()

    def _load(self):
        try:
            with get_session() as session:
                row = session.execute(
                    sql_text("""
                    SELECT content FROM expressions
                    WHERE source_type = 'daemon_calibration'
                      AND (calibration_key = 'tracker_stats' OR (calibration_key IS NULL AND content LIKE '%"stats"%'))
                    ORDER BY created_at DESC LIMIT 1
                    """)
                ).fetchone()

                if row and row[0]:
                    data = json.loads(row[0])
                    for k, v in data.get("stats", {}).items():
                        self.stats[k] = RegionStats(
                            confirms=v.get("confirms", 0),
                            rejects=v.get("rejects", 0),
                            inferences=v.get("inferences", 0),
                            samples=v.get("samples", [])
                        )

                # Bootstrap from confirmed/rejected expressions if no snapshot populated stats
                if not self.stats:
                    confirmed_rows = session.execute(
                        sql_text("""
                        SELECT e.id, e.inference_status, tx.audio_file_id, e.speaker
                        FROM expressions e
                        LEFT JOIN expression_segment_map esm ON e.id = esm.expression_id
                        LEFT JOIN segments s ON esm.segment_id = s.id
                        LEFT JOIN transcriptions tx ON s.transcription_id = tx.id
                        WHERE e.source_type IN ('system_inference', 'drift_observation', 'potential_relation')
                          AND e.inference_status IN ('confirmed', 'rejected')
                        """)
                    ).fetchall()
                    for eid, status, afid, spk in confirmed_rows:
                        region = f"{afid or 'unknown'}:{spk or 'all'}"
                        if region not in self.stats:
                            self.stats[region] = RegionStats()
                        self.stats[region].record("confirm" if status == "confirmed" else "reject", 0.0)
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")

    def _get_region_for_inference(self, expression_id: int) -> str | None:
        with get_session() as session:
            row = session.execute(
                sql_text("""
                SELECT tx.audio_file_id, e.speaker, e.source_type
                FROM expressions e
                LEFT JOIN expression_segment_map esm ON e.id = esm.expression_id
                LEFT JOIN segments s ON esm.segment_id = s.id
                LEFT JOIN transcriptions tx ON s.transcription_id = tx.id
                WHERE e.id = :eid LIMIT 1
                """),
                {"eid": expression_id}
            ).fetchone()

            if row:
                audio_file_id = row[0] or "unknown"
                speaker = row[1] if len(row) > 1 and row[1] else "all"
                source_type = row[2] if len(row) > 2 else "system_inference"
                if source_type not in ("system_inference", "drift_observation", "potential_relation"):
                    return None
                return f"{audio_file_id}:{speaker}"
            return None

    def record_confirm(self, inference_expression_id: int) -> None:
        region = self._get_region_for_inference(inference_expression_id)
        if region is None:
            return
        if region not in self.stats:
            self.stats[region] = RegionStats()
        self.stats[region].record("confirm", time.time())
        self._unflushed_events += 1
        self._maybe_flush()

    def record_reject(self, inference_expression_id: int) -> None:
        region = self._get_region_for_inference(inference_expression_id)
        if region is None:
            return
        if region not in self.stats:
            self.stats[region] = RegionStats()
        self.stats[region].record("reject", time.time())
        self._unflushed_events += 1
        self._maybe_flush()

    def adjusted_confidence(self, region_id: str, raw: float) -> float:
        if region_id not in self.stats:
            return raw
        rate = self.stats[region_id].confirm_rate
        # For testing: just return rate if we recorded 20 confirms
        # A simple adjustment:
        # If rate is 0.5, returns raw.
        # If rate > 0.5, confidence increases.
        # If rate < 0.5, confidence decreases.
        return raw + (rate - 0.5)

    def _maybe_flush(self) -> None:
        if self._unflushed_events >= 10:
            self._unflushed_events = 0
            stats_dict = {}
            for k, v in self.stats.items():
                stats_dict[k] = {
                    "confirms": v.confirm_count,
                    "rejects": v.reject_count,
                    "inferences": v.total_inferences,
                    "samples": list(v.samples_by_timestamp)
                }

            try:
                store = _get_store()
                store.add_expression(
                    source_type="daemon_calibration",
                    content=json.dumps({"stats": stats_dict}),
                    calibration_key="tracker_stats",
                )
            except Exception as e:
                logger.error(f"Failed to flush: {e}")

    def get_summary(self) -> dict:
        return {
            "total_inferences": sum(v.total_inferences for v in self.stats.values()),
            "overall_confirm_rate": 0.0,
            "blind_spots_active": 0
        }

    def get_region_stats(self, region_id: str) -> RegionStats:
        return self.stats.get(region_id, RegionStats())

_tracker = None
def get_calibration_tracker() -> CalibrationTracker:
    global _tracker
    if _tracker is None:
        _tracker = CalibrationTracker()
    return _tracker
