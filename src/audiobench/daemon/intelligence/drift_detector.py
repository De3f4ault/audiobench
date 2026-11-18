import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.server import _get_store

from .scheduler import IntelligenceTask

logger = logging.getLogger("audiobench.daemon.intelligence.drift_detector")

@dataclass
class RunningCentroid:
    dim: int = 768
    count: int = 0
    sum_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sum_vector = None

    def add(self, vector: list[float] | np.ndarray) -> None:
        v = np.array(vector, dtype=np.float32)
        if self.sum_vector is None:
            self.dim = v.shape[0]
            self.sum_vector = np.zeros(self.dim, dtype=np.float32)
        self.sum_vector += v
        self.count += 1

    def get_centroid(self) -> np.ndarray:
        if self.count == 0 or self.sum_vector is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self.sum_vector / self.count


class DriftDetector(IntelligenceTask):
    INTERVAL_SECONDS = 86400
    DRIFT_THRESHOLD = 0.15
    MIN_ITEMS = 10
    IS_INFERENCE_TASK = False

    async def run(self) -> None:
        logger.info("Running DriftDetector...")
        store = _get_store()
        import datetime as _dt
        now = time.time()

        def _to_iso(unix_ts: float) -> str:
            return _dt.datetime.fromtimestamp(unix_ts, tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        window_b_start = now - 15 * 86400
        window_a_start = now - 30 * 86400

        iso_now = _to_iso(now)
        iso_b_start = _to_iso(window_b_start)
        iso_a_start = _to_iso(window_a_start)

        with get_session() as session:
            # Window A: days 15–30 ago (most recent 500, for centroid stability)
            rows_a = session.execute(
                sql_text("""
                SELECT id FROM expressions
                WHERE source_type = 'audio_transcript'
                  AND graph_role = 'sweep_chunk'
                  AND created_at >= :start AND created_at < :end
                ORDER BY created_at DESC
                LIMIT 500
                """),
                {"start": iso_a_start, "end": iso_b_start}
            ).mappings().all()

            # Window B: days 0–15 ago (most recent 500)
            rows_b = session.execute(
                sql_text("""
                SELECT id FROM expressions
                WHERE source_type = 'audio_transcript'
                  AND graph_role = 'sweep_chunk'
                  AND created_at >= :start AND created_at <= :end
                ORDER BY created_at DESC
                LIMIT 500
                """),
                {"start": iso_b_start, "end": iso_now}
            ).mappings().all()

        ids_a = [r["id"] for r in rows_a]
        ids_b = [r["id"] for r in rows_b]

        if len(ids_a) < self.MIN_ITEMS or len(ids_b) < self.MIN_ITEMS:
            logger.info("Not enough expressions for drift detection.")
            return

        rc_a = RunningCentroid()
        vectors_a = store.get_vectors(ids_a)
        for eid in ids_a:
            if eid in vectors_a:
                rc_a.add(vectors_a[eid])

        rc_b = RunningCentroid()
        vectors_b = store.get_vectors(ids_b)
        for eid in ids_b:
            if eid in vectors_b:
                rc_b.add(vectors_b[eid])

        if rc_a.count < self.MIN_ITEMS or rc_b.count < self.MIN_ITEMS:
            return

        c_a = rc_a.get_centroid()
        c_b = rc_b.get_centroid()

        # Calculate cosine distance: 1 - cosine_similarity
        norm_a = np.linalg.norm(c_a)
        norm_b = np.linalg.norm(c_b)
        if norm_a == 0 or norm_b == 0:
            return

        sim = np.dot(c_a, c_b) / (norm_a * norm_b)
        dist = 1.0 - float(sim)

        if dist > self.DRIFT_THRESHOLD:
            earliest_id = min(ids_b) if ids_b else None

            content = (
                f"Semantic centroid shift detected over the past 30 days.\n"
                f"Cosine distance: {dist:.3f} (threshold: {self.DRIFT_THRESHOLD:.2f}).\n"
                f"Window A centroid covers {rc_a.count} expressions. Window B covers {rc_b.count}.\n"
                f"Earliest diverging source: expression_id={earliest_id}."
            )

            store.add_expression(
                source_type="system_inference",
                content=content,
                inference_status="proposed"
            )
