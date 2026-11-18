import json
import logging
import time

from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.server import _get_store
from audiobench.memory.enums import SourceType

from .math_utils import pairwise_cosine_above_threshold
from .scheduler import IntelligenceTask

logger = logging.getLogger("audiobench.daemon.intelligence.pattern_detector")

class PatternDetector(IntelligenceTask):
    INTERVAL_SECONDS = 3600
    BATCH_SIZE = 100
    IS_INFERENCE_TASK = False

    async def run(self) -> None:
        logger.info("Running PatternDetector...")
        store = _get_store()

        # 1. Load seen_pairs from latest daemon_calibration expression
        seen_pairs = set()
        with get_session() as session:
            calib_row = session.execute(
                sql_text("""
                SELECT content FROM expressions 
                WHERE source_type = 'daemon_calibration' 
                  AND (calibration_key = 'seen_pairs' OR (calibration_key IS NULL AND content LIKE '%"seen_pairs"%'))
                ORDER BY created_at DESC LIMIT 1
                """)
            ).fetchone()
            if calib_row and calib_row[0]:
                try:
                    data = json.loads(calib_row[0])
                    for pair in data.get("seen_pairs", []):
                        seen_pairs.add(tuple(sorted(pair)))
                except json.JSONDecodeError:
                    pass

        # 2. Query last 100 audio_transcript expressions from past 7 days
        # We need their work_id and author to formulate inferences.
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT e.id, tx.audio_file_id, e.work_id, af.file_name as source_title, w.author as author_name
                FROM expressions e
                JOIN transcriptions tx ON e.source_id = tx.id
                JOIN audio_files af ON tx.audio_file_id = af.id
                LEFT JOIN works w ON e.work_id = w.id
                WHERE e.source_type = :stype
                  AND e.created_at >= :since
                ORDER BY e.created_at DESC
                LIMIT :limit
                """),
                {
                    "stype": SourceType.AUDIO_TRANSCRIPT.value,
                    "since": __import__("datetime").datetime.fromtimestamp(
                        time.time() - 7 * 86400,
                        tz=__import__("datetime").timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "limit": self.BATCH_SIZE
                }
            ).mappings().all()

            if len(rows) < 2:
                return

            rows_by_id = {r["id"]: dict(r) for r in rows}
            ids = [r["id"] for r in rows]

        # 3. Fetch vectors
        vector_dict = store.get_vectors(ids)

        # 4. Build input lists for math_utils
        valid_ids = []
        vectors = []
        for eid in ids:
            if eid in vector_dict:
                valid_ids.append(eid)
                vectors.append(vector_dict[eid])

        if len(vectors) < 2:
            return

        # 5. Compute similarities
        from audiobench.daemon.index import get_shared_hnsw_index
        shared_index = get_shared_hnsw_index()

        pairs_above_threshold = []
        if shared_index.is_ready():
            k_query = min(11, shared_index.item_count())
            labels, dists = shared_index.query_batch(vectors, k=k_query, chunk_size=25)
            seen_local = set()
            for idx_a, id_a in enumerate(valid_ids):
                for col in range(labels.shape[1]):
                    id_b = int(labels[idx_a, col])
                    dist = float(dists[idx_a, col])
                    if id_b == id_a or id_b not in rows_by_id:
                        continue
                    if dist > 0.15:  # cosine distance <= 0.15 corresponds to sim >= 0.85
                        continue
                    sim = 1.0 - dist
                    pair_key = tuple(sorted((id_a, id_b)))
                    if pair_key not in seen_local:
                        seen_local.add(pair_key)
                        pairs_above_threshold.append((id_a, id_b, sim))
        else:
            logger.info("PatternDetector: shared HNSW index not ready — using pairwise cosine")
            pairs_above_threshold = pairwise_cosine_above_threshold(vectors, valid_ids, 0.85)

        new_pairs_added = False
        inferences_emitted = 0

        # 6. Filter and emit
        for id_a, id_b, score in pairs_above_threshold:
            pair_key = tuple(sorted((id_a, id_b)))
            if pair_key in seen_pairs:
                continue

            row_a = rows_by_id[id_a]
            row_b = rows_by_id[id_b]

            # Cross-file and cross-work requirement (must be different works)
            if row_a["audio_file_id"] == row_b["audio_file_id"]:
                continue
            if row_a["work_id"] is None or row_b["work_id"] is None:
                continue
            if row_a["work_id"] == row_b["work_id"]:
                continue

            seen_pairs.add(pair_key)
            new_pairs_added = True

            content = (
                f"Topic convergence detected: expression {id_a} and expression {id_b} share\n"
                f"a semantic similarity of {score:.2f}. Both from the past 7 days.\n"
                f"Sources: {row_a['source_title']} (by {row_a['author_name']}), "
                f"{row_b['source_title']} (by {row_b['author_name']}). Similarity threshold: 0.85."
            )

            store.add_expression(
                source_type=SourceType.SYSTEM_INFERENCE.value,
                content=content,
                inference_status="proposed"
            )
            inferences_emitted += 1

        # 7. Write updated seen_pairs
        if new_pairs_added:
            store.add_expression(
                source_type="daemon_calibration",
                content=json.dumps({"seen_pairs": list(seen_pairs)}),
                inference_status="system",
                calibration_key="seen_pairs",
            )

        return inferences_emitted
