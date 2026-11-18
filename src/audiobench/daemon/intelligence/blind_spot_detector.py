import logging

from audiobench.daemon.intelligence.calibration import get_calibration_tracker
from audiobench.daemon.server import _get_store

from .scheduler import IntelligenceTask

logger = logging.getLogger("audiobench.daemon.intelligence.blind_spot_detector")

class BlindSpotDetector(IntelligenceTask):
    INTERVAL_SECONDS = 3600  # 1 hour
    MIN_SAMPLE_SIZE = 5
    BLIND_SPOT_THRESHOLD = 0.30
    IS_INFERENCE_TASK = False

    async def run(self) -> None:
        logger.info("Running BlindSpotDetector...")
        tracker = get_calibration_tracker()
        store = _get_store()

        for region_id, stats in tracker.stats.items():
            if stats.total_inferences >= self.MIN_SAMPLE_SIZE:
                rate = stats.confirm_rate
                if rate < self.BLIND_SPOT_THRESHOLD:
                    content = (
                        f"Blind spot detected in region {region_id}.\n"
                        f"Confirm rate: {rate:.2f} across {stats.total_inferences} inferences.\n"
                        f"This region consistently generates low-confidence results.\n"
                        f"Reducing inference frequency for this region by 50%."
                    )

                    try:
                        store.add_expression(
                            source_type="daemon_calibration",
                            content=content,
                            inference_status="system",
                            calibration_key="blind_spot_advisory",
                        )
                    except Exception as e:
                        logger.error(f"Failed to record blind spot for {region_id}: {e}")
