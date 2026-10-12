import json
import logging

from pydantic import ValidationError
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.calibration import get_calibration_tracker

from .operator_templates import _TEMPLATE_CLASSES

logger = logging.getLogger("audiobench.daemon.intelligence.operator_registry")

class OperatorRegistry:
    def __init__(self):
        self.dynamic_operators = {}

    def load_from_db(self, store=None) -> None:
        logger.info("Loading dynamic operators from confirmed daemon_proposal records...")
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT id, content FROM expressions
                WHERE source_type = 'daemon_proposal'
                AND inference_status = 'confirmed'
                """)
            ).fetchall()

            for row in rows:
                content = (row[1] or "").strip()
                if not content.startswith("{"):
                    # Non-operator advisory proposal (e.g. natural language knowledge gap proposal)
                    logger.debug("Skipping non-operator proposal %d (not JSON)", row[0])
                    continue
                try:
                    data = json.loads(content)
                except Exception as e:
                    logger.error(f"Failed to load operator {row[0]}: {e}")
                    continue

                if not isinstance(data, dict):
                    continue

                proposal_type = data.get("proposal_type")
                if proposal_type is not None and proposal_type != "operator":
                    logger.debug("Skipping proposal %d: proposal_type is %s", row[0], proposal_type)
                    continue

                payload = data.get("payload", data) if proposal_type == "operator" else data
                if not isinstance(payload, dict) or "operator_template" not in payload:
                    logger.debug("Skipping proposal %d: missing operator_template", row[0])
                    continue

                self._register_proposal(row[0], payload, session)

    def hot_register(self, expression_id: int) -> None:
        logger.info("Hot-registering dynamic operator from proposal %d", expression_id)
        with get_session() as session:
            row = session.execute(
                sql_text("""
                SELECT content FROM expressions
                WHERE id = :eid AND source_type = 'daemon_proposal'
                """),
                {"eid": expression_id}
            ).fetchone()

            if row:
                content = (row[0] or "").strip()
                if not content.startswith("{"):
                    logger.debug("Proposal %d is not an operator specification (plain text)", expression_id)
                    return
                try:
                    data = json.loads(content)
                except Exception as e:
                    logger.error(f"Failed to hot register operator {expression_id}: {e}")
                    return

                if not isinstance(data, dict):
                    return

                proposal_type = data.get("proposal_type")
                if proposal_type is not None and proposal_type != "operator":
                    logger.debug("Skipping hot register for proposal %d: proposal_type is %s", expression_id, proposal_type)
                    return

                payload = data.get("payload", data) if proposal_type == "operator" else data
                if not isinstance(payload, dict) or "operator_template" not in payload:
                    logger.debug("Proposal %d has no operator_template", expression_id)
                    return

                self._register_proposal(expression_id, payload, session)

    def _register_proposal(self, expression_id: int, proposal_data: dict, session) -> None:
        template_name = proposal_data.get("operator_template")
        schema_version = proposal_data.get("schema_version")
        parameters = proposal_data.get("parameters", {})

        if template_name not in _TEMPLATE_CLASSES:
            logger.warning(
                f"rejected: unknown operator_template '{template_name}' for proposal "
                f"{expression_id}. Template may have been removed or renamed."
            )
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'rejected' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()
            return

        template_class = _TEMPLATE_CLASSES[template_name]

        template_version = getattr(template_class, "SCHEMA_VERSION", 1)
        if schema_version != template_version:
            logger.warning(f"Schema mismatch for {template_name}. Expected {template_version}, got {schema_version}")
            return

        try:
            instance = template_class(**parameters)
            self.dynamic_operators[expression_id] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for proposal {expression_id}: {e}")
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'rejected' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()

    def authorize(self, expression_id: int) -> None:
        """Mark a proposal confirmed in DB then hot-register it as a live operator.

        This is the single call the server handler makes. Status write and
        operator instantiation are kept together so the registry is never in
        a state where confirmed exists in DB but the operator is not resident.
        """
        with get_session() as session:
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'confirmed' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()
        self.hot_register(expression_id)

    def deactivate(self, expression_id: int) -> None:
        if expression_id in self.dynamic_operators:
            del self.dynamic_operators[expression_id]
        with get_session() as session:
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'revoked' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()

    def get_pending_operator_proposals(self, min_age_seconds: float = 7 * 86400) -> list[dict]:
        """Fetch pending operator proposals and run staleness checks against CalibrationTracker.

        Staleness rules:
        1. Performance degradation: If confirm_rate < 0.70, proposal is immediately expired.
        2. Inactivity abandonment: If age >= min_age_seconds (default 7 days) AND new_samples < 5,
           proposal is expired.
        Proposals younger than min_age_seconds are never expired due to low sample counts alone,
        preventing premature eviction of freshly generated proposals.

        Explicitly skips non-operator proposals (e.g. knowledge gaps or plain text) without
        error and without touching their status. Only genuine operator proposals are evaluated
        for staleness.
        """
        logger.info("Fetching pending operator proposals and running staleness check...")
        import datetime as _dt
        import time as _time

        now = _time.time()
        valid_proposals = []
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT id, content, created_at FROM expressions
                WHERE source_type = 'daemon_proposal'
                AND inference_status IN ('proposed', 'deferred')
                """)
            ).fetchall()

            tracker = get_calibration_tracker()

            for row in rows:
                content = (row[1] or "").strip()
                if not content.startswith("{"):
                    # Plain-text legacy proposal — never an operator proposal, ignore cleanly
                    continue
                try:
                    data = json.loads(content)
                except Exception:
                    # Malformed JSON — ignore cleanly without marking expired
                    continue

                if not isinstance(data, dict):
                    continue

                proposal_type = data.get("proposal_type")
                # If explicit proposal_type exists and is not 'operator', skip without touching status
                if proposal_type is not None and proposal_type != "operator":
                    continue

                # If no proposal_type, check if it's an operator proposal (has 'operator_template')
                payload = data.get("payload", data) if proposal_type == "operator" else data
                if not isinstance(payload, dict) or "operator_template" not in payload:
                    continue

                # Parse created_at timestamp
                raw_created = row[2]
                if raw_created is None:
                    created_ts = now
                elif isinstance(raw_created, (int, float)):
                    created_ts = float(raw_created)
                elif hasattr(raw_created, "timestamp"):
                    created_ts = float(raw_created.timestamp())
                elif isinstance(raw_created, str):
                    try:
                        clean_str = raw_created.replace("Z", "+00:00")
                        created_ts = _dt.datetime.fromisoformat(clean_str).timestamp()
                    except Exception:
                        try:
                            created_ts = float(raw_created)
                        except Exception:
                            created_ts = now
                else:
                    created_ts = now

                age_seconds = now - created_ts
                region_id = payload.get("region_id", "")
                stats = tracker.get_region_stats(region_id)
                new_samples = stats.total_since(created_ts)

                # Staleness check on genuine operator proposal
                is_degraded = stats.confirm_rate < 0.70
                is_abandoned = (age_seconds >= min_age_seconds) and (new_samples < 5)

                if is_degraded or is_abandoned:
                    logger.info(
                        f"Operator proposal {row[0]} is stale (degraded={is_degraded}, abandoned={is_abandoned}, "
                        f"rate={stats.confirm_rate:.2f}, age_days={age_seconds/86400:.1f}, new_samples={new_samples})"
                    )
                    session.execute(
                        sql_text("UPDATE expressions SET inference_status = 'expired' WHERE id = :eid"),
                        {"eid": row[0]}
                    )
                    continue

                valid_proposals.append({
                    "id": row[0],
                    "data": data,
                    "created_at": row[2]
                })

            session.commit()

        return valid_proposals

    get_pending_proposals = get_pending_operator_proposals

_registry = None
def get_operator_registry() -> OperatorRegistry:
    global _registry
    if _registry is None:
        _registry = OperatorRegistry()
    return _registry
