"""src/audiobench/daemon/intelligence/proposal_cluster.py

Canonical clustering and deduplication utilities for daemon proposals.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_cluster_key(content: str | dict[str, Any]) -> tuple:
    """Return a hashable canonical cluster key for a daemon proposal.

    For knowledge gap proposals with stranded queries:
      Returns ('knowledge_gap', frozenset(normalized_queries))
      where normalized_queries are lowercased and whitespace-trimmed.
    For operator proposals:
      Returns ('operator', operator_template, region_id)
    Fallback:
      Returns ('fallback', proposal_type, title)
    """
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except Exception:
            return ("fallback", "raw", content.strip()[:100])
    elif isinstance(content, dict):
        data = content
    else:
        return ("fallback", "unknown", "")

    if not isinstance(data, dict):
        return ("fallback", "raw", str(data)[:100])

    payload = data.get("payload", data)
    if not isinstance(payload, dict):
        payload = data

    queries = payload.get("stranded_queries") or data.get("stranded_queries", [])
    if queries and isinstance(queries, list):
        norm_queries = frozenset(str(q).strip().lower() for q in queries if str(q).strip())
        if norm_queries:
            return ("knowledge_gap", norm_queries)

    proposal_type = payload.get("proposal_type") or data.get("proposal_type")
    if proposal_type == "operator" or "operator_template" in payload or "operator_template" in data:
        op_template = (
            payload.get("operator_template")
            or data.get("operator_template")
            or payload.get("operator_name")
            or data.get("operator_name")
        )
        region_id = payload.get("region_id") or data.get("region_id")
        return ("operator", str(op_template), region_id)

    title = payload.get("title") or payload.get("gap_title") or data.get("title") or data.get("gap_title")
    return ("fallback", str(proposal_type), str(title))
