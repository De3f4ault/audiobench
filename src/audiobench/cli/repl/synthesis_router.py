"""synthesis_router.py — Synthesis-scoped retrieval router.

After ``\\synthesize``, the user's mental context is the topic of that synthesis.
Subsequent ``\\search`` and ``.search`` commands benefit from knowing which works
were cited.  This module decides *how strongly* to apply that scope:

- ``mode="off"``  : No citations, or citations are stale.  No scoping.
- ``mode="soft"`` : Citations exist but evidence is weak.  Apply a +0.05 MMR
                    boost to in-scope segments.  Never drops results.
- ``mode="hard"`` : A verified 3-clique exists among cited expressions.  Apply
                    a WHERE work_id IN (...) filter to retrieval queries.  Only
                    fires for ~10% of synthesis sessions.

**Invariant:** fail toward soft on any uncertainty.  Hard-filter only when a
triangle is *verified*.  Any lookup failure, stale session, or insufficient
evidence → soft.  A silent hard-filter on bad evidence destroys recall invisibly;
a silent soft-filter on good evidence merely delays a feature.  The asymmetry
is severe.

See implementation_plan.md for the full edge-case spec (EC-1 through EC-11).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from audiobench.cli.repl.session import ReplSession

log = logging.getLogger("audiobench.synthesis_router")

# Commands after which a synthesis citation window is considered stale.
STALE_AFTER: int = 6

# Minimum BM25-style edge weight to count an edge in the triangle check.
# Consistent with the existing clique boost in synthesis_retriever.py.
TRIANGLE_WEIGHT_THRESHOLD: float = 0.78


@dataclass
class RouterResult:
    """Output of ``get_synthesis_scope``.

    Attributes:
        mode:     "off" | "soft" | "hard"
        work_ids: Work IDs whose segments are in scope (empty when mode="off").
        reason:   Short diagnostic label for logging/debugging.
    """

    mode: Literal["off", "soft", "hard"]
    work_ids: list[int] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_synthesis_scope(session: "ReplSession") -> RouterResult:
    """Return the router result for the current session state.

    Called by retrieval commands (``.search``, ``\\search``) before they query
    the DB.  Pure function over session state plus DB reads — no side effects.
    """
    citations: dict[int, int] = getattr(session, "last_synthesis_citations", {})
    snapshot: int = getattr(session, "_synthesis_command_snapshot", 0)
    command_count: int = getattr(session, "_command_count", 0)

    # EC-1: No citations.
    if not citations:
        return RouterResult(mode="off", reason="no_citations")

    # EC-2: Stale citations.
    # snapshot == 0 means no synthesis has run this session (pre-rollout guard).
    if snapshot == 0 or (command_count - snapshot) > STALE_AFTER:
        _clear_citations(session)
        return RouterResult(mode="off", reason="stale")

    citation_eids = list(set(citations.values()))

    # EC-3: Insufficient unique expression IDs.
    if len(citation_eids) < 3:
        return RouterResult(mode="soft", reason="insufficient_citations")

    # EC-4/5: Resolve work_ids, filtering ghosts (work_id=NULL) and deleted rows.
    try:
        from audiobench.core.db_session import get_session as _get_db
        from sqlalchemy import text as sa_text

        with _get_db() as db:
            ph = ", ".join(f":eid{i}" for i in range(len(citation_eids)))
            pm = {f"eid{i}": eid for i, eid in enumerate(citation_eids)}
            rows = db.execute(
                sa_text(
                    f"SELECT id, work_id FROM expressions WHERE id IN ({ph})"
                ),
                pm,
            ).fetchall()

        eid_to_wid: dict[int, int] = {
            r[0]: r[1] for r in rows if r[1] is not None
        }
    except Exception as exc:
        log.warning("synthesis_router: work_id lookup failed: %s", exc)
        return RouterResult(mode="soft", reason="db_error")

    clean_ids = [eid for eid in citation_eids if eid in eid_to_wid]

    if len(clean_ids) < 3:
        return RouterResult(mode="soft", reason="ghost_or_deleted_citations")

    # Triangle verification: check all C(n,3) triples, accumulate verified work_ids.
    try:
        from audiobench.core.db_session import get_session as _get_db

        verified_work_ids: set[int] = set()

        with _get_db() as db:
            for triple in combinations(clean_ids, 3):
                result = _check_triple(triple, eid_to_wid, db)
                if result:
                    verified_work_ids.update(result)
                    # Do NOT short-circuit — accumulate all verified triples (see spec).

    except Exception as exc:
        log.warning("synthesis_router: triangle check failed: %s", exc)
        return RouterResult(mode="soft", reason="db_error")

    if verified_work_ids:
        return RouterResult(
            mode="hard",
            work_ids=sorted(verified_work_ids),
            reason="triangle_verified",
        )

    return RouterResult(
        mode="soft",
        work_ids=sorted(eid_to_wid[eid] for eid in clean_ids),
        reason="no_triangle",
    )


def apply_scope_to_query(
    result: RouterResult,
    base_sql: str,
    params: dict,
    community_id: int | None = None,
) -> tuple[str, dict]:
    """Inject scope constraints into a retrieval SQL fragment.

    Hard mode: appends ``AND work_id IN (...)``.
    Soft mode: returns base_sql unchanged — the caller applies the MMR boost.
    Off mode:  returns base_sql unchanged.

    The community_id filter (access scope) is *orthogonal* to the topic scope
    expressed by work_ids — both apply independently (EC-8).
    """
    if result.mode != "hard" or not result.work_ids:
        return base_sql, params

    wid_ph = ", ".join(f":swid{i}" for i in range(len(result.work_ids)))
    wid_params = {f"swid{i}": wid for i, wid in enumerate(result.work_ids)}

    scope_clause = f"AND work_id IN ({wid_ph})"
    if community_id is not None:
        scope_clause += " AND (community_id = :scope_cid OR :scope_cid IS NULL)"
        wid_params["scope_cid"] = community_id

    return base_sql + "\n" + scope_clause, {**params, **wid_params}


def soft_boost(result: RouterResult, work_id: int | None) -> float:
    """Return the MMR score boost for a candidate in soft mode.

    Returns +0.05 if the candidate's work_id is in the scope set, else 0.0.
    Safe to call in all modes — returns 0.0 for "off" and "hard".
    """
    if result.mode == "soft" and work_id is not None and work_id in result.work_ids:
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_triple(
    triple: tuple[int, int, int],
    eid_to_wid: dict[int, int],
    db,
) -> set[int] | None:
    """Verify that (a, b, c) form a cross-work triangle.

    Returns the set of 3 work_ids if verified, else None.

    Work-distinctness is enforced both in SQL (belt) and in Python (suspenders).
    NULL work_id safety: explicit IS NOT NULL guards prevent NULL != NULL from
    silently evaluating to NULL (which would exclude rows rather than error).
    Making this explicit means a future COALESCE "simplification" can't silently
    break the NULL-safety property.
    """
    from sqlalchemy import text as sa_text

    a, b, c = triple
    w_a = eid_to_wid.get(a)
    w_b = eid_to_wid.get(b)
    w_c = eid_to_wid.get(c)

    # Python-side work-distinctness guard (fast, no DB round-trip).
    if None in (w_a, w_b, w_c) or len({w_a, w_b, w_c}) != 3:
        return None

    row = db.execute(
        sa_text("""
            SELECT COUNT(*) FROM expression_relations r
            JOIN expressions e1 ON e1.id = :a
            JOIN expressions e2 ON e2.id = :b
            JOIN expressions e3 ON e3.id = :c
            WHERE r.relation_type IN ('semantic', 'thematic')
              AND r.weight >= :threshold
              -- Explicit NULL guards: NULL != NULL is NULL (not TRUE) in SQL.
              -- Without these, ghost expressions with NULL work_id would be
              -- silently excluded from the != check instead of erroring.
              AND e1.work_id IS NOT NULL
              AND e2.work_id IS NOT NULL
              AND e3.work_id IS NOT NULL
              AND e1.work_id != e2.work_id
              AND e2.work_id != e3.work_id
              AND e1.work_id != e3.work_id
              AND (
                (r.from_expression_id = :a AND r.to_expression_id = :b) OR
                (r.from_expression_id = :b AND r.to_expression_id = :a) OR
                (r.from_expression_id = :b AND r.to_expression_id = :c) OR
                (r.from_expression_id = :c AND r.to_expression_id = :b) OR
                (r.from_expression_id = :a AND r.to_expression_id = :c) OR
                (r.from_expression_id = :c AND r.to_expression_id = :a)
              )
        """),
        {"a": a, "b": b, "c": c, "threshold": TRIANGLE_WEIGHT_THRESHOLD},
    ).scalar()

    if row is not None and row >= 3:
        return {w_a, w_b, w_c}
    return None


def _clear_citations(session: "ReplSession") -> None:
    """Reset synthesis citation state on the session."""
    session.last_synthesis_citations = {}
    session._synthesis_command_snapshot = 0
