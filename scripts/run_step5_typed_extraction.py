"""Step 5 — Typed Relation Extraction Runner.

Classifies semantic edges within triadic cross-work cliques as one of:
  ELABORATES   — A develops/extends/operationalises B's premise
  CONTRADICTS  — A and B make incompatible empirical or normative claims
  INSPIRED_BY  — A explicitly cites, draws from, or was inspired by B
  INDEPENDENT  — A and B converge without direct influence or opposition

SCHEMA DECISIONS (not oversights):
  • INDEPENDENT is never persisted — the absence of an ELABORATES/CONTRADICTS/
    INSPIRED_BY row for a pair IS the signal. Do not "fix" missing INDEPENDENT
    rows by inserting them.
  • CONTRADICTS is stored with canonical direction: lower expression_id → higher.
    The 'subject' field in the LLM response is ignored for CONTRADICTS.
  • ELABORATES  → stored as subject → other  (subject is the elaborating passage)
  • INSPIRED_BY → stored as subject → other  (subject is the citing/influenced passage)

IDEMPOTENCY + PROMPT VERSIONING (Gaps 6 & 7):
  Each row is tagged with created_by='step5_triadic_extraction' AND prompt_version.
  The idempotency skip-check includes prompt_version, so a prompt change between
  iterations naturally re-classifies all pairs under the new version while a
  crash-and-restart under the same version safely skips already-classified pairs.

DEDUP (Gap 5):
  Phase 1 builds a pending classification table in memory.
  Conflicts (same pair, same prompt_version) are resolved by keeping the
  classification from the triangle with the highest min_edge weight.
  Phase 2 persists only the surviving classifications.

Usage:
  python scripts/run_step5_typed_extraction.py [--dry-run] [--first N]

  --dry-run   : classify and log but do not write to expression_relations
  --first N   : stop after N LLM calls (useful for prompt calibration checks)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audiobench.chat.providers.ollama_provider import OllamaClient
from audiobench.core.db_session import get_session
from audiobench.memory.boilerplate import is_pure_boilerplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"
MIN_TRIANGLE_WEIGHT = 0.82
DUAL_SAMPLE_MIN_SPREAD = 0.015

# Single-triangle author triples: take 1 triangle each (highest min-edge)
SINGLE_SAMPLE_QUOTA = 38
# Multi-triangle author triples: take peak + floor if spread >= threshold
MULTI_SAMPLE_QUOTA = 30

CREATED_BY = "step5_triadic_extraction"

# Bump this string whenever the extraction prompt changes so that
# the idempotency check correctly re-runs all pairs under the new prompt.
# Format: v<major>.<minor> — increment minor for wording tweaks, major for
# schema or option-set changes.
PROMPT_VERSION = "v1.1"

VALID_RELATIONS = {"ELABORATES", "CONTRADICTS", "INSPIRED_BY", "INDEPENDENT"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step5")

# ---------------------------------------------------------------------------
# Mining helpers (inlined from deep_test_multihop_cliques.py)
# ---------------------------------------------------------------------------

from collections import defaultdict as _dd
from sqlalchemy import text as _sql


def mine_cross_work_triangles(
    min_weight: float = MIN_TRIANGLE_WEIGHT,
) -> tuple[list[dict], list[dict]]:
    """Return (total_triangles, secular_triangles) dict lists.

    Each triangle dict has keys:
      a_id, b_id, c_id          — expression IDs (integers)
      w_ab, w_bc, w_ca          — edge weights
      min_w, avg_w              — triangle quality metrics
      a_wid, b_wid, c_wid      — work IDs
      a_author, b_author, c_author
      a_title, b_title, c_title
      a_content, b_content, c_content
    """
    with get_session() as session:
        edges = session.execute(
            _sql("""
                SELECT from_expression_id, to_expression_id, weight
                FROM expression_relations
                WHERE relation_type IN ('semantic', 'thematic') AND weight >= :t
            """),
            {"t": min_weight},
        ).fetchall()

        adj: dict[int, set[int]] = _dd(set)
        edge_weights: dict[tuple[int, int], float] = {}
        for u, v, w in edges:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
            edge_weights[(u, v)] = max(edge_weights.get((u, v), 0.0), float(w))
            edge_weights[(v, u)] = max(edge_weights.get((v, u), 0.0), float(w))

        rows = session.execute(
            _sql("""
                SELECT e.id, e.work_id, w.author, w.title, e.content
                FROM expressions e
                JOIN works w ON e.work_id = w.id
                WHERE e.id IN (
                    SELECT from_expression_id FROM expression_relations
                    WHERE relation_type IN ('semantic', 'thematic') AND weight >= :t
                    UNION
                    SELECT to_expression_id FROM expression_relations
                    WHERE relation_type IN ('semantic', 'thematic') AND weight >= :t
                )
            """),
            {"t": min_weight},
        ).fetchall()

        meta: dict[int, dict] = {}
        for eid, wid, author, title, content in rows:
            meta[eid] = {
                "work_id": wid,
                "author": author or "Unknown Author",
                "title": title or "Unknown Title",
                "content": content or "",
            }

    total_triangles: list[dict] = []
    secular_triangles: list[dict] = []

    for u in adj:
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w <= v:
                    continue
                mu, mv, mw = meta.get(u), meta.get(v), meta.get(w)
                if not mu or not mv or not mw:
                    continue
                wu, wv, ww = mu["work_id"], mv["work_id"], mw["work_id"]
                if not (wu and wv and ww and wu != wv and wv != ww and wu != ww):
                    continue
                if (
                    is_pure_boilerplate(mu["content"])
                    or is_pure_boilerplate(mv["content"])
                    or is_pure_boilerplate(mw["content"])
                ):
                    continue

                w_uv = edge_weights[(u, v)]
                w_vw = edge_weights[(v, w)]
                w_uw = edge_weights[(u, w)]
                min_e = min(w_uv, w_vw, w_uw)
                avg_e = (w_uv + w_vw + w_uw) / 3.0

                item = {
                    "a_id": u, "b_id": v, "c_id": w,
                    "w_ab": w_uv, "w_bc": w_vw, "w_ca": w_uw,
                    "min_w": min_e, "avg_w": avg_e,
                    "a_wid": wu, "b_wid": wv, "c_wid": ww,
                    "a_author": mu["author"], "b_author": mv["author"], "c_author": mw["author"],
                    "a_title": mu["title"],   "b_title": mv["title"],   "c_title": mw["title"],
                    "a_content": mu["content"], "b_content": mv["content"], "c_content": mw["content"],
                }
                total_triangles.append(item)

                au, av, aw = mu["author"], mv["author"], mw["author"]
                if (
                    au != av and av != aw and au != aw
                    and "Unknown Author" not in (au, av, aw)
                ):
                    secular_triangles.append(item)

    return total_triangles, secular_triangles


# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------

def _is_scripture_triple(triple: tuple[str, ...]) -> bool:
    """Return True if the author triple is dominated by scriptural/devotional works.

    These triangles have high edge weights because they share verbatim text
    (Bible verses, liturgical passages), not because of cross-disciplinary idea
    transfer. They are correctly classified by the LLM but add no value to the
    intellectual graph.

    Mirrors the showcase filter in deep_test_multihop_cliques.py.
    """
    # Keywords that identify devotional/scriptural channels (case-insensitive)
    SCRIPTURE_KEYWORDS = {
        "holy bible", "the bible", "repentant soul", "godseekers",
        "breathe bible", "bible study", "devotional", "kjv",
    }
    auth_str = " ".join(t.lower() for t in triple)
    # Block if 2+ authors in the triple are scripture-domain
    scripture_count = sum(
        1 for author in triple
        if any(kw in author.lower() for kw in SCRIPTURE_KEYWORDS)
    )
    # Allow Ehrman-style academic biblical studies (keeps scholarly triangles)
    has_scholar = any(
        kw in auth_str for kw in ("ehrman", "brakke", "pagels", "elaine", "bart")
    )
    return scripture_count >= 2 and not has_scholar


def build_work_queue(
    secular_triangles: list[dict],
    single_quota: int = SINGLE_SAMPLE_QUOTA,
    multi_quota: int = MULTI_SAMPLE_QUOTA,
    spread_threshold: float = DUAL_SAMPLE_MIN_SPREAD,
) -> list[dict]:
    """Group triangles by author-triple, then apply the dual-sampling strategy.

    Single-triangle triples  → emit the triangle with the highest min_edge.
    Multi-triangle triples   → emit:
        • peak  = triangle with highest min_edge (always)
        • floor = triangle with lowest min_edge, only if
                  (peak.min_w - floor.min_w) >= spread_threshold
    Returns list of triangle dicts, each augmented with _sample_role.
    """
    by_triple: dict[tuple, list[dict]] = defaultdict(list)
    for tr in secular_triangles:
        triple = tuple(sorted([tr["a_author"], tr["b_author"], tr["c_author"]]))
        by_triple[triple].append(tr)

    # Filter out scripture-only triples before sampling
    filtered_out = sum(1 for t in by_triple if _is_scripture_triple(t))
    by_triple = {t: v for t, v in by_triple.items() if not _is_scripture_triple(t)}
    if filtered_out:
        log.info(
            "  Filtered %d scripture-only author triples (verbatim text, not idea transfer)",
            filtered_out,
        )

    single_triples = {t: v for t, v in by_triple.items() if len(v) == 1}
    multi_triples  = {t: v for t, v in by_triple.items() if len(v) > 1}

    queue: list[dict] = []

    # Multi first (more signal per call)
    for triple, trs in sorted(multi_triples.items(), key=lambda x: max(t["min_w"] for t in x[1]), reverse=True):
        if len(queue) >= multi_quota * 2:  # rough cap; exact quota enforced below
            break
        peak  = max(trs, key=lambda t: t["min_w"])
        floor = min(trs, key=lambda t: t["min_w"])
        entry = {**peak, "_sample_role": "peak", "_triple": triple}
        queue.append(entry)
        if (peak["min_w"] - floor["min_w"]) >= spread_threshold and peak is not floor:
            queue.append({**floor, "_sample_role": "floor", "_triple": triple})

    multi_calls = len(queue)

    # Single triples fill the remainder
    for triple, trs in sorted(single_triples.items(), key=lambda x: x[1][0]["min_w"], reverse=True):
        if len([q for q in queue if q["_sample_role"] == "single"]) >= single_quota:
            break
        entry = {**trs[0], "_sample_role": "single", "_triple": triple}
        queue.append(entry)

    log.info(
        "Work queue: %d total calls (%d multi-triple, %d single-triple)",
        len(queue),
        multi_calls,
        len(queue) - multi_calls,
    )
    return queue


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a Semantic Relation Classifier for an intellectual graph.

Your task: given three passages from different books, classify the pairwise semantic relation for each of the three pairs.

For each pair, output one of these relation types:
  ELABORATES   — one passage develops, extends, operationalises, or concretises the other's premise
  CONTRADICTS  — the passages make mutually incompatible empirical or normative claims
  INSPIRED_BY  — one passage explicitly cites, attributes ideas to, or was clearly influenced by the other
  INDEPENDENT  — the passages converge thematically but show no direct influence or opposition

Critical rule: if a passage cites or references another work primarily to argue against or refute its claims, classify the pair as CONTRADICTS — not ELABORATES or INSPIRED_BY.

For ELABORATES and INSPIRED_BY, you must also identify the 'subject': which passage (A, B, or C) is doing the elaborating or citing. The 'subject' points FROM the derivative/citing work TO the foundational/cited work.

Output Rules:
1. Return ONLY a JSON object with a single key "edges" containing a list of exactly 3 edge objects.
2. Each edge object must have exactly these fields:
   - "pair": ["X", "Y"]  where X and Y are "A", "B", or "C"
   - "relation": one of ELABORATES | CONTRADICTS | INSPIRED_BY | INDEPENDENT
   - "subject": "A" | "B" | "C" for ELABORATES and INSPIRED_BY; null for CONTRADICTS and INDEPENDENT
   - "justification": one sentence citing specific text from both passages
3. Cover all three pairs: [A,B], [A,C], [B,C].
4. Do not add any text outside the JSON object."""


def build_extraction_prompt(triangle: dict) -> str:
    """Build the user-turn extraction prompt for a single triangle."""
    return f"""Classify the pairwise semantic relations between these three passages:

Passage A (ID #{triangle['a_id']}) — {triangle['a_author']}, "{triangle['a_title']}":
\"\"\"{triangle['a_content'][:600].strip()}\"\"\"

Passage B (ID #{triangle['b_id']}) — {triangle['b_author']}, "{triangle['b_title']}":
\"\"\"{triangle['b_content'][:600].strip()}\"\"\"

Passage C (ID #{triangle['c_id']}) — {triangle['c_author']}, "{triangle['c_title']}":
\"\"\"{triangle['c_content'][:600].strip()}\"\"\"

Classify each pair. Output only the JSON object."""


# ---------------------------------------------------------------------------
# Idempotency check (Gap 6 + prompt_version from Gap 7)
# ---------------------------------------------------------------------------

def already_classified(
    session,
    id_a: int,
    id_b: int,
    prompt_version: str = PROMPT_VERSION,
    min_edge: float = 0.0,
) -> bool:
    """Return True if this pair already has a classification under the same
    prompt_version with an equal-or-stronger triangle min_edge.

    Uses OR on both directions so a pair (A→B) and (B→A) are treated as equivalent.
    """
    rows = session.execute(
        _sql("""
            SELECT weight FROM expression_relations
            WHERE created_by = :created_by
              AND prompt_version = :pv
              AND (
                (from_expression_id = :a AND to_expression_id = :b)
                OR
                (from_expression_id = :b AND to_expression_id = :a)
              )
        """),
        {
            "created_by": CREATED_BY,
            "pv": prompt_version,
            "a": id_a,
            "b": id_b,
        },
    ).fetchall()

    if not rows:
        return False

    existing_min_edge = max(float(r[0]) for r in rows)
    return existing_min_edge >= min_edge


# ---------------------------------------------------------------------------
# LLM call + response validation
# ---------------------------------------------------------------------------

def call_llm(client: OllamaClient, system: str, user: str) -> str:
    """Call the Ollama client and return the raw text response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    # OllamaClient.chat() returns {"content": str, "thinking": str|None}
    # think=False: cloud-routed endpoints reject the think field; structured
    # JSON output also doesn't benefit from chain-of-thought separation.
    resp = client.chat(messages=messages, temperature=0.1, think=False)
    return resp.get("content", "").strip()


def _extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from the raw LLM response."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def validate_response(parsed: dict, triangle: dict) -> list[dict] | None:
    """Validate the parsed LLM response and return a clean list of edge dicts.

    Returns None on hard validation failure (caller should log and skip).
    Each returned dict has: pair, relation, subject, justification, from_id, to_id.
    """
    id_map = {"A": triangle["a_id"], "B": triangle["b_id"], "C": triangle["c_id"]}

    if "edges" not in parsed:
        return None
    edges = parsed["edges"]
    if not isinstance(edges, list) or len(edges) != 3:
        return None

    expected_pairs = {("A", "B"), ("A", "C"), ("B", "C")}
    seen_pairs: set[tuple] = set()
    result: list[dict] = []

    for edge in edges:
        pair = edge.get("pair", [])
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        x, y = str(pair[0]).upper(), str(pair[1]).upper()
        # Normalize order
        canonical = (min(x, y), max(x, y))
        if canonical not in expected_pairs:
            return None
        if canonical in seen_pairs:
            return None
        seen_pairs.add(canonical)

        relation = str(edge.get("relation", "")).upper()
        if relation not in VALID_RELATIONS:
            return None

        subject = edge.get("subject")
        if relation in ("ELABORATES", "INSPIRED_BY"):
            if subject not in ("A", "B", "C"):
                return None
        else:
            # CONTRADICTS and INDEPENDENT must have null subject
            subject = None

        justification = str(edge.get("justification", ""))[:500]

        # Map subject/other to expression IDs
        from_id: int
        to_id: int
        if relation == "CONTRADICTS":
            # Canonical direction: lower → higher expression ID
            from_id = min(id_map[x], id_map[y])
            to_id   = max(id_map[x], id_map[y])
        elif relation in ("ELABORATES", "INSPIRED_BY"):
            subject_upper = str(subject).upper()
            other = y if subject_upper == x else x
            from_id = id_map[subject_upper]
            to_id   = id_map[other]
        else:  # INDEPENDENT
            from_id = id_map[x]
            to_id   = id_map[y]

        result.append({
            "pair": [x, y],
            "relation": relation,
            "subject": subject,
            "justification": justification,
            "from_id": from_id,
            "to_id": to_id,
        })

    return result


# ---------------------------------------------------------------------------
# Pending classification table (in-memory, Gap 5 dedup)
# ---------------------------------------------------------------------------

class PendingClassifications:
    """In-memory accumulator with highest-min-edge-wins dedup.

    Key: (from_id, to_id, relation, prompt_version)
    Value: {'min_edge': float, 'justification': str, ...}

    For INDEPENDENT we accumulate but Phase 3 drops them before persisting.
    """

    def __init__(self) -> None:
        # (from_id, to_id) → best classification so far
        self._by_pair: dict[tuple[int, int], dict] = {}

    def add(
        self,
        from_id: int,
        to_id: int,
        relation: str,
        min_edge: float,
        justification: str,
        triangle: dict,
    ) -> bool:
        """Add a classification. Returns True if this entry wins (is kept)."""
        key = (from_id, to_id)
        existing = self._by_pair.get(key)
        if existing is None or min_edge > existing["min_edge"]:
            self._by_pair[key] = {
                "from_id": from_id,
                "to_id": to_id,
                "relation": relation,
                "min_edge": min_edge,
                "justification": justification,
                "triangle_a_id": triangle["a_id"],
                "triangle_b_id": triangle["b_id"],
                "triangle_c_id": triangle["c_id"],
            }
            return True
        return False

    def winners(self) -> list[dict]:
        return list(self._by_pair.values())


# ---------------------------------------------------------------------------
# Phase 3 — Persist winners
# ---------------------------------------------------------------------------

def persist_winners(winners: list[dict], dry_run: bool) -> dict[str, int]:
    """Write non-INDEPENDENT winners to expression_relations.

    Returns counts by relation_type.
    """
    counts: dict[str, int] = defaultdict(int)
    now = datetime.now(timezone.utc).isoformat()

    to_persist = [w for w in winners if w["relation"] != "INDEPENDENT"]

    if dry_run:
        log.info("[DRY-RUN] Would persist %d edges (skipping %d INDEPENDENT)",
                 len(to_persist), len(winners) - len(to_persist))
        for w in to_persist:
            log.info("[DRY-RUN] %s %d → %d  (min_edge=%.3f)",
                     w["relation"], w["from_id"], w["to_id"], w["min_edge"])
            counts[w["relation"]] += 1
        return dict(counts)

    with get_session() as session:
        for w in to_persist:
            # Idempotency: check if row already exists under CURRENT prompt_version
            existing_row = session.execute(
                _sql("""
                    SELECT id FROM expression_relations
                    WHERE from_expression_id = :from_id
                      AND to_expression_id = :to_id
                      AND created_by = :created_by
                      AND prompt_version = :pv
                """),
                {
                    "from_id": w["from_id"],
                    "to_id": w["to_id"],
                    "created_by": CREATED_BY,
                    "pv": PROMPT_VERSION,
                },
            ).fetchone()

            if existing_row:
                # Same version already exists — update weight/relation in-place
                session.execute(
                    _sql("""
                        UPDATE expression_relations
                        SET relation_type = :rel,
                            weight = :weight,
                            created_at = :now
                        WHERE id = :row_id
                    """),
                    {
                        "rel": w["relation"].lower(),
                        "weight": w["min_edge"],
                        "now": now,
                        "row_id": existing_row[0],
                    },
                )
            else:
                # Check for a prior-version row with same (from, to, relation_type).
                # If found, UPDATE it to current version rather than INSERT (which
                # would violate the UNIQUE(from, to, relation_type) constraint).
                prior_same_rel = session.execute(
                    _sql("""
                        SELECT id FROM expression_relations
                        WHERE from_expression_id = :from_id
                          AND to_expression_id = :to_id
                          AND relation_type = :rel
                          AND created_by = :created_by
                    """),
                    {
                        "from_id": w["from_id"],
                        "to_id": w["to_id"],
                        "rel": w["relation"].lower(),
                        "created_by": CREATED_BY,
                    },
                ).fetchone()

                if prior_same_rel:
                    # Existing row has same label under old version — bump to new version
                    session.execute(
                        _sql("""
                            UPDATE expression_relations
                            SET prompt_version = :pv,
                                weight = :weight,
                                created_at = :now
                            WHERE id = :row_id
                        """),
                        {
                            "pv": PROMPT_VERSION,
                            "weight": w["min_edge"],
                            "now": now,
                            "row_id": prior_same_rel[0],
                        },
                    )
                else:
                    # Genuinely new (from, to, relation_type) — INSERT.
                    # May replace an old row with a different relation_type (e.g.
                    # INSPIRED_BY → CONTRADICTS across prompt versions). In that case
                    # the old row remains in the DB under its old prompt_version;
                    # consumers should filter by prompt_version to get the current view.
                    session.execute(
                        _sql("""
                            INSERT INTO expression_relations
                                (from_expression_id, to_expression_id, relation_type,
                                 weight, created_by, created_at, prompt_version)
                            VALUES
                                (:from_id, :to_id, :rel,
                                 :weight, :created_by, :now, :pv)
                        """),
                        {
                            "from_id": w["from_id"],
                            "to_id": w["to_id"],
                            "rel": w["relation"].lower(),
                            "weight": w["min_edge"],
                            "created_by": CREATED_BY,
                            "now": now,
                            "pv": PROMPT_VERSION,
                        },
                    )
            counts[w["relation"]] += 1

        session.commit()

    return dict(counts)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, first_n: int | None = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path("scratch") / f"step5_extraction_{ts}.jsonl"
    log_path.parent.mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("Step 5 — Typed Relation Extraction  (prompt_version=%s, model=%s)",
             PROMPT_VERSION, MODEL)
    if dry_run:
        log.info("DRY-RUN mode — no writes to expression_relations")
    log.info("=" * 70)

    # ── Phase 1: Load & sample ──────────────────────────────────────────────
    log.info("Phase 1: Mining cross-work triangles (threshold=%.2f)…", MIN_TRIANGLE_WEIGHT)
    t0 = time.monotonic()
    total_trs, secular_trs = mine_cross_work_triangles(MIN_TRIANGLE_WEIGHT)
    log.info("  Mined %d total / %d secular triangles in %.2fs",
             len(total_trs), len(secular_trs), time.monotonic() - t0)

    queue = build_work_queue(secular_trs, SINGLE_SAMPLE_QUOTA, MULTI_SAMPLE_QUOTA)
    if first_n is not None:
        queue = queue[:first_n]
        log.info("  --first %d: truncated queue to %d calls", first_n, len(queue))

    # ── Phase 2: Classify ───────────────────────────────────────────────────
    log.info("Phase 2: Classifying %d triangles…", len(queue))
    client = OllamaClient(model=MODEL)

    pending = PendingClassifications()
    audit_records: list[dict] = []

    stats = {"llm_calls": 0, "skipped_existing": 0, "errors": 0, "edges_pending": 0}

    with get_session() as session:
        for call_idx, triangle in enumerate(queue, 1):
            min_edge = triangle["min_w"]
            triple   = " ↔ ".join(str(a) for a in triangle["_triple"])

            # --- Idempotency check: skip all three pairs if the dominant pair
            #     is already strongly classified under this prompt_version ---
            pairs = [
                (triangle["a_id"], triangle["b_id"]),
                (triangle["a_id"], triangle["c_id"]),
                (triangle["b_id"], triangle["c_id"]),
            ]
            skip_count = sum(
                1 for (a, b) in pairs
                if already_classified(session, a, b, PROMPT_VERSION, min_edge)
            )
            if skip_count == 3:
                log.info(
                    "[%d/%d] SKIP  %s  (all pairs already classified, min_edge=%.3f)",
                    call_idx, len(queue), triple, min_edge,
                )
                stats["skipped_existing"] += 1
                audit_records.append({
                    "call_idx": call_idx, "action": "skip_existing",
                    "triple": triple, "min_edge": min_edge,
                    "triangle_ids": [triangle["a_id"], triangle["b_id"], triangle["c_id"]],
                })
                continue

            # --- LLM call ---
            user_prompt = build_extraction_prompt(triangle)
            t_call = time.monotonic()
            try:
                raw = call_llm(client, _EXTRACTION_SYSTEM_PROMPT, user_prompt)
                elapsed = time.monotonic() - t_call
            except Exception as exc:
                elapsed = time.monotonic() - t_call
                log.warning("[%d/%d] ERROR  %s  LLM call failed: %s", call_idx, len(queue), triple, exc)
                stats["errors"] += 1
                audit_records.append({
                    "call_idx": call_idx, "action": "llm_error", "triple": triple,
                    "min_edge": min_edge, "error": str(exc), "elapsed_s": elapsed,
                })
                continue

            stats["llm_calls"] += 1
            log.info(
                "[%d/%d] %.2fs  %s  (min_edge=%.3f, role=%s)",
                call_idx, len(queue), elapsed, triple, min_edge,
                triangle.get("_sample_role", "?"),
            )

            # --- Parse & validate ---
            parsed = _extract_json(raw)
            if parsed is None:
                log.warning("[%d/%d] PARSE_FAIL  raw=%r", call_idx, len(queue), raw[:300])
                stats["errors"] += 1
                audit_records.append({
                    "call_idx": call_idx, "action": "parse_fail", "triple": triple,
                    "min_edge": min_edge, "raw": raw[:600], "elapsed_s": elapsed,
                })
                continue

            edges = validate_response(parsed, triangle)
            if edges is None:
                log.warning("[%d/%d] VALIDATE_FAIL  parsed=%s", call_idx, len(queue), json.dumps(parsed)[:400])
                stats["errors"] += 1
                audit_records.append({
                    "call_idx": call_idx, "action": "validate_fail", "triple": triple,
                    "min_edge": min_edge, "parsed": parsed, "elapsed_s": elapsed,
                })
                continue

            # --- Accumulate into pending (with dedup) ---
            for edge in edges:
                won = pending.add(
                    from_id=edge["from_id"],
                    to_id=edge["to_id"],
                    relation=edge["relation"],
                    min_edge=min_edge,
                    justification=edge["justification"],
                    triangle=triangle,
                )
                stats["edges_pending"] += 1
                log.debug(
                    "  %s %d → %d  (%s, min_edge=%.3f) %s",
                    edge["relation"], edge["from_id"], edge["to_id"],
                    "won" if won else "lost_dedup", min_edge, edge["justification"][:80],
                )

            audit_records.append({
                "call_idx": call_idx, "action": "classified", "triple": triple,
                "min_edge": min_edge, "elapsed_s": elapsed,
                "triangle_ids": [triangle["a_id"], triangle["b_id"], triangle["c_id"]],
                "edges": edges,
            })

    # ── Phase 3: Resolve & persist ──────────────────────────────────────────
    log.info("Phase 3: Resolving %d pending edges…", stats["edges_pending"])
    winners = pending.winners()
    relation_counts = defaultdict(int)
    for w in winners:
        relation_counts[w["relation"]] += 1

    log.info(
        "  Dedup resolved: %d unique pairs  (%s)",
        len(winners),
        ", ".join(f"{k}={v}" for k, v in sorted(relation_counts.items())),
    )

    persisted = persist_winners(winners, dry_run)

    # ── Phase 4: Report ─────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Phase 4: Results")
    log.info("  LLM calls made:        %d", stats["llm_calls"])
    log.info("  Skipped (existing):    %d", stats["skipped_existing"])
    log.info("  Errors:                %d", stats["errors"])
    log.info("  Edges pending (pre-dedup): %d", stats["edges_pending"])
    log.info("  Unique pairs (post-dedup): %d", len(winners))
    log.info("  INDEPENDENT (not persisted): %d", relation_counts.get("INDEPENDENT", 0))
    log.info("  Persisted:")
    for rel in sorted(persisted):
        log.info("    %-14s %d", rel, persisted[rel])
    log.info("=" * 70)

    # Sample 5 of each persisted relation type for human review
    for rel in sorted(persisted):
        samples = [w for w in winners if w["relation"] == rel][:5]
        log.info("Sample %ss:", rel)
        for w in samples:
            log.info(
                "  %d → %d  min_edge=%.3f  justification=%r",
                w["from_id"], w["to_id"], w["min_edge"], w["justification"][:100],
            )

    # Write audit log
    with log_path.open("w", encoding="utf-8") as fh:
        for rec in audit_records:
            fh.write(json.dumps(rec, default=str) + "\n")
    log.info("Audit log → %s", log_path)

    # Distribution warning: check for CONTRADICTS anomaly on first_n runs
    if first_n is not None:
        contradicts = relation_counts.get("CONTRADICTS", 0)
        total_non_independent = sum(v for k, v in relation_counts.items() if k != "INDEPENDENT")
        if total_non_independent > 0:
            contradicts_pct = contradicts / max(total_non_independent, 1) * 100
            if contradicts_pct > 20:
                log.warning(
                    "CONTRADICTS=%.0f%% (>20%%) — prompt may be over-eager. "
                    "Inspect and consider refining before full run.",
                    contradicts_pct,
                )
            elif contradicts == 0 and total_non_independent >= 10:
                log.warning(
                    "CONTRADICTS=0 across %d non-INDEPENDENT classifications — "
                    "prompt may be missing epistemic disagreement. Inspect before full run.",
                    total_non_independent,
                )
            else:
                log.info(
                    "Distribution check: CONTRADICTS=%.0f%% (%d/%d) — within expected 0–20%% range.",
                    contradicts_pct, contradicts, total_non_independent,
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",   action="store_true", help="Classify but do not write to DB")
    parser.add_argument("--first",     type=int, default=None, metavar="N",
                        help="Stop after N LLM calls (for prompt calibration)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, first_n=args.first)
