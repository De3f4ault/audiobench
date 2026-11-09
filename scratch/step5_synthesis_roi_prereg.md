# Step 5 — Option 2 Synthesis ROI Pre-Registration

**Written:** 2026-11-09, before any Option 2 synthesis results are observed.
**Purpose:** Pre-register the success criteria and measurement plan for the
Option 2 typed-edge injection test, so the result is interpretable and
the goalposts cannot be moved after the fact.

---

## What Is Being Tested

**Option 2 — Typed-edge metadata injection:**  After `retrieve_synthesis_context`
returns its top-k fragments, query `expression_relations` for any pair where
both `from_expression_id` and `to_expression_id` are in the retrieved set and
`created_by = 'step5_triadic_extraction'`. Inject those relation types as
"library graph hints" into the synthesis prompt before the LLM call.

**Injection format (pre-registered):**
```
Library graph hints (verify against passage text before citing):
  fragments [{i}, {j}] — {RELATION_TYPE} (prompt_v=1.1, weight={weight:.3f})
```

The "verify against passage text" instruction is mandatory. It prevents the LLM
from stating the edge label as fact if the passages don't support it, which
matters because v1.1 has a 1.4% residual error rate.

---

## Conditions

| Condition | Description |
|---|---|
| **Baseline** | `generate_synthesis(query, top_k=20)` — Structure 2 prompt, no typed-edge injection |
| **Injected** | Same, but with typed-edge annotation prepended to the user message |

**Query:** Q3 — `"Graham's stock selection rules and whether they still apply in modern markets"`

**Why Q3:** It has the most co-retrieved typed pairs at top_k=20 (5 pairs including
3 CONTRADICTS edges from Housel→Graham). It is the highest-signal test case.

**N per condition:** 5 (10 total synthesis calls). Temperature=0.3 (default).

**All 5 runs per condition use the same query and top_k.** Non-determinism at
temperature 0.3 is expected but should be small. The metric is computed on each
individual synthesis, not an average of embeddings.

---

## Pre-Registered Metrics

### Primary: Named-attribution density (M1)

Count sentences of the form *"[Author A] claims/argues/states X, while [Author B]
argues/claims Y"* where X and Y are explicitly contrasted.

**Procedure:** Read each synthesis output, count qualifying sentences manually.
A sentence qualifies if it:
1. Names at least two authors in the same sentence or consecutive sentences
2. Presents their positions as opposed ("while," "whereas," "in contrast," "however")
3. The opposition is substantive (not just "both discuss X but from different angles")

**Success threshold (pre-registered):** Injected M1 ≥ Baseline M1 × 1.3
(30% more named attributions on average across 5 runs). If the gap is smaller
than 30%, injection is not producing distinguishable value over Structure 2 alone.

### Secondary: Citation co-occurrence density (M2)

For each synthesis, count sentences containing two or more distinct `[N]` citation
markers. Example: "Housel [3] argues X while Graham [7] argues Y" counts as 1.

**Procedure:** Regex count — `len(re.findall(r'\[\d+\].*\[\d+\]', sentence))`
across all sentences in sections 2 (Dialectic Tension) and 3 (Integrated Synthesis).

**Success threshold (pre-registered):** Injected M2 ≥ Baseline M2 × 1.2
(20% more multi-citation sentences).

### Tertiary: Blind quality score (M3)

Read 5 baseline vs 5 injected syntheses for Q3, blind to condition (randomly
interleaved, no labels visible), score each 1–5 on:
*"Does this synthesis identify a specific, named disagreement the user could
pursue further?"*

**Success threshold (pre-registered):** Mean injected M3 ≥ Mean baseline M3 + 0.5.

---

## Decision Criteria (pre-registered)

**Ship Option 2 as v1.1 of the router if:** M1 passes AND (M2 passes OR M3 passes).
Requires at least 2 of 3 metrics to pass, with M1 (the primary) mandatory.

**Do not ship, reconsider architecture if:** M1 fails. If the injection doesn't
increase named-attribution density, the typed edges aren't changing what the LLM
says — they're being ignored, overridden by Structure 2's inherent friction-finding,
or producing the wrong labels loudly enough to suppress correct inferences.

**Ambiguous result — investigate before deciding if:** M1 passes marginally (1.0x–1.3x)
and M2/M3 are mixed. In this case, read the 5 injected syntheses carefully for
active harm: does any synthesis state a typed-edge label that contradicts what
the passages actually say? If yes, the 1.4% residual error rate is producing
synthesis errors, and the injection format needs strengthening before the threshold
question becomes meaningful.

---

## Known Confounds (pre-registered)

1. **top_k change from 8 → 20 is a confound.** The baseline at top_k=20 may
   already be better than the baseline at top_k=8, independent of typed edges.
   This is acceptable for the first test — the question being answered is
   "does typed-edge injection + top_k=20 beat pure-Structure-2 + top_k=20,"
   not "does injection beat the current production config." A separate measurement
   of top_k=8 vs top_k=20 without injection should be run afterward.

2. **Q3 is a named-claim query, not an abstract query.** Results generalize to
   named-claim and named-author queries only. Q1 (abstract) is structurally
   excluded (Outcome C confirmed). Do not generalize to Q1-type queries.

3. **v1.1 prompt was used for the 2 re-scored mislabeled edges.** All other 140
   edges were scored under v1.0. The injection pool is therefore a mix of v1.0
   and v1.1 classifications. Edge selection filters to `prompt_version IN ('v1.0', 'v1.1')`
   for the injection query.

---

## What Happens After

- **If ship:** Wire Option 2 into `synthesis_engine.py` properly, behind a flag
  (`use_typed_edges: bool = True`). Write tests. Commit.
- **If don't ship:** Document which metric failed and why. The typed edges remain
  in the graph as inert data — they're still correct, still queryable, still useful
  for future integration approaches (entity-anchored retrieval, v2 architecture).
- **If ambiguous:** Strengthen injection format (e.g., add passage fragment text
  to the hint so the LLM can verify inline), re-run M1 only, re-apply criterion.

---

*This file was written before any Option 2 synthesis runs. Do not modify the
thresholds after seeing results. If the thresholds turn out to be wrong in
principle (not in outcome), write a new pre-registration and document why the
old one was invalidated.*
