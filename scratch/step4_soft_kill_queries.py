# Step 4 — Soft-Mode Kill/Keep Query List
# Pre-registration protocol: written BEFORE any \synthesize is run against these.
# Commit timestamp: 2026-11-06T13:52
# 
# Kill threshold (pre-registered): remove soft if <30% of soft-mode queries
# show rank shift >=1 for an in-scope result.
#
# Control: these 15 queries are written now. No query added after \synthesize
# results are seen. Post-hoc additions invalidate the measurement.
#
# Procedure:
#   1. Run \synthesize "luck and randomness in success"
#   2. Run each query below as .search "<query>" with router active
#   3. For each: log router.mode, boosted_count, shifted, reason
#      (from synthesis_router logger, see dot_commands.py instrumentation)
#   4. After ALL 15: count queries where shifted >= 1
#   5. Keep soft only if count >= 5 (30% of 15, rounded up)
#
# Queries are ordered by expected topic distance from "luck and randomness":
# Near (1-5): high chance of in-scope boost being relevant
# Mid (6-10): partial overlap expected
# Far (11-15): out-of-scope; expect 0 boost here — these are the control group
#
# NEAR (expected topic overlap with Taleb/Munger/Gould triangle)
Q1  = "survivorship bias in success stories"
Q2  = "randomness versus skill in performance"
Q3  = "the role of luck in financial outcomes"
Q4  = "narrative fallacy and hindsight"
Q5  = "fooled by skill when it was randomness"

# MID (partial overlap — probabilistic reasoning, bias, decision-making)
Q6  = "cognitive bias in decision making"
Q7  = "black swan events and tail risk"
Q8  = "uncertainty and human judgment"
Q9  = "mean reversion and regression to the mean"
Q10 = "sample size and statistical significance"

# FAR (control group — expect 0 rank shift; in-scope boost should do nothing)
Q11 = "daily habits and morning routines"
Q12 = "negotiation tactics and salary"
Q13 = "software architecture and design patterns"
Q14 = "meditation and mindfulness practice"
Q15 = "stoic philosophy and discipline"

# DO NOT MODIFY BELOW THIS LINE AFTER FIRST COMMIT
# Any result interpretation must use these exact queries against the same session.
