# Exercises — Week 12 — Capstone

Do these after reading [Week 12 — Capstone](../week-12.md).

**1. Time wall.** Load Customer 360. Split on `signup_date` (train = earlier 80% of signups, test = later 20%). Train the same GBT on a *shuffled* split and on the time split. Report both AUCs. If they differ, write one sentence about why.

**2. `predict()` contract.** Write `validate(payload: dict) -> CustomerFeatures` that rejects unknown keys and missing required fields. Write `predict(payload) -> dict` that returns `{churn_score, model_version}`. Call it 80 times. Print p50 / p95 latency.

**3. Capacity, not 0.5.** From the time-split test set, pick the threshold that flags **at most 80** customers (CS budget). Report precision and recall at that cut. Compare to 0.5.

**4. Drift sketch.** Overlay histograms of `mrr`, `log_usage`, `tenure_days` for train vs later signups. One sentence: did the world move?

**5. One-page write-up.** (1) the time wall, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

Starter: `python exercises/ml/week-12/starter.py` from the repo root.
