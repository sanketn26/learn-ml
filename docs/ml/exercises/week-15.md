# Exercises — Week 15 — The Pickle

Do these after reading [Week 15](../week-15.md).

**1. Time wall.** `build_features` + `label_eventual_churn` (or `label_churn_in_horizon`). Split on `signup_date` (train = earlier 80% of signups, test = later 20%). Train the same GBT (with `make_preprocessor()` — `plan_type` is a string) on a *shuffled* split and on the time split. Report both AUCs. If they differ, write one sentence about why.

**2. `predict()` contract.** Import `validate` and `predict` from `pipelines.contract`. Do **not** invent a `CustomerFeatures` type. `validate` already rejects unknown keys and NaN. `predict` returns `{churn_score, flag_for_cs, model_version}`. Call it 80 times. Print p50 / p95 latency.

**3. Capacity, not 0.5.** From the time-split test set, pick the threshold that flags **at most 80** customers (CS budget). Report precision and recall at that cut. Compare to 0.5.

**4. Drift sketch.** Overlay histograms of `mrr`, `log_usage`, `tenure_so_far` for train vs later signups. One sentence: did the world move?

**5. One-page write-up.** (1) the time wall, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

Starter: `python exercises/ml/week-15/starter.py` from the repo root. Dump with `joblib` into `artifacts/<version>/model.joblib`, the same layout as `pipelines.train`.
