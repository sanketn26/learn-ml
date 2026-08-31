# Exercise — Week 15 — The Pickle

## What you are building

A time-split vs shuffled AUC, 80 `predict()` calls with p50/p95 latency, a capacity threshold of 80 names, a drift overlay, and a one-page write-up.

## Predict before you run

1. Will shuffled AUC exceed time-split AUC?
2. Does `validate` accept `email` on the payload?
3. Will a 0.5 threshold flag more or fewer than 80 customers?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-15/starter.py
```

See the [exercise page](../../../docs/ml/exercises/week-15.md) for the five tasks. Use `pipelines.contract.validate` / `predict` and `joblib`, not a homemade `CustomerFeatures` type.

**1. Time wall.** `build_features` + `label_eventual_churn` (or `label_churn_in_horizon`). Split on `signup_date` (train = earlier 80% of signups, test = later 20%). Train the same GBT (with `make_preprocessor()` — `plan_type` is a string) on a *shuffled* split and on the time split. Report both AUCs. If they differ, write one sentence about why.

**2. `predict()` contract.** Import `validate` and `predict` from `pipelines.contract`. Do **not** invent a `CustomerFeatures` type. `validate` already rejects unknown keys and NaN. `predict` returns `{churn_score, flag_for_cs, model_version}`. Call it 80 times. Print p50 / p95 latency.

**3. Capacity, not 0.5.** From the time-split test set, pick the threshold that flags **at most 80** customers (CS budget). Report precision and recall at that cut. Compare to 0.5.

**4. Drift sketch.** Overlay histograms of `mrr`, `log_usage`, `tenure_so_far` for train vs later signups. One sentence: did the world move?

**5. One-page write-up.** (1) the time wall, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

Dump with `joblib` into `artifacts/<version>/model.joblib`, the same layout as `pipelines.train`.

## Success criteria

- Two AUCs (time vs shuffle).
- 80 `predict()` latencies and a response with `model_version`.
- Threshold for ≤80 flags vs 0.5.
- Artifact layout matches train.

## Debugging clues

- Extra keys (`user_id`, `email`) must 400.
- Dump the pipeline, not a naked forest.
- `plan_type` must be `str`.

## After you run

A pickle is an artifact plus a contract plus a budget. Random splits measure yesterday. Time splits measure next month.

## Lesson link

[Week 15 — The Pickle](../../../docs/ml/week-15.md)
