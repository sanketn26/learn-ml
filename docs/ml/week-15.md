# Week 15 — The Pickle: A Training Script Is Not Production

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers about to pickle a model and get paged for it.

This is still a **training script**. We will be explicit about what it is not: no feature store, no canary, no GDPR review, no CI. The GPU coding-specialist capstone is a different page (`docs/ml/capstone.md`).

---

## 🎯 What you will be able to do

- Wire `build_features` → time-based split → GBT → a versioned artifact
- Call `pipelines.contract.validate` / `predict` — extra keys and NaN fail loud
- Pick a threshold from **CS capacity**, not from 0.5
- Draw a crude drift picture (`tenure_so_far`, this month vs train)
- Time 80 `predict()` calls so you have a latency number, not a vibe
- List what you are *not* deploying

!!! think "Think of it like… a build artifact + an API contract."

    The model file is a binary. The scaler is part of that binary. The request body is `FEATURE_COLS`. If any of those three drift independently, production is a silent wrong-number generator.

## If you already write software

A training script is a scratchpad. Production is a contract.

This week is the one that separates “I trained a thing” from “we can call it on Tuesday.”

```
Explore (weeks 6–14)
    │  time-based split, pick a model, write predict()
    ▼
Ship
    predict(payload) -> {churn_score, flag_for_cs, model_version}
    the same features as training (tenure_so_far, not lifetime tenure_days)
    the scaler travels with the joblib
    a latency budget (we time 80 calls)
    a drift picture (did live traffic stop looking like train?)
```

### Time-based split is not optional

`train_test_split(..., shuffle=True)` is fine for a homework iris set. It is a lie for SaaS. Customers in the “test” set would include people from the same week as train — and tomorrow’s traffic is *next* week. Split on signup or on event time. Train on the past. Test on the future. Same rule as backtesting a trading strategy, or as not using tomorrow’s logs to tune today’s alert.

### Picture the contract

```python
from pipelines.contract import predict

# payload keys == FEATURE_COLS
# extra key → ValueError, missing field → ValueError, NaN → ValueError
# returns {"churn_score": float, "flag_for_cs": bool, "model_version": str}
```

If you cannot call that function without reaching back into a throwaway script, you do not have a model. You have a souvenir.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all ~49k rows.

```python
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pipelines.contract import predict, validate
from pipelines.features import (
    AS_OF_DEFAULT,
    FEATURE_COLS,
    build_features,
    make_preprocessor,
)
from pipelines.labels import drop_unlabelled, label_eventual_churn
```

## Architecture (the only diagram that matters)

```
 warehouse CSVs
      │  nightly job
      ▼
 build_features(as_of)  ──►  time split  ──►  train pipeline  ──►  artifact (joblib + metrics)
                         │                                      │
                         └── holdout report                     ▼
                                                         predict(payload)  {score, flag, version}
                                                                │
                                                                ▼
                                                         monitor: score volume,
                                                         feature histograms, weekly PR-AUC
```

!!! engineer "Engineer mental model"

    Train on the *past*, test on the *more recent past*. Random shuffle is a unit test. A time wall is an integration test against reality.

```python
as_of = AS_OF_DEFAULT
df = build_features(as_of=as_of, n=None, at_risk_only=True)
y = label_eventual_churn(df, as_of)
df, y = drop_unlabelled(df, y)
# This file only has tens of 30-day cancels. Eventual-after-as_of is the
# question it can supervise. Write that on the artifact. Horizon is still
# the product question (Week 8 / pipelines.train --label).

df = df.sort_values("signup_date")
cutoff = df["signup_date"].quantile(0.80)
train_df = df[df["signup_date"] <= cutoff]
test_df = df[df["signup_date"] > cutoff]
y_train, y_test = y.loc[train_df.index], y.loc[test_df.index]
print(f"Time wall at {cutoff.date()}  train={len(train_df):,}  test={len(test_df):,}")
print("Train rate", float(y_train.mean()), "Test rate", float(y_test.mean()))

pipe = Pipeline(
    [
        ("prep", make_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=40, learning_rate=0.1, max_depth=2, random_state=42
        )),
    ]
)
pipe.fit(train_df[FEATURE_COLS], y_train)

proba = pipe.predict_proba(test_df[FEATURE_COLS])[:, 1]
print(f"Holdout AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"Holdout PR-AUC: {average_precision_score(y_test, proba):.3f}")
print(f"dummy PR-AUC: {float(y_test.mean()):.3f}")
```

## Threshold from a staffing number

CS can call **80** accounts from this test window. We take the 80 highest scores and measure precision. That is the meeting.

```python
BUDGET = 80
order = np.argsort(-proba)
top = order[:BUDGET]
picked = y_test.to_numpy()[top]
hits = picked.sum()
print(f"Calling {BUDGET} highest-risk test users catches {int(hits)} actuals "
      f"({hits/BUDGET:.0%} precision at this budget).")
print(f"There were {int(y_test.sum())} events in the window; "
      f"recall={hits / max(float(y_test.sum()), 1):.0%}.")
threshold = float(np.partition(proba, -BUDGET)[-BUDGET]) if len(proba) >= BUDGET else 1.0

prec, rec, _thr = precision_recall_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(rec, prec, color="#1d4ed8")
ax.set_xlabel("recall (catch rate)")
ax.set_ylabel("precision (when we call, we were right)")
ax.set_title("Precision–recall — pick a point your team can staff")
ax.scatter([hits / max(float(y_test.sum()), 1)], [hits / BUDGET], color="#b91c1c", zorder=5)
ax.annotate("80-call budget", xy=(hits / max(float(y_test.sum()), 1), hits / BUDGET),
            xytext=(0.35, 0.55), textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#b91c1c"})
plt.tight_layout()
plt.show()
```

## The contract: `validate` + `predict`

!!! warning "Watch out"

    If the API re-implements feature math differently from `build_features`, you have two products. One function. The handler imports `pipelines.contract`. Extra keys (`churn_date`, `email`) and NaN fail loud — they do not become a silent 0.

```python
version = as_of.strftime("%Y%m%d")
artifact = {
    "pipeline": pipe,
    "metrics": {"threshold": round(threshold, 4), "model_version": version},
}

demo = {k: train_df.iloc[0][k] for k in FEATURE_COLS}
demo["plan_type"] = str(demo["plan_type"])
print("demo payload →", predict(demo, artifact))

try:
    validate({"mrr": 10})
except Exception as exc:
    print("missing keys →", type(exc).__name__, exc)
try:
    validate({**demo, "churn_date": "2024-07-01"})
except Exception as exc:
    print("unknown key →", type(exc).__name__, exc)

t0 = time.perf_counter()
for _ in range(80):
    predict(demo, artifact)
print("80 predict() calls", round(time.perf_counter() - t0, 3), "s")
```

`predict` already returns `churn_score`, `flag_for_cs`, and `model_version`. Do not invent a `CustomerFeatures` type unless you define it.

## Drift — did this month stop looking like train?

We will not implement a full PSI monitor. We will overlay histograms. If the orange fill walks away from the blue, someone should get a Slack.

!!! math "Math, translated (optional)"

    Population Stability Index is a fancy “how different are two histograms.” If you want a number, compare bin shares: `sum( (p − q) * log(p / q) )`. If you want a decision, look at the picture first.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, col in zip(axes, ["mrr", "log_usage", "tenure_so_far"]):
    ax.hist(train_df[col], bins=30, density=True, alpha=0.55, label="train", color="#3b82f6")
    ax.hist(test_df[col], bins=30, density=True, alpha=0.55, label="later signups", color="#f59e0b")
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle("If orange leaves blue, the world moved — re-check PR-AUC before celebrating")
plt.tight_layout()
plt.show()

dest = Path.cwd() / "artifacts" / version
dest.mkdir(parents=True, exist_ok=True)
joblib.dump({"pipeline": pipe, "features": FEATURE_COLS}, dest / "model.joblib")
print("Wrote", dest / "model.joblib")
print("Ship the joblib AND contract.py AND this week's commit hash. Same layout as pipelines.train.")
```

## What this lesson is not

| Claimed in many “production” tutorials | Reality here |
|---|---|
| Feature versioning | `FEATURE_COLS` in a joblib |
| Error handling | `validate()` on types, extras, NaN |
| Drift detection | Three histograms of `tenure_so_far` etc. |
| Retraining triggers | “Look at the histograms + weekly PR-AUC” |
| Deployment | A directory on disk (`artifacts/<date>/`) |

!!! success "Ship / don’t ship"

    You can ship a *batch score* from this pipeline: score tonight’s accounts, hand CS a CSV of 80 names. Do not ship a public HTTP API until the contract lives in a tested module, the artifact is in a registry, and someone owns the weekly PR-AUC dashboard.

## Course recap (the actual skills)

| Week | How to think about it |
|---|---|
| 0 Python | Glue: dataclasses, a `fit`/`predict` class |
| 1 NumPy | SIMD / SQL on a typed column |
| 2 Pandas | Joins you already know; never explode the grain |
| 3 SQL / as_of | The warehouse is source of truth |
| 4 Charts | Pick the shape that matches the question |
| 5 Stats | “How often would luck look like this?” |
| 6 Features | API contract + a wall against the future |
| 7 Classifiers | Score, then a staffed threshold |
| 8 Labels | Horizon, imbalance, calibration |
| 9 Regression | MAE in real units |
| 10 Clusters | Personas, not APIs |
| 11 Ranking | Precision@k; beat a SQL sort |
| 12 PCA | JPEG / rotate the cloud |
| 13 Ensembles | Reviewers voting vs hunting leftovers |
| 14 Nets | Mixers + the four-line training step |
| 15 Pickle | Artifact + contract + capacity |

## ✍️ Write-up

In one page: (1) the time wall you used, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-15.md). Starter: `python exercises/ml/week-15/starter.py` from the repo root.

## 🔗 Next: the job, then (optionally) deep learning

You can refuse a leak, demand a baseline, and staff a threshold. Next:

- **16** the job pipeline: train → gate → prod → tonight’s CSV
- **17** on-call + the score as a bot tool
- **18–20** optional pictures (CNN / RNN / attention) — not how CloudWave ships churn
- **Capstone** (optional, GPU): a coding-tool-use specialist, not this pickle
