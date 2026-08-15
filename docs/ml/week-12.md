# Week 12 — Capstone: A Training Notebook Is Not Production

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers about to pickle a model and get paged for it.

This is still a **training script**. We will be explicit about what it is not: no feature store, no canary, no GDPR review, no CI.

---

## 🎯 What you will be able to do

- Wire Customer 360 → time-based split → GBT → a versioned artifact
- Expose a `predict(payload)` function with a schema check
- Pick a threshold from **CS capacity**, not from 0.5
- Draw a crude drift picture (this month vs train)
- List what you are *not* deploying

!!! think "Think of it like… a build artifact + an API contract."

    The model file is a binary. The scaler is part of that binary. The request body is the feature list from Week 5. If any of those three drift independently, production is a silent wrong-number generator.

## If you already write software

A notebook is a scratchpad. Production is a contract.

This week is the one that separates “I trained a thing” from “we can call it on Tuesday.”

```
Explore (notebooks, this course so far)
    │  time-based split, pick a model, write predict()
    ▼
Ship
    predict(payload) -> {score, version}
    the same features as training
    the scaler travels with the pickle
    a latency budget (we use 80 calls as a stand-in)
    a drift picture (did live traffic stop looking like train?)
```

### Time-based split is not optional

`train_test_split(..., shuffle=True)` is fine for a homework iris set. It is a lie for SaaS. Customers in the “test” set would include people from the same week as train — and tomorrow’s traffic is *next* week. Split on signup or on event time. Train on the past. Test on the future. Same rule as backtesting a trading strategy, or as not using tomorrow’s logs to tune today’s alert.

### Picture the contract

```python
def predict(payload: CustomerFeatures) -> dict:
    # payload fields == training columns
    # missing field → 400, not a silent 0 unless 0 is honest
    return {"churn_score": float(score), "model_version": "2026-08-14"}
```

If you cannot write that function without reaching back into the notebook, you do not have a model. You have a souvenir.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all 50k rows.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the shared style kit importable from the repo root

from pathlib import Path
import sys
from lib.course_data import find_data_dir

DATA = find_data_dir()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             precision_recall_curve)
import pickle
from datetime import datetime
```

## Architecture (the only diagram that matters)

```
 warehouse CSVs
      │  nightly job
      ▼
 Customer 360  ──►  time split  ──►  train pipeline  ──►  artifact (model+prep, versioned)
                         │                                      │
                         └── holdout report                     ▼
                                                         POST /predict {user features}
                                                                │
                                                                ▼
                                                         monitor: score volume,
                                                         feature histograms, weekly AUC
```

!!! engineer "Engineer mental model"

    Train on the *past*, test on the *more recent past*. Random shuffle is a unit test. A time wall is an integration test against reality.

```python
df = load_customer_360(DATA)
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "n_support"]
categorical = ["plan_type"]
feature_cols = numeric + categorical
label = "is_churned"

df = df.sort_values("signup_date")
cutoff = df["signup_date"].quantile(0.80)
train_df = df[df["signup_date"] <= cutoff]
test_df = df[df["signup_date"] > cutoff]
print(f"Time wall at {cutoff.date()}  train={len(train_df):,}  test={len(test_df):,}")
print("Train churn rate", train_df[label].mean().round(3),
      "Test churn rate", test_df[label].mean().round(3))

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1,
                                   max_depth=2, random_state=42)
pipe = Pipeline([("prep", prep), ("model", model)])
pipe.fit(train_df[feature_cols], train_df[label])

proba = pipe.predict_proba(test_df[feature_cols])[:, 1]
print(f"Holdout AUC: {roc_auc_score(test_df[label], proba):.3f}")
```

## Threshold from a staffing number

CS can call **80** accounts from this test window. We take the 80 highest scores and measure precision. That is the meeting.

```python
BUDGET = 80
order = np.argsort(-proba)
top = order[:BUDGET]
picked = test_df.iloc[top]
hits = picked[label].sum()
print(f"Calling {BUDGET} highest-risk test users catches {hits} actual churners "
      f"({hits/BUDGET:.0%} precision at this budget).")
print(f"There were {test_df[label].sum()} churners in the window; recall={hits/test_df[label].sum():.0%}.")

# Picture the tradeoff
prec, rec, thr = precision_recall_curve(test_df[label], proba)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(rec, prec, color="#1d4ed8")
ax.set_xlabel("recall (catch rate)")
ax.set_ylabel("precision (when we call, we were right)")
ax.set_title("Precision–recall — pick a point your team can staff")
ax.scatter([hits / test_df[label].sum()], [hits / BUDGET], color="#b91c1c", zorder=5)
ax.annotate("80-call budget", xy=(hits / max(test_df[label].sum(), 1), hits / BUDGET),
            xytext=(0.35, 0.55), textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#b91c1c"})
plt.tight_layout()
plt.show()
```

## The contract: `predict(payload)`

!!! warning "Watch out"

    If the API re-implements feature math differently from this lesson, you have two products. One function. Import it from a module in a real repo. Here it lives next to the training so you can see the whole story.

```python
REQUIRED = {
    "mrr": (int, float),
    "tenure_days": (int, float),
    "log_usage": (int, float),
    "features_adopted": (int, float),
    "total_events": (int, float),
    "n_support": (int, float),
    "plan_type": (str,),
}

def validate(payload: dict) -> None:
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    for key, types in REQUIRED.items():
        if not isinstance(payload[key], types):
            raise TypeError(f"{key} should be {types}, got {type(payload[key])}")
    if payload["plan_type"] not in {"free", "starter", "pro", "enterprise"}:
        raise ValueError(f"unknown plan_type {payload['plan_type']}")

def predict(payload: dict, threshold: float = 0.35) -> dict:
    validate(payload)
    row = pd.DataFrame([payload])
    score = float(pipe.predict_proba(row[feature_cols])[0, 1])
    return {"churn_score": round(score, 4), "flag_for_cs": score >= threshold}

# Smoke tests
demo = train_df.iloc[0][feature_cols].to_dict()
print("demo payload →", predict(demo))
try:
    predict({"mrr": 10})
except Exception as exc:
    print("schema catch →", type(exc).__name__, exc)
```

## Drift — did this month stop looking like train?

We will not implement a full PSI monitor. We will overlay histograms. If the orange fill walks away from the blue, someone should get a Slack.

!!! math "Math, translated (optional)"

    Population Stability Index is a fancy “how different are two histograms.” If you want a number, compare bin shares: `sum( (p − q) * log(p / q) )`. If you want a decision, look at the picture first.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, col in zip(axes, ["mrr", "log_usage", "tenure_days"]):
    ax.hist(train_df[col], bins=30, density=True, alpha=0.55, label="train", color="#3b82f6")
    ax.hist(test_df[col], bins=30, density=True, alpha=0.55, label="later signups", color="#f59e0b")
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle("If orange leaves blue, the world moved — re-check AUC before celebrating")
plt.tight_layout()
plt.show()

version = datetime.now().strftime("%Y%m%d")
artifact_dir = Path.cwd() / "artifacts"
artifact_dir.mkdir(exist_ok=True)
artifact = artifact_dir / f"cloudwave_churn_{version}.pkl"
with artifact.open("wb") as f:
    pickle.dump({"pipeline": pipe, "features": feature_cols, "trained_through": str(cutoff.date())}, f)
print("Wrote", artifact)
print("Ship the pickle AND the predict()/validate() code AND this lesson's commit hash.")
```

## What this lesson is not

| Claimed in many “production” tutorials | Reality here |
|---|---|
| Feature versioning | A Python list in a pickle |
| Error handling | A `validate()` on types |
| Drift detection | Three histograms |
| Retraining triggers | “Look at the histograms + weekly AUC” |
| Deployment | A file on disk |

!!! success "Ship / don’t ship"

    You can ship a *batch score* from this pipeline: score tonight’s accounts, hand CS a CSV of 80 names. Do not ship a public HTTP API until the contract lives in a tested module, the artifact is in a registry, and someone owns the weekly AUC dashboard.

## Course recap (the actual skills)

| Week | How to think about it |
|---|---|
| 1 NumPy | SIMD / SQL on a typed column |
| 2 Pandas | Joins you already know; never explode the grain |
| 3 Charts | Pick the shape that matches the question |
| 4 Stats | “How often would luck look like this?” |
| 5 Features | API contract + a wall against the future |
| 6 Classifiers | Score, then a staffed threshold |
| 7 Regression | Trendline; MAE in real units; no fake CLV |
| 8 Clusters | Pins on a map; personas, not APIs |
| 9 PCA | JPEG / rotate the cloud |
| 10 Ensembles | Reviewers voting vs hunting leftovers |
| 0 Python | Glue language: dataclasses, a `fit`/`predict` class |
| 1 NumPy | SIMD / SQL on a typed column |
| 2 Pandas | Joins you already know; never explode the grain |
| 3 Charts | Pick the shape that matches the question |
| 4 Stats | “How often would luck look like this?” |
| 5 Features | API contract + a wall against the future |
| 6 Classifiers | Score, then a staffed threshold; bias vs variance |
| 7 Regression | Trendline; MAE in real units; no fake CLV |
| 8 Clusters | Pins on a map; personas, not APIs |
| 9 PCA | JPEG / rotate the cloud |
| 10 Ensembles | Reviewers voting vs hunting leftovers |
| 11 Nets + PyTorch | Mixers + the four-line training step |
| 12 Capstone | Artifact + contract + capacity + humility |

## ✍️ Capstone write-up

In one page: (1) the time wall you used, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-12.md). Starter: `python exercises/ml/week-12/starter.py` from the repo root.

## 🔗 Next: the deep-learning track

You can now refuse a leak, demand a baseline, and staff a threshold. Weeks **13–15** are the other half of “I know deep learning”:

- **13 CNNs** — sliding detectors (1-D on usage, 2-D on images)
- **14 RNNs** — a clipboard that walks a sequence
- **15 Transformers** — every token looks at every other token

Then the LangChain / LangGraph / CrewAI tracks in this repo.
