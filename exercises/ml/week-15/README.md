# Exercise — Week 15 — The Pickle

Ana's question from the lesson still stands: what happens when your laptop is off and it's Tuesday night? Turn the training script into an artifact with a contract, a latency number, and a threshold sized to Priya's 80-call budget — something that survives without you in the room.

## What you are building

Test AUC under three splits, 80 `predict()` calls with p50/p95 latency, a capacity threshold of 80 names, a drift overlay, and a one-page write-up.

## Predict before you run

1. Rank the three splits in task 1 by the AUC you expect, before you run any of them.
2. Does `validate` accept `email` on the payload?
3. Will a 0.5 threshold flag more or fewer than 80 customers?

## Before you start

- `plan_type` must be a `str` in every payload, not a pandas NA or a category.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-15/starter.py
```

**1. Time wall.** Train the same GBT (with `make_preprocessor()` — `plan_type` is a string) three ways and report test AUC for each: (a) one `as_of` snapshot, shuffled 80/20; (b) the same snapshot cut on `signup_date` (earlier 80% of signups train, later 20% test); (c) `snapshot_split` — train on the snapshot 90 days earlier, test on `as_of`. Use a 90-day horizon label for all three. Rank the three by how much you'd trust them, in one sentence each.

<details>
<summary>Hint 1 — a nudge</summary>

In production you stand on a date, score everyone at risk, and find out later who left. Which of the three splits looks like that — and does (b) really train on "the past," or on something else?

</details>

<details>
<summary>Hint 2 — the approach</summary>

For (a) and (b), label one snapshot with `label_churn_in_horizon(df, as_of, 90)` and `drop_unlabelled`. (a) is `train_test_split(..., stratify=y)`; (b) is `df["signup_date"].quantile(0.80)`. For (c), `snapshot_split(as_of, horizon_days=90)` returns all four pieces. After (b), print the `tenure_so_far` range on each side.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipelines.contract import predict, validate
from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon
from pipelines.split import snapshot_split

def gbt() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ])

def test_auc(tr, y_tr, te, y_te) -> float:
    return roc_auc_score(y_te, gbt().fit(tr[FEATURE_COLS], y_tr).predict_proba(te[FEATURE_COLS])[:, 1])

snap = build_features(as_of=AS_OF_DEFAULT, n=None)
snap, y_snap = drop_unlabelled(snap, label_churn_in_horizon(snap, AS_OF_DEFAULT, horizon_days=90))

a_tr, a_te, ya_tr, ya_te = train_test_split(snap, y_snap, test_size=0.2, random_state=42, stratify=y_snap)
cut = snap["signup_date"].quantile(0.80)
b_tr, b_te = snap[snap["signup_date"] <= cut], snap[snap["signup_date"] > cut]
train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=90)

print(f"(a) shuffled     AUC={test_auc(a_tr, ya_tr, a_te, ya_te):.3f}")
print(f"(b) signup cut   AUC={test_auc(b_tr, y_snap.loc[b_tr.index], b_te, y_snap.loc[b_te.index]):.3f}  "
      f"tenure train {b_tr['tenure_so_far'].min()}–{b_tr['tenure_so_far'].max()}, "
      f"test {b_te['tenure_so_far'].min()}–{b_te['tenure_so_far'].max()}")
print(f"(c) backtest     AUC={test_auc(train, y_train, test, y_test):.3f}")
```
The trust ranking — and why (b) behaves the way it does — is yours.

</details>

**2. `predict()` contract.** Import `validate` and `predict` from `pipelines.contract`. Do **not** invent a `CustomerFeatures` type — read `validate` before you build a payload. `predict` returns `{churn_score, flag_for_cs, model_version}`. Call it 80 times. Print p50 / p95 latency.

<details>
<summary>Hint 1 — a nudge</summary>

`predict` wants two things: a payload that passes `validate`, and an artifact dict. What keys does `predict` read out of that artifact?

</details>

<details>
<summary>Hint 2 — the approach</summary>

The artifact is `{"pipeline": ..., "metrics": {"threshold": ..., "model_version": ...}}` — the same shape `load_artifact` returns. Build payloads from test rows with only `FEATURE_COLS`, numbers as `float`, `plan_type` as `str`. Time each call with `time.perf_counter()` and take `np.percentile(..., [50, 95])`.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
time_model = gbt().fit(train[FEATURE_COLS], y_train)
time_scores = time_model.predict_proba(test[FEATURE_COLS])[:, 1]
version = AS_OF_DEFAULT.strftime("%Y%m%d")
artifact = {"pipeline": time_model, "metrics": {"threshold": 0.5, "model_version": version}}

def payload(row) -> dict:
    return {k: (str(row[k]) if k == "plan_type" else float(row[k])) for k in FEATURE_COLS}

latencies = []
for _, row in test.head(80).iterrows():
    t0 = time.perf_counter()
    response = predict(payload(row), artifact)
    latencies.append((time.perf_counter() - t0) * 1000)
p50, p95 = np.percentile(latencies, [50, 95])
print(f"p50={p50:.2f} ms  p95={p95:.2f} ms  last response={response}")
```
Then dump it the way `pipelines.train` does — the whole pipeline, not the bare model, so prod can still one-hot a plan:
```python
dest = Path("artifacts") / version
dest.mkdir(parents=True, exist_ok=True)
joblib.dump({"pipeline": time_model, "features": FEATURE_COLS}, dest / "model.joblib")
```

</details>

**3. Capacity, not 0.5.** From the backtest test set, pick the threshold that flags **at most 80** customers (CS budget). Report precision and recall at that cut. Compare to 0.5.

<details>
<summary>Hint 1 — a nudge</summary>

Priya's budget is a *count*. Turn the count into a score: what is the score of the 80th-riskiest customer?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Sort the test scores descending and take the 80th value (`np.partition` works too — see `pipelines.train._threshold_for_budget`). Flag `scores >= cut`; compute flagged count, precision, recall. Do the same at 0.5.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
cut_80 = float(np.sort(time_scores)[::-1][79])
for name, cut in [("budget-80", cut_80), ("0.5", 0.5)]:
    flag = time_scores >= cut
    print(f"{name:<10} cut={cut:.4f} flagged={int(flag.sum()):>5} "
          f"precision={precision_score(y_test, flag, zero_division=0):.3f} "
          f"recall={recall_score(y_test, flag, zero_division=0):.3f}")
```

</details>

**4. Drift sketch.** Overlay histograms of `mrr`, `log_usage`, `tenure_so_far` for the train snapshot vs today's snapshot. One sentence: did the world move?

<details>
<summary>Hint 1 — a nudge</summary>

The two snapshots are 90 days apart. Which column shifts just because the calendar moved — and does that count as the world moving?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Three subplots; in each, `hist(..., density=True, alpha=0.5)` for `train` and `test` with shared bins. Density, not counts — the groups are different sizes.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, col in zip(axes, ["mrr", "log_usage", "tenure_so_far"]):
    bins = np.histogram_bin_edges(np.r_[train[col], test[col]], bins=30)
    ax.hist(train[col], bins=bins, density=True, alpha=0.5, label="train snapshot")
    ax.hist(test[col], bins=bins, density=True, alpha=0.5, label="today's snapshot")
    ax.set_title(col)
axes[0].legend()
fig.savefig("drift.png", dpi=120)
```

</details>

**5. One-page write-up.** (1) the time wall and why you chose it, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

<details>
<summary>Hint 1 — a nudge</summary>

Write it for Ana, who wasn't in the room. Every claim should point at a number you printed above.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Five short sections, one per item. A dummy's AUC is 0.5 by definition — the interesting comparison is PR-AUC or precision@80 against the base rate.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
# Churn model <version> — ship note
1. Time wall:   trained on the <date> snapshot, tested on <as_of> (+<h> days).
2. Holdout:     backtest AUC <x> vs shuffled <y> vs signup cut <z>; base rate <r>.
3. Budget:      top-80 precision <p> → ~<k> real churners reached per week.
4. Drift risk:  <column> moved because <reason>; watch <what>.
5. Not claimed: <the thing you are refusing to say>.
```

</details>

Dump with `joblib` into `artifacts/<version>/model.joblib`, the same layout as `pipelines.train`.

## Success criteria

- Three AUCs (shuffled, signup cut, backtest) with a one-line trust ranking.
- 80 `predict()` latencies and a response with `model_version`.
- Threshold for ≤80 flags vs 0.5.
- Artifact layout matches train.

## After you run

A pickle is an artifact plus a contract plus a budget. Random splits measure yesterday. Time splits measure next month. Next week it stops being a file you run by hand.

## Lesson link

[Week 15 — The Pickle](../../../docs/ml/week-15.md)
