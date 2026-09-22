---
description: Compare shuffled vs time-split AUC, benchmark predict() latency, and pick a capacity-constrained decision threshold for a churn model artifact.
---

# Exercises — Week 15 — The Pickle

Ana's question from the lesson still stands: what happens when your laptop is off and it's Tuesday night? Turn the training script into an artifact with a contract, a latency number, and a threshold sized to Priya's 80-call budget — something that survives without you in the room.

## What you are building

A time-split vs shuffled AUC, 80 `predict()` calls with p50/p95 latency, a capacity threshold of 80 names, a drift overlay, and a one-page write-up.

## Predict before you run

1. Will shuffled AUC exceed time-split AUC?
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

**1. Time wall.** `build_features` + `label_eventual_churn` (or `label_churn_in_horizon`). Split on `signup_date` (train = earlier 80% of signups, test = later 20%). Train the same GBT (with `make_preprocessor()` — `plan_type` is a string) on a *shuffled* split and on the time split. Report both AUCs. If they differ, write one sentence about why.

??? tip "Hint 1 — a nudge"
    In production you train on everyone who exists today and score people who sign up next month. Which of the two splits looks like that?

??? tip "Hint 2 — the approach"
    `cutoff = df["signup_date"].quantile(0.80)` gives the time split; `train_test_split(..., stratify=y)` gives the shuffled one with the same test size. Fit the same pipeline on each and compare test AUC.

??? example "Hint 3 — most of the code"
    ```python
    import json
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
    from pipelines.labels import drop_unlabelled, label_eventual_churn

    df = build_features(as_of=AS_OF_DEFAULT, n=None, at_risk_only=True)
    df, y = drop_unlabelled(df, label_eventual_churn(df, AS_OF_DEFAULT))


    def gbt() -> Pipeline:
        return Pipeline([
            ("prep", make_preprocessor()),
            ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
        ])


    cutoff = df["signup_date"].quantile(0.80)
    train, test = df[df["signup_date"] <= cutoff], df[df["signup_date"] > cutoff]
    y_train, y_test = y.loc[train.index], y.loc[test.index]
    time_model = gbt().fit(train[FEATURE_COLS], y_train)
    time_scores = time_model.predict_proba(test[FEATURE_COLS])[:, 1]

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(df[FEATURE_COLS], y, test_size=len(test), random_state=42, stratify=y)
    shuffled_auc = roc_auc_score(ys_te, gbt().fit(Xs_tr, ys_tr).predict_proba(Xs_te)[:, 1])
    print(f"time-split AUC={roc_auc_score(y_test, time_scores):.3f}   shuffled AUC={shuffled_auc:.3f}")
    ```

**2. `predict()` contract.** Import `validate` and `predict` from `pipelines.contract`. Do **not** invent a `CustomerFeatures` type — read `validate` before you build a payload. `predict` returns `{churn_score, flag_for_cs, model_version}`. Call it 80 times. Print p50 / p95 latency.

??? tip "Hint 1 — a nudge"
    `predict` wants two things: a payload that passes `validate`, and an artifact dict. What keys does `predict` read out of that artifact?

??? tip "Hint 2 — the approach"
    The artifact is `{"pipeline": ..., "metrics": {"threshold": ..., "model_version": ...}}` — the same shape `load_artifact` returns. Build payloads from test rows with only `FEATURE_COLS`, numbers as `float`, `plan_type` as `str`. Time each call with `time.perf_counter()` and take `np.percentile(..., [50, 95])`.

??? example "Hint 3 — most of the code"
    ```python
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

**3. Capacity, not 0.5.** From the time-split test set, pick the threshold that flags **at most 80** customers (CS budget). Report precision and recall at that cut. Compare to 0.5.

??? tip "Hint 1 — a nudge"
    Priya's budget is a *count*. Turn the count into a score: what is the score of the 80th-riskiest customer?

??? tip "Hint 2 — the approach"
    Sort the test scores descending and take the 80th value (`np.partition` works too — see `pipelines.train._threshold_for_budget`). Flag `scores >= cut`; compute flagged count, precision, recall. Do the same at 0.5.

??? example "Hint 3 — most of the code"
    ```python
    cut_80 = float(np.sort(time_scores)[::-1][79])
    for name, cut in [("budget-80", cut_80), ("0.5", 0.5)]:
        flag = time_scores >= cut
        print(f"{name:<10} cut={cut:.4f} flagged={int(flag.sum()):>5} "
              f"precision={precision_score(y_test, flag, zero_division=0):.3f} "
              f"recall={recall_score(y_test, flag, zero_division=0):.3f}")
    ```

**4. Drift sketch.** Overlay histograms of `mrr`, `log_usage`, `tenure_so_far` for train vs later signups. One sentence: did the world move?

??? tip "Hint 1 — a nudge"
    One of these three columns *has* to differ between early and late signups, by construction. Which one — and does that count as the world moving?

??? tip "Hint 2 — the approach"
    Three subplots; in each, `hist(..., density=True, alpha=0.5)` for `train` and `test` with shared bins. Density, not counts — the groups are different sizes.

??? example "Hint 3 — most of the code"
    ```python
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col in zip(axes, ["mrr", "log_usage", "tenure_so_far"]):
        bins = np.histogram_bin_edges(df[col], bins=30)
        ax.hist(train[col], bins=bins, density=True, alpha=0.5, label="train (earlier)")
        ax.hist(test[col], bins=bins, density=True, alpha=0.5, label="later signups")
        ax.set_title(col)
    axes[0].legend()
    fig.savefig("drift.png", dpi=120)
    ```

**5. One-page write-up.** (1) the time wall, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

??? tip "Hint 1 — a nudge"
    Write it for Ana, who wasn't in the room. Every claim should point at a number you printed above.

??? tip "Hint 2 — the approach"
    Five short sections, one per item. A dummy's AUC is 0.5 by definition — the interesting comparison is PR-AUC or precision@80 against the base rate.

??? example "Hint 3 — a skeleton"
    ```text
    # Churn model <version> — ship note
    1. Time wall:   trained on signups ≤ <date>, tested on <n> later signups.
    2. Holdout:     AUC <x> (time) vs <y> (shuffled); base rate <r>.
    3. Budget:      top-80 precision <p> → ~<k> real churners reached per week.
    4. Drift risk:  <column> moved because <reason>; watch <what>.
    5. Not claimed: <the thing you are refusing to say>.
    ```

Dump with `joblib` into `artifacts/<version>/model.joblib`, the same layout as `pipelines.train`.

## Success criteria

- Two AUCs (time vs shuffle).
- 80 `predict()` latencies and a response with `model_version`.
- Threshold for ≤80 flags vs 0.5.
- Artifact layout matches train.

## After you run

A pickle is an artifact plus a contract plus a budget. Random splits measure yesterday. Time splits measure next month. Next week it stops being a file you run by hand.

## Lesson link

[Week 15 — The Pickle](../week-15.md)
