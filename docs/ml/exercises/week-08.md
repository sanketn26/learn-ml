---
description: Compare horizon vs lifetime churn labels under censoring, evaluate PR-AUC against ROC-AUC on rare positives, and check a model's calibration curve.
---

# Exercises — Week 8 — Labels Lie

Ana already blocked the handoff in the lesson: last week's model answers "did they ever churn," not "will they churn in the next 30 days." Before Priya gets a queue built on it, rebuild the label properly and prove the old ranking metric was flattering you.

## What you are building

Horizon vs lifetime rates, a censoring count, PR-AUC vs ROC-AUC vs precision@80, a forbidden-key `validate()` raise, and a calibration glance.

## Predict before you run

1. About 9% of today's customers ever cancel; about 2% cancel in the next 30 days. If you raise the threshold 0.5 → 0.8, which way do precision and recall move, and why is the swing sharper than on a 50/50 label?
2. Which rate is legal at score time?
3. Would you put ROC-AUC or PR-AUC in the Monday email?

## Before you start

- Labels come back with NaNs (already churned, or censored). Call `drop_unlabelled` before handing them to sklearn.
- `plan_type` is a string — use `make_preprocessor()`, not a raw estimator.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-08/starter.py
pytest tests/test_labels.py tests/test_contract.py
```

**0. Predict first.** About 9% of the customers active on 2024-06-01 ever cancel (lifetime). Before running anything: a model trained on the horizon-30 label instead sees roughly 530 positives among ~28,000 rows. If you raise the classification threshold from 0.5 to 0.8 on that model, which direction do precision and recall move, and why does a rare positive class make that swing sharper than it would on a 50/50 label? Write your guess, then check it against exercise 3.

**1. Two rates.** On `as_of=2024-06-01`, print horizon-30 churn rate vs lifetime `is_churned` on the same at-risk people. Which one is legal at score time?

??? tip "Hint 1 — a nudge"
    At noon on `as_of`, which of these two columns could a scoring job actually know — and which one is a souvenir from the future?

??? tip "Hint 2 — the approach"
    The starter already prints both. Read them next to each other on the *same* labelled rows: `label_churn_in_horizon`, `drop_unlabelled`, then compare `y.mean()` with `labelled["is_churned"].mean()`.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.calibration import calibration_curve
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from pipelines.contract import validate
    from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
    from pipelines.labels import drop_unlabelled, label_churn_in_horizon, label_eventual_churn

    as_of = AS_OF_DEFAULT
    df = build_features(as_of=as_of, n=None)
    labelled, y_h = drop_unlabelled(df, label_churn_in_horizon(df, as_of))
    print(f"horizon-30 rate={y_h.mean():.4f} (positives={int(y_h.sum())})  "
          f"lifetime rate on the same rows={labelled['is_churned'].mean():.4f}")
    ```

**2. Censoring.** Call `label_churn_in_horizon` with `observation_end=as_of + 10 days` and `horizon_days=30`. Rows we have not watched through the horizon become NaN — that is censoring, per row. A cancel you already saw inside those 10 days stays **1**. How many labels are NaN, and how many observed cancels survived? (A Saturday signup with a fully observed window is noisy, not censored.)

??? tip "Hint 1 — a nudge"
    If you stopped watching 10 days in, can you say "this customer did *not* churn within 30 days"? Can you say "this one *did*"?

??? tip "Hint 2 — the approach"
    Pass `observation_end=as_of + pd.Timedelta(days=10)`. Count `isna()` and `== 1` on the result. Compare the NaN count to `len(df)`.

??? example "Hint 3 — most of the code"
    ```python
    short = label_churn_in_horizon(df, as_of, horizon_days=30, observation_end=as_of + pd.Timedelta(days=10))
    print(f"NaN labels={int(short.isna().sum()):,} of {len(short):,}   observed 1s={int((short == 1).sum())}")
    ```

**3. PR vs ROC.** Train the small GBT from the lesson on the **30-day** label (`label_churn_in_horizon`). Print ROC-AUC, PR-AUC, dummy PR-AUC, precision@80. With ~2% positives, 80 calls hold only a handful of churners — so precision@80 moves a lot between runs. Which number would you put in the Monday email, and what would you write next to it?

??? tip "Hint 1 — a nudge"
    ROC-AUC asks "are churners ranked above non-churners on average?" Priya's question is "of my 80 calls, how many land?" Which metric is closer to hers, and what does a dummy score on it?

??? tip "Hint 2 — the approach"
    `label_churn_in_horizon` → `drop_unlabelled` → stratified split → `Pipeline([make_preprocessor(), GradientBoostingClassifier(n_estimators=40, max_depth=2)])`. The dummy PR-AUC is `average_precision_score` on a constant score (the train base rate). Precision@80 is `y_test` at `np.argsort(-proba)[:80]`.

??? example "Hint 3 — most of the code"
    ```python
    frame, y = drop_unlabelled(df, label_churn_in_horizon(df, as_of))
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLS], y, test_size=0.2, random_state=42, stratify=y
    )
    gbt = Pipeline([
        ("prep", make_preprocessor()),
        ("model", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ]).fit(X_train, y_train)
    proba = gbt.predict_proba(X_test)[:, 1]

    dummy_pr = average_precision_score(y_test, np.full(len(y_test), y_train.mean()))
    prec80 = y_test.to_numpy()[np.argsort(-proba)[:80]].mean()
    print(f"ROC-AUC={roc_auc_score(y_test, proba):.3f}  PR-AUC={average_precision_score(y_test, proba):.3f}  "
          f"dummy PR-AUC={dummy_pr:.3f}  precision@80={prec80:.3f}")
    ```

**4. Forbidden.** `pytest tests/test_labels.py tests/test_contract.py`. Then try `validate({..., "churn_date": "2024-07-01"})` and show it raises.

??? tip "Hint 1 — a nudge"
    `validate` is an allowlist, not a blocklist. What does it do with a key it has never heard of?

??? tip "Hint 2 — the approach"
    Build a legal payload from one test row (numbers as `float`, `plan_type` as `str`), confirm it validates, then add `churn_date` and catch the exception.

??? example "Hint 3 — most of the code"
    ```python
    row = X_test.iloc[0]
    payload = {k: (str(row[k]) if k == "plan_type" else float(row[k])) for k in FEATURE_COLS}
    validate(payload)
    try:
        validate({**payload, "churn_date": "2024-07-01"})
    except ValueError as exc:
        print("rejected:", exc)
    ```

**5. Calibration glance.** Draw the reliability curve. One sentence: would you let finance treat the score as a probability?

??? tip "Hint 1 — a nudge"
    "0.3" is a probability only if, among customers scored around 0.3, about 30% actually leave. How would you check that?

??? tip "Hint 2 — the approach"
    `calibration_curve(y_test, proba, n_bins=8, strategy="quantile")` returns observed and predicted per bin. Plot one against the other with the diagonal for reference.

??? example "Hint 3 — most of the code"
    ```python
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"predicted={mp:.3f}  observed={fp:.3f}")
    ```
    The sentence for finance is yours.

## Success criteria

- Both rates printed; legal one named.
- NaN count and surviving 1s under a short observation window.
- PR-AUC vs dummy, precision@80, pytest green, extra key rejected.

## After you run

Eventual-after-as_of is the question this file can supervise. Say so in `metrics.json`. Lifetime `churn_date` on a payload is a 400. Now the label is legal — Priya's queue can be built on it.

## Lesson link

[Week 8 — Labels Lie](../week-08.md)
