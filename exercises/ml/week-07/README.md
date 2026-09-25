# Exercise — Week 7 — Classification: A Score, Then a Threshold

Priya can call 100 test users this week. She wants the list of names, ranked by risk — not an AUC number in a Slack thread. Before you hand it over, prove the model beats doing nothing, and know exactly what you're trading away as you move the cutoff.

## What you are building

Precision at a 100-call budget, a threshold sweep, and an ablation that drops `tenure_so_far`.

## Predict before you run

1. About 9% of the customers still active on 2024-06-01 ever cancel in this file. Accuracy of “nobody churns”? AUC of that dummy?
2. As you sweep the threshold 0.5 → 0.8, which way do precision and recall move?
3. How much AUC dies without `tenure_so_far`?

## Before you start

- This week still trains on lifetime `is_churned`. Week 8 replaces the label — keep your AUC number for comparison.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-07/starter.py
```

**0. Predict first.** About 9% of the customers still active on 2024-06-01 ever cancel (the lifetime flag, restricted to people still around). A classifier that predicts "nobody churns" for every customer — before running the dummy baseline in the lesson, guess its accuracy and its AUC. Then guess: as you sweep the threshold from 0.5 up to 0.8 in exercise 2, which direction do precision and recall move? Write both guesses down before you run anything.

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

<details>
<summary>Hint 1 — a nudge</summary>

Forget thresholds for a moment. Priya has 100 calls. Which 100 names does the model hand her, and how many of those calls reach someone who actually left?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Fit the lesson's forest pipeline (`make_preprocessor()` + `RandomForestClassifier`) on a stratified split of `build_features`. `np.argsort(-proba)[:100]` is the call list; the mean of `y_test` at those positions is precision@100.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipelines.features import FEATURE_COLS, NUMERIC, build_features, make_preprocessor

df = build_features(as_of="2024-06-01")
X, y = df[FEATURE_COLS], df["is_churned"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

forest = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2)),
]).fit(X_train, y_train)
proba = forest.predict_proba(X_test)[:, 1]
y_np = y_test.to_numpy()

top = np.argsort(-proba)[:100]
print(f"precision@100 = {y_np[top].mean():.3f}  ({int(y_np[top].sum())} of 100 calls)")
```

</details>

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

<details>
<summary>Hint 1 — a nudge</summary>

A threshold is the same cut as "top 100," written as a score instead of a headcount. Which column of your table tells you whether CS can actually staff that row?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Loop `np.arange(0.1, 1.0, 0.1)`: `pred = proba >= cut`, then `pred.sum()`, `precision_score`, `recall_score`. A high cut can flag nobody — pass `zero_division=0` so sklearn doesn't warn.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
print(f"{'cut':>5} {'flagged':>8} {'prec':>6} {'rec':>6}")
for cut in np.arange(0.1, 1.0, 0.1):
    pred = proba >= cut
    print(f"{cut:5.1f} {int(pred.sum()):8d} "
          f"{precision_score(y_np, pred, zero_division=0):6.3f} "
          f"{recall_score(y_np, pred, zero_division=0):6.3f}")
```
Circling the row is yours.

</details>

**3. Ablation.** Retrain the forest without `tenure_so_far`. How much AUC dies?

<details>
<summary>Hint 1 — a nudge</summary>

If one column does most of the work, is it a signal about *behavior*, or just a signal about *how new the account is*? Brand-new users haven't had time to churn.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Same pipeline, with `tenure_so_far` removed from both the numeric list and `X`. Compare `roc_auc_score` on the same test rows. Note: `tenure_so_far` is a little circular for new users — but lifetime `tenure_days` is the real leak, and Week 8 kills it.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

keep = [c for c in NUMERIC if c != "tenure_so_far"]
prep = ColumnTransformer([
    ("num", StandardScaler(), keep),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]),
])
ablated = Pipeline([
    ("prep", prep),
    ("model", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2)),
]).fit(X_train[keep + ["plan_type"]], y_train)
auc_ab = roc_auc_score(y_np, ablated.predict_proba(X_test[keep + ["plan_type"]])[:, 1])
print(f"AUC full={roc_auc_score(y_np, proba):.3f}  without tenure_so_far={auc_ab:.3f}")
```

</details>

## Success criteria

- Dummy accuracy vs AUC guessed first.
- Precision@100 and a circled threshold with a flagged count CS could staff.
- AUC with and without `tenure_so_far`.

## After you run

You ship a list of names, not a textbook 0.5 cutoff. Ranking quality (AUC) and desk precision are different emails — Priya only reads one of them. (Save the AUC number: Week 8 is about to tell you the label behind it was wrong.)

## Lesson link

[Week 7 — Classification: A Score, Then a Threshold](../../../docs/ml/week-07.md)
