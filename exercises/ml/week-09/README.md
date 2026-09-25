# Exercise — Week 9 — Regression: Predict a Number, Not a Category

## What you are building

A log-target forest, residual MAE by plan, and an R² audit of a constructed `fake_clv` target.

## Predict before you run

1. Does `log1p` / `expm1` help MAE on the original dollar scale (whales)?
2. Is the model worse on `free` or `enterprise`?
3. What R² do you expect from `fake_clv = mrr * (tenure_so_far / 30)` using those same two columns as features?

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-09/starter.py
```

**1. Log target.** Train the forest on `log1p(y)` and `expm1` the predictions. Does MAE on the original scale improve?

<details>
<summary>Hint 1 — a nudge</summary>

Usage has a long right tail. A model that minimizes squared error on raw usage spends most of its effort on a handful of whales. What would you do to a skewed column in Week 6?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Reuse the lesson's backtest (train on last month's snapshot, test on this month's) and its next-30-day usage target. Fit one forest on `y_train` and one on `np.log1p(y_train)`; `np.expm1` the second forest's predictions. Compare both with `mean_absolute_error` on the *original* scale — log-space MAE is not a number a PM can read.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

from lib.course_data import find_data_dir
from pipelines.features import AS_OF_DEFAULT, build_features

usage = pd.read_csv(find_data_dir() / "feature_usage.csv",
                    usecols=["user_id", "usage_count", "date"], parse_dates=["date"])

def next_30d_usage(frame, as_of):
    window = usage[(usage["date"] > as_of) & (usage["date"] <= as_of + pd.Timedelta(days=30))]
    return frame["user_id"].map(window.groupby("user_id")["usage_count"].sum()).fillna(0)

num = ["mrr", "tenure_so_far", "log_usage", "features_adopted", "total_events", "n_support"]
train_as_of = AS_OF_DEFAULT - pd.Timedelta(days=30)          # the lesson's backtest
train = build_features(as_of=train_as_of, n=8000)
test = build_features(as_of=AS_OF_DEFAULT, n=8000, random_state=7)
X_train, X_test = train[num + ["plan_type"]], test[num + ["plan_type"]]
y_train, y_test = next_30d_usage(train, train_as_of), next_30d_usage(test, AS_OF_DEFAULT)

def forest() -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]),
    ])
    return Pipeline([("prep", prep), ("m", RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42, n_jobs=2))])

raw_pred = forest().fit(X_train, y_train).predict(X_test)
log_pred = np.expm1(forest().fit(X_train, np.log1p(y_train)).predict(X_test))
print(f"MAE raw target={mean_absolute_error(y_test, raw_pred):,.1f}  "
      f"log target={mean_absolute_error(y_test, log_pred):,.1f}")
```

</details>

**2. Residual slices.** MAE for `free` vs `enterprise`. Where is the model actually bad?

<details>
<summary>Hint 1 — a nudge</summary>

One overall MAE averages the easy customers with the hard ones. Which plans have the widest range of usage to predict?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Put `y_test`, the predictions, and `plan_type` in one frame, then `groupby("plan_type")` and take the mean absolute residual per plan. Add the plan's median usage for scale.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
slices = test[["plan_type"]].assign(actual=y_test.to_numpy(), pred=log_pred)
slices["abs_err"] = (slices["actual"] - slices["pred"]).abs()
print(slices.groupby("plan_type").agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                                      median_usage=("actual", "median")).round(1))
```

</details>

**3. An alternative target.** Create `fake_clv = mrr * (tenure_so_far / 30)` and train a linear model using `mrr` and `tenure_so_far`. Print R². Then decide whether this code path ships.

<details>
<summary>Hint 1 — a nudge</summary>

Write out what `fake_clv` is made of. Now write out what the model's inputs are. What is the model being asked to learn?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Build the column on `train` and `test`, fit `LinearRegression` on `[mrr, tenure_so_far]`, and print `r2_score` on test. Then ask whether a high R² here means *prediction* or *arithmetic*. (Lifetime `tenure_days` would be worse still — it already knows who left.)

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
cols = ["mrr", "tenure_so_far"]
fake_train = train["mrr"] * (train["tenure_so_far"] / 30)
fake_test = test["mrr"] * (test["tenure_so_far"] / 30)
lin = LinearRegression().fit(train[cols], fake_train)
print(f"fake_clv R² on test = {r2_score(fake_test, lin.predict(test[cols])):.3f}")
```
The verdict — and what you do with this code afterwards — is yours.

</details>

## Success criteria

- MAE raw vs log-target reported on the original scale.
- Slice MAE for at least two plans.
- Fake CLV R² printed, with a one-line ship / don't-ship verdict on the code path that produced it.

## After you run

Regression is dollars and leftovers. If the target is a function of the features, you have a calculator, not a model.

## Lesson link

[Week 9 — Regression: Predict a Number, Not a Category](../../../docs/ml/week-09.md)
