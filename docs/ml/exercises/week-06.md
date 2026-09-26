---
description: Compare two scaler fitting strategies across a train/test split, test a missingness flag, and write a contract test for a model's score payload.
---

# Exercises — Week 6 — Features Are the Model's API

## What you are building

Two scaler fits compared across a split, a missingness flag, and `assert_score_payload` — a contract test, not another transformer.

## Predict before you run

1. If a feature is built from behavior *seven days after* `as_of`, what happens to offline AUC vs production AUC?
2. Will the all-rows scaler mean for `mrr` differ much from the train-only mean on an 8k sample?
3. Does `has_usage` split churn even if `total_usage` is already in the matrix?

## Before you start

- Keys and labels never go in the payload: no `user_id`, no `churn_date`.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-06/starter.py
```

**0. Predict first.** Suppose someone adds a feature built from customer behavior recorded *seven days after* the prediction timestamp (`as_of`). Before touching any code: what happens to offline holdout AUC — better, worse, unchanged? What happens to production AUC once that feature is scored on customers who have not lived those seven days yet? Write both guesses, then read the "time machine" picture in Week 8 to check the second one.

**1. Two scaler fits.** Fit a scaler on all rows, then only on train. Print the two means used for `mrr`. How far apart are they?

??? tip "Hint 1 — a nudge"
    The test set stands in for customers you will score next month. Which of the two scalers has already "seen" them?

??? tip "Hint 2 — the approach"
    Split first (`train_test_split`, stratified), then fit one `StandardScaler` on all of `X` and one on `X_train` only. `scaler.mean_` holds the number each one would subtract. The size of the gap on an 8k sample is one question; whether the habit is acceptable is a separate one.

??? example "Hint 3 — most of the code"
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from pipelines.features import NUMERIC, build_features

    df = build_features(as_of="2024-06-01")
    X, y = df[NUMERIC], df["is_churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    all_rows = StandardScaler().fit(X)
    train_only = StandardScaler().fit(X_train)
    i = NUMERIC.index("mrr")
    print(f"mrr mean  all rows={all_rows.mean_[i]:.3f}  train only={train_only.mean_[i]:.3f}")
    ```

**2. Missingness flag.** Add `has_usage = total_usage > 0`. Does churn differ? Would you keep the flag *and* the count?

??? tip "Hint 1 — a nudge"
    `build_features` fills "no usage rows" with 0. Is a customer with zero usage the same story as a customer with a little usage?

??? tip "Hint 2 — the approach"
    Add the boolean column, then `groupby("has_usage")["is_churned"].agg(["count", "mean"])`. Keep both columns only if *zero vs some* tells a different story than the count already does.

??? example "Hint 3 — most of the code"
    ```python
    df["has_usage"] = df["total_usage"] > 0
    print(df.groupby("has_usage")["is_churned"].agg(["count", "mean"]).round(3))
    ```

**3. Contract test.** Write `assert_score_payload(payload: dict)` that checks required keys and types. This is more production-shaped than another transformer.

??? tip "Hint 1 — a nudge"
    What does the service on the other side of `/predict` rely on? List it — then treat anything else as an error, not a bonus.

??? tip "Hint 2 — the approach"
    A dict of `key → allowed types`. Check missing keys, unknown keys, then each type. Prove it with two bad payloads — one missing a key, one with a string where a number belongs — each wrapped in `try` / `except`.

??? example "Hint 3 — most of the code"
    ```python
    NUM = (int, float)
    REQUIRED = {"mrr": NUM, "tenure_so_far": NUM, "log_usage": NUM, "plan_type": (str,)}


    def assert_score_payload(payload: dict) -> None:
        missing = REQUIRED.keys() - payload.keys()
        extra = payload.keys() - REQUIRED.keys()
        if missing or extra:
            raise ValueError(f"missing={sorted(missing)} extra={sorted(extra)}")
        # check each value against REQUIRED[key]; raise TypeError on a mismatch


    good = {"mrr": 49.0, "tenure_so_far": 120, "log_usage": 3.2, "plan_type": "pro"}
    assert_score_payload(good)
    ```

## Success criteria

- Two scaler means printed.
- Churn rates with/without usage.
- Contract test rejects a missing key and a wrong type.

## After you run

Features are the model's public API. Week 15's `validate()` is this function with a worse mood.

## Lesson link

[Week 6 — Features Are the Model's API](../week-06.md)
