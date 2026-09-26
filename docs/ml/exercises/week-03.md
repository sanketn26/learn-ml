---
description: Build a grain-tested as-of Customer 360 in SQL and Pandas, compare tenure_days vs tenure_so_far, and find the latest legal as_of date.
---

# Exercises — Week 3 — SQL / as_of

## What you are building

A date-bounded usage count, a grain-tested as-of Customer 360, a tenure_days vs tenure_so_far comparison, and the latest legal as_of in this universe.

## Predict before you run

1. Does `load_customer_360()` silently use *all* usage rows, including after 2024-06-01?
2. When do `tenure_days` and `tenure_so_far` disagree?
3. What happens if you ask for `as_of=2025-01-01`?

## Before you start

- `build_features` samples ~8k rows by default. Grain and row-count checks need `n=None`.
- `at_risk_only=True` (the default) drops customers who had already churned by `as_of`.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-03/starter.py
pytest tests/test_features.py
```

**1. Date bound.** Using DuckDB or Pandas, count `feature_usage` rows with `date <= 2024-06-01` vs all rows. The second number is what `load_customer_360()` silently uses. Write both.

??? tip "Hint 1 — a nudge"
    Every row has a date. The as-of wall is a `WHERE` clause. How many rows sit on each side of it?

??? tip "Hint 2 — the approach"
    DuckDB reads the CSV in place: one `SELECT` with a `count(*)` and a `count(*) FILTER (WHERE date <= DATE '2024-06-01')`. Pandas works too — parse `date`, compare, `.sum()`.

??? example "Hint 3 — most of the code"
    ```python
    import duckdb
    import pandas as pd

    from lib.course_data import find_data_dir
    from pipelines.features import build_features

    DATA = find_data_dir()
    cut, total = duckdb.sql(f"""
        SELECT count(*) FILTER (WHERE date <= DATE '2024-06-01'), count(*)
        FROM read_csv_auto('{DATA / "feature_usage.csv"}')
    """).fetchone()
    print(f"usage rows as of 2024-06-01: {cut:,}   all rows: {total:,}")
    ```

**2. Grain test.** Build the `as_of=2024-06-01` 360 (SQL from the lesson, or `build_features(as_of="2024-06-01", n=None)`). Assert unique `user_id` and `len(frame) ==` the at-risk count from `subscriptions`.

??? tip "Hint 1 — a nudge"
    "At risk on 2024-06-01" means two conditions on `subscriptions`: they had signed up by then, and they had not already left by then.

??? tip "Hint 2 — the approach"
    Count the at-risk rows straight from `subscriptions.csv` — `signup_date <= as_of` and (`churn_date` is null or after `as_of`). Then `build_features(..., n=None)` and assert `is_unique` plus the length match.

??? example "Hint 3 — most of the code"
    ```python
    as_of = pd.Timestamp("2024-06-01")
    subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["signup_date", "churn_date"])
    at_risk = (subs["signup_date"] <= as_of) & ~(subs["churn_date"] <= as_of)

    frame = build_features(as_of=as_of, n=None)
    assert frame["user_id"].is_unique
    assert len(frame) == int(at_risk.sum()), (len(frame), int(at_risk.sum()))
    ```

**3. tenure_days vs tenure_so_far.** For five users, print both. When do they disagree?

??? tip "Hint 1 — a nudge"
    One of these columns answers "how long had they been here on the morning of `as_of`?" What question does the other one answer, and on which date was it measured?

??? tip "Hint 2 — the approach"
    Pick your five users on purpose: a couple who churned *after* `as_of` and a couple who are still active. `tenure_days` was stamped when the snapshot was taken (or when they left); `tenure_so_far` stops at `as_of`.

??? example "Hint 3 — most of the code"
    ```python
    later_churn = frame[frame["churn_date"].notna()].head(2)
    still_here = frame[frame["churn_date"].isna()].head(3)
    five = pd.concat([later_churn, still_here])
    print(five[["user_id", "signup_date", "churn_date", "tenure_days", "tenure_so_far"]].to_string(index=False))
    ```
    Write the rule for *when* they disagree in your own words.

**4. Freshness.** Print min/max of usage and events. What is the latest legal `as_of` in this universe?

??? tip "Hint 1 — a nudge"
    An `as_of` is only a question the warehouse can answer if the logs reach that far.

??? tip "Hint 2 — the approach"
    `min()` / `max()` of `feature_usage.date` and `user_events.timestamp`. The latest legal `as_of` is bounded by the earlier of the two maxima — and a horizon label needs room *after* it, too.

??? example "Hint 3 — most of the code"
    ```python
    usage_dates = pd.read_csv(DATA / "feature_usage.csv", usecols=["date"], parse_dates=["date"])["date"]
    event_ts = pd.read_csv(DATA / "user_events.csv", usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
    print("usage ", usage_dates.min().date(), "→", usage_dates.max().date())
    print("events", event_ts.min().date(), "→", event_ts.max().date())
    ```

## Success criteria

- Two usage counts, cut vs all.
- Unique user_id, row count matches at-risk subs (`n=None`).
- Five-row tenure comparison.
- `pytest tests/test_features.py` passes.

## After you run

`build_features` is the as-of path. `load_customer_360` is a convenience sample that does not cut time. Lifetime `tenure_days` is not a legal score-time feature.

## Lesson link

[Week 3 — SQL / as_of](../week-03.md)
