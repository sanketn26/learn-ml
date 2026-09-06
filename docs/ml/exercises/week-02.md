---
description: Build a churn-by-plan snapshot, a region mix via groupby and join, and a row-count join validator using Pandas on CloudWave subscription data.
---

# Exercises — Week 2 — Pandas: SQL You Already Know, in Python

## What you are building

A plan snapshot (churn, ARPU, headcount), a region mix after collapsing events to one row per user, and a five-line join validator.

## Predict before you run

1. Which plan is the leaky bucket?
2. If you join raw `user_events` onto subscriptions, do output rows stay ~49k or explode?
3. Will churn differ by region enough to change a staffing plan, or is it a small mix shift?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-02/starter.py
```

**1. Plan snapshot.** Churn rate, mean MRR, and user count by `plan_type`. Which plan is the leaky bucket?

**2. Region mix.** From `user_events`, each user's most-common `region`. Left-join onto subscriptions. Does churn differ by region?

**3. Quality check.** Write a 5-line join validator: input rows, output rows, duplicate `user_id` count, null share of a key metric, and a `raise` if output rows > 1.01 × input rows.

??? tip "✅ One possible plan snapshot"

    ```python
    subs.groupby("plan_type").agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        arpu=("mrr", "mean"),
    ).sort_values("churn_rate", ascending=False)
    ```

## Success criteria

- Snapshot table sorted by churn.
- Region is one value per user before the join.
- Validator raises on a fan-out.

## Debugging clues

- Events are many-per-user. Join them raw and the grain dies.
- `mode()` can return two values; take one.
- Null region after a left-join is a segment, not a crash.

## After you run

The leaky bucket is usually free in this file. The validator is the habit Week 3's `as_of` 360 depends on.

## Lesson link

[Week 2 — Pandas: SQL You Already Know, in Python](../week-02.md)
