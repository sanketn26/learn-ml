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

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-02/starter.py
```

**1. Plan snapshot.** Churn rate, mean MRR, and user count by `plan_type`. Which plan is the leaky bucket?

??? tip "Hint 1 — a nudge"
    One row per plan, three numbers per row. `is_churned` is already 0/1 — what is the mean of a column of 0s and 1s?

??? tip "Hint 2 — the approach"
    One `groupby("plan_type").agg(...)` with named aggregations — `count` for users, `mean` for churn and MRR — then `sort_values` so the worst plan is on top.

??? example "Hint 3 — most of the code"
    ```python
    import pandas as pd

    from lib.course_data import find_data_dir

    DATA = find_data_dir()
    subs = pd.read_csv(DATA / "subscriptions.csv")
    snap = subs.groupby("plan_type").agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        arpu=("mrr", "mean"),
    ).sort_values("churn_rate", ascending=False)
    print(snap.round(3))
    ```

**2. Region mix.** From `user_events`, each user's most-common `region`. Left-join onto subscriptions. Does churn differ by region?

??? tip "Hint 1 — a nudge"
    `user_events` has many rows per user; `subscriptions` has one. What happens to the row count if you join them as they are?

??? tip "Hint 2 — the approach"
    Collapse first: one region per `user_id` (the mode, or `value_counts().idxmax()`). Drop null regions before the mode (an all-null group has no mode), and `mode()` can return a tie — take the first value. Then left-join. Users with no events get a null region; that is a segment to report, not a crash to paper over.

??? example "Hint 3 — most of the code"
    ```python
    events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"])
    region = (
        events.dropna(subset=["region"])
        .groupby("user_id")["region"]
        .agg(lambda s: s.mode().iat[0])
        .rename("region")
    )
    subs_r = subs.merge(region, on="user_id", how="left")
    print("null region share:", round(subs_r["region"].isna().mean(), 3))
    print(subs_r.groupby("region", dropna=False)["is_churned"].agg(["count", "mean"]).round(3))
    ```
    Whether the gap changes a staffing plan is your call.

**3. Quality check.** Write a 5-line join validator: input rows, output rows, duplicate `user_id` count, null share of a key metric, and a `raise` if output rows > 1.01 × input rows.

??? tip "Hint 1 — a nudge"
    This is the grain test you'll reuse every week: count in, count out, duplicates, nulls. Test it by feeding it a join you *know* is wrong.

??? tip "Hint 2 — the approach"
    A function that takes the left frame, the joined frame, a key, and a metric column. Print four numbers, then `raise` on fan-out. Prove it fires by merging raw events onto `subs`.

??? example "Hint 3 — most of the code"
    ```python
    def validate_join(left: pd.DataFrame, out: pd.DataFrame, key: str, metric: str) -> None:
        n_in, n_out = len(left), len(out)
        dups = int(out.duplicated(key).sum())
        nulls = out[metric].isna().mean()
        print(f"in={n_in:,} out={n_out:,} dup_{key}={dups:,} null_{metric}={nulls:.1%}")
        # raise if the join fanned out


    validate_join(subs, subs_r, "user_id", "region")
    ```

## Success criteria

- Snapshot table sorted by churn.
- Region is one value per user before the join.
- Validator raises on a fan-out.

## After you run

The leaky bucket is usually free in this file. The validator is the habit Week 3's `as_of` 360 depends on.

## Lesson link

[Week 2 — Pandas: SQL You Already Know, in Python](../week-02.md)
