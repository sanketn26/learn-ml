# Week 02 — recovery writeup

Lesson: [docs/ml/week-02.md](../../../docs/ml/week-02.md)
Exercise: [docs/ml/exercises/week-02.md](../../../docs/ml/exercises/week-02.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-02/starter.py` first.

## Hint 1

??? tip "Hint 1"

    A plan snapshot is one `groupby.agg`. Region mix is a two-step: collapse
    events to *one row per user*, then left-join. The join validator is the
    grain test you will reuse every week — count in, count out, duplicates,
    nulls.

## Hint 2

??? tip "Hint 2"

    `subs.groupby("plan_type").agg(users=..., churn_rate=..., arpu=...)`.
    Most-common region: `events.groupby("user_id")["region"].agg(lambda s: s.mode().iat[0])`
    (or `value_counts().idxmax()`). After the merge, `duplicated("user_id").sum()`
    must be 0.

## Debugging clues

??? warning "Debugging clues"

    - Events are many-per-user. Join them raw onto subscriptions and the
      table explodes (output rows >> input rows).
    - `mode()` can return more than one value; take the first.
    - Left-join: users with no events get a null region — that is a real
      segment, not a bug. Report the null share; don't silently fill.
    - `is_churned` as 0/1 already averages to a rate.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-02/solution.py
```

```python
snap = subs.groupby("plan_type").agg(
    users=("user_id", "count"),
    churn_rate=("is_churned", "mean"),
    arpu=("mrr", "mean"),
).sort_values("churn_rate", ascending=False)
```

## Why this decision

The leaky bucket is almost always the free plan in this file — not because
free users are "worse people," but because they have the least lock-in. The
validator exists so the *next* join (Week 3 `as_of`) cannot silently fan out.
If output rows exceed ~1% of input, you joined at the wrong grain.
