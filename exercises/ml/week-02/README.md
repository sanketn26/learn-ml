# Exercise — Week 2 — Pandas: SQL You Already Know, in Python

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-02/starter.py
```

## ✍️ Exercises

**1. Plan snapshot.** Churn rate, mean MRR, and user count by `plan_type`. Which plan is the leaky bucket?

**2. Region mix.** From `user_events`, each user’s most-common `region`. Left-join onto subscriptions. Does churn differ by region?

**3. Quality check.** Write a 5-line join validator: input rows, output rows, duplicate `user_id` count, null share of a key metric, and a `raise` if output rows > 1.01 × input rows.

??? tip "✅ One possible plan snapshot"



    ```python
    subs.groupby("plan_type").agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        arpu=("mrr", "mean"),
    ).sort_values("churn_rate", ascending=False)
    ```
