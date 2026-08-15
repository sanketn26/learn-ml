# Week 2 — Pandas: SQL You Already Know, in Python

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written `SELECT / JOIN / GROUP BY`. You do not need statistics yet.

---

## 🎯 What you will be able to do

- Map every Pandas verb to the SQL you already know
- Build a **Customer 360** row from four messy systems
- Catch a join that secretly exploded 10×
- Decide what to do with missing values (0 vs median vs “missing is a signal”)

!!! think "Think of it like… a typed DataFrame is a SQL table that lives in RAM."

    A Series is a column. An index is a primary key (sometimes sloppy). `merge` is `JOIN`. `groupby` is `GROUP BY`. If you can write the query, you can write the Pandas.

```python
from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
```


## If you already write software

Pandas is the ORM you already know, except the table is in RAM and the query language is method calls.

```
SQL                            Pandas
─────────────────────────      ─────────────────────────────
FROM subscriptions             subs = pd.read_csv(...)
SELECT user_id, mrr            subs[["user_id", "mrr"]]
WHERE mrr > 50                 subs[subs["mrr"] > 50]
LEFT JOIN usage USING (id)     subs.merge(usage, on="user_id", how="left")
GROUP BY plan                  subs.groupby("plan_type").agg(...)
COUNT(*)                       .size()
```

The habit that saves careers: **aggregate the many-side before you join.** Joining a 50k customer table to 160k usage rows is the same bug as a SQL join that blows up a report and then you `SUM(mrr)` on the exploded grain.

### Picture the grain

Every frame has a *grain* — “what does one row mean?”

```
subscriptions     one row = one customer      ← start here
feature_usage     one row = one user×feature×day
user_events       one row = one click
feedback          one row = one comment
        │
        ▼  groupby(user_id) first
customer_360      one row = one customer      ← this is what ML wants
```

If two tables have different grains, you do not join them raw. You collapse the many side. Same review comment you would leave on a dbt model.

## 🏢 Scenario — four systems, one customer

CloudWave’s data is not one warehouse table. It is:

| System | File | Grain |
|---|---|---|
| Billing | `subscriptions.csv` | one row per user |
| Product | `feature_usage.csv` | one row per user × feature × day |
| Telemetry | `user_events.csv` | one row per event |
| Support | `feedback.json` | one JSON object per comment (JSON Lines) |

Your job: **one row per customer** the ML weeks can train on.

```
subscriptions ──┐
feature_usage ──┼──► customer_360  (one row = one user)
user_events  ───┤
feedback     ───┘
```

## SQL → Pandas cheat sheet

| You already write | Pandas |
|---|---|
| `SELECT cols` | `df[["user_id", "mrr"]]` |
| `WHERE mrr > 50` | `df[df["mrr"] > 50]` |
| `ORDER BY mrr DESC` | `df.sort_values("mrr", ascending=False)` |
| `GROUP BY plan_type` | `df.groupby("plan_type")` |
| `COUNT(*)` | `.size()` or `.agg(n=("user_id", "count"))` |
| `LEFT JOIN` | `left.merge(right, on="user_id", how="left")` |
| `COALESCE(x, 0)` | `df["x"].fillna(0)` |

!!! engineer "Engineer mental model"

    Prefer **named aggregations** over chained mystery columns. Treat `merge` like a code review: assert row counts before and after, the same way you would check a SQL join in a PR.

## Visual: what each join keeps

```
LEFT (subscriptions)     RIGHT (feedback)

 user_1 ●───────────● comment_a     INNER: only matches
 user_2 ●                           LEFT:  keep user_2, feedback = NaN
 user_3 ●───────────● comment_b     OUTER: everyone, holes filled with NaN
                    ● comment_orphan
```

**SaaS default is LEFT JOIN from the customer table.** You do not want to drop a paying user because they never left feedback.

```python
subs = pd.read_csv(
    DATA / "subscriptions.csv",
    usecols=["user_id", "plan_type", "mrr", "signup_date", "churn_date", "is_churned", "tenure_days"],
    parse_dates=["signup_date", "churn_date"],
)
usage = pd.read_csv(
    DATA / "feature_usage.csv",
    usecols=["user_id", "feature_name", "usage_count", "avg_session_seconds", "date"],
    parse_dates=["date"],
)
events = pd.read_csv(
    DATA / "user_events.csv",
    usecols=["event_id", "user_id", "event_type", "timestamp", "device", "region"],
    parse_dates=["timestamp"],
)
# feedback.json is JSON Lines (one object per line), not a JSON array
feedback = pd.read_json(DATA / "feedback.json", lines=True)

print("subscriptions", subs.shape, list(subs.columns))
print("feature_usage", usage.shape, list(usage.columns))
print("user_events   ", events.shape, list(events.columns))
print("feedback      ", feedback.shape, list(feedback.columns))
print("\nplan_type counts:\n", subs["plan_type"].value_counts().to_string())
print("churn rate   ", subs["is_churned"].mean().round(3))
```

## Worked example — Customer 360

Aggregate the *many* side down to *one row per user* **before** you join. That is the single most important ETL habit in this course.

```python
# 1) collapse usage and events to user grain
usage_by_user = usage.groupby("user_id").agg(
    total_usage=("usage_count", "sum"),
    features_adopted=("feature_name", "nunique"),
    last_usage_date=("date", "max"),
).reset_index()

events_by_user = events.groupby("user_id").agg(
    total_events=("event_id", "count"),
    n_devices=("device", "nunique"),
    n_regions=("region", "nunique"),
).reset_index()

feedback_by_user = feedback.groupby("user_id").agg(
    n_feedback=("feedback_text", "count"),
    avg_sentiment=("sentiment_score", "mean"),
).reset_index()

print("before join, subscriptions rows:", len(subs))

customer = (
    subs.merge(usage_by_user, on="user_id", how="left")
        .merge(events_by_user, on="user_id", how="left")
        .merge(feedback_by_user, on="user_id", how="left")
)

print("after  join, customer rows:     ", len(customer))
print("row-count ratio (want ~1.0):    ", round(len(customer) / len(subs), 3))

# missingness is information
customer["has_feedback"] = customer["n_feedback"].fillna(0).gt(0).astype(int)
customer["total_usage"] = customer["total_usage"].fillna(0)
customer["total_events"] = customer["total_events"].fillna(0)
customer["features_adopted"] = customer["features_adopted"].fillna(0)

print("\nCustomer 360 sample:")
print(customer[["user_id", "plan_type", "mrr", "is_churned",
                "total_usage", "features_adopted", "total_events",
                "has_feedback"]].head())
```

## Watch the join explode on purpose

If you join subscriptions to **raw** feature_usage (many rows per user), you duplicate every billing field.

!!! warning "Watch out — fan-out"

    A 50,000-row customer table joined to 160,000 usage rows becomes ~160,000 rows, and `mrr.sum()` will lie by a factor of ~3. Always aggregate the many-side first. Always print `len(left)` vs `len(result)`.

```python
exploded = subs.merge(usage[["user_id", "usage_count"]], on="user_id", how="left")
print(f"subscriptions: {len(subs):,}")
print(f"joined to raw usage: {len(exploded):,}   ← {len(exploded)/len(subs):.1f}× blow-up")
print(f"true total MRR:     ${subs['mrr'].sum():,.0f}")
print(f"exploded MRR sum:   ${exploded['mrr'].sum():,.0f}   ← do not ship this number")
```

## Missing values — a decision, not a default

```
Is “missing” actually zero?
   │
   ├─ YES (they never used the feature) → fillna(0)
   │
   └─ NO (we never observed it)
         │
         ├─ The hole itself predicts the outcome → keep a has_* flag
         └─ The model needs a number            → median of the *training* set
```

!!! success "Ship / don’t ship"

    `fillna(0)` on *usage* is honest: no events means no usage. `fillna(0)` on *sentiment* is a lie: “no review” is not “neutral review.” Use a flag.

```python
print("Null share in customer_360 (after left joins, before our fills):")
probe = (
    subs.merge(usage_by_user, on="user_id", how="left")
        .merge(feedback_by_user, on="user_id", how="left")
)
print((probe.isna().mean() * 100).round(1).astype(str) + "%")

print("\nChurn rate by 'left any feedback':")
probe["has_feedback"] = probe["n_feedback"].notna()
print(probe.groupby("has_feedback")["is_churned"].mean().round(3))
```


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-02.md). Starter: `python exercises/ml/week-02/starter.py` from the repo root.

## 🤔 Reflection

1. Your exploded MRR was 3× too big. What code review comment do you leave?
2. A PM says “customers who write feedback churn less.” Is that product magic, or selection (happy people write reviews)?
3. When would you *want* an inner join from subscriptions to events?

## 🔗 Next week

Pictures. We will actually plot the dashboard the CFO asked for — not hide it inside a collapsed solution.
