---
description: Choose the right chart for the question at hand, building dashboards and cohort heatmaps that avoid dishonest, truncated-axis visuals.
---

# Week 4 — Charts That Change a Decision

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who will paste a chart into Slack or a board deck. Not a design class.

---

## 🎯 What you will be able to do

- Pick a chart the way you pick a data structure — by the *question*
- Build a one-page CloudWave dashboard that actually renders
- Read a cohort heatmap without being fooled by “new customers look loyal”
- Spot a dishonest chart (truncated axis, rainbow pie)

!!! think "Think of it like… picking an API response shape."

    A line chart is a time series. A bar chart is a group-by. A scatter is a join between two metrics. A heatmap is a group-by on *two* keys. A pie chart is an unreadable JSON blob with a color theme.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.course_data import find_data_dir

DATA = find_data_dir()
```


## If you already write software

A chart is an API response. You pick the shape for the question, not because a library has a pretty default.

| Question | Chart | Same idea as |
|---|---|---|
| How is this changing? | line | a time-series metric |
| Which group is worst? | bar | a `GROUP BY` |
| Do these two numbers move together? | scatter | a join of two metrics |
| Two keys at once (signup month × age)? | heatmap | a pivot table |
| Share of a whole? | almost never a pie | an unreadable JSON blob |

### What makes a chart honest

Truncating a y-axis from 47% to 51% is the same as returning `{ "uptime": 99.99 }` while hiding that you redefined the denominator. The axis is part of the contract.

A cohort heatmap fools people the same way a “new users are so engaged!” dashboard does: the newest signups have not had time to churn yet. This week’s grid is signup month × {30d, 60d, 90d}. Read **across a row** (same signup month, different ages). To compare product quality, read **down a column** (same age, different signup months). A classic month-by-calendar-month grid is what people read diagonally — that is not this picture.

### Picture the dashboard as a PR

If you would not ship a SQL query without a denominator, do not ship a chart without:

1. a clear question in the title
2. axes that start at a truthful place
3. a sample size (n=) somewhere visible

## 🏢 Scenario — one page for the CFO

Your CFO wants, on one screen:

1. Is churn getting worse?
2. Which plan is the leak?
3. Do engaged customers stay?
4. Do older signup cohorts retain?

!!! tip "Visual cue — chart chooser"

    **Trend over time** → line. **Compare categories** → bar. **Relationship** → scatter. **Two categorical axes + a rate** → heatmap. **Distribution** → histogram or box. **Share of 100%** → still a bar, not a pie (people cannot compare slice angles).

## Why not a pie chart

```
Pie:     [/////####......]   which slice is bigger — #### or .....?
Bar:     Pro         ████
         Starter     ██████
         Free        ████████████
```

If the point is “free is the biggest bucket,” a sorted bar wins in 200ms of eye time.

```python
subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["signup_date", "churn_date"])
usage = pd.read_csv(DATA / "feature_usage.csv")

subs["signup_month"] = subs["signup_date"].dt.to_period("M").dt.to_timestamp()

usage_by_user = usage.groupby("user_id").agg(
    total_usage=("usage_count", "sum"),
    features_adopted=("feature_name", "nunique"),
).reset_index()
df = subs.merge(usage_by_user, on="user_id", how="left")
df["total_usage"] = df["total_usage"].fillna(0)
df["features_adopted"] = df["features_adopted"].fillna(0)

# 1) churn by plan
churn_plan = df.groupby("plan_type")["is_churned"].mean().reindex(
    ["free", "starter", "pro", "enterprise"]
)

# 2) churn by signup month (careful: recent months have had less time to churn)
churn_month = df.groupby("signup_month")["is_churned"].mean()

# 3) engagement vs outcome
sample = df.sample(min(4000, len(df)), random_state=7)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

churn_plan.plot(kind="bar", ax=axes[0, 0], color="#6366f1", rot=0)
axes[0, 0].set_title("Churn rate by plan — free is the leaky bucket")
axes[0, 0].set_ylabel("churn rate")
axes[0, 0].set_ylim(0, 0.15)

churn_month.plot(ax=axes[0, 1], marker="o", color="#0f766e")
axes[0, 1].set_title("Churn by signup month — recent months look 'better'")
axes[0, 1].set_ylabel("churn rate")
axes[0, 1].annotate("haven't had time\nto churn yet →",
                    xy=(0.72, 0.2), xycoords="axes fraction", color="#b45309")

axes[1, 0].scatter(sample["features_adopted"], sample["is_churned"]
                   + np.random.default_rng(0).normal(0, 0.03, len(sample)),
                   alpha=0.15, s=12, c="#1d4ed8")
axes[1, 0].set_title("Features adopted vs churned (jittered 0/1)")
axes[1, 0].set_xlabel("distinct features used")
axes[1, 0].set_ylabel("churned (jittered)")

# 4) tenure is a real clock: signup → churn, or signup → 2024-11-30 if still around
df.boxplot(column="tenure_days", by="is_churned", ax=axes[1, 1], grid=False)
axes[1, 1].set_title("Tenure by churn flag — churners leave earlier")
axes[1, 1].set_xlabel("is_churned")
axes[1, 1].get_figure().suptitle("")

plt.tight_layout()
plt.show()
```

## Cohort heatmap — the chart that fools smart people

**30-day retention** for the cohort that signed up *last week* is not comparable to the cohort from a year ago. Young cohorts have not had time to die.

!!! warning "Watch out"

    A heatmap that is bright-green in the bottom rows (newest signups) often means *they are still new*, not that you suddenly built a better product. Gray-out cells that have not reached that age.


!!! engineer "Engineer mental model"

    A cohort chart is a two-key group-by: `(signup_month, age_bucket) → retained_rate`. Same as a SQL cube. The visual is optional; the grain is not.

```python
# tenure_days is a real clock: signup → churn, or signup → 2024-11-30 if still around.

def retention_at(days: int, frame: pd.DataFrame) -> pd.Series:
    # Eligible = old enough to have reached `days`.
    # Still-active users with short tenure have not had time to churn — drop them.
    eligible = frame.copy()
    too_young = (eligible["is_churned"] == 0) & (eligible["tenure_days"] < days)
    eligible = eligible[~too_young]
    retained = eligible["tenure_days"] >= days
    return eligible.assign(retained=retained).groupby("signup_month")["retained"].mean()

ages = [30, 60, 90]
heat = pd.DataFrame({f"{d}d": retention_at(d, df) for d in ages})

show = heat.tail(18)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.ma.masked_invalid(show.to_numpy()), aspect="auto",
               cmap="YlGnBu", vmin=0.7, vmax=1)
ax.set_xticks(range(show.shape[1]), show.columns)
ax.set_yticks(
    range(show.shape[0]),
    [pd.Timestamp(i).strftime("%Y-%m") for i in show.index],
    fontsize=8,
)
for i in range(show.shape[0]):
    for j in range(show.shape[1]):
        val = show.iloc[i, j]
        if pd.notna(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Retention by signup month × age — read across a row")
ax.set_ylabel("signup month")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

print("Read a row: of people who signed up that month and are old enough,")
print("what fraction lived at least 30 / 60 / 90 days.")
```

## Bad chart vs honest chart

Same data. One lies with a truncated axis.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
churn_plan.plot(kind="bar", ax=axes[0], color="#ef4444", rot=0)
axes[0].set_ylim(0.03, 0.10)
axes[0].set_title("❌ Dishonest: axis starts at 3%")
axes[0].set_ylabel("churn rate")

churn_plan.plot(kind="bar", ax=axes[1], color="#22c55e", rot=0)
axes[1].set_ylim(0, 0.15)
axes[1].set_title("✅ Honest: axis starts at 0")
axes[1].set_ylabel("churn rate")
plt.tight_layout()
plt.show()

print("The left chart makes starter vs pro look like a crisis.")
print("The right chart says: all paid plans are similar; free is different.")
```

!!! success "Ship / don’t ship"

    - **Slack / debugging:** default Pandas/matplotlib is fine. Label axes. Start bars at 0.

    - **Board deck:** one sentence title that is the insight, not the chart type. “Free churn is 2× paid,” not “Churn by plan.”

    - **Never:** dual axes, 3-D bars, pie-of-pies, rainbow on unordered categories.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-04.md). Starter: `python exercises/ml/week-04/starter.py` from the repo root.

## 🤔 Reflection

1. A region’s churn *looks* high. List five non-product reasons (pricing, support hours, competitor, sales quality, data bug).
2. Feature adoption correlates with lower churn. Draw the causation arrow both ways.
3. Which one chart would you send the CEO, and which sentence sits above it?

## 🔗 Next week

The CFO asks: “16% vs 20% churn — is that real?” We will answer without turning you into a statistician.
