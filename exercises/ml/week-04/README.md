# Exercise — Week 4 — Charts That Change a Decision

Helen's board meeting is Monday. She asked for one screen, not a notebook. Rebuild the deck's plan-churn bar honestly, and give her the adoption and region cuts she'll ask about when someone pushes back.

## What you are building

Three charts: adoption by plan, churn by region, and an honest plan-churn bar whose title is a claim and whose y-axis starts at 0.

## Predict before you run

1. Which plan adopts the most features?
2. If the y-axis starts at 14% instead of 0, what lie does the screenshot tell?
3. Should region churn be computed on events or on users?

## Before you start

- Use the `Agg` backend (`matplotlib.use("Agg")`) in a terminal so matplotlib writes PNGs instead of blocking on a window.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-04/starter.py
```

**1. Adoption curve.** For users with a `signup_date`, plot average `features_adopted` by `plan_type` as a bar. Annotate the winner.

<details>
<summary>Hint 1 — a nudge</summary>

A chart is an API response. If Helen has to squint to find the winner, the response is incomplete — put the number on the bar.

</details>

<details>
<summary>Hint 2 — the approach</summary>

`load_customer_360()` already has `features_adopted` and `signup_date`. `groupby("plan_type")["features_adopted"].mean()`, `ax.bar`, then `ax.annotate` the `idxmax()` with its value.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
df = load_customer_360(DATA)
adopt = df[df["signup_date"].notna()].groupby("plan_type")["features_adopted"].mean().sort_values()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(adopt.index, adopt.values, color="#6366f1")
top = adopt.idxmax()
ax.annotate(f"{adopt[top]:.2f}", (top, adopt[top]), ha="center", va="bottom")
ax.set_title("Average features adopted by plan")  # rewrite as a claim
fig.savefig("adoption_by_plan.png", dpi=120)
```

</details>

**2. Region bars.** Most-common region per user from events, then churn rate by region. Horizontal bars, sorted.

<details>
<summary>Hint 1 — a nudge</summary>

Should a region with chatty users count more than a region with quiet ones? Decide what one "vote" is before you group.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Same collapse as Week 2: drop null regions, one mode per `user_id`, left-join onto the 360. Only then `groupby("region")["is_churned"].mean()`, `sort_values()`, `ax.barh`.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
events = pd.read_csv(DATA / "user_events.csv", usecols=["user_id", "region"]).dropna()
region = events.groupby("user_id")["region"].agg(lambda s: s.mode().iat[0]).rename("region")
by_user = df.merge(region, on="user_id", how="left")
churn = by_user.groupby(by_user["region"].fillna("unknown"))["is_churned"].mean().sort_values()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(churn.index, churn.values, color="#0f766e")
ax.set_xlabel("churn rate (users)")
fig.savefig("churn_by_region.png", dpi=120)
```

</details>

**3. Honest title.** Rebuild the plan-churn bar so the title is a claim ("Free churn is ~2× paid") and the y-axis starts at 0.

<details>
<summary>Hint 1 — a nudge</summary>

Screenshot the chart and cover the axis labels. What would a board member conclude from the bar heights alone?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Compute churn by plan, `ax.bar`, then `ax.set_ylim(0, ...)` explicitly — matplotlib autoscaling is not a promise. The title is a sentence someone could disagree with.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
plan_churn = df.groupby("plan_type")["is_churned"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(plan_churn.index, plan_churn.values, color="#b91c1c")
ax.set_ylim(0, plan_churn.max() * 1.15)
ax.set_ylabel("churn rate")
ax.set_title("TODO: one sentence the numbers support")
fig.savefig("plan_churn_honest.png", dpi=120)
```

</details>

## Success criteria

- Winner annotated on the adoption bars.
- Region chart is user-grain.
- Honest chart: title is a sentence, ylim starts at 0.

## After you run

A chart is an API response. If the PM can misquote the title, rewrite the title. This is the screen Helen actually shows the board.

## Lesson link

[Week 4 — Charts That Change a Decision](../../../docs/ml/week-04.md)
