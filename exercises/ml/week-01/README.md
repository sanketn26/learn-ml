# Exercise — Week 1 — NumPy: Fast Math on Whole Columns

## What you are building

Feature totals as a 1-D array, a user-normalized usage matrix via broadcasting, and a whale list (top 1% of users by usage).

## Predict before you run

1. Will mean feature-total usage sit above or below the median?
2. What shape must the row-mean have so `(users, features) / (users, 1)` broadcasts?
3. Will the top 1% of users account for closer to 1% of usage, or a lot more?

## Before you start

- Pivoting every user × feature is a big matrix on a laptop. Sample users first.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-01/starter.py
```

Use the real files — the lesson's DAU picture is 7 days × 4 regions.

**1. Feature ranking.** Load `feature_usage.csv`. For each `feature_name`, compute total `usage_count` with a group-by, then convert the totals to a NumPy array and print mean / median / p90 of *those feature totals*.

<details>
<summary>Hint 1 — a nudge</summary>

The stats are over one number per *feature*, not one per usage row. How many numbers should your array hold before you call `np.mean`?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Pandas groups the ragged keys, NumPy does the math: `groupby("feature_name")["usage_count"].sum()`, then `.to_numpy(dtype=float)`, then `np.mean` / `np.median` / `np.quantile(..., 0.9)`. p90 of per-row usage is a different question — don't answer that one by accident.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import numpy as np
import pandas as pd

from lib.course_data import find_data_dir

usage = pd.read_csv(find_data_dir() / "feature_usage.csv")
totals = usage.groupby("feature_name")["usage_count"].sum().to_numpy(dtype=float)
print(f"{len(totals)} features  mean={np.mean(totals):,.0f}  median={np.median(totals):,.0f}  "
      f"p90={np.quantile(totals, 0.9):,.0f}")
```

</details>

**2. Broadcasting on a real pivot.** Pivot a *sample* of users × features into a 2-D usage matrix (`fillna(0)`). Divide each row by that row's mean (user-normalized usage). Shapes: `(users, features) / (users, 1)`.

<details>
<summary>Hint 1 — a nudge</summary>

Broadcasting is a shape contract. You want one divisor per *row*. What shape does `mat.mean(axis=1)` return, and what shape does NumPy need to line it up against `(users, features)`?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Sample a few thousand `user_id`s, then `pivot_table(index="user_id", columns="feature_name", values="usage_count", aggfunc="sum", fill_value=0)`. `mean(axis=1, keepdims=True)` gives `(users, 1)`. A user whose row mean is 0 divides by 0 — swap those means for 1.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
rng = np.random.default_rng(0)
some_users = rng.choice(usage["user_id"].unique(), size=2000, replace=False)
pivot = usage[usage["user_id"].isin(some_users)].pivot_table(
    index="user_id", columns="feature_name", values="usage_count", aggfunc="sum", fill_value=0
)
mat = pivot.to_numpy(dtype=float)
row_means = mat.mean(axis=1, keepdims=True)  # shape (n_users, 1)
normalized = mat / np.where(row_means == 0, 1, row_means)
print(pivot.shape, normalized.shape, np.isfinite(normalized).all())
```

</details>

**3. Whale hunt.** Per `user_id`, sum usage. List user ids in the top 1%. How many are they? What share of all usage do they account for?

<details>
<summary>Hint 1 — a nudge</summary>

"Top 1%" is a cut on a 1-D vector of per-user totals — one line, not a loop over users.

</details>

<details>
<summary>Hint 2 — the approach</summary>

`groupby("user_id")["usage_count"].sum()`, then `np.quantile(per_user, 0.99)` as the cut. The share is the whales' sum over everyone's sum.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
per_user = usage.groupby("user_id")["usage_count"].sum().astype(float)
cut = np.quantile(per_user.to_numpy(), 0.99)
whales = per_user[per_user >= cut]
print(f"{len(whales)} whales (cut={cut:,.0f})  share of usage={whales.sum() / per_user.sum():.1%}")
```

</details>

## Success criteria

- Stats are over *feature totals*, not raw rows.
- Normalized matrix shape matches the pivot; zero-mean rows did not explode.
- Whale count and usage share are printed.

## After you run

Pandas groups the ragged keys. NumPy does the slab math. Broadcasting is a shape contract, not a trick.

## Lesson link

[Week 1 — NumPy: Fast Math on Whole Columns](../../../docs/ml/week-01.md)
