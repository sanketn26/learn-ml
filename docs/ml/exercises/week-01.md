---
description: NumPy exercises computing feature-usage totals, normalizing a user-by-feature matrix with broadcasting, and identifying the top 1% usage whales.
---

# Exercises — Week 1 — NumPy: Fast Math on Whole Columns

## What you are building

Feature totals as a 1-D array, a user-normalized usage matrix via broadcasting, and a whale list (top 1% of users by usage).

## Predict before you run

1. Will mean feature-total usage sit above or below the median? (Whales pull the mean.)
2. What shape must the row-mean have so `(users, features) / (users, 1)` broadcasts?
3. Will the top 1% of users account for closer to 1% of usage, or a lot more?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-01/starter.py
```

Use the real files — the lesson's DAU picture is 7 days × 4 regions.

**1. Feature ranking.** Load `feature_usage.csv`. For each `feature_name`, compute total `usage_count` with a group-by, then convert the totals to a NumPy array and print mean / median / p90 of *those feature totals*.

**2. Broadcasting on a real pivot.** Pivot a *sample* of users × features into a 2-D usage matrix (`fillna(0)`). Divide each row by that row's mean (user-normalized usage). Shapes: `(users, features) / (users, 1)`.

**3. Whale hunt.** Per `user_id`, sum usage. List user ids in the top 1%. How many are they? What share of all usage do they account for?

??? tip "💡 Hint — row-normalize with broadcasting"

    ```python
    mat = pivot.to_numpy()
    row_means = mat.mean(axis=1, keepdims=True)  # shape (n_users, 1)
    normalized = mat / np.where(row_means == 0, 1, row_means)
    ```

## Success criteria

- Stats are over *feature totals*, not raw rows.
- Normalized matrix shape matches the pivot; zero-mean rows did not explode.
- Whale count and usage share are printed.

## Debugging clues

- Pivoting every user can be huge — sample.
- Row mean 0 → divide by 1.
- p90 of row usage is a different question than p90 of feature totals.

## After you run

Pandas groups the ragged keys. NumPy does the slab math. Broadcasting is a shape contract, not a trick.

## Lesson link

[Week 1 — NumPy: Fast Math on Whole Columns](../week-01.md)
