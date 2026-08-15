# Exercises — Week 1 — NumPy: Fast Math on Whole Columns

Do these after reading [Week 1 — NumPy: Fast Math on Whole Columns](../week-01.md).

**1. Feature ranking.** Load `feature_usage.csv`. For each `feature_name`, compute total `usage_count` with a group-by, then convert the totals to a NumPy array and print mean / median / p90 of *those feature totals*.

**2. Broadcasting on a real pivot.** Pivot a *sample* of users × features into a 2-D usage matrix (`fillna(0)`). Divide each row by that row’s mean (user-normalized usage). Shapes: `(users, features) / (users, 1)`.

**3. Whale hunt.** Per `user_id`, sum usage. List user ids in the top 1%. How many are they? What share of all usage do they account for?

??? tip "💡 Hint — row-normalize with broadcasting"


    ```python
    mat = pivot.to_numpy()
    row_means = mat.mean(axis=1, keepdims=True)  # shape (n_users, 1)
    normalized = mat / np.where(row_means == 0, 1, row_means)
    ```
