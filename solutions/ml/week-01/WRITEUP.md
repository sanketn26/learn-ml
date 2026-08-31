# Week 01 — recovery writeup

Lesson: [docs/ml/week-01.md](../../../docs/ml/week-01.md)
Exercise: [docs/ml/exercises/week-01.md](../../../docs/ml/exercises/week-01.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-01/starter.py` first.

## Hint 1

??? tip "Hint 1"

    Group in Pandas (or a dict of sums), then hop to NumPy for the
    statistics. Broadcasting is a shape problem: one mean per *row*, not one
    mean for the whole matrix. Whales are a percentile cut on a 1-D usage
    vector, not a loop over users.

## Hint 2

??? tip "Hint 2"

    `usage.groupby("feature_name")["usage_count"].sum().to_numpy()` then
    `np.mean` / `np.median` / `np.quantile(..., 0.9)`. For the pivot,
    `keepdims=True` on `mean(axis=1)` so the divisor is `(users, 1)`. Top 1%
    is `np.quantile(totals, 0.99)`.

## Debugging clues

??? warning "Debugging clues"

    - Dividing by a row mean of 0 → `inf`. Replace 0 means with 1.
    - Pivoting the *full* user × feature matrix can be huge. Sample users.
    - `p90` of the *row* usage is a different question than p90 of *feature totals*.
    - `quantile` on an integer array can surprise you; use float.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-01/solution.py
```

Row-normalize with broadcasting:

```python
mat = pivot.to_numpy(dtype=float)
row_means = mat.mean(axis=1, keepdims=True)
normalized = mat / np.where(row_means == 0, 1, row_means)
```

## Why this decision

NumPy earns its keep when the work is the *same math on a whole slab*. The
groupby still belongs in Pandas (ragged keys). Once you have a rectangular
matrix, shapes are the API: `(users, features) / (users, 1)` is the contract
that "each customer is compared to themselves, not to a global mean."
