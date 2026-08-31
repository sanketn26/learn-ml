# Week 03 — recovery writeup

Lesson: [docs/ml/week-03.md](../../../docs/ml/week-03.md)
Exercise: [docs/ml/exercises/week-03.md](../../../docs/ml/exercises/week-03.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-03/starter.py` first.

## Hint 1

??? tip "Hint 1"

    `load_customer_360()` does **not** cut usage at a date. `build_features(as_of=...)`
    does. Count the usage rows on each side of that wall. Grain is "one row
    per at-risk `user_id`." `tenure_days` is a lifetime souvenir;
    `tenure_so_far` is "how long have they been here *as of this morning*."

## Hint 2

??? tip "Hint 2"

    DuckDB or Pandas: `date <= '2024-06-01'` vs all rows in `feature_usage.csv`.
    `build_features(as_of="2024-06-01", n=None)` then
    `assert df["user_id"].is_unique`. Compare five users' `tenure_days` vs
    `tenure_so_far`. Print min/max of usage `date` and events `timestamp`.

## Debugging clues

??? warning "Debugging clues"

    - Sampling (`n=8000`) makes `len(frame)` fail the at-risk count. Use `n=None`.
    - Already-churned customers are dropped when `at_risk_only=True`.
    - `tenure_days` disagrees for anyone who later churns, or whose snapshot
      was not taken at 2024-11-30.
    - An `as_of` after the last log is a question the warehouse cannot answer.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-03/solution.py
```

```python
df = build_features(as_of="2024-06-01", n=None, at_risk_only=True)
assert df["user_id"].is_unique
```

## Why this decision

Training and scoring must build the same row. Cutting usage and events at
`as_of` is the feature-time equivalent of a SQL `WHERE date <= :as_of`.
Lifetime `tenure_days` already knows who left — that is a label leak
waiting for Week 8, so it stays off the model contract.
