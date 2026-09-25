# Week 15 — recovery writeup

Lesson: [docs/ml/week-15.md](../../../docs/ml/week-15.md)
Exercise: [docs/ml/exercises/week-15.md](../../../docs/ml/exercises/week-15.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-15/starter.py` first.

## Hint 1

??? tip "Hint 1"

    A pickle is not production. Stand on an earlier date and score a later
    one — not a shuffled row index, and not a `signup_date` cut (at a fixed
    `as_of` that is a tenure cut). The handler talks to `validate` / `predict` in
    `pipelines.contract`, not a homemade dataclass. CS has 80 calls, not a
    0.5 threshold. Overlay histograms to see whether the world moved.

## Hint 2

??? tip "Hint 2"

    `snapshot_split(as_of, horizon_days=30)` from `pipelines.split`. Train
    the GBT with `make_preprocessor()` on FEATURE_COLS, and compare it with a
    shuffled split and a `signup_date` cut of one snapshot. Time 80 `predict()` calls with
    `time.perf_counter`. Threshold = the 80th-highest test score (same idea
    as `pipelines.train._threshold_for_budget`). Dump with `joblib` into
    `artifacts/<version>/model.joblib`.

## Debugging clues

??? warning "Debugging clues"

    - A shuffled split scores rows from the same day on both sides — it
      flatters. A `signup_date` cut puts all short tenures in test and
      none in train — it collapses. Print `tenure_so_far` ranges per side.
    - `validate` rejects unknown keys — don't send `user_id`.
    - `plan_type` must be a `str`, not a pandas NA.
    - Dumping only the forest without the preprocessor means prod cannot
      one-hot a plan.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-15/solution.py
```

```python
from pipelines.contract import predict, validate
from pipelines.split import snapshot_split
train, y_train, test, y_test = snapshot_split(as_of, horizon_days=30)
```

## Why this decision

Random splits measure "can we rank customers who look like the ones we
already have." A backtest measures "standing on a date, can we rank who
leaves next?" A `signup_date` cut measures neither — it trains on old
customers and tests on new ones.
That is the number a pickle has to beat a dummy on before it is allowed
near `artifacts/prod`. Capacity, not 0.5, is how the desk actually works.
