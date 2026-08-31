# Week 15 — recovery writeup

Lesson: [docs/ml/week-15.md](../../../docs/ml/week-15.md)
Exercise: [docs/ml/exercises/week-15.md](../../../docs/ml/exercises/week-15.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-15/starter.py` first.

## Hint 1

??? tip "Hint 1"

    A pickle is not production. Split on *when they signed up*, not on a
    shuffled row index. The handler talks to `validate` / `predict` in
    `pipelines.contract`, not a homemade dataclass. CS has 80 calls, not a
    0.5 threshold. Overlay histograms to see whether the world moved.

## Hint 2

??? tip "Hint 2"

    `cutoff = frame["signup_date"].quantile(0.80)`. Train the GBT with
    `make_preprocessor()` on FEATURE_COLS. Time 80 `predict()` calls with
    `time.perf_counter`. Threshold = the 80th-highest test score (same idea
    as `pipelines.train._threshold_for_budget`). Dump with `joblib` into
    `artifacts/<version>/model.joblib`.

## Debugging clues

??? warning "Debugging clues"

    - A shuffled split leaks tomorrow's mix into today's fit. The AUCs
      should differ, even if only a little on this file.
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
cutoff = frame["signup_date"].quantile(0.80)
```

## Why this decision

Random splits measure "can we rank customers who look like the ones we
already have." Time splits measure "can we rank *next month's* signups."
That is the number a pickle has to beat a dummy on before it is allowed
near `artifacts/prod`. Capacity, not 0.5, is how the desk actually works.
