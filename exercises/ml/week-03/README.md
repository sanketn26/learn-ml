# Exercise — Week 3 — SQL / as_of

## What you are building

A date-bounded usage count, a grain-tested as-of Customer 360, a tenure_days vs tenure_so_far comparison, and the latest legal as_of in this universe.

## Predict before you run

1. Does `load_customer_360()` silently use *all* usage rows, including after 2024-06-01?
2. When do `tenure_days` and `tenure_so_far` disagree?
3. What happens if you ask for `as_of=2025-01-01`?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-03/starter.py
pytest tests/test_features.py
```

**1. Date bound.** Using DuckDB or Pandas, count `feature_usage` rows with `date <= 2024-06-01` vs all rows. The second number is what `load_customer_360()` silently uses. Write both.

**2. Grain test.** Build the `as_of=2024-06-01` 360 (SQL from the lesson, or `build_features(as_of="2024-06-01", n=None)`). Assert unique `user_id` and `len(frame) ==` the at-risk count from `subscriptions`.

**3. tenure_days vs tenure_so_far.** For five users, print both. When do they disagree? (Anyone who later churns, or whose snapshot `tenure_days` was not "as of today.")

**4. Freshness.** Print min/max of usage and events. What is the latest legal `as_of` in this universe?

## Success criteria

- Two usage counts, cut vs all.
- Unique user_id, row count matches at-risk subs (`n=None`).
- Five-row tenure comparison.
- `pytest tests/test_features.py` passes.

## Debugging clues

- `n=8000` will fail the at-risk length assert.
- Already-churned rows are dropped when `at_risk_only=True`.
- Usage and events clip at 2024-11-30.

## After you run

`build_features` is the as-of path. `load_customer_360` is a convenience sample that does not cut time. Lifetime `tenure_days` is not a legal score-time feature.

## Lesson link

[Week 3 — SQL / as_of](../../../docs/ml/week-03.md)
