# Exercises — Week 3 — SQL Is the Source of Truth

Do these after reading [Week 3](../week-03.md).

**1. Date bound.** Using DuckDB or Pandas, count `feature_usage` rows with `date <= 2024-06-01` vs all rows. The second number is what `load_customer_360()` silently uses. Write both.

**2. Grain test.** Build the `as_of=2024-06-01` 360 (SQL from the lesson, or `build_features(as_of="2024-06-01", n=None)`). Assert unique `user_id` and `len(frame) ==` the at-risk count from `subscriptions`.

**3. tenure_days vs tenure_so_far.** For five users, print both. When do they disagree? (Anyone who later churns, or whose snapshot `tenure_days` was not “as of today.”)

**4. Freshness.** Print min/max of usage and events. What is the latest legal `as_of` in this universe?

```bash
pytest tests/test_features.py
```
