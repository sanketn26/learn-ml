# Exercise — Week 0 — Strong Python for AI Engineers

## What you are building

A plan-level churn report with the standard library, a dataclass that round-trips to a JSON payload, a `fit` / `predict` class, and a fixed mutable-default foot-gun.

## Predict before you run

1. Which plan will have the highest churn rate, and why (lock-in, not morality)?
2. Will `MeanBaseline().predict(1)` return zeros or raise if you forgot `fit`?
3. After two `add_tag` calls with a mutable default, does the second user inherit the first tag?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-00/starter.py
```

**1. Plan report.** Using only `csv` + `Counter`, print churn rate per `plan_type` from `subscriptions.csv`.

**2. Dataclass round-trip.** Build a `CustomerFeatures` from a subscription row. Write `to_payload(self) -> dict` that a JSON API could accept.

**3. MeanBaseline tests.** `assert` that `fit([2, 4, 6]).predict(2)` returns `[4.0, 4.0]`. `assert` that `predict` before `fit` raises.

**4. Foot-gun hunt.** Deliberately write the mutable-default version of `add_tag` and show the second call is dirty. Then fix it.

## Success criteria

- One churn rate per plan, denominators visible.
- `to_payload()` is a dict of JSON-safe types.
- Both MeanBaseline asserts pass.
- Buggy `add_tag` is dirty; the fix is not.

## Debugging clues

- `is_churned` in the CSV is `"0"` / `"1"` strings.
- `predict` before `fit` must raise, not return `[0]`.
- If the second `add_tag` already contains `vip`, the default list is shared.

## After you run

Python is glue. The dataclass is next week's row and Week 15's `/predict` body. A model that answers before `fit` is a handler that 200s an empty payload.

## Lesson link

[Week 0 — Strong Python for AI Engineers](../../../docs/ml/week-00.md)
