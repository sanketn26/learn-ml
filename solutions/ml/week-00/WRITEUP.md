# Week 00 — recovery writeup

Lesson: [docs/ml/week-00.md](../../../docs/ml/week-00.md)
Exercise: [docs/ml/exercises/week-00.md](../../../docs/ml/exercises/week-00.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-00/starter.py` first.

## Hint 1

??? tip "Hint 1"

    Treat `subscriptions.csv` as a stream of dicts, not a DataFrame. Count
    two things per plan (heads, and how many of those heads cancelled). A
    dataclass is a JSON body waiting to happen. A model is an object that
    refuses to answer before it has been fitted.

## Hint 2

??? tip "Hint 2"

    `csv.DictReader` + `collections.Counter` for task 1. `CustomerFeatures`
    needs a `to_payload` that returns plain Python types. `MeanBaseline.fit`
    should `return self` so you can chain. The mutable-default version of
    `add_tag` uses `tags: list = []`; the fix is `tags is None` then a new
    list.

## Debugging clues

??? warning "Debugging clues"

    - `is_churned` in the CSV is the string `"0"` / `"1"`, not a bool.
    - Dividing by zero if a plan never appears — guard the rate.
    - `predict` before `fit` must **raise**, not return zeros.
    - If the second `add_tag` call already contains the first tag, you still
      have a shared default list.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-00/solution.py
```

Key pieces — the payload and the fit/predict contract:

```python
def to_payload(self) -> dict:
    return {
        "user_id": self.user_id,
        "mrr": self.mrr,
        "tenure_days": self.tenure_days,
        "plan_type": self.plan_type,
        "log_usage": self.log_usage,
    }

def predict(self, n: int) -> list[float]:
    if self.mean_ is None:
        raise RuntimeError("call fit() before predict()")
    return [self.mean_] * n
```

## Why this decision

A scoring API will not accept a dataclass instance; it accepts a dict of
JSON types. Putting `to_payload` on the struct now is the same contract
Week 15's `validate()` will enforce later. `fit` returning `self` is how
every sklearn estimator chains, and raising before `fit` is the "400, not a
quietly defaulted 0" habit.
