---
description: Python fundamentals exercises: a churn report from a CSV, a JSON-serializable dataclass, a fit/predict baseline class, and a mutable-default experiment.
---

# Exercises — Week 0 — Strong Python for AI Engineers

Priya wants plan-level churn numbers in Slack this afternoon. No Pandas yet — that's next week. Standard library only.

## What you are building

A plan-level churn report with the standard library, a dataclass that round-trips to a JSON payload, a `fit` / `predict` class, and a fixed mutable-default foot-gun.

## Predict before you run

1. Which plan will have the highest churn rate, and why (lock-in, not morality)?
2. Will `MeanBaseline().predict(1)` return zeros or raise if you forgot `fit`?
3. After two `add_tag` calls with a mutable default, does the second user inherit the first tag?

## Before you start

- `is_churned` in the CSV is the string `"0"` / `"1"`, not a bool. `csv` gives you strings for every column.

Each task has three hints, closed by default. Open only as far as you need — each one costs you a bit of the puzzle.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-00/starter.py
```

**1. Plan report.** Using only `csv` + `Counter`, print churn rate per `plan_type` from `subscriptions.csv`.

??? tip "Hint 1 — a nudge"
    A churn rate is two counts per plan: how many rows, and how many of those rows cancelled. You never need the whole file in memory.

??? tip "Hint 2 — the approach"
    `csv.DictReader` yields one dict per row. Keep two `Counter`s keyed by `plan_type` — heads and churned heads — and divide at the end. Print the denominator next to the rate so nobody quotes 50% off two customers.

??? example "Hint 3 — most of the code"
    ```python
    import csv
    from collections import Counter

    from lib.course_data import find_data_dir

    heads, churned = Counter(), Counter()
    with open(find_data_dir() / "subscriptions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            heads[row["plan_type"]] += 1
            churned[row["plan_type"]] += row["is_churned"] == "1"

    for plan in sorted(heads):
        print(f"{plan:<11} {churned[plan]:>5} / {heads[plan]:>6} = {churned[plan] / heads[plan]:.1%}")
    ```
    The *why* behind the top plan is yours to write.

**2. Dataclass round-trip.** Build a `CustomerFeatures` from a subscription row. Write `to_payload(self) -> dict` that a JSON API could accept.

??? tip "Hint 1 — a nudge"
    A dataclass is a request body waiting to happen. What would `json.dumps` choke on if you passed the raw CSV row straight through?

??? tip "Hint 2 — the approach"
    Convert types at the edge: a `from_row` classmethod turns strings into `float` / `int` once. `to_payload` then returns plain Python types — `dataclasses.asdict` gets you most of the way. Prove it with `json.dumps`.

??? example "Hint 3 — most of the code"
    ```python
    import json
    from dataclasses import asdict, dataclass


    @dataclass
    class CustomerFeatures:
        user_id: str
        plan_type: str
        mrr: float
        tenure_days: int

        @classmethod
        def from_row(cls, row: dict) -> "CustomerFeatures":
            return cls(row["user_id"], row["plan_type"], float(row["mrr"]), int(row["tenure_days"]))

        def to_payload(self) -> dict:
            return asdict(self)


    with open(find_data_dir() / "subscriptions.csv", newline="") as fh:
        first = next(csv.DictReader(fh))
    print(json.dumps(CustomerFeatures.from_row(first).to_payload()))
    ```

**3. MeanBaseline tests.** `assert` that `fit([2, 4, 6]).predict(2)` returns `[4.0, 4.0]`. `assert` that `predict` before `fit` raises.

??? tip "Hint 1 — a nudge"
    A model has state that does not exist until `fit` runs. What should an HTTP handler do when it is asked a question it has no data for — answer anyway, or fail loudly?

??? tip "Hint 2 — the approach"
    Set `self.mean_ = None` in `__init__`. `fit` stores the mean and `return self` so the calls chain. `predict` checks for `None` first. The "must raise" assert is a `try` / `except` with an `else: raise AssertionError`.

??? example "Hint 3 — most of the code"
    ```python
    class MeanBaseline:
        def __init__(self) -> None:
            self.mean_: float | None = None

        def fit(self, y: list[float]) -> "MeanBaseline":
            self.mean_ = sum(y) / len(y)
            return self

        def predict(self, n: int) -> list[float]:
            if self.mean_ is None:
                raise RuntimeError("call fit() before predict()")
            return [self.mean_] * n


    assert MeanBaseline().fit([2, 4, 6]).predict(2) == [4.0, 4.0]
    try:
        MeanBaseline().predict(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("predict before fit should raise")
    ```

**4. Foot-gun hunt.** Deliberately write the mutable-default version of `add_tag` and show the second call is dirty. Then fix it.

??? tip "Hint 1 — a nudge"
    Call it twice for two different users, passing no list, and print the second result. Then ask: where did that list object come from, and *when* was it created?

??? tip "Hint 2 — the approach"
    A default value is built once, when the `def` line runs — not on every call. Every call that omits the argument shares that one object. The fix is a `None` sentinel and a fresh list inside the body.

??? example "Hint 3 — most of the code"
    ```python
    def add_tag_buggy(user_id: str, tag: str, tags: list = []) -> dict:
        tags.append(tag)
        return {"user_id": user_id, "tags": tags}


    add_tag_buggy("user_1", "vip")
    print("second call:", add_tag_buggy("user_2", "trial"))


    def add_tag(user_id: str, tag: str, tags: list | None = None) -> dict:
        tags = [] if tags is None else list(tags)
        # append and return, same shape as the buggy version
    ```

## Success criteria

- One churn rate per plan, denominators visible.
- `to_payload()` is a dict of JSON-safe types.
- Both MeanBaseline asserts pass.
- Buggy `add_tag` is dirty; the fix is not.

## After you run

Python is glue. The dataclass is next week's row and Week 15's `/predict` body. A model that answers before `fit` is a handler that 200s an empty payload. Send Priya the numbers.

## Lesson link

[Week 0 — Strong Python for AI Engineers](../week-00.md)
