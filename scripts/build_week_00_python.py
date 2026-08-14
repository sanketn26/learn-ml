#!/usr/bin/env python3
"""Week 0 — Strong Python for AI engineers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbformat_util import BOOT, LAPTOP_BOX, code_cell, md_cell, write_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def week0():
    cells = [
        md_cell(
            """# Week 0 — Strong Python for AI Engineers

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Developers who already write code (Java, TypeScript, Go, …) and need Python to be *the* language they think in for AI work. You do not need prior Python.

Python is not “the AI.” It is the **glue**. NumPy, Pandas, and PyTorch are the engines. If the glue is sloppy, the engines leak.

---

## 🎯 What you will be able to do

- Read and write idiomatic Python: names, functions, type hints, dataclasses, classes
- Choose the right collection (list / dict / set / comprehension) and know when it becomes a NumPy array next week
- Treat a model as an object with a contract (`fit` / `predict`) — you will write a tiny one
- Load CloudWave CSVs with the standard library, then see why Pandas exists
- Avoid the five foot-guns that waste a week on an ML team
- Know where PyTorch will plug in later (a tensor is “NumPy that remembers how it was made”)

<div class="think-box">
<strong>Think of it like… TypeScript for a data pipeline, not a website.</strong>
<p>Python is dynamically typed, but you still design types. A <code>user_id: str</code> that silently becomes an <code>int</code> will corrupt a join three notebooks from now. Type hints are documentation the computer can check later (<code>mypy</code>) — they are not optional politeness.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(BOOT),
        md_cell(
            """## The stack you are joining

```
Your product code          Python (this week)
tabular numbers            NumPy          ← Week 1
tables, joins, dates       Pandas         ← Week 2
classical ML               scikit-learn   ← Weeks 5–10
gradients + GPU + nets     PyTorch        ← Weeks 11, 13–15
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p><strong>NumPy</strong> = a typed C array with math verbs.<br>
<strong>Pandas</strong> = a SQL table in RAM.<br>
<strong>PyTorch</strong> = NumPy + a tape recorder (autograd) + optional GPU.<br>
You will bounce between all three. That is the job.</p>
</div>
"""
        ),
        md_cell(
            """## Names, values, and mutability

In Python a name is a **sticky note on an object**, not a box that holds a copy.

```
a = [1, 2, 3]
b = a          # b is another sticky note on the SAME list
b.append(4)    # a is now [1, 2, 3, 4]  ← this surprises Java/C# people
```
"""
        ),
        code_cell(
            """a = [1, 2, 3]
b = a
b.append(4)
print("same object?", a is b, "  a =", a)

c = a.copy()          # shallow copy — new list, same inner objects
c.append(5)
print("after copy:  a =", a, "  c =", c)

# is  = same object in memory
# ==  = same value
print("[] == []", [] == [], "   [] is []", [] is [])
"""
        ),
        md_cell(
            """## Functions — the unit of work

Type hints are not enforced at runtime. Write them anyway. They are how your future teammate (and your editor) know the contract.

<div class="watch-box">
<strong>Watch out — mutable defaults</strong>
<p><code>def add_flag(user, flags=[])</code> reuses the <em>same</em> list forever. Use <code>flags=None</code> and create a new list inside. This bug has shipped in production more times than anyone will admit.</p>
</div>
"""
        ),
        code_cell(
            """from dataclasses import dataclass
from typing import Iterable


def churn_rate(churned: int, n: int) -> float:
    \"\"\"Share of customers who left. n == 0 → 0.0, not an exception.\"\"\"
    if n <= 0:
        return 0.0
    return churned / n


def add_tag(user: dict, tag: str, tags: list[str] | None = None) -> list[str]:
    tags = list(tags) if tags is not None else []
    tags.append(tag)
    return tags


print(churn_rate(8, 50))
print(add_tag({"id": "u1"}, "vip"))
print(add_tag({"id": "u2"}, "vip"))  # must NOT contain two 'vip' from a shared default
"""
        ),
        md_cell(
            """## Dataclasses — the feature payload

A dataclass is a typed struct. Next week this becomes a Pandas row. In Week 12 it becomes the `/predict` JSON body.
"""
        ),
        code_cell(
            """@dataclass
class CustomerFeatures:
    user_id: str
    mrr: float
    tenure_days: int
    plan_type: str
    log_usage: float = 0.0

    def is_paid(self) -> bool:
        return self.plan_type != "free"


row = CustomerFeatures("user_0001", mrr=29.0, tenure_days=80, plan_type="starter")
print(row)
print("paid?", row.is_paid())
print("as dict (API body):", row.__dict__)
"""
        ),
        md_cell(
            """## Collections and comprehensions

| Want | Use |
|---|---|
| Ordered bag | `list` |
| Unique ids | `set` |
| Lookup by key | `dict` |
| Immutable row | `tuple` |
| “Map this list” | `[f(x) for x in xs if pred(x)]` |

Comprehensions are Python’s `map` + `filter`. When the list is 160,000 numbers and the work is math, **stop** and use NumPy (Week 1).
"""
        ),
        code_cell(
            """plans = ["free", "starter", "pro", "free", "enterprise"]
paid = [p for p in plans if p != "free"]
counts = {p: plans.count(p) for p in set(plans)}
print("paid plans:", paid)
print("counts:", counts)

# dict as a tiny join
mrr_by_plan = {"free": 0.0, "starter": 29.0, "pro": 99.0, "enterprise": 499.0}
print("ARPU of paid:", [mrr_by_plan[p] for p in paid])
"""
        ),
        md_cell(
            """## A class is a model-shaped object

scikit-learn, PyTorch `nn.Module`, and your future API all share this shape: **hold config in `__init__`, learn in `fit`, answer in `predict`.**
"""
        ),
        code_cell(
            """class MeanBaseline:
    \"\"\"The model you must beat. Predicts the training-set average forever.\"\"\"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y: Iterable[float]) -> "MeanBaseline":
        values = list(y)
        self.mean_ = sum(values) / len(values)
        return self

    def predict(self, n: int) -> list[float]:
        if self.mean_ is None:
            raise RuntimeError("call fit() before predict()")
        return [self.mean_] * n


model = MeanBaseline().fit([10, 20, 30, 40])
print("learned mean:", model.mean_)
print("predictions:", model.predict(3))
"""
        ),
        md_cell(
            """## Load CloudWave with the standard library

Before Pandas, feel the grain of the file. `csv.DictReader` is “one dict per row.”
"""
        ),
        code_cell(
            """import csv
from collections import Counter

path = DATA / "subscriptions.csv"
with path.open() as f:
    rows = list(csv.DictReader(f))

print(f"{len(rows):,} rows  columns={list(rows[0])}")
print("first row:", rows[0])

plans = Counter(r["plan_type"] for r in rows)
churned = sum(r["is_churned"] == "1" for r in rows)
print("plans:", dict(plans))
print(f"churn rate: {churned / len(rows):.3f}")

# This loop is fine for 50k rows. It is the thing NumPy/Pandas will replace when
# the work is 'mean of a numeric column' or 'join three files.'
"""
        ),
        md_cell(
            """## Errors, context managers, and “fail loud”

ML code fails as **silent NaNs** more often than as exceptions. Raise when a contract breaks. Use `with` so files close even when you throw.
"""
        ),
        code_cell(
            """def parse_mrr(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"mrr is not a number: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"mrr cannot be negative: {value}")
    return value


print(parse_mrr("29.01"))
try:
    parse_mrr("n/a")
except ValueError as exc:
    print("caught:", exc)
"""
        ),
        md_cell(
            """## Five foot-guns (print this on a sticky note)

1. **Mutable default arguments** — see above.
2. **`is` vs `==`** — `is` is identity. Use `==` for values. Exception: `x is None`.
3. **Integer division in your head** — `3 / 2` is `1.5`. `3 // 2` is `1`.
4. **Modifying a list while looping over it** — iterate over a copy, or build a new list.
5. **Assuming dict order is insertion order** — it is (3.7+), but do not use a dict as a vector. That is NumPy.

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Ship a <code>.py</code> module when a function is used in two notebooks (loaders, <code>predict()</code>, validators). Stay in a notebook when you are exploring. The moment you paste the same 30 lines a third time, it is a module.</p>
</div>
"""
        ),
        md_cell(
            """## PyTorch in one paragraph (preview)

You will not train a net today. You need the *shape* in your head:

```
NumPy array     torch.tensor     # same mental model: a slab of numbers
x * 2           x * 2            # same spelling
—               x.backward()     # NEW: fill in .grad by walking the tape
—               x.to("cuda")     # NEW: same tensor, different device
```

Week 11 writes a training loop. Weeks 13–15 put CNNs, RNNs, and Transformers on that loop.

<div class="cue-box">
<strong>Visual cue — what “strong Python” means on an ML team</strong>
<p>Not leetcode. It means: dataclasses for payloads, one class per model, no mutable defaults, paths via <code>pathlib</code>, and you can read a stack trace without panic.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Plan report.** Using only `csv` + `Counter`, print churn rate per `plan_type` from `subscriptions.csv`.

**2. Dataclass round-trip.** Build a `CustomerFeatures` from a subscription row. Write `to_payload(self) -> dict` that a JSON API could accept.

**3. MeanBaseline tests.** `assert` that `fit([2, 4, 6]).predict(2)` returns `[4.0, 4.0]`. `assert` that `predict` before `fit` raises.

**4. Foot-gun hunt.** Deliberately write the mutable-default version of `add_tag` and show the second call is dirty. Then fix it.

## 🤔 Reflection

1. When would you keep a Python loop even after you know NumPy?
2. Why is `MeanBaseline` a class instead of a function?
3. Which of the five foot-guns have you already shipped in another language?

## 🔗 Next week

NumPy: the same CloudWave numbers, but as a typed column the CPU can chew in one gulp.
"""
        ),
    ]
    write_notebook(OUT / "week-00-saas.ipynb", cells, "Week 0 — Python")


if __name__ == "__main__":
    week0()
