# Week 0 — Strong Python for AI Engineers

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

!!! think "Think of it like… TypeScript for a data pipeline, not a website."

    Python is dynamically typed, but you still design types. A `user_id: str` that silently becomes an `int` will corrupt a join three weeks from now. Type hints are documentation the computer can check later (`mypy`) — they are not optional politeness.

## If you already write software

Python is going to feel sloppy if you came from Java, TypeScript, or Go. That sloppiness is real — and it is also why the ML ecosystem lives here. You ship faster, you shoot your foot faster.

The move is not “become a Pythonista.” The move is: treat Python the way you already treat a glue language in a backend.

- A **name** is a pointer. `b = a` does not copy. Same as `let b = a` for objects in JS.
- A **dataclass** is a struct / DTO. Use it for anything that will later be a JSON body or a table row.
- A **class with `fit` / `predict`** is an interface. scikit-learn, PyTorch, and your future scoring service all share that shape.
- **Type hints** are the API doc. Runtime will not save you. `mypy` and your editor will.

### What this week is not

Not a language tour. Not leetcode. Not “learn 40 standard-library modules.” If you can write a dataclass, not mutate a default argument, and load a CSV without crying, you are ready for NumPy.

```
Java / TS mental model          Python this week
─────────────────────          ─────────────────
class User { ... }             @dataclass class CustomerFeatures
interface Model { predict }    class MeanBaseline: fit / predict
DTO → JSON                     row.__dict__  or  asdict(row)
null-safe copy                 a.copy()  /  copy.deepcopy
```

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all ~49k rows.

```python
from lib.course_data import find_data_dir

DATA = find_data_dir()
```


## The stack you are joining

```
Your product code          Python (this week)
tabular numbers            NumPy          ← Week 1
tables, joins, dates       Pandas         ← Week 2
classical ML               scikit-learn   ← Weeks 7–13
gradients + GPU + nets     PyTorch        ← Weeks 14, 18–20
```

!!! engineer "Engineer mental model"

    **NumPy** = a typed C array with math verbs.

    **Pandas** = a SQL table in RAM.

    **PyTorch** = NumPy + a tape recorder (autograd) + optional GPU.

    You will bounce between all three. That is the job.

## Names, values, and mutability

In Python a name is a **sticky note on an object**, not a box that holds a copy.

```
a = [1, 2, 3]
b = a          # b is another sticky note on the SAME list
b.append(4)    # a is now [1, 2, 3, 4]  ← this surprises Java/C# people
```

```python
a = [1, 2, 3]
b = a
b.append(4)
print("same object?", a is b, "  a =", a)

c = a.copy()          # shallow copy — new list, same inner objects
c.append(5)
print("after copy:  a =", a, "  c =", c)

# is  = same object in memory
# ==  = same value
print("[] == []", [] == [], "   [] is []", [] is [])
```

## Functions — the unit of work

Type hints are not enforced at runtime. Write them anyway. They are how your future teammate (and your editor) know the contract.

!!! warning "Watch out — mutable defaults"

    `def add_flag(user, flags=[])` reuses the *same* list forever. Use `flags=None` and create a new list inside. This bug has shipped in production more times than anyone will admit.

```python
from dataclasses import dataclass
from typing import Iterable

def churn_rate(churned: int, n: int) -> float:
    """Share of customers who left. n == 0 → 0.0, not an exception."""
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
```

## Dataclasses — the feature payload

A dataclass is a typed struct. Next week this becomes a Pandas row. In Week 15 it becomes the `/predict` JSON body.

```python
@dataclass
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
```

## Collections and comprehensions

| Want | Use |
|---|---|
| Ordered bag | `list` |
| Unique ids | `set` |
| Lookup by key | `dict` |
| Immutable row | `tuple` |
| “Map this list” | `[f(x) for x in xs if pred(x)]` |

Comprehensions are Python’s `map` + `filter`. When the list is 160,000 numbers and the work is math, **stop** and use NumPy (Week 1).

```python
plans = ["free", "starter", "pro", "free", "enterprise"]
paid = [p for p in plans if p != "free"]
counts = {p: plans.count(p) for p in set(plans)}
print("paid plans:", paid)
print("counts:", counts)

# dict as a tiny join
mrr_by_plan = {"free": 0.0, "starter": 29.0, "pro": 99.0, "enterprise": 499.0}
print("ARPU of paid:", [mrr_by_plan[p] for p in paid])
```

## A class is a model-shaped object

scikit-learn, PyTorch `nn.Module`, and your future API all share this shape: **hold config in `__init__`, learn in `fit`, answer in `predict`.**

```python
class MeanBaseline:
    """The model you must beat. Predicts the training-set average forever."""

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
```

## Load CloudWave with the standard library

Before Pandas, feel the grain of the file. `csv.DictReader` is “one dict per row.”

```python
import csv
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

# This loop is fine for ~49k rows. It is the thing NumPy/Pandas will replace when
# the work is 'mean of a numeric column' or 'join three files.'

# Pandas contrast (Week 2 is the full course):
import pandas as pd
print(pd.read_csv(path)["plan_type"].value_counts())
```

## Errors, context managers, and “fail loud”

ML code fails as **silent NaNs** more often than as exceptions. Raise when a contract breaks. Use `with` so files close even when you throw.

```python
def parse_mrr(raw: str) -> float:
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
```

## Five foot-guns (print this on a sticky note)

1. **Mutable default arguments** — see above.
2. **`is` vs `==`** — `is` is identity. Use `==` for values. Exception: `x is None`.
3. **Integer division in your head** — `3 / 2` is `1.5`. `3 // 2` is `1`.
4. **Modifying a list while looping over it** — iterate over a copy, or build a new list.
5. **Assuming dict order is insertion order** — it is (3.7+), but do not use a dict as a vector. That is NumPy.

!!! success "Ship / don’t ship"

    Ship a `.py` module when a function is used in two lessons (loaders, `predict()`, validators). Stay in a scratch file when you are exploring. The moment you paste the same 30 lines a third time, it is a module.

## PyTorch in one paragraph (preview)

You will not train a net today. You need the *shape* in your head:

```
NumPy array     torch.tensor     # same mental model: a slab of numbers
x * 2           x * 2            # same spelling
—               x.backward()     # NEW: fill in .grad by walking the tape
—               x.to("cuda")     # NEW: same tensor, different device
```

Week 14 writes a training loop. Weeks 18–20 put CNNs, RNNs, and Transformers on that loop.

!!! tip "Visual cue — what “strong Python” means on an ML team"

    Not leetcode. It means: dataclasses for payloads, one class per model, no mutable defaults, paths via `pathlib`, and you can read a stack trace without panic.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-00.md). Starter: `python exercises/ml/week-00/starter.py` from the repo root.

## 🤔 Reflection

1. When would you keep a Python loop even after you know NumPy?
2. Why is `MeanBaseline` a class instead of a function?
3. Which of the five foot-guns have you already shipped in another language?

## 🔗 Next week

NumPy: the same CloudWave numbers, but as a typed column the CPU can chew in one gulp.
