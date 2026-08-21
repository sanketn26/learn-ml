# Exercises — Week 1 — Branching state

Do these after reading [Week 1](../week-01.md). The “done when” is two inputs taking different paths. Return **partial** dicts. Use a list reducer.

```python
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
```

## 1. Two paths

Build a CloudWave ticket graph: `classify` → `route` → `docs` **or** `refund_queue`. Keyword `if`s are fine.

**Checks:**

- `"How do I rotate an API key?"` ends with a docs-path label
- `"I want a refund"` ends with a refund-path label and never ran the docs node (count a `RUNS` counter like the week-3 lesson, or a path field)

## 2. Reducer

State has `log: Annotated[list[str], operator.add]`. `classify` returns `{"log": ["classified"]}`; the branch node returns `{"log": ["docs"]}` or `{"log": ["refund"]}`.

**Checks:**

- After a docs ticket, `log` is `["classified", "docs"]` (both entries present)
- You did not `state["log"].append(...)` as the only write

## 3. Partial updates

A node that only knows the decision returns `{"decision": "docs"}` and does not pass the rest of the state through by mutation.

**Checks:**

- `classify(...)` returns a dict whose keys are a subset of the state schema
- Two invokes do not share a list object (no leaked labels from ticket 1 into ticket 2)
