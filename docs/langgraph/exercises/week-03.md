---
description: Exercises using LangGraph's MemorySaver checkpointer to crash a graph after node two, then resume the same thread without rerunning finished nodes.
---

# Exercises — Week 3 — MemorySaver resume

Do these after reading [Week 3](../week-03.md). Use LangGraph 0.2’s checkpointer, not a homemade `Checkpoint` class.

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "t1"}}
```

## 1. Crash after node 2

Three nodes. Node 3 raises when `crash is True`. Invoke once, catch the error.

**Checks:**

- `RUNS["n1"] == 1` and `RUNS["n2"] == 1` after the crash
- `app.get_state(config).values["log"]` contains `n1` and `n2`, not a successful `n3`

## 2. Resume

`update_state(config, {"crash": False})` then `invoke(None, config)`.

**Checks:**

- `RUNS["n1"]` and `RUNS["n2"]` are still 1
- `RUNS["n3"] == 2` (failed + succeeded)
- Final log ends with `n3`

## 3. Wrong thread

Invoke a **different** `thread_id` after the crash (do not resume `t1`).

**Checks:**

- The new thread starts at node 1 (`RUNS["n1"]` increments)
- You can explain in one sentence why `thread_id` is the resume key

## Predict before you run

After a crash in node 3, are `RUNS["n1"]` and `RUNS["n2"]` still 1? A new `thread_id` — does it resume n3 or start at n1?

## Runnable command

```bash
python your_memorysaver_resume.py
```

Use LangGraph 0.2's `MemorySaver`, not a homemade `Checkpoint` class.

## Expected observation

Crash leaves n1/n2 at 1. Resume with `crash=False` runs n3 (count 2). A different thread increments n1.

## Self-check

`thread_id` is the resume key. `invoke(None, config)` continues; a new thread does not.
