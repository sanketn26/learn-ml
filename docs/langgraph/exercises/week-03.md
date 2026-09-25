---
description: Exercises using LangGraph's InMemorySaver checkpointer to crash a graph after node two, then resume the same thread without rerunning finished nodes.
---

# Exercises — Week 3 — InMemorySaver resume

Do these after reading [Week 3](../week-03.md). Use LangGraph’s checkpointer, not a homemade `Checkpoint` class.

```python
from langgraph.checkpoint.memory import InMemorySaver

app = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}
```

## Predict before you run

After a crash in node 3, are `RUNS["n1"]` and `RUNS["n2"]` still 1? A new `thread_id` — does it resume n3 or start at n1?

## Runnable command

```bash
python your_memorysaver_resume.py
```

Use LangGraph's `InMemorySaver`, not a homemade `Checkpoint` class.

Each task has three hints, closed by default. Open only as far as you need.

## 1. Crash after node 2

Three nodes. Node 3 raises when `crash is True`. Invoke once, catch the error.

**Checks:**

- `RUNS["n1"] == 1` and `RUNS["n2"] == 1` after the crash
- `app.get_state(config).values["log"]` contains `n1` and `n2`, not a successful `n3`

??? tip "Hint 1 — a nudge"
    A checkpointer saves state *between* nodes. After the crash, what is the last thing it could have saved?

??? tip "Hint 2 — the approach"
    Three nodes in a line, each bumping `RUNS[name]` and appending to a reducer `log`. Compile with `checkpointer=InMemorySaver()`, invoke with a `thread_id` config and `crash=True`, catch the exception, then read `app.get_state(config).values`.

??? example "Hint 3 — most of the code"
    ```python
    import operator
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    RUNS = {"n1": 0, "n2": 0, "n3": 0}


    class Job(TypedDict):
        log: Annotated[list[str], operator.add]
        crash: bool


    def make_node(name: str):
        def node(state: Job) -> dict:
            RUNS[name] += 1
            if name == "n3" and state["crash"]:
                raise RuntimeError("billing API unavailable")
            return {"log": [name]}
        return node


    g = StateGraph(Job)
    for name in RUNS:
        g.add_node(name, make_node(name))
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n3")
    g.add_edge("n3", END)
    app = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    try:
        app.invoke({"log": [], "crash": True}, config)
    except RuntimeError as exc:
        print("crashed:", exc)
    print(RUNS, app.get_state(config).values["log"])
    ```

## 2. Resume

`update_state(config, {"crash": False})` then `invoke(None, config)`.

**Checks:**

- `RUNS["n1"]` and `RUNS["n2"]` are still 1
- `RUNS["n3"] == 2` (failed + succeeded)
- Final log ends with `n3`

??? tip "Hint 1 — a nudge"
    Passing a fresh input starts a new run. What input means "carry on from where this thread stopped"?

??? tip "Hint 2 — the approach"
    Fix the cause first with `update_state(config, {"crash": False})`, then `invoke(None, config)` — `None` means "no new input, resume." Print `RUNS` before and after.

??? example "Hint 3 — most of the code"
    ```python
    app.update_state(config, {"crash": False})
    final = app.invoke(None, config)
    print(RUNS, final["log"])
    ```

## 3. Wrong thread

Invoke a **different** `thread_id` after the crash (do not resume `t1`).

**Checks:**

- The new thread starts at node 1 (`RUNS["n1"]` increments)
- You can explain in one sentence why `thread_id` is the resume key

??? tip "Hint 1 — a nudge"
    `InMemorySaver` is a dict of checkpoints. What's the key?

??? tip "Hint 2 — the approach"
    Build a second config with `thread_id="t2"` and invoke it with a full input. Watch which counters move. Then try `invoke(None, ...)` on a thread that has no checkpoint at all and see what you get.

??? example "Hint 3 — most of the code"
    ```python
    other = {"configurable": {"thread_id": "t2"}}
    before = dict(RUNS)
    app.invoke({"log": [], "crash": False}, other)
    print("before", before, "after", RUNS)
    ```
    The one-sentence explanation is yours.

## Expected observation

??? success "Open after you run"
    Crash leaves n1/n2 at 1. Resume with `crash=False` runs n3 (count 2). A different thread increments n1.

## Self-check

`thread_id` is the resume key. `invoke(None, config)` continues; a new thread does not.
