# Week 3 — Checkpoint, crash, resume

**Course:** LangGraph  
**Who this is for:** Engineers who have lost 20 minutes of a job because step 3 died and they reran from step 1.

The reason to use a graph is that the **runtime owns the state**. LangGraph 0.2’s in-memory checkpointer is `MemorySaver`. A homemade dict of snapshots is useful intuition; it is not what you compile.

---

## 🎯 What you will be able to do

- Compile with `MemorySaver` and a `thread_id`
- Run until a node fails **after node 2**
- Resume the same thread so nodes 1–2 do not run again
- Know that resume is at-least-once for the failed node (week 5 keys the write)

!!! think "Think of it like… a debugger’s snapshot, not a backup disk."

    `thread_id` is the workflow id. Each completed node writes a checkpoint. Crash = restore that row and continue. `MemorySaver` lives in process RAM — enough to prove resume. Durable production stores are a different class (Postgres, etc.) and out of this week’s scope.

## Picture the crash

```
node1  →  ✓ checkpoint
node2  →  ✓ checkpoint
node3  →  boom
              │
              ▼
         resume(thread_id)
              │
              ▼
         node3 again     ← node1 and node2 must not re-run
```

## MemorySaver, not a homemade store

```python
from typing import Annotated, TypedDict
import operator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

RUNS = {"n1": 0, "n2": 0, "n3": 0}


class Job(TypedDict):
    log: Annotated[list[str], operator.add]
    crash: bool


def node1(state: Job) -> dict:
    RUNS["n1"] += 1
    return {"log": ["n1"]}


def node2(state: Job) -> dict:
    RUNS["n2"] += 1
    return {"log": ["n2"]}


def node3(state: Job) -> dict:
    RUNS["n3"] += 1
    if state["crash"]:
        raise RuntimeError("node3 exploded")
    return {"log": ["n3"]}


g = StateGraph(Job)
g.add_node("node1", node1)
g.add_node("node2", node2)
g.add_node("node3", node3)
g.add_edge(START, "node1")
g.add_edge("node1", "node2")
g.add_edge("node2", "node3")
g.add_edge("node3", END)

app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "t1"}}

try:
    app.invoke({"log": [], "crash": True}, config)
except RuntimeError:
    pass

assert RUNS == {"n1": 1, "n2": 1, "n3": 1}
snap = app.get_state(config)
assert "n1" in snap.values["log"] and "n2" in snap.values["log"]
assert "n3" not in snap.values["log"]

app.update_state(config, {"crash": False})
final = app.invoke(None, config)

assert final["log"][-1] == "n3"
assert RUNS["n1"] == 1 and RUNS["n2"] == 1
assert RUNS["n3"] == 2  # failed once, succeeded once — did not replay n1/n2
```

`invoke(None, config)` means “continue this thread.” It is not a new run.

!!! warning "Watch out — resume re-enters the failed node"

    Node 3 ran, threw, and will run again. If node 3 had charged a card before raising, you now have a double charge. Checkpoints are necessary and **not sufficient**. Week 5 puts an idempotency key on the write.

!!! success "Ship / don’t ship"

    **Ship** a `thread_id` + checkpointer when a crash must not replay completed **pure** work. **Don’t ship** a homemade `Checkpoint` class as if it were LangGraph, and don’t tell anyone resume is exactly-once.

## What this week is not

- Not Postgres. `MemorySaver` dies with the process — that is fine for the concept demo.
- Not human approval (week 4 uses the same `MemorySaver` + `interrupt_before`).
- Not a 30-minute Spark job. The three-node graph is the whole point.

## ✍️ Exercise

[Exercises](exercises/week-03.md).

## 🤔 Reflection

1. After the crash, which node is next? How do you know without printing `RUNS`?
2. Why must `thread_id` be stable across the crash?
3. What happens if you `invoke({...}, config)` with a **new** dict instead of `invoke(None, config)`?

## 🔗 Next week

Pause before a write: `interrupt_before=["approve"]`, then approve / reject / needs-info.
