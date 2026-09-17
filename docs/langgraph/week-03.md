---
description: Use LangGraph's MemorySaver checkpointer to persist state and resume a crashed graph run without re-executing already-completed nodes.
---

# Week 3 — Checkpoint, crash, resume

The `CW-1847` refund pipeline dies on the third step, mid-run. Nobody wants to re-fetch the ticket and re-check the refund policy just to retry the one node that actually crashed.

??? note "Course details"

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
fetch_ticket        →  ✓ checkpoint
check_refund_policy →  ✓ checkpoint
issue_credit        →  boom
              │
              ▼
         resume(thread_id)
              │
              ▼
    issue_credit again     ← fetch_ticket and check_refund_policy must not re-run
```

## MemorySaver, not a homemade store

```python
from typing import Annotated, TypedDict
import operator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

RUNS = {"fetch_ticket": 0, "check_refund_policy": 0, "issue_credit": 0}


class Job(TypedDict):
    log: Annotated[list[str], operator.add]
    crash: bool


def fetch_ticket(state: Job) -> dict:
    RUNS["fetch_ticket"] += 1
    return {"log": ["fetch_ticket"]}


def check_refund_policy(state: Job) -> dict:
    RUNS["check_refund_policy"] += 1
    return {"log": ["check_refund_policy"]}


def issue_credit(state: Job) -> dict:
    RUNS["issue_credit"] += 1
    if state["crash"]:
        raise RuntimeError("billing API unavailable — issue_credit failed")
    return {"log": ["issue_credit"]}


g = StateGraph(Job)
g.add_node("fetch_ticket", fetch_ticket)
g.add_node("check_refund_policy", check_refund_policy)
g.add_node("issue_credit", issue_credit)
g.add_edge(START, "fetch_ticket")
g.add_edge("fetch_ticket", "check_refund_policy")
g.add_edge("check_refund_policy", "issue_credit")
g.add_edge("issue_credit", END)

app = g.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "CW-1847"}}

try:
    app.invoke({"log": [], "crash": True}, config)
except RuntimeError:
    pass

assert RUNS == {"fetch_ticket": 1, "check_refund_policy": 1, "issue_credit": 1}
snap = app.get_state(config)
assert "fetch_ticket" in snap.values["log"] and "check_refund_policy" in snap.values["log"]
assert "issue_credit" not in snap.values["log"]

app.update_state(config, {"crash": False})
final = app.invoke(None, config)

assert final["log"][-1] == "issue_credit"
assert RUNS["fetch_ticket"] == 1 and RUNS["check_refund_policy"] == 1
assert RUNS["issue_credit"] == 2  # failed once, succeeded once — did not replay the first two steps
```

`invoke(None, config)` means “continue this thread.” It is not a new run.

!!! warning "Watch out — resume re-enters the failed node"

    `issue_credit` ran, threw, and will run again. If it had actually moved money before raising, you now have a double payout. Checkpoints are necessary and **not sufficient** — resume is at-least-once, not exactly-once, for the node that crashed. Week 5 shows the fix (an idempotency key) on a different CloudWave write; the same key-the-write pattern applies here.

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

Pause before a write: `interrupt_before=["approve"]`, then approve / reject / needs-info. Two weeks after that, `issue_credit` gets the idempotency key that makes its replay safe.
