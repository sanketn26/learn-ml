---
description: Exercises building a LangGraph human-in-the-loop approval graph with interrupt_before, testing approve, reject, and needs-info resume paths.
---

# Exercises — Week 4 — Approve / reject / needs-info

Do these after reading [Week 4](../week-04.md). One small approval graph. Weeks 1–4; week 5 (idempotency) is next — mention it, do not skip it later.

```python
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["approve"])
# ...
app.invoke(payload, config)          # pauses
app.update_state(config, {"decision": ...})
app.invoke(None, config)             # continues
```

## Predict before you run

After the first `invoke` with `interrupt_before=["approve"]`, does `get_state(config).next` include `approve`? Has `executed` already been logged?

## Runnable command

```bash
python your_refund_approval.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Pause

CloudWave refund: `draft` then `approve`. Compile with `interrupt_before=["approve"]`.

**Checks:**

- After the first `invoke`, `get_state(config).next` includes `approve`
- `log` has the draft line and **not** `executed`

??? tip "Hint 1 — a nudge"
    An interrupt is a checkpoint that *waits*. Where does the paused state live, and what tells you which node runs next?

??? tip "Hint 2 — the approach"
    Two nodes, `draft` → `approve`, a `MemorySaver`, and `interrupt_before=["approve"]`. The first `invoke` returns early; `get_state(config)` shows `.values` (what's done) and `.next` (what's waiting).

??? example "Hint 3 — most of the code"
    ```python
    import operator
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph


    class Refund(TypedDict):
        request: str
        decision: str
        log: Annotated[list[str], operator.add]


    def draft(state: Refund) -> dict:
        return {"log": [f"drafted:{state['request']}"]}


    def apply(state: Refund) -> dict:
        outcome = {"approve": "executed", "reject": "cancelled"}.get(state["decision"], "asked-for-info")
        return {"log": [outcome]}


    g = StateGraph(Refund)
    g.add_node("draft", draft)
    g.add_node("approve", apply)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", END)
    app = g.compile(checkpointer=MemorySaver(), interrupt_before=["approve"])

    config = {"configurable": {"thread_id": "refund-1"}}
    app.invoke({"request": "refund $50 user_0001", "decision": "", "log": []}, config)
    paused = app.get_state(config)
    print("next:", paused.next, "log:", paused.values["log"])
    ```

## 2. Three paths

Separate `thread_id`s for `approve`, `reject`, `needs_info`.

**Checks:**

- approve → last log `executed`
- reject → last log `cancelled`
- needs-info → last log `asked-for-info` (or your equivalent)

??? tip "Hint 1 — a nudge"
    The human's answer is just a state update. What two calls turn "paused" into "finished with this decision"?

??? tip "Hint 2 — the approach"
    One helper per run: fresh `thread_id`, first `invoke` (pauses), `update_state(config, {"decision": ...})`, then `invoke(None, config)`. Call it three times with three decisions.

??? example "Hint 3 — most of the code"
    ```python
    def run_path(thread_id: str, decision: str) -> list[str]:
        config = {"configurable": {"thread_id": thread_id}}
        app.invoke({"request": "refund $50 user_0001", "decision": "", "log": []}, config)
        app.update_state(config, {"decision": decision})
        return app.invoke(None, config)["log"]


    for decision in ("approve", "reject", "needs_info"):
        print(decision, "→", run_path(f"t-{decision}", decision))
    ```

## 3. Weeks 1–4, not a loan platform

In five lines, list which week each piece is (branch / reducer, optional extra node, MemorySaver, interrupt). Add one sentence: the write is still at-least-once until week 5 keys it.

**Checks:**

- No `ApprovalRequest` class
- No loan-underwriting project

??? tip "Hint 1 — a nudge"
    Point at each line of your graph and ask which week introduced it.

??? tip "Hint 2 — the approach"
    Four pieces, four weeks, one line each — then the sentence about what happens to `executed` if the process dies right after the refund call but before the checkpoint.

??? example "Hint 3 — a skeleton"
    ```text
    Week 1: <the piece>   — <where it is in this graph>
    Week 2: <the piece>   — <…>
    Week 3: <the piece>   — <…>
    Week 4: <the piece>   — <…>
    Still at-least-once: if <what happens when>, <what the customer sees> — week 5 fixes it with <…>.
    ```

## Expected observation

??? success "Open after you run"
    Pause before approve. Three `thread_id`s: executed / cancelled / asked-for-info. No `ApprovalRequest` class.

## Self-check

Weeks 1–4 pieces named. The write is still at-least-once until week 5 keys it.
