---
description: Pause a LangGraph run with interrupt_before for human approval, then resume it down approve, reject, or needs-info paths.
---

# Week 4 — Interrupt for a human

**Course:** LangGraph  
**Who this is for:** Engineers who have put a ticket in `pending_approval` and waited for a Slack reaction.

A homemade `ApprovalRequest` class is a to-do list. LangGraph’s version is the same `StateGraph` + `MemorySaver`, paused with `interrupt_before=["approve"]`. Resume with `invoke(None, config)` after `update_state`.

---

## 🎯 What you will be able to do

- Pause a CloudWave refund graph before the write
- Resume three tested paths: **approve**, **reject**, **needs-info**
- Keep the graph small (draft → approve gate → act)
- Know that week 5 still has to key the write

!!! think "Think of it like… a GitHub required reviewer."

    CI is green; merge is blocked until a human hits approve. `interrupt_before=["approve"]` is that required check. The checkpoint is the PR. `thread_id` is the PR number.

## Picture the gate

```
START → draft → ⏸ interrupt_before approve
                     │
              update_state(decision=...)
                     │
                     ▼
                  approve node
                 /     |      \
           execute   cancel   ask
              │        │       │
             END      END     END
```

## One small approval graph

```python
from typing import Annotated, Literal, TypedDict
import operator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class Refund(TypedDict):
    request: str
    decision: str  # "", "approve", "reject", "needs_info"
    log: Annotated[list[str], operator.add]


def draft(state: Refund) -> dict:
    return {"log": [f"drafted:{state['request']}"]}


def apply(state: Refund) -> dict:
    d = state["decision"]
    if d == "approve":
        return {"log": ["executed"]}
    if d == "reject":
        return {"log": ["cancelled"]}
    return {"log": ["asked-for-info"]}


def after_gate(state: Refund) -> Literal["apply"]:
    return "apply"


g = StateGraph(Refund)
g.add_node("draft", draft)
g.add_node("approve", apply)
g.add_edge(START, "draft")
g.add_edge("draft", "approve")
g.add_edge("approve", END)

app = g.compile(checkpointer=MemorySaver(), interrupt_before=["approve"])


def run_path(thread_id: str, decision: str) -> list[str]:
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke(
        {"request": "refund $50 user_0001", "decision": "", "log": []},
        config,
    )
    paused = app.get_state(config)
    assert paused.next == ("approve",) or "approve" in paused.next
    app.update_state(config, {"decision": decision})
    final = app.invoke(None, config)
    return final["log"]


assert run_path("t-approve", "approve")[-1] == "executed"
assert run_path("t-reject", "reject")[-1] == "cancelled"
assert run_path("t-info", "needs_info")[-1] == "asked-for-info"
```

Three `thread_id`s, three decisions, three last log lines. That is the week.

If `get_state(...).next` is empty, the interrupt did not fire — you compiled without `interrupt_before`.

!!! warning "Watch out — update_state is the human"

    Putting `decision="approve"` in the **first** `invoke` skips the point of the gate. The human (or a test) writes the decision after the pause. Also: interrupting does not make `executed` idempotent. Week 5.

!!! success "Ship / don’t ship"

    **Ship** a pause in front of refunds, deletes, and mail that cannot be unsent, with tests for approve / reject / needs-info. **Don’t ship** an `ApprovalRequest` class that is not the graph, and don’t combine this week with a loan-underwriting mega-project.

## Weeks 1–4 together (not “all 4 weeks” as a capstone)

A small refund graph already uses:

1. Branching state (week 1)
2. A join or extra notify node if you add one (week 2)
3. `MemorySaver` (week 3)
4. `interrupt_before` (week 4)

Week 5 (idempotency) is **not** done. Mention it on the write: resume will re-enter `approve` if the process dies after the HTTP 200.

## ✍️ Exercise

[Exercises](exercises/week-04.md).

## 🤔 Reflection

1. Who is allowed to call `update_state` in a real service (which authz)?
2. `needs_info` — do you loop back to `draft` or END with a question ticket?
3. Why is a loan-approval “platform” the wrong exercise for this interrupt?

## 🔗 Next week

Resume is at-least-once. Keys make the charge happen once.
