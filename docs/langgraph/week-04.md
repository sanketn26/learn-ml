---
description: Pause a LangGraph run with interrupt() for human approval, then resume it with Command(resume=...) down approve, reject, or needs-info paths.
---

# Week 4 — Interrupt for a human

`CW-1847`’s customer wants their money back while the export is still broken. The bot can draft that refund. It cannot approve it — that write waits for a person to look.

??? note "Course details"

    **Course:** LangGraph
    **Who this is for:** Engineers who have put a ticket in `pending_approval` and waited for a Slack reaction.

A homemade `ApprovalRequest` class is a to-do list. LangGraph’s version is the same `StateGraph` + checkpointer, paused by calling `interrupt(...)` inside the node that needs a human. Resume with `invoke(Command(resume=decision), config)`.

---

## 🎯 What you will be able to do

- Pause a CloudWave refund graph before the write
- Resume three tested paths: **approve**, **reject**, **needs-info**
- Keep the graph small (draft → approve gate → act)
- Know that week 5 still has to key the write

!!! think "Think of it like… a GitHub required reviewer."

    CI is green; merge is blocked until a human hits approve. `interrupt(...)` is that required check. The checkpoint is the PR. `thread_id` is the PR number. `Command(resume="approve")` is the reviewer's click.

## Picture the gate

```
START → draft → approve node ──⏸ interrupt({"refund": ...})
                     ▲               │  (checkpointed; the caller gets __interrupt__)
                     │               ▼
                     └── invoke(Command(resume="approve" | "reject" | "needs_info"))
                                     │
                          execute / cancel / ask → END
```

## One small approval graph

```python
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class Refund(TypedDict):
    request: str
    log: Annotated[list[str], operator.add]


def draft(state: Refund) -> dict:
    return {"log": [f"drafted:{state['request']}"]}


def approve(state: Refund) -> dict:
    # Pauses here. The payload is what the reviewer sees; the return value is what they decided.
    decision = interrupt({"refund": state["request"], "ask": "approve, reject, or needs_info?"})
    if decision == "approve":
        return {"log": ["executed"]}
    if decision == "reject":
        return {"log": ["cancelled"]}
    return {"log": ["asked-for-info"]}


g = StateGraph(Refund)
g.add_node("draft", draft)
g.add_node("approve", approve)
g.add_edge(START, "draft")
g.add_edge("draft", "approve")
g.add_edge("approve", END)

app = g.compile(checkpointer=InMemorySaver())   # no checkpointer → nothing to resume


def run_path(thread_id: str, decision: str) -> list[str]:
    config = {"configurable": {"thread_id": thread_id}}
    paused = app.invoke({"request": "refund $50 user_0001", "log": []}, config)
    assert "__interrupt__" in paused                       # stopped at the gate, payload attached
    assert "executed" not in paused["log"]                 # nothing written yet
    final = app.invoke(Command(resume=decision), config)   # the human's answer
    return final["log"]


assert run_path("t-approve", "approve")[-1] == "executed"
assert run_path("t-reject", "reject")[-1] == "cancelled"
assert run_path("t-info", "needs_info")[-1] == "asked-for-info"
```

Three `thread_id`s, three decisions, three last log lines. That is the week.

`paused["__interrupt__"][0].value` is the payload you would render in Slack or an admin page. The reviewer's click becomes `Command(resume=...)` on the same `thread_id`.

!!! warning "Watch out — resume re-runs the node from the top"

    On resume, LangGraph does not jump to the line after `interrupt()`. It **re-executes the whole `approve` node**, and this time `interrupt()` returns the decision instead of pausing. Anything above the `interrupt()` call runs twice. Keep side effects *after* the interrupt, or make them idempotent (week 5). The same rule, from the other side: a decision passed in the **first** `invoke` skips the gate entirely — only the resume carries the human's answer.

### The older spelling: a static breakpoint

You will also see `compile(checkpointer=..., interrupt_before=["approve"])`, with the decision written by `app.update_state(config, {"decision": ...})` and the run resumed by `app.invoke(None, config)`. It still works in 1.x and the [agent capstone](../ml/capstone-agent.md) uses it. The difference: a static breakpoint pauses *before* a node no matter what, and the human edits state from outside; `interrupt()` pauses *inside* the node, can be conditional (“only ask above $500”), and hands the answer straight back to the code that needs it. Prefer `interrupt()` for approvals; keep `interrupt_before` for debugging a graph step by step.

!!! success "Ship / don’t ship"

    **Ship** a pause in front of refunds, deletes, and mail that cannot be unsent, with tests for approve / reject / needs-info. **Don’t ship** an `ApprovalRequest` class that is not the graph, and don’t combine this week with a loan-underwriting mega-project.

## Weeks 1–4 together (not “all 4 weeks” as a capstone)

A small refund graph already uses:

1. Branching state (week 1)
2. A join or extra notify node if you add one (week 2)
3. A checkpointer (week 3)
4. `interrupt()` + `Command(resume=...)` (week 4)

Week 5 (idempotency) is **not** done. Mention it on the write: resume re-runs `approve` from the top, and so does recovery if the process dies after the HTTP 200.

## ✍️ Exercise

[Exercises](exercises/week-04.md).

## 🤔 Reflection

1. Who is allowed to send `Command(resume=...)` for a refund in a real service (which authz)?
2. `needs_info` — do you loop back to `draft` or END with a question ticket?
3. Why is a loan-approval “platform” the wrong exercise for this interrupt?

## 🔗 Next week

Resume is at-least-once. Keys make the charge happen once.
