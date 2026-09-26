---
description: Learn LangGraph's StateGraph as a branching state machine, using reducers and partial state updates to route two tickets down different paths.
---

# Week 1 — A graph is a state machine that branches

Two tickets hit the bot in the same minute. One says “I want a refund.” The other asks “how do I rotate an API key?” They cannot both go to the same place — one needs `refund_queue`, the other just needs `docs`.

??? note "Course details"

    **Course:** LangGraph
    **Who this is for:** Engineers who have drawn a ticket’s lifecycle on a whiteboard: `open → triage → {reject, review, publish}`.

A LangChain chain is a straight pipe. A graph is a **state machine**: nodes are functions, edges are `if`s, state is the request-scoped dict you already thread through a saga. If two inputs cannot take different paths, you did not need a graph.

---

## 🎯 What you will be able to do

- Define typed state and **return partial dicts** (reducers merge them)
- Use `Annotated[list, operator.add]` so two writes append instead of overwrite
- Compile a graph whose main example **branches**
- Show two CloudWave tickets — a refund ask and a docs question — taking different visible paths
- Know when a chain is enough

!!! think "Think of it like… a vending machine, not a novel."

    Coin in, state = `idle`. Select soda → `vend`. Select refund → `return_coin`. Same machine, two paths. A five-node “always classify then extract then summarize then route then log” is a chain with extra ceremony. The “done when” for this week is **two inputs, two paths**.

## Picture the machine

```
          START
            │
            ▼
        [classify]     returns {labels: [...], decision: ...}
            │
            ▼
         route(state)
        /           \
  refund_queue      docs
  ("I want a       ("how do I
   refund")         rotate a key?")
        \           /
         ▼         ▼
            END
```

A refund ticket never runs the docs node. A docs ticket never enters `refund_queue`. If both always run, you built a list.

## Partial updates + a reducer

Do not mutate `state["labels"].append(...)` and return the same object as your only strategy. Return a **partial** dict. Lists that must accumulate use `operator.add`.

```python
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class Ticket(TypedDict):
    content: str
    labels: Annotated[list[str], operator.add]
    decision: str


def classify(state: Ticket) -> dict:
    text = state["content"].lower()
    if "refund" in text or "cancel" in text:
        return {"labels": ["refund"], "decision": "refund_queue"}
    return {"labels": ["question"], "decision": "docs"}


def refund_queue(state: Ticket) -> dict:
    return {"labels": ["queued-for-human"]}


def docs(state: Ticket) -> dict:
    return {"labels": ["answered-from-docs"]}


def route(state: Ticket) -> Literal["refund_queue", "docs"]:
    return state["decision"]  # type: ignore[return-value]


graph = StateGraph(Ticket)
graph.add_node("classify", classify)
graph.add_node("refund_queue", refund_queue)
graph.add_node("docs", docs)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("refund_queue", END)
graph.add_edge("docs", END)
app = graph.compile()

refund = app.invoke({"content": "I want a refund", "labels": [], "decision": ""})
question = app.invoke({"content": "How do I rotate an API key?", "labels": [], "decision": ""})

assert refund["decision"] == "refund_queue"
assert "queued-for-human" in refund["labels"]
assert question["decision"] == "docs"
assert "answered-from-docs" in question["labels"]
assert "queued-for-human" not in question["labels"]
```

Two inputs, two paths. `labels` is a list reducer: `classify` writes `["refund"]`, `refund_queue` writes `["queued-for-human"]`, the merge is `["refund", "queued-for-human"]`. Last-writer-wins would have dropped the first label.

!!! warning "Watch out — mutating state in place"

    If node A does `state["labels"].append("x"); return state` and node B does the same on a shared list without a reducer, you will get lost updates or phantom appends when you later fan-out (week 2). Return `{"labels": ["x"]}` and let `operator.add` merge.

!!! success "Ship / don’t ship"

    **Ship** a graph when you can point at a conditional edge and a test where a refund and a docs question diverge. **Don’t ship** a linear five-node “document novel” and call it LangGraph. Three sequential LLM calls are a chain (LangChain week 1).

## What this week is not

- Not persistence (week 3), not a human gate (week 4), not idempotency (week 5).
- Not moderation, not spam filtering. Hypothetical CloudWave: inbound support tickets.
- Not `FakeListChatModel`. Keyword `if`s prove the machine. Add a model later at `classify` if you want — import `from langchain_core.language_models import FakeListChatModel` (the old `langchain_community` fakes are being sunset).

## ✍️ Exercise

[Exercises](exercises/week-01.md).

## 🤔 Reflection

1. Which field is last-writer-wins in `Ticket`, and which uses a reducer?
2. Why is `return state` after mutating it a problem the first time you add a parallel node?
3. This graph decides `refund_queue` vs `docs` on keyword match alone. What is the cost of a false positive in each direction?

## 🔗 Next week

Fan-out + reducer, a subgraph as a node, retry on the node that talks to the world.
