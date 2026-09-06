---
description: Learn LangGraph's StateGraph as a branching state machine, using reducers and partial state updates to route two tickets down different paths.
---

# Week 1 — A graph is a state machine that branches

**Course:** LangGraph  
**Who this is for:** Engineers who have drawn a ticket’s lifecycle on a whiteboard: `open → triage → {reject, review, publish}`.

A LangChain chain is a straight pipe. A graph is a **state machine**: nodes are functions, edges are `if`s, state is the request-scoped dict you already thread through a saga. If two inputs cannot take different paths, you did not need a graph.

---

## 🎯 What you will be able to do

- Define typed state and **return partial dicts** (reducers merge them)
- Use `Annotated[list, operator.add]` so two writes append instead of overwrite
- Compile a graph whose main example **branches**
- Show two CloudWave tickets taking different visible paths
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
   reject          approve
   (spam)          (clean)
        \           /
         ▼         ▼
            END
```

Spam ticket never runs `approve`. Clean ticket never runs `reject`. If both always run, you built a list.

## Partial updates + a reducer

Do not mutate `state["labels"].append(...)` and return the same object as your only strategy. Return a **partial** dict. Lists that must accumulate use `operator.add`.

```python
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class Moderation(TypedDict):
    content: str
    labels: Annotated[list[str], operator.add]
    decision: str


def classify(state: Moderation) -> dict:
    text = state["content"].lower()
    if "buy now" in text:
        return {"labels": ["spam"], "decision": "reject"}
    if "hate" in text:
        return {"labels": ["toxic"], "decision": "review"}
    return {"labels": ["clean"], "decision": "approve"}


def reject(state: Moderation) -> dict:
    return {"labels": ["auto-reject"]}


def review(state: Moderation) -> dict:
    return {"labels": ["human-queue"]}


def approve(state: Moderation) -> dict:
    return {"labels": ["publish"]}


def route(state: Moderation) -> Literal["reject", "review", "approve"]:
    return state["decision"]  # type: ignore[return-value]


graph = StateGraph(Moderation)
graph.add_node("classify", classify)
graph.add_node("reject", reject)
graph.add_node("review", review)
graph.add_node("approve", approve)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("reject", END)
graph.add_edge("review", END)
graph.add_edge("approve", END)
app = graph.compile()

spam = app.invoke({"content": "BUY NOW limited offer", "labels": [], "decision": ""})
clean = app.invoke({"content": "CloudWave export is documented here", "labels": [], "decision": ""})

assert spam["decision"] == "reject"
assert "auto-reject" in spam["labels"]
assert clean["decision"] == "approve"
assert "publish" in clean["labels"]
assert "auto-reject" not in clean["labels"]
```

Two inputs, two paths. `labels` is a list reducer: `classify` writes `["spam"]`, `reject` writes `["auto-reject"]`, the merge is `["spam", "auto-reject"]`. Last-writer-wins would have dropped the first label.

!!! warning "Watch out — mutating state in place"

    If node A does `state["labels"].append("x"); return state` and node B does the same on a shared list without a reducer, you will get lost updates or phantom appends when you later fan-out (week 2). Return `{"labels": ["x"]}` and let `operator.add` merge.

!!! success "Ship / don’t ship"

    **Ship** a graph when you can point at a conditional edge and a test where spam and clean diverge. **Don’t ship** a linear five-node “document novel” and call it LangGraph. Three sequential LLM calls are a chain (LangChain week 1).

## What this week is not

- Not persistence (week 3), not a human gate (week 4), not idempotency (week 5).
- Not a moderation vendor case study. Hypothetical CloudWave: inbound tickets, not Reddit.
- Not `FakeListLLM`. Keyword `if`s prove the machine. Add a model later at `classify` if you want — import `from langchain_community.llms import FakeListLLM`, not `langchain.llms.fake`.

## ✍️ Exercise

[Exercises](exercises/week-01.md).

## 🤔 Reflection

1. Which field is last-writer-wins in `Moderation`, and which uses a reducer?
2. Why is `return state` after mutating it a problem the first time you add a parallel node?
3. Draw CloudWave “refund vs docs question” as two paths. Where is the `if`?

## 🔗 Next week

Fan-out + reducer, a subgraph as a node, retry on the node that talks to the world.
