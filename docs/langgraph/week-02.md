---
description: Build LangGraph fan-out branches that merge with a reducer, embed a compiled subgraph as a node, and retry a flaky external call.
---

# Week 2 — Fan-out, subgraph, retry

`CW-1847` just got triaged into `refund_queue`. Two things need to happen at once: the customer gets an acknowledgment email, and Ana's on-call channel gets a Slack ping. Neither should wait on the other, and neither write should stomp the other's log entry.

??? note "Course details"

    **Course:** LangGraph
    **Who this is for:** Engineers who have written a CI workflow with `if:` jobs, or `asyncio.gather` three calls and a join.

Week 1 branched. This week: **do independent work in parallel**, **reuse a subgraph like a function**, **retry the node that talks to the world**.

---

## 🎯 What you will be able to do

- Compile a fan-out whose reducer **merges** partial list updates
- Drop a compiled subgraph in as one node
- Retry a flaky node (library `RetryPolicy`, or a labeled Python wrapper)
- Keep routing predicates in functions you can unit-test

!!! think "Think of it like… a CI `workflow.yml`."

    `add_edge` is `needs: [build]`. `add_conditional_edges` is `if:`. A subgraph is `uses:`. Fan-out is three jobs with no `needs:` on each other, then a join job.

## Conditional routing (from week 1, kept short)

```
analyze → is_urgent? → priority_q | standard_q → END
```

`route` is a pure function of state. “Use your judgment to escalate” is not a route.

## Fan-out + reducer (compiles)

Independent CloudWave side-work on `CW-1847`: email the customer, post to the on-call channel. Both write `notes`. Without `operator.add`, the last node wins and you “lose” the email.

```python
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class Ticket(TypedDict):
    text: str
    notes: Annotated[list[str], operator.add]


def email(state: Ticket) -> dict:
    return {"notes": ["email: queued"]}


def slack(state: Ticket) -> dict:
    return {"notes": ["slack: on-call"]}


def join(state: Ticket) -> dict:
    return {}


fan = StateGraph(Ticket)
fan.add_node("email", email)
fan.add_node("slack", slack)
fan.add_node("join", join)
fan.add_edge(START, "email")
fan.add_edge(START, "slack")
fan.add_edge("email", "join")
fan.add_edge("slack", "join")
fan.add_edge("join", END)
fan_app = fan.compile()

out = fan_app.invoke({"text": "export timing out, CW-1847", "notes": []})
assert set(out["notes"]) == {"email: queued", "slack: on-call"}
```

!!! engineer "Engineer mental model"

    Fan-out is cheap only if the branches do not share a mutable write. Three nodes that `UPDATE users SET ...` without a version column is a race, graph or not.

## Subgraph as a node

An account-status lookup that shows up when a ticket opens *and* again when a refund escalates is a subgraph: compile it, add it as one node.

```python
class Account(TypedDict):
    user_id: str
    notes: Annotated[list[str], operator.add]


def account_lookup(state: Account) -> dict:
    return {"notes": [f"account:{state['user_id']}"]}


inner = StateGraph(Account)
inner.add_node("account_lookup", account_lookup)
inner.add_edge(START, "account_lookup")
inner.add_edge("account_lookup", END)
lookup_app = inner.compile()

outer = StateGraph(Account)
outer.add_node("lookup", lookup_app)  # compiled graph as a node
outer.add_edge(START, "lookup")
outer.add_edge("lookup", END)
outer_app = outer.compile()

got = outer_app.invoke({"user_id": "user_041906", "notes": []})
assert got["notes"] == ["account:user_041906"]
```

Same contract as extracting a function: typed in, typed out, no secret globals.

## Retry: library spelling + concept demo

A billing GET will 503. Retry **that node**, not the email you already sent.

**Library spelling** (langgraph 0.2):

```python
from langgraph.pregel import RetryPolicy

# graph.add_node("charge", charge, retry=RetryPolicy(max_attempts=3))
```

**Concept demo** (no API key, proves the policy with ordinary Python):

```python
from typing import Callable


def with_retry(fn: Callable[[dict], dict], max_attempts: int = 3) -> Callable[[dict], dict]:
    def wrapped(state: dict) -> dict:
        last: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn({**state, "attempt": attempt})
            except Exception as e:
                last = e
        raise last  # type: ignore[misc]
    return wrapped


calls = {"n": 0}


def flaky(state: dict) -> dict:
    calls["n"] += 1
    if calls["n"] < 3:
        raise RuntimeError("billing 503")
    return {"notes": ["charged"]}


stable = with_retry(flaky, max_attempts=3)
assert stable({"notes": []})["notes"] == ["charged"]
assert calls["n"] == 3
```

Do not retry the node that already sent mail unless that send is keyed (week 5).

!!! warning "Watch out — hidden branches in the prompt"

    “Escalate if needed” is a coin flip. Put the rule in `route()`. Log it.

!!! success "Ship / don’t ship"

    **Ship** a branch whose predicate is five lines and tested; **ship** fan-out when writes cannot stomp each other. **Don’t ship** “spawn N agents dynamically” as v1. That is a fork bomb with a system prompt.

## ✍️ Exercise

[Exercises](exercises/week-02.md).

## 🤔 Reflection

1. Draw CloudWave “refund request” as a graph. Which nodes talk to the world?
2. Fan-out of email + slack failed on slack. Retry all three, or only slack?
3. When is a subgraph premature copy-paste insurance?

## 🔗 Next week

Checkpoints: crash after node 2, resume without repeating node 1.
