# Week 2 — Workflows: Branches, Fan-out, Subgraphs

**Course:** LangGraph  
**Who this is for:** Engineers who have written a CI pipeline with `if:` jobs, or a saga that fans out to three services and waits.

Week 1 was a straight line. This week is everything that makes a graph worth the ceremony: **if/else**, **do these in parallel**, **reuse a subgraph**, **retry without losing the state**.

---

## If you already write software

LangGraph is a **state machine**. Nodes are functions. Edges are control flow. State is the request-scoped object you thread through.

You have written this as:

- a workflow engine
- a Redux store + reducers
- a CI pipeline with conditional jobs
- an XState chart
- a saga

```
graph = StateGraph(State)
graph.add_node("parse", parse)       # a function: State -> partial State
graph.add_node("act", act)
graph.add_edge("parse", "act")       # always
graph.add_conditional_edges("act", route)   # if / else
```

The payoff versus a pile of `if` statements: you can **checkpoint**, **replay**, and **pause for a human** because the runtime owns the state. That is the point of the next two weeks. If your flow is three sequential LLM calls with no branch, a chain is enough — do not pay for a graph yet.

## 🎯 What you will be able to do

- Route on state (`if urgent → fast path else → slow path`)
- Fan-out work that does not depend on itself, then fan-in
- Extract a subgraph you can reuse like a function
- Retry a node without replaying the whole request from zero
- Know when a branch is a real product decision vs. prompt spaghetti

!!! think "Think of it like… a CI workflow.yml"
    `add_edge` is `needs: [build]`. `add_conditional_edges` is `if: github.event_name == 'push'`. A subgraph is a reusable workflow you `uses:`. The `State` object is the artifact bag that every job can read.

## Conditional routing

Not every ticket takes the same path. That is not “AI.” That is an `if`.

```
                 analyze
                    │
                    ▼
              is_urgent(state)
               /           \
            yes             no
             │               │
             ▼               ▼
        priority_q       standard_q
             \               /
              \             /
               ▼           ▼
                  respond
```

```python
from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph


class Ticket(TypedDict):
    text: str
    urgent: bool
    answer: str


def analyze(state: Ticket) -> Ticket:
    text = state["text"].lower()
    return {"urgent": "down" in text or "refund" in text}


def priority(state: Ticket) -> Ticket:
    return {"answer": "paging on-call"}


def standard(state: Ticket) -> Ticket:
    return {"answer": "queued for next business day"}


def route(state: Ticket) -> Literal["priority", "standard"]:
    return "priority" if state["urgent"] else "standard"


graph = StateGraph(Ticket)
graph.add_node("analyze", analyze)
graph.add_node("priority", priority)
graph.add_node("standard", standard)
graph.add_edge(START, "analyze")
graph.add_conditional_edges("analyze", route)
graph.add_edge("priority", END)
graph.add_edge("standard", END)
app = graph.compile()
```

`route` is a **pure function of state**. Keep it boring. If the routing rule lives in a 40-line prompt, you will not be able to test it.

!!! warning "Watch out — hidden branches in the prompt"
    “Use your judgment to escalate if needed” is not a route. It is a coin flip. Put the rule in `route()`. Log the decision. If product wants a different rule tomorrow, you change a function, not a paragraph.

## Fan-out / fan-in

Independent work should not sit in a queue behind itself.

```
            start
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
   email   dashboard  assign_csm
     └────────┼────────┘
              ▼
            notify
```

In a backend you would `asyncio.gather` three calls. In LangGraph you add three edges from the same node and a join later. The state reducer must **merge** the partial updates (week 1) or the last writer wins and you “lose” the email.

!!! engineer "Engineer mental model"
    Fan-out is cheap only if the branches do not share a mutable side effect. Three nodes that each `UPDATE users SET ...` without a version column is a race, graph or not. Treat branches like microservices: disjoint writes, merge in the join node.

## Subgraphs are functions

A “KYC check” of five nodes will show up in onboarding *and* in a limit-raise flow. That is a subgraph: compile a graph, drop it in as one node of a bigger graph.

```
onboarding
  parse → kyc_subgraph → provision → welcome

limit_raise
  parse → kyc_subgraph → decide_limit
```

Same contract as extracting a function. Same test: the subgraph has a typed state in, a typed state out, and no secret globals.

## Retries

A node that calls billing will 503. You do not restart the whole ticket. You retry **that node**, with the checkpointed state from just before it (week 3 makes this real).

```
analyze  →  charge  →  email
              ▲  │
              └──┘  retry 3×, then fail the graph
```

```python
# sketch — exact API moves; the idea does not
# attach a retry policy to the node that talks to the world
# do not retry the node that already sent the email
```

!!! success "Ship / don’t ship"
    **Ship** a branch when you can write the predicate in five lines and unit-test it. **Ship** fan-out when the branches cannot stomp each other. **Don’t ship** a graph that “decides dynamically how many agents to spawn” as a first version. That is a fork bomb with a system prompt.

## Picture the review comment

When you read a teammate’s graph, ask the same three questions you ask of a workflow.yml:

1. What is the state, and who is allowed to write each field?
2. Which edges are unconditional, which are predicates, and are the predicates tested?
3. If this dies after node 3 of 6, can we resume without double-charging?

If they cannot answer #3, you are not ready for week 3 — you are ready to go add checkpoints.

## 🤔 Reflection

1. Draw the CloudWave “refund request” flow as a graph. Which nodes talk to the world? Which are pure?
2. When is a subgraph better than copy-pasting five nodes? When is it premature?
3. A fan-out of “email + slack + ticket” failed on slack. Do you retry all three, or only slack? Why?

## 🔗 Next week

Persistence: the runtime writes the state after every node so a crash is a resume, not a restart. That is the actual reason to use a graph.
