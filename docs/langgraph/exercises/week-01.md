---
description: Hands-on exercises building a branching LangGraph ticket-routing graph with partial state updates and a list reducer for append-only logging.
---

# Exercises — Week 1 — Branching state

Do these after reading [Week 1](../week-01.md). The “done when” is two inputs taking different paths. Return **partial** dicts. Use a list reducer.

```python
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
```

## Predict before you run

Does `"I want a refund"` ever enter the docs node? After a docs ticket, is `log` `["classified", "docs"]` or did you `.append` in place?

## Starter / TODO

```python
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
# TODO: classify → route → docs OR refund_queue
# TODO: log: Annotated[list[str], operator.add]
```

## Runnable command

```bash
python your_ticket_graph.py   # compile + invoke, no API key
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Two paths

Build a CloudWave ticket graph: `classify` → `route` → `docs` **or** `refund_queue`. Keyword `if`s are fine.

**Checks:**

- `"How do I rotate an API key?"` ends with a docs-path label
- `"I want a refund"` ends with a refund-path label and never ran the docs node (count a `RUNS` counter like the week-3 lesson, or a path field)

??? tip "Hint 1 — a nudge"
    A graph is a `switch` statement you can inspect. Which function *returns the case label*, and which nodes are the cases?

??? tip "Hint 2 — the approach"
    `classify` writes a `decision`. `add_conditional_edges("classify", route)` — where `route` just returns `state["decision"]` — picks the next node by name. Bump a module-level `RUNS` counter inside each branch node so you can prove which ones ran.

??? example "Hint 3 — most of the code"
    ```python
    import operator
    from typing import Annotated, Literal, TypedDict

    from langgraph.graph import END, START, StateGraph

    RUNS = {"docs": 0, "refund_queue": 0}


    class Ticket(TypedDict):
        content: str
        decision: str
        log: Annotated[list[str], operator.add]


    def classify(state: Ticket) -> dict:
        decision = "refund_queue" if "refund" in state["content"].lower() else "docs"
        return {"decision": decision, "log": ["classified"]}


    def docs(state: Ticket) -> dict:
        RUNS["docs"] += 1
        return {"log": ["docs"]}


    def refund_queue(state: Ticket) -> dict:
        RUNS["refund_queue"] += 1
        return {"log": ["refund"]}


    def route(state: Ticket) -> Literal["docs", "refund_queue"]:
        return state["decision"]  # type: ignore[return-value]


    g = StateGraph(Ticket)
    for name, fn in (("classify", classify), ("docs", docs), ("refund_queue", refund_queue)):
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_edge("docs", END)
    g.add_edge("refund_queue", END)
    app = g.compile()

    for text in ("How do I rotate an API key?", "I want a refund"):
        print(text, "→", app.invoke({"content": text, "decision": "", "log": []})["log"], RUNS)
    ```

## 2. Reducer

State has `log: Annotated[list[str], operator.add]`. `classify` returns `{"log": ["classified"]}`; the branch node returns `{"log": ["docs"]}` or `{"log": ["refund"]}`.

**Checks:**

- After a docs ticket, `log` is `["classified", "docs"]` (both entries present)
- You did not `state["log"].append(...)` as the only write

??? tip "Hint 1 — a nudge"
    Two nodes both write `log`. Without instructions, the graph keeps the last write. What tells it to *combine* them instead?

??? tip "Hint 2 — the approach"
    `Annotated[list[str], operator.add]` is the instruction: new values are `+`-ed onto the old list. Each node returns a *one-item list*, never the whole list and never an in-place `.append`. To see why it matters, build the same graph with a plain `list[str]` field and compare.

??? example "Hint 3 — most of the code"
    ```python
    class NoReducer(TypedDict):
        content: str
        decision: str
        log: list[str]


    g2 = StateGraph(NoReducer)
    for name, fn in (("classify", classify), ("docs", docs), ("refund_queue", refund_queue)):
        g2.add_node(name, lambda state, fn=fn: fn(state))  # unannotated: LangGraph reads schemas from type hints
    g2.add_edge(START, "classify")
    # Unannotated too: LangGraph 1.x also reads the router's type hints, and `route` names Ticket.
    g2.add_conditional_edges("classify", lambda state: route(state), ["docs", "refund_queue"])
    g2.add_edge("docs", END)
    g2.add_edge("refund_queue", END)
    print("no reducer:", g2.compile().invoke({"content": "rotate key", "decision": "", "log": []})["log"])
    ```

## 3. Partial updates

A node that only knows the decision returns `{"decision": "docs"}` and does not pass the rest of the state through by mutation.

**Checks:**

- `classify(...)` returns a dict whose keys are a subset of the state schema
- Two invokes do not share a list object (no leaked labels from ticket 1 into ticket 2)

??? tip "Hint 1 — a nudge"
    A node returns a *diff*, not the new state. What happens to the fields it doesn't mention?

??? tip "Hint 2 — the approach"
    Call `classify` directly and compare its keys to `Ticket.__annotations__`. Then invoke the app twice and check the second result's `log` has no labels from the first.

??? example "Hint 3 — most of the code"
    ```python
    diff = classify({"content": "I want a refund", "decision": "", "log": []})
    print("keys returned:", set(diff), "schema:", set(Ticket.__annotations__))

    first = app.invoke({"content": "I want a refund", "decision": "", "log": []})
    second = app.invoke({"content": "rotate key", "decision": "", "log": []})
    print(first["log"], second["log"], first["log"] is second["log"])
    ```

## Expected observation

??? success "Open after you run"
    Two inputs, two paths. Refund path never increments a docs counter. Partial updates return a subset of the state keys.

## Self-check

Return **partial** dicts. Use a list reducer. Two invokes do not share a list object.
