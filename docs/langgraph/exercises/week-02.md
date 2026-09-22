---
description: Exercises for compiling a LangGraph fan-out graph with a subgraph node and a retry policy for a flaky, real-world-facing node.
---

# Exercises — Week 2 — Fan-out, subgraph, retry

Do these after reading [Week 2](../week-02.md). Compile the examples; do not leave retry as a comment.

## Predict before you run

If you drop `operator.add` on the notes list, does one branch's string disappear? Does a string-formatting node get retried?

## Runnable command

```bash
python your_fanout_graph.py   # compile the graph; fail twice, succeed third
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Fan-out + reducer

Compile the lesson’s email + slack + join graph (or the same shape for CloudWave: `notify_user` + `notify_csm`).

**Checks:**

- `set(result["notes"])` contains both branch strings
- If you temporarily drop `operator.add`, you can show that one note disappears (optional)

??? tip "Hint 1 — a nudge"
    Two edges out of `START` means two nodes run in the same step. Both write `notes`. What decides how the two writes combine?

??? tip "Hint 2 — the approach"
    Two `add_edge(START, ...)` calls fan out; both branches edge into one `join` node. With `operator.add` on `notes` the writes concatenate. For the optional check, build the same graph with a plain `list[str]` and catch what happens.

??? example "Hint 3 — most of the code"
    ```python
    import operator
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.pregel import RetryPolicy


    class Notify(TypedDict):
        text: str
        notes: Annotated[list[str], operator.add]


    def notify_user(state: Notify) -> dict:
        return {"notes": ["user: emailed"]}


    def notify_csm(state: Notify) -> dict:
        return {"notes": ["csm: paged"]}


    def fan_out(schema) -> StateGraph:
        g = StateGraph(schema)
        g.add_node("notify_user", notify_user)
        g.add_node("notify_csm", notify_csm)
        g.add_node("join", lambda state: {})
        g.add_edge(START, "notify_user")
        g.add_edge(START, "notify_csm")
        g.add_edge("notify_user", "join")
        g.add_edge("notify_csm", "join")
        g.add_edge("join", END)
        return g


    print(fan_out(Notify).compile().invoke({"text": "CW-1847", "notes": []})["notes"])
    ```
    The no-reducer experiment is yours — wrap the invoke in `try` / `except`.

## 2. Subgraph as a node

Compile a one-node inner graph (`kyc_check`) and add `inner.compile()` as a node of an outer graph.

**Checks:**

- `outer.invoke(...)` returns the inner node’s log line
- The inner graph is passed to `add_node`, not copy-pasted as five functions

??? tip "Hint 1 — a nudge"
    A compiled graph is a callable that takes state and returns state. What else in LangGraph has that shape?

??? tip "Hint 2 — the approach"
    Build and compile the inner graph on its own, then `outer.add_node("kyc", inner_app)`. Share the state keys (at least the `notes` reducer) between inner and outer so the inner write shows up in the outer result.

??? example "Hint 3 — most of the code"
    ```python
    def kyc_check(state: Notify) -> dict:
        return {"notes": ["kyc: passed"]}


    inner = StateGraph(Notify)
    inner.add_node("kyc_check", kyc_check)
    inner.add_edge(START, "kyc_check")
    inner.add_edge("kyc_check", END)
    inner_app = inner.compile()

    outer = StateGraph(Notify)
    outer.add_node("kyc", inner_app)
    outer.add_edge(START, "kyc")
    outer.add_edge("kyc", END)
    print(outer.compile().invoke({"text": "user_041906", "notes": []})["notes"])
    ```

## 3. Retry

Use `RetryPolicy` on a node **or** the lesson’s `with_retry` wrapper (label it concept demo). Fail twice, succeed third.

**Checks:**

- Call counter is 3
- A node that only formats a string is **not** retried (it is not wrapped)

??? tip "Hint 1 — a nudge"
    Retry is a per-node decision. Which nodes talk to something that can flake, and which are pure functions that would fail the same way every time?

??? tip "Hint 2 — the approach"
    `add_node("charge", flaky, retry=RetryPolicy(max_attempts=3, initial_interval=0.01))`. Careful: the default `retry_on` only retries *transient-looking* errors — `ConnectionError` yes, `RuntimeError` / `ValueError` no. Raise the kind of error a flaky network actually raises, or pass your own `retry_on`. Leave the formatting node without a policy.

??? example "Hint 3 — most of the code"
    ```python
    CALLS = {"charge": 0, "format": 0}


    def flaky_charge(state: Notify) -> dict:
        CALLS["charge"] += 1
        if CALLS["charge"] < 3:
            raise ConnectionError("billing 503")
        return {"notes": ["charged"]}


    def format_receipt(state: Notify) -> dict:
        CALLS["format"] += 1
        return {"notes": [f"receipt for {state['text']}"]}


    g = StateGraph(Notify)
    g.add_node("charge", flaky_charge, retry=RetryPolicy(max_attempts=3, initial_interval=0.01))
    g.add_node("format", format_receipt)
    g.add_edge(START, "charge")
    g.add_edge("charge", "format")
    g.add_edge("format", END)
    print(g.compile().invoke({"text": "inv_9", "notes": []})["notes"], CALLS)
    ```

## Expected observation

??? success "Open after you run"
    `set(result["notes"])` has both branch strings. Retry call counter is 3. Inner `kyc_check` graph is passed to `add_node`.

## Self-check

Retry is compiled, not a comment. You did not copy-paste five functions instead of a subgraph.
