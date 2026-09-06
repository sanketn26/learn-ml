---
description: Exercises for compiling a LangGraph fan-out graph with a subgraph node and a retry policy for a flaky, real-world-facing node.
---

# Exercises — Week 2 — Fan-out, subgraph, retry

Do these after reading [Week 2](../week-02.md). Compile the examples; do not leave retry as a comment.

## 1. Fan-out + reducer

Compile the lesson’s email + slack + join graph (or the same shape for CloudWave: `notify_user` + `notify_csm`).

**Checks:**

- `set(result["notes"])` contains both branch strings
- If you temporarily drop `operator.add`, you can show that one note disappears (optional)

## 2. Subgraph as a node

Compile a one-node inner graph (`kyc_check`) and add `inner.compile()` as a node of an outer graph.

**Checks:**

- `outer.invoke(...)` returns the inner node’s log line
- The inner graph is passed to `add_node`, not copy-pasted as five functions

## 3. Retry

Use `RetryPolicy` on a node **or** the lesson’s `with_retry` wrapper (label it concept demo). Fail twice, succeed third.

**Checks:**

- Call counter is 3
- A node that only formats a string is **not** retried (it is not wrapped)

## Predict before you run

If you drop `operator.add` on the notes list, does one branch's string disappear? Does a string-formatting node get retried?

## Runnable command

```bash
python your_fanout_graph.py   # compile the graph; fail twice, succeed third
```

## Expected observation

`set(result["notes"])` has both branch strings. Retry call counter is 3. Inner `kyc_check` graph is passed to `add_node`.

## Self-check

Retry is compiled, not a comment. You did not copy-paste five functions instead of a subgraph.
