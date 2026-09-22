---
description: Build a six-case golden evaluation set for a LangChain app, scoring accuracy, tool correctness, and SLA independently, with no LangSmith.
---

# Exercises — Week 5 — Golden set

Do these after reading [Week 5](../week-05.md). Local `Trace` dict only. No LangSmith account.

## Predict before you run

Can a row be `quality_pass True` and `sla_ok False` at the same time? If angry billing is patched onto `documentation_search`, which flag flips?

## Runnable command

```bash
python your_golden_suite.py   # local Trace dicts, no LangSmith
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Six cases, two scores

Add four cases to the lesson’s two (password, angry billing). Cover: happy path, escalate, unknown, injection (“ignore previous”). Score **accuracy** and **tool_ok** separately from **sla_ok**.

**Checks:**

- A row can have `quality_pass True` and `sla_ok False` (slow but correct)
- You never assign `relevance_score = 1.0 if latency_ok else 0.5`

??? tip "Hint 1 — a nudge"
    A slow correct answer and a fast wrong one are different bugs owned by different people. How many columns do you need so neither hides the other?

??? tip "Hint 2 — the approach"
    Copy the lesson's `Trace`, `handle`, `overlap`, and `evaluate`. Add four dicts to `GOLDEN`. To show "correct but slow," give one case an impossible `max_latency_ms` (like `0`) — `quality_pass` shouldn't move.

??? example "Hint 3 — most of the code"
    ```python
    import time
    from dataclasses import dataclass, field


    @dataclass
    class Trace:
        query: str
        steps: list[dict] = field(default_factory=list)

        def span(self, name: str, **data):
            self.steps.append({"name": name, **data})


    def handle(query: str):
        tr, t0, q = Trace(query), time.perf_counter(), query.lower()
        if "ignore previous" in q:
            tool, output = "none", "I can't help with that."
        elif "billing" in q or "angry" in q:
            tool, output = "escalate_to_human", "Escalate to human support"
        elif "password" in q:
            tool, output = "documentation_search", "Go to Settings > Security > Change Password"
        else:
            tool, output = "documentation_search", "See the docs"
        tr.span("route", tool=tool)
        tr.span("generate", output=output)
        return output, tool, tr, (time.perf_counter() - t0) * 1000


    def overlap(expected: str, actual: str) -> float:
        e, a = set(expected.lower().split()), set(actual.lower().split())
        return len(e & a) / len(e) if e else 0.0


    GOLDEN = [
        {"id": "g1", "query": "How do I reset my password?", "expected_output": "Settings > Security > Change Password",
         "expected_tool": "documentation_search", "max_latency_ms": 2000},
        {"id": "g2", "query": "I'm extremely angry about billing", "expected_output": "Escalate to human support",
         "expected_tool": "escalate_to_human", "max_latency_ms": 500},
        # g3 happy path, g4 escalate, g5 unknown, g6 injection — and one with max_latency_ms=0
    ]


    def evaluate(cases=GOLDEN, handler=handle):
        rows = []
        for case in cases:
            output, tool, tr, latency_ms = handler(case["query"])
            accuracy = overlap(case["expected_output"], output)
            tool_ok = tool == case["expected_tool"]
            rows.append({"id": case["id"], "quality_pass": tool_ok and accuracy >= 0.5, "accuracy": accuracy,
                         "tool_ok": tool_ok, "sla_ok": latency_ms <= case["max_latency_ms"], "trace": tr.steps})
        return rows


    for row in evaluate():
        print({k: v for k, v in row.items() if k != "trace"})
    ```

## 2. Forced failure

Patch the handler so angry billing uses `documentation_search`. Run the suite.

**Checks:**

- The billing case is `quality_pass False` because `tool_ok` is False
- The trace for that case still has a `route` span naming the wrong tool

??? tip "Hint 1 — a nudge"
    A test suite you've never seen fail is a test suite you don't know works. What's the smallest change that should turn exactly one row red?

??? tip "Hint 2 — the approach"
    Write `broken_handle` that wraps `handle` but swaps the tool for billing queries, and still records a `route` span with the tool it actually used. Pass it into `evaluate(handler=...)`.

??? example "Hint 3 — most of the code"
    ```python
    def broken_handle(query: str):
        output, tool, tr, latency_ms = handle(query)
        if tool == "escalate_to_human":
            tr = Trace(query)
            tool, output = "documentation_search", "See the docs"
            tr.span("route", tool=tool)
        return output, tool, tr, latency_ms


    for row in evaluate(handler=broken_handle):
        print(row["id"], row["quality_pass"], row["tool_ok"], row["trace"])
    ```

## 3. Trace dict

Print `trace.steps` for one pass and one fail.

**Checks:**

- Each step is a dict with `"name"`
- You did not require `LANGCHAIN_API_KEY`

??? tip "Hint 1 — a nudge"
    A trace is just structured logging with one row per step. What would you want to grep for when a case fails at 2 a.m.?

??? tip "Hint 2 — the approach"
    Pick one passing row from `evaluate()` and one failing row from `evaluate(handler=broken_handle)`, print their `trace` lists, and assert every step has a `"name"` key.

??? example "Hint 3 — most of the code"
    ```python
    passing = evaluate()[0]
    failing = next(r for r in evaluate(handler=broken_handle) if not r["quality_pass"])
    for label, row in (("pass", passing), ("fail", failing)):
        print(label, row["trace"])
    ```

## Expected observation

??? success "Open after you run"
    Six cases. Accuracy and `tool_ok` scored separately from `sla_ok`. Forced failure still has a `route` span naming the wrong tool.

## Self-check

You never wrote `relevance_score = 1.0 if latency_ok else 0.5`. You did not create a LangSmith account.
