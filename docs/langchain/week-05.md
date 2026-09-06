---
description: Evaluate a LangChain app with a golden test file, scoring tool choice and accuracy separately from latency instead of eyeballing playground output.
---

# Week 5 — Eval is a golden file

**Course:** LangChain  
**Who this is for:** Engineers who already write pytest fixtures and do not ship on “it looked good in the playground.”

You cannot ship what you cannot fail. A LangChain app is a function: input in, dict out. Score the dict.

---

## 🎯 What you will be able to do

- Write a tiny golden set (query, expected tool, expected substring)
- Score **accuracy** and **tool choice** separately from **latency**
- Record a local **trace dict** (no vendor account)
- Force one realistic failure and watch the suite go red
- Know when you are measuring the wrong thing

!!! think "Think of it like… pytest, not a dashboard."

    `expected_tool` is the assertion. A trace is the captured log. Latency is an SLO, not a relevance score. Mixing them is how a slow-but-correct escalate looks “worse” than a fast wrong answer.

## Picture a run

```
golden.jsonl
    │
    ▼
for case in cases:
    t0 = now()
    out, tool, trace = app(case.query)     # your function
    latency_ms = now() - t0
    accuracy  = substring / overlap on output     # quality
    tool_ok   = (tool == case.expected_tool)      # quality
    sla_ok    = latency_ms <= case.max_latency_ms # operations
    pass      = accuracy high AND tool_ok         # do not AND sla into "relevance"
```

No LangSmith required. If you later want a hosted tracer, that is an env var and a vendor — not this week’s objective.

## A local trace dict is enough

```python
from dataclasses import dataclass, field
import time

@dataclass
class Trace:
    query: str
    steps: list[dict] = field(default_factory=list)

    def span(self, name: str, **data):
        self.steps.append({"name": name, **data})


def handle(query: str) -> tuple[str, str, Trace, float]:
    """Stand-in CloudWave handler. Concept demo — no API key."""
    tr = Trace(query=query)
    t0 = time.perf_counter()
    q = query.lower()
    if "billing" in q or "angry" in q:
        tool, output = "escalate_to_human", "Escalate to human support"
    elif "password" in q:
        tool, output = "documentation_search", "Go to Settings > Security > Change Password"
    else:
        tool, output = "documentation_search", "See the docs"
    tr.span("route", tool=tool)
    tr.span("generate", output=output)
    latency_ms = (time.perf_counter() - t0) * 1000
    return output, tool, tr, latency_ms
```

## Score the right thing

```python
GOLDEN = [
    {
        "id": "g1",
        "query": "How do I reset my password?",
        "expected_output": "Settings > Security > Change Password",
        "expected_tool": "documentation_search",
        "max_latency_ms": 2000,
    },
    {
        "id": "g2",
        "query": "I'm extremely angry about billing",
        "expected_output": "Escalate to human support",
        "expected_tool": "escalate_to_human",
        "max_latency_ms": 500,
    },
]


def overlap(expected: str, actual: str) -> float:
    e, a = set(expected.lower().split()), set(actual.lower().split())
    return len(e & a) / len(e) if e else 0.0


def evaluate(cases=GOLDEN):
    rows = []
    for case in cases:
        output, tool, tr, latency_ms = handle(case["query"])
        accuracy = overlap(case["expected_output"], output)
        tool_ok = tool == case["expected_tool"]
        sla_ok = latency_ms <= case["max_latency_ms"]
        quality_pass = tool_ok and accuracy >= 0.5
        rows.append({
            "id": case["id"],
            "quality_pass": quality_pass,
            "accuracy": accuracy,
            "tool_ok": tool_ok,
            "latency_ms": latency_ms,
            "sla_ok": sla_ok,          # reported, not folded into accuracy
            "trace": tr.steps,
        })
    return rows

rows = evaluate()
assert rows[0]["quality_pass"] is True
assert rows[0]["tool_ok"] is True
print(rows[0]["trace"])
```

Latency can fail the **SLA** column while quality still passes. Do not rename `sla_ok` to `relevance`.

## Forced failure

Break the router on purpose. The suite must go red.

```python
def broken_handle(query: str):
    # Always search the docs — even for angry billing. This is the bug.
    return "See the docs", "documentation_search", Trace(query), 1.0

output, tool, _, _ = broken_handle(GOLDEN[1]["query"])
assert tool != GOLDEN[1]["expected_tool"], "suite should fail when escalate is skipped"
```

That red is the point. Fix the router, re-run, watch g2 go green. A dashboard that only charts p95 will not catch this.

Optional 10-line env sketch if you later add a hosted tracer (not required):

```python
# import os
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "..."   # not in this repo
# os.environ["LANGCHAIN_PROJECT"] = "cloudwave-week5"
```

The local `Trace` dict is what the exercise grades.

!!! warning "Watch out — overlap is a blunt instrument"

    Word overlap will pass “Escalate to human support” vs “Escalating to our support team” and fail a correct paraphrase that uses different words. For this week, keep gold short and literal. For week 7, the golden file checks **tools**, not prose.

!!! success "Ship / don’t ship"

    **Ship** a golden set that fails when the wrong tool fires, with latency as a separate SLO. **Don’t ship** “95% quality” that mixes speed into relevance, and don’t block the week on a LangSmith account.

## ✍️ Exercise

[Exercises](exercises/week-05.md).

## 🤔 Reflection

1. A correct escalate takes 3 seconds; SLA is 500ms. Pass or fail? On which column?
2. What is one CloudWave query you would add that the overlap scorer would mishandle?
3. Where does the trace live if the process crashes before you print it?

## 🔗 Next week

Timeouts, fallbacks, a local FastAPI wrapper. A golden file still beats a Dockerfile.
