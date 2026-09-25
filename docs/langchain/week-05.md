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
# os.environ["LANGSMITH_TRACING"] = "true"     # older code: LANGCHAIN_TRACING_V2
# os.environ["LANGSMITH_API_KEY"] = "..."      # not in this repo
# os.environ["LANGSMITH_PROJECT"] = "cloudwave-week5"
```

The local `Trace` dict is what the exercise grades.

## One run is one draw

The stand-in `handle` above is deterministic. A real model is not: it samples, and the same query can route to `escalate_to_human` nine times and `documentation_search` the tenth. So a golden case is not pass/fail — it has a **pass rate**, and five cases run once each is a very small sample. This is ML Week 5 and Week 11 again, with prompts instead of customers.

Concept demo: a simulated model that gets each case right 85% of the time (prompt A) or 80% (prompt B).

```python
import numpy as np
from scipy.stats import binomtest

rng = np.random.default_rng(0)


def run_suite(p_correct: float, cases: int, repeats: int) -> tuple[int, int]:
    """(passes, attempts) for a model that passes any single attempt with p_correct."""
    attempts = cases * repeats
    return int(rng.binomial(attempts, p_correct)), attempts


for label, p in [("prompt A", 0.85), ("prompt B", 0.80)]:
    for repeats in (1, 20):
        passes, n = run_suite(p, cases=5, repeats=repeats)
        ci = binomtest(passes, n).proportion_ci(method="exact")
        print(f"{label}  5 cases × {repeats:>2} runs: {passes}/{n} = {passes / n:.0%}  "
              f"95% CI {ci.low:.0%}–{ci.high:.0%}")
```

With one run per case, each prompt's interval spans most of the range; “A scored 5/5 and B scored 4/5” is noise. Twenty runs per case narrow it — and on this seed the *worse* prompt (B, 80% true) still comes out ahead of A (85% true), inside overlapping intervals. Before you tell anyone a prompt change helped, run each case several times and compare **intervals**, not two single scores. Sampling temperature is part of the test setup: evaluate at the temperature you ship.

## LLM-as-judge is a model — test it too

Word overlap cannot grade “Escalating to our support team.” The common fix is a second model that grades the first (“does this reply escalate billing anger to a human? yes/no”). That judge is a classifier, and it makes mistakes. Before its verdict gates CI, grade 30–50 real replies **by hand** and measure how often the judge agrees with you:

```python
from sklearn.metrics import cohen_kappa_score

human = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
                  1, 1, 0, 0, 1, 1, 0, 1, 0, 1])            # you, reading 30 replies
judge = human.copy()
judge[[2, 5, 11, 17, 24]] = 1 - judge[[2, 5, 11, 17, 24]]  # a judge that disagrees 5 times
agree = (human == judge).mean()
print(f"agreement {agree:.0%}   Cohen's kappa {cohen_kappa_score(human, judge):.2f}")
print("judge said pass, human said fail:", int(((judge == 1) & (human == 0)).sum()))
```

Raw agreement flatters a judge when most replies pass; **kappa** corrects for agreement you would get by chance (1.0 perfect, 0 chance). The row that matters is *judge said pass, human said fail*: those are the failures your CI will wave through. If that count is not small, fix the judge's rubric before you trust its green checks. Re-check it whenever you change the judge model or prompt.

!!! warning "Watch out — overlap is a blunt instrument"

    Word overlap will pass “Escalate to human support” vs “Escalating to our support team” and fail a correct paraphrase that uses different words. For this week, keep gold short and literal. For week 7, the golden file checks **tools**, not prose.

!!! success "Ship / don’t ship"

    **Ship** a golden set that fails when the wrong tool fires, with latency as a separate SLO, several runs per case against a sampled model, and — if a model grades the replies — a measured agreement with human labels. **Don’t ship** “95% quality” that mixes speed into relevance, and don’t block the week on a LangSmith account.

## ✍️ Exercise

[Exercises](exercises/week-05.md).

## 🤔 Reflection

1. A correct escalate takes 3 seconds; SLA is 500ms. Pass or fail? On which column?
2. What is one CloudWave query you would add that the overlap scorer would mishandle?
3. Where does the trace live if the process crashes before you print it?
4. Prompt B passes 19/20 runs and prompt A passes 17/20. Is B better? What would you run next?

## 🔗 Next week

Timeouts, fallbacks, a local FastAPI wrapper. A golden file still beats a Dockerfile.
