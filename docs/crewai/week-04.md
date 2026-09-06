---
description: Run a bounded comparison of one worker versus a small CrewAI crew on quality, latency, and call count before adding more agents.
---

# Week 4 — One worker vs a crew

**Course:** CrewAI  
**Who this is for:** Engineers who have been asked to “add more agents” the way people add more microservices.

The track’s “done when” is a **bounded comparison**: one worker versus a small crew, on quality, latency, and call count. Kubernetes, 500 RPS, and Prometheus are not this week.

This venv has **no LangChain**. Compare ordinary functions that stand in for workers. A real `kickoff()` would add provider latency on top of the same shape.

---

## 🎯 What you will be able to do

- Run the same CloudWave changelog job as **one function** and as **three role functions**
- Record quality (required keys present), elapsed time, and call count
- Decide whether the extra roles paid for themselves
- Leave worker pools and Docker out of the required path

!!! think "Think of it like… one senior vs a three-person squad on a one-page note."

    Sometimes the squad catches a missing `risks` field. Sometimes it triples latency and still drops it. Measure; don’t staff by vibe.

## Picture the comparison

```
ONE WORKER                         CREW (simulated)
ticket → write_all                 t1 summarize  (1 call)
      1 call                             │
      1 latency                      t2 draft    (1 call)
                                         │
                                     t3 qa       (1 call)
                                   3 calls, 3 latencies
quality = keys present             quality = keys present
```

## The measurement (concept demo)

```python
import time

REQUIRED_NOTE = ("title", "body", "review_required")
REQUIRED_SUM = ("customer_changes", "operator_changes", "risks")
ROWS = ["export timeout on 150k rows", "raised worker memory"]


def one_worker(rows: list[str]) -> dict:
    # One pass: summary + note. Fast. Easy to drop risks.
    return {
        "title": "Export retry",
        "body": "We retry timed-out exports.",
        "review_required": False,
        # risks omitted on purpose — the bug a second role might catch
        "calls": 1,
    }


def crew(rows: list[str]) -> dict:
    calls = 0
    t0 = time.perf_counter()
    calls += 1
    summary = {
        "customer_changes": ["export retry"],
        "operator_changes": ["worker memory"],
        "risks": ["still fails >200k rows"],
    }
    calls += 1
    note = {
        "title": "Export retry",
        "body": f"{summary['customer_changes']}. Risk: {summary['risks']}",
        "review_required": True,
    }
    calls += 1
    missing = [k for k in REQUIRED_SUM if k not in summary] + [
        k for k in REQUIRED_NOTE if k not in note
    ]
    elapsed = time.perf_counter() - t0
    note.update({"calls": calls, "missing": missing, "elapsed": elapsed})
    return note


def quality(result: dict) -> bool:
    if "risks" in result:
        return True
    return "Risk:" in result.get("body", "") or "risks" in result.get("body", "").lower()


solo = one_worker(ROWS)
squad = crew(ROWS)

print("one_worker", "calls", solo["calls"], "quality", quality(solo), "keys", set(solo))
print("crew", "calls", squad["calls"], "quality", quality(squad), "missing", squad["missing"])

assert solo["calls"] == 1
assert squad["calls"] == 3
assert quality(solo) is False
assert quality(squad) is True
```

Fill a table (the exercise asks for this on a real or simulated crew):

| Measure | One worker | Crew |
|---|---:|---:|
| Model / function calls | 1 | 3 |
| Elapsed (this process) | lower | higher |
| Required keys / risks present | often no | if QA exists, yes |
| Trace easy to read? | one stack | three tickets |

A crew that is slower *and* still drops `risks` loses. Two functions with a validator (week 2) may beat both.

!!! warning "Watch out — more workers ≠ more quality"

    Extra roles add calls and merge bugs. Quality is “did the contract hold,” not “did we have a researcher persona.” p95 at 500 RPS is a different job; this week does not measure it.

!!! success "Ship / don’t ship"

    **Ship** a crew when the comparison shows a quality win you can name (QA caught `risks`) and the extra calls are acceptable. **Don’t ship** a worker pool, a Docker chart, or a Prometheus board as the proof you staffed well. The conclusion “use one function” is allowed.

## What this week is not

- Not Kubernetes, not 500 RPS, not 99.9%.
- Not a required Docker/worker-pool exercise.
- Not LangGraph persistence. If the write needs a human, go there.

## ✍️ Exercise

[Exercises](exercises/week-04.md).

## 🤔 Reflection

1. Which extra call in the crew actually changed `quality`?
2. If latency triples and quality is unchanged, what do you delete?
3. Where does this comparison live in CI so “add more agents” has to beat a baseline?

## 🔗 After

You have the CrewAI introduction: role, ticket, process, a comparison. Official docs for the rest. LangGraph week 5 if a write must not run twice.
