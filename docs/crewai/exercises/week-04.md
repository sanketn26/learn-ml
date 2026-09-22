---
description: Measure calls, latency, and quality for a single-function worker versus a three-role CrewAI crew on the same changelog job.
---

# Exercises — Week 4 — One worker vs crew

Do these after reading [Week 4](../week-04.md). The “done when” is the comparison. No Docker, no worker pool, no 500 RPS.

## Predict before you run

Will the crew of three beat one worker on *elapsed time*, or only on “risks present”? Is `calls == 3` a win?

## Runnable command

```bash
python your_one_vs_crew.py
```

Time both with `time.perf_counter`. No Docker, no 500 RPS.

Each task has three hints, closed by default. Open only as far as you need.

## 1. One worker

Implement `one_worker(rows) -> dict` for a CloudWave changelog (title, body, review_required). Count `calls = 1`.

**Checks:**

- `calls == 1`
- Elapsed time recorded (`time.perf_counter`)

??? tip "Hint 1 — a nudge"
    One worker does the summary and the note in a single pass. What's the easiest thing for a single pass to drop?

??? tip "Hint 2 — the approach"
    A function that returns the three note fields plus `calls` and `elapsed`. Wrap the body in `perf_counter` start/stop. Keep the draft realistic — a quick one-pass draft often omits the risk line.

??? example "Hint 3 — most of the code"
    ```python
    import time

    ROWS = ["export timeout on 150k rows", "raised worker memory", "exports >200k rows still fail"]


    def one_worker(rows: list[str]) -> dict:
        t0 = time.perf_counter()
        note = {"title": "Export retry", "body": "We retry timed-out exports.", "review_required": False}
        return {**note, "calls": 1, "elapsed": time.perf_counter() - t0}


    print(one_worker(ROWS))
    ```

## 2. Crew of three role functions

`summarize` → `draft` → `qa` as three functions (or a Crew you do not have to `kickoff()`). Count calls.

**Checks:**

- `calls == 3`
- QA asserts `risks` is present (fail if the draft dropped it)

??? tip "Hint 1 — a nudge"
    The QA role only earns its call if it can *fail* something. What exactly does it check?

??? tip "Hint 2 — the approach"
    Three functions, one counter. `qa(summary, note)` checks that every risk from the summary appears in the note's body and returns `{"ok": ..., "missing": [...]}`. Run it once on a good draft and once on a draft that drops the risk.

??? example "Hint 3 — most of the code"
    ```python
    def summarize(rows: list[str]) -> dict:
        return {"customer_changes": ["export retry"], "operator_changes": ["worker memory"],
                "risks": ["exports >200k rows still fail"]}


    def draft(summary: dict) -> dict:
        return {"title": "Export retry", "body": f"{summary['customer_changes']}. Risk: {summary['risks']}",
                "review_required": True}


    def qa(summary: dict, note: dict) -> dict:
        missing = [r for r in summary["risks"] if r not in note["body"]]
        return {"ok": not missing, "missing": missing}


    def crew(rows: list[str]) -> dict:
        t0, calls = time.perf_counter(), 0
        summary = summarize(rows); calls += 1
        note = draft(summary); calls += 1
        check = qa(summary, note); calls += 1
        return {**note, **check, "calls": calls, "elapsed": time.perf_counter() - t0}


    print(crew(ROWS))
    ```

## 3. Table

Fill:

| Measure | One worker | Crew |
|---|---:|---:|
| Calls | | |
| Elapsed | | |
| Risks present? | | |
| Keep this version? | | |

**Checks:**

- Both rows filled
- The conclusion may be “keep the one worker” or “keep two functions from week 2”
- No Kubernetes / Prometheus / Docker requirement

??? tip "Hint 1 — a nudge"
    With real models, elapsed time is dominated by the number of serial calls. What does that predict for your two columns?

??? tip "Hint 2 — the approach"
    Fill the table from your two results. For "Keep this version?", weigh the one thing the crew caught against 3× the calls — and ask whether a plain `assert` in the one-worker version would catch it for free.

??? example "Hint 3 — most of the code"
    ```python
    solo, squad = one_worker(ROWS), crew(ROWS)
    for name, r in (("one worker", solo), ("crew", squad)):
        risks = "Risk:" in r["body"]
        print(f"{name:<11} calls={r['calls']} elapsed={r['elapsed'] * 1000:.3f}ms risks_present={risks}")
    ```
    The "keep" row is yours.

## Expected observation

??? success "Open after you run"
    Table filled: calls, elapsed, risks present, keep-this-version. Conclusion may be “keep the one worker.”

## Self-check

The “done when” is the comparison, not a worker pool. No Kubernetes.
