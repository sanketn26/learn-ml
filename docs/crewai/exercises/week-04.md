# Exercises — Week 4 — One worker vs crew

Do these after reading [Week 4](../week-04.md). The “done when” is the comparison. No Docker, no worker pool, no 500 RPS.

## 1. One worker

Implement `one_worker(rows) -> dict` for a CloudWave changelog (title, body, review_required). Count `calls = 1`.

**Checks:**

- `calls == 1`
- Elapsed time recorded (`time.perf_counter`)

## 2. Crew of three role functions

`summarize` → `draft` → `qa` as three functions (or a Crew you do not have to `kickoff()`). Count calls.

**Checks:**

- `calls == 3`
- QA asserts `risks` is present (fail if the draft dropped it)

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
