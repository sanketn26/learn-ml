---
description: Build a two-task CrewAI release-note pipeline with a declared context dependency, then compare it against plain Python functions.
---

# Exercises — Week 2 — Tasks and Dependencies

Do these after reading [Week 2 — Task Management & Dependencies](../week-02.md).

This is a **concept exercise**. Ordinary Python functions are enough — do not import `langchain` or `crewai_tools`. The point is the task contract and dependency, not prose quality.

## Predict before you run

If `summarize_changes` drops `risks`, does `draft_release_note` still run? Which version — functions or crew — makes fewer model calls?

## Starter / TODO

Ordinary Python functions are enough. Do not import `langchain` or `crewai_tools`.

## Runnable command

```bash
python your_task_contract.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Two tasks, one artifact

Build a tiny CloudWave release-note crew:

1. `summarize_changes` receives three change records and returns exactly these fields: `customer_changes`, `operator_changes`, and `risks`.
2. `draft_release_note` consumes that artifact and returns `title`, `body`, and `review_required`.

Do not let the second task reread the raw change records. Its only input is the first task's declared output.

??? tip "Hint 1 — a nudge"
    A task dependency is a function signature. What's the *only* argument `draft_release_note` should accept?

??? tip "Hint 2 — the approach"
    Two functions with dict contracts, plus the crew version as *declarations*: two `Task`s where the second has `context=[t1]`. The functions are what you run; the crew shows the same dependency in CrewAI's vocabulary.

??? example "Hint 3 — most of the code"
    ```python
    import time

    from crewai import Agent, Crew, Process, Task

    SUMMARY_KEYS = ("customer_changes", "operator_changes", "risks")
    ROWS = ["#123 export timeout retry", "#124 raise worker memory", "#125 exports >200k rows still fail"]


    def summarize_changes(rows: list[str]) -> dict:
        return {
            "customer_changes": ["export timeout retry"],
            "operator_changes": ["raised worker memory"],
            "risks": ["exports over 200k rows still fail"],
        }


    def draft_release_note(artifact: dict) -> dict:
        # validate the artifact here (task 2), then:
        return {
            "title": "Export retry",
            "body": f"Fix: {artifact['customer_changes']}. Risk: {artifact['risks']}",
            "review_required": True,
        }


    writer_agent = Agent(role="writer", goal="draft from the summary only", backstory="never rereads raw rows",
                         allow_delegation=False, verbose=False)
    researcher = Agent(role="researcher", goal="summarize changes", backstory="extracts, no marketing",
                       allow_delegation=False, verbose=False)
    t1 = Task(description="Summarize the change records.", expected_output="customer_changes, operator_changes, risks",
              agent=researcher)
    t2 = Task(description="Draft a release note from the summary.", expected_output="title, body, review_required",
              agent=writer_agent, context=[t1])
    crew = Crew(agents=[researcher, writer_agent], tasks=[t1, t2], process=Process.sequential)

    note = draft_release_note(summarize_changes(ROWS))
    print(note, t2.context == [t1])
    ```

## 2. Break the contract

Remove `risks` from the first result. Show that validation stops the workflow before the writer runs. A vague prompt asking the writer to “figure it out” does not count as validation.

??? tip "Hint 1 — a nudge"
    Where does a contract live — in a sentence the writer reads, or in code that runs before the writer does?

??? tip "Hint 2 — the approach"
    At the top of `draft_release_note`, compute the missing keys against `SUMMARY_KEYS` and `raise ValueError` if any. Count writer runs with a counter placed *after* the check so you can prove it stayed at zero.

??? example "Hint 3 — most of the code"
    ```python
    broken = {k: v for k, v in summarize_changes(ROWS).items() if k != "risks"}
    missing = [k for k in SUMMARY_KEYS if k not in broken]
    print("missing:", missing)
    ```
    Moving that check into `draft_release_note` — and proving the writer never ran — is yours.

## 3. Is the crew buying anything?

Implement the same two steps as two ordinary Python functions. Compare the function version and the crew version:

| Measure | Functions | Crew |
|---|---:|---:|
| Model calls | | |
| Elapsed time | | |
| Contract failures caught | | |
| Trace easy to understand? | | |

Write three sentences explaining which version you would keep for this job and why.

??? tip "Hint 1 — a nudge"
    Count what each version *would* cost when it runs for real: how many model calls does each task make?

??? tip "Hint 2 — the approach"
    Time the function version with `time.perf_counter`. For the crew, you don't need `kickoff` to fill the calls row — count the tasks (one model call each, at minimum). "Contract failures caught" is whatever code you wrote in task 2, in either version.

??? example "Hint 3 — most of the code"
    ```python
    t0 = time.perf_counter()
    draft_release_note(summarize_changes(ROWS))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"functions: model calls=0, elapsed={elapsed_ms:.3f} ms")
    print(f"crew:      model calls≥{len(crew.tasks)}, elapsed=<one LLM round-trip per task>")
    ```
    The three sentences are yours.

## Completion checks

- The happy path produces all six required fields.
- A missing upstream field prevents the downstream task from running.
- The downstream task consumes the upstream artifact rather than hidden global state.
- The comparison reports calls and elapsed time for both implementations.
- The conclusion is allowed to be “use two functions.”

## Expected observation

??? success "Open after you run"
    Happy path has all six fields. Missing `risks` stops the writer. Comparison table has calls and elapsed time.

## Self-check

The downstream task consumes the upstream artifact, not a hidden global. “Use two functions” is an allowed conclusion.
