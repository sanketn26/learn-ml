# Exercises — Week 2 — Tasks and Dependencies

Do these after reading [Week 2 — Task Management & Dependencies](../week-02.md).

This is a **concept exercise**. Ordinary Python functions are enough — do not import `langchain` or `crewai_tools`. The point is the task contract and dependency, not prose quality.

## 1. Two tasks, one artifact

Build a tiny CloudWave release-note crew:

1. `summarize_changes` receives three change records and returns exactly these fields: `customer_changes`, `operator_changes`, and `risks`.
2. `draft_release_note` consumes that artifact and returns `title`, `body`, and `review_required`.

Do not let the second task reread the raw change records. Its only input is the first task's declared output.

## 2. Break the contract

Remove `risks` from the first result. Show that validation stops the workflow before the writer runs. A vague prompt asking the writer to “figure it out” does not count as validation.

## 3. Is the crew buying anything?

Implement the same two steps as two ordinary Python functions. Compare the function version and the crew version:

| Measure | Functions | Crew |
|---|---:|---:|
| Model calls | | |
| Elapsed time | | |
| Contract failures caught | | |
| Trace easy to understand? | | |

Write three sentences explaining which version you would keep for this job and why.

## Completion checks

- The happy path produces all six required fields.
- A missing upstream field prevents the downstream task from running.
- The downstream task consumes the upstream artifact rather than hidden global state.
- The comparison reports calls and elapsed time for both implementations.
- The conclusion is allowed to be “use two functions.”

## Predict before you run

If `summarize_changes` drops `risks`, does `draft_release_note` still run? Which version — functions or crew — makes fewer model calls?

## Starter / TODO

Ordinary Python functions are enough. Do not import `langchain` or `crewai_tools`.

## Runnable command

```bash
python your_task_contract.py
```

## Expected observation

Happy path has all six fields. Missing `risks` stops the writer. Comparison table has calls and elapsed time.

## Self-check

The downstream task consumes the upstream artifact, not a hidden global. “Use two functions” is an allowed conclusion.
