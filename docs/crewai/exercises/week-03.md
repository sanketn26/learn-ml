# Exercises — Week 3 — Sequential vs hierarchical

Do these after reading [Week 3](../week-03.md). Sketch both process types. Do not implement a voting/CEO consensus loop that never runs.

```python
from crewai import Agent, Crew, Process, Task
```

## 1. Sequential crew

Two agents, two tasks, `context=[t1]` on the second, `process=Process.sequential`.

**Checks:**

- `crew.process == Process.sequential`
- `t2.context == [t1]`
- No `kickoff()` required

## 2. Hierarchical flag

Build a second `Crew` with `process=Process.hierarchical` (manager LLM omitted unless you have a real provider — do not pass a LangChain fake).

**Checks:**

- `hier.process == Process.hierarchical`
- In two sentences: who owns the final string in sequential vs hierarchical

## 3. No ballot

Do **not** add vote/re-vote/CEO-aggregate code.

**Checks:**

- The file has no `votes =` / `consensus` loop
- One sentence: a human gate is LangGraph week 4, not a CrewAI poll

## Predict before you run

Does building a `Crew(..., process=Process.sequential)` call an LLM? Who owns the final string in sequential vs hierarchical?

## Runnable command

```bash
python your_crew_process.py   # construct only — no kickoff()
```

## Expected observation

`crew.process == Process.sequential` and `t2.context == [t1]`. Hierarchical flag set without a fake manager LLM.

## Self-check

No vote/re-vote/CEO loop. A human gate is LangGraph week 4, not a CrewAI poll.
