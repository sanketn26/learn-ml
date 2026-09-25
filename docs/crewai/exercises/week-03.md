---
description: Sketch sequential and hierarchical CrewAI Process crews side by side and explain who owns the final output in each mode.
---

# Exercises — Week 3 — Sequential vs hierarchical

Do these after reading [Week 3](../week-03.md). Sketch both process types. Do not implement a voting/CEO consensus loop that never runs.

```python
from crewai import Agent, Crew, Process, Task
```

## Predict before you run

Does building a `Crew(..., process=Process.sequential)` call an LLM? Who owns the final string in sequential vs hierarchical?

## Runnable command

```bash
python your_crew_process.py   # construct only — no kickoff()
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Sequential crew

Two agents, two tasks, `context=[t1]` on the second, `process=Process.sequential`.

**Checks:**

- `crew.process == Process.sequential`
- `t2.context == [t1]`
- No `kickoff()` required

??? tip "Hint 1 — a nudge"
    In sequential mode, what decides the order the work happens in?

??? tip "Hint 2 — the approach"
    Two `Agent`s, two `Task`s (the second with `context=[t1]`), and `Crew(..., process=Process.sequential)`. The task list *is* the order; there's no manager.

??? example "Hint 3 — most of the code"
    ```python
    from crewai import Agent, Crew, Process, Task

    researcher = Agent(role="researcher", goal="list churn reasons", backstory="extracts facts", verbose=False)
    writer = Agent(role="writer", goal="five-line summary", backstory="writes from facts only", verbose=False)
    t1 = Task(description="List three churn reasons from tickets.", expected_output="bullet list", agent=researcher)
    t2 = Task(description="Write a 5-line summary.", expected_output="five lines", agent=writer, context=[t1])

    seq = Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential)
    print(seq.process, t2.context == [t1])
    ```

## 2. Hierarchical flag

Build a second `Crew` with `process=Process.hierarchical` and a `manager_agent` whose goal names the manager's one job. CrewAI 1.x refuses to construct a hierarchical crew without a manager.

**Checks:**

- `hier.process == Process.hierarchical`
- In two sentences: who owns the final string in sequential vs hierarchical

??? tip "Hint 1 — a nudge"
    Hierarchical adds a role that isn't in your `agents` list. Who is it, and what does it need that the others don't?

??? tip "Hint 2 — the approach"
    Same agents and tasks, `process=Process.hierarchical`, plus `manager_agent=Agent(..., allow_delegation=True)`. Keep the manager out of `agents=`. Try it once *without* a manager first and read the `ValidationError` — it fails at construction, before any tokens are spent. Your two sentences contrast "the last task's agent" with "the manager."

??? example "Hint 3 — most of the code"
    ```python
    try:
        Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.hierarchical)
    except Exception as exc:  # CrewAI 1.x validates the manager at construction
        print("no manager:", type(exc).__name__)

    manager = Agent(role="release manager", goal="Assign each ticket; do not rewrite the output.",
                    backstory="Owns the order of work, not the words.", allow_delegation=True, verbose=False)
    hier = Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.hierarchical, manager_agent=manager)
    assert hier.process == Process.hierarchical
    ```

## 3. No ballot

Do **not** add vote/re-vote/CEO-aggregate code.

**Checks:**

- The file has no `votes =` / `consensus` loop
- One sentence: a human gate is LangGraph week 4, not a CrewAI poll

??? tip "Hint 1 — a nudge"
    If two agents disagree, who should decide — a third agent, or a person with a pager?

??? tip "Hint 2 — the approach"
    Grep your file for `vote` and `consensus`. Then write the sentence pointing at `interrupt_before` from LangGraph week 4.

??? example "Hint 3 — a skeleton"
    ```text
    When the crew's output needs a decision, <who> approves it via <LangGraph mechanism>, because <why a poll of models isn't accountability>.
    ```

## Expected observation

??? success "Open after you run"
    `crew.process == Process.sequential` and `t2.context == [t1]`. Hierarchical flag set without a fake manager LLM.

## Self-check

No vote/re-vote/CEO loop. A human gate is LangGraph week 4, not a CrewAI poll.
