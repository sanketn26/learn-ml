---
description: Hands-on exercise building a single CrewAI Agent with a local dict-lookup tool, asserting fields without calling kickoff or any network.
---

# Exercises — Week 1 — One agent, one local tool

Do these after reading [Week 1](../week-01.md). `crewai==0.80.0` only — **no** `langchain`, **no** `crewai_tools`, **no** `SerperDevTool`, **no** `!pip`. Do not `kickoff()`.

```python
from crewai import Agent
```

## Predict before you run

Does constructing an `Agent` call a model? Does `lookup_plan("nope")` raise or return `"unknown"`?

## Runnable command

```bash
python -c "from crewai import Agent; print(Agent.__name__)"
```

If CrewAI is not installed: `make setup-crewai` (or `pip install -r requirements-crewai.txt` in its own venv). Do not `kickoff()`.

Each task has three hints, closed by default. Open only as far as you need.

## 1. Construct and assert

Build one CloudWave `Agent(role, goal, backstory)`.

**Checks:**

- `agent.role`, `agent.goal`, `agent.backstory` are the strings you passed
- `allow_delegation is False`
- You never called `crew.kickoff()`

??? tip "Hint 1 — a nudge"
    An `Agent` is a config object — a job description. What would you expect a constructor like that to do on the network?

??? tip "Hint 2 — the approach"
    Pass `role`, `goal`, `backstory`, `allow_delegation=False`, `verbose=False`, then assert each attribute back. If you want proof there's no model call, time the constructor.

??? example "Hint 3 — most of the code"
    ```python
    import time

    from crewai import Agent, Crew, Process, Task

    t0 = time.perf_counter()
    agent = Agent(
        role="CloudWave plan lookup",
        goal="Answer which plan a user_id is on, from the local catalog only",
        backstory="You read the plan table. You never guess a plan.",
        allow_delegation=False,
        verbose=False,
    )
    print(f"constructed in {time.perf_counter() - t0:.3f}s", agent.role, agent.allow_delegation)
    ```

## 2. Local `@tool` dict lookup

Plan catalog in a dict. Implement `lookup_plan(user_id) -> str`. Prefer `from crewai.tools import tool` if it imports; otherwise a plain function is the same idea.

**Checks:**

- `lookup_plan("user_0001")` is a known plan (`Enterprise` / `Pro` / `Free`)
- Unknown id returns `"unknown"` (string), not an exception
- No HTTP, no Serper, no `crewai_tools` import

??? tip "Hint 1 — a nudge"
    A tool a model calls must never crash the model's turn. What should a lookup return for a key it doesn't have?

??? tip "Hint 2 — the approach"
    Write the plain function first with `dict.get(user_id, "unknown")` and test it. Then wrap it with `@tool("lookup_plan")` — the decorator turns it into a tool object whose `.name` and `.description` come from the decorator and docstring.

??? example "Hint 3 — most of the code"
    ```python
    from crewai.tools import tool

    PLANS = {"user_0001": "Enterprise", "user_0002": "Pro", "user_0003": "Free"}


    def lookup_plan(user_id: str) -> str:
        """Return the CloudWave plan for a user_id from the local catalog."""
        return PLANS.get(user_id, "unknown")


    lookup_plan_tool = tool("lookup_plan")(lookup_plan)
    print(lookup_plan("user_0001"), lookup_plan("nope"), lookup_plan_tool.name)
    ```

## 3. Roster, not a run

Optional: wrap the agent in a `Task` + `Crew(process=Process.sequential)` and assert `len(crew.agents) == 1`.

**Checks:**

- `crew.tasks[0].agent is agent` (or same role)
- Still no `kickoff()`

??? tip "Hint 1 — a nudge"
    A `Crew` is a roster plus a work order. What's the one method that turns it into a run?

??? tip "Hint 2 — the approach"
    `Task(description=..., expected_output=..., agent=agent)`, then `Crew(agents=[agent], tasks=[task], process=Process.sequential)`. Assert on the objects; never call `kickoff`.

??? example "Hint 3 — most of the code"
    ```python
    task = Task(
        description="Which plan is {user_id} on?",
        expected_output="One word: Enterprise, Pro, Free, or unknown",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    print(len(crew.agents), crew.tasks[0].agent.role, crew.process)
    ```

## Expected observation

??? success "Open after you run"
    `agent.role` is the string you passed. `allow_delegation is False`. Unknown user id is the string `"unknown"`.

## Self-check

No `langchain`, no `crewai_tools`, no Serper, no `!pip`. Roster ≠ a run.
