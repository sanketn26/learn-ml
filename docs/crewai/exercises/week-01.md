# Exercises — Week 1 — One agent, one local tool

Do these after reading [Week 1](../week-01.md). `crewai==0.80.0` only — **no** `langchain`, **no** `crewai_tools`, **no** `SerperDevTool`, **no** `!pip`. Do not `kickoff()`.

```python
from crewai import Agent
```

## 1. Construct and assert

Build one CloudWave `Agent(role, goal, backstory)`.

**Checks:**

- `agent.role`, `agent.goal`, `agent.backstory` are the strings you passed
- `allow_delegation is False`
- You never called `crew.kickoff()`

## 2. Local `@tool` dict lookup

Plan catalog in a dict. Implement `lookup_plan(user_id) -> str`. Prefer `from crewai.tools import tool` if it imports; otherwise a plain function is the same idea.

**Checks:**

- `lookup_plan("user_0001")` is a known plan (`Enterprise` / `Pro` / `Free`)
- Unknown id returns `"unknown"` (string), not an exception
- No HTTP, no Serper, no `crewai_tools` import

## 3. Roster, not a run

Optional: wrap the agent in a `Task` + `Crew(process=Process.sequential)` and assert `len(crew.agents) == 1`.

**Checks:**

- `crew.tasks[0].agent is agent` (or same role)
- Still no `kickoff()`

## Predict before you run

Does constructing an `Agent` call a model? Does `lookup_plan("nope")` raise or return `"unknown"`?

## Runnable command

```bash
python -c "from crewai import Agent; print(Agent.__name__)"
```

If CrewAI is not installed: `pip install -r requirements-crewai.txt` in `.venv-crewai`. Do not `kickoff()`.

## Expected observation

`agent.role` is the string you passed. `allow_delegation is False`. Unknown user id is the string `"unknown"`.

## Self-check

No `langchain`, no `crewai_tools`, no Serper, no `!pip`. Roster ≠ a run.
