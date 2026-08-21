# Week 1 — An agent is a job description

**Course:** CrewAI  
**Who this is for:** Engineers who have written a role into a hiring doc and an IAM policy.

CrewAI is **jobs + tickets + a sprint team**. An agent is a worker with a system prompt (role, goal, backstory), tools (IAM), and a task (a ticket). It is ceremony when one function would do.

This environment pins `crewai==0.80.0` with **no LangChain** and **no crewai-tools**. Concept demos construct objects and assert fields. They do not `kickoff()` a ReAct loop against a fake LangChain LLM.

---

## 🎯 What you will be able to do

- Construct `Agent(role, goal, backstory)` and assert the fields
- Write a tool as a **local dict lookup** (no Serper, no network)
- Know when a second agent is staffing theatre
- Leave `Crew.kickoff()` for a run that has a real model — not this demo

!!! think "Think of it like… a hiring packet, not a hive mind."

    Role = job title. Goal = OKR. Backstory = the prompt’s few-shot personality. Tools = IAM. If you cannot say what *output contract* the worker owes, you are not ready to add a second worker.

## Picture one worker

```
Agent
  role:      "CloudWave ticket analyst"
  goal:      "Extract category and whether to escalate"
  backstory: "You read tickets. You do not invent MRR."
  tools:     lookup_plan(user_id) → local dict
                │
                ▼
         (later) Task + Crew
                │
                ▼
         kickoff()  ← needs a real model; not this week’s demo
```

Hypothetical CloudWave: a few ticket analysts. Not a HubSpot/Jasper case study.

## Construct and assert — no kickoff

```python
from crewai import Agent

analyst = Agent(
    role="CloudWave ticket analyst",
    goal="Extract category (bug, billing, question) and escalate: bool",
    backstory="You read support tickets. You do not invent metrics or refunds.",
    verbose=False,
    allow_delegation=False,
)

assert analyst.role == "CloudWave ticket analyst"
assert "escalate" in analyst.goal.lower()
assert "do not invent" in analyst.backstory.lower()
assert analyst.allow_delegation is False
```

If construction tries to bind a default OpenAI client, set a dummy `OPENAI_API_KEY` in the shell for the **constructor** only — still do not `kickoff()`. The fields are what this week proves.

A `Crew` without a run is a roster:

```python
from crewai import Crew, Process, Task

task = Task(
    description="Classify this CloudWave ticket: '{ticket}'",
    expected_output="JSON with keys category, escalate",
    agent=analyst,
)
crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)
assert crew.agents[0].role == analyst.role
assert crew.tasks[0].expected_output.startswith("JSON")
# do not crew.kickoff() in this concept demo
```

## A tool is a local lookup

No `crewai_tools`, no `SerperDevTool`, no LangChain tool wrapper.

```python
PLANS = {"user_0001": "Enterprise", "user_0002": "Free"}


def lookup_plan(user_id: str) -> str:
    """Return the CloudWave plan for a user_id from a local dict."""
    return PLANS.get(user_id, "unknown")


assert lookup_plan("user_0001") == "Enterprise"
assert lookup_plan("nope") == "unknown"
```

Library spelling if your 0.80 install exposes it (optional):

```python
from crewai.tools import tool

@tool("lookup_plan")
def lookup_plan_tool(user_id: str) -> str:
    """Look up a CloudWave plan. Local dict only."""
    return PLANS.get(user_id, "unknown")
```

Attach tools only if you can do it without importing `langchain` or `crewai_tools`. Otherwise keep the function and assert it — that is the IAM idea.

!!! warning "Watch out — kickoff is a ReAct loop"

    `crew.kickoff()` will prompt a model, possibly call tools, possibly loop. A `FakeListLLM` from LangChain is **out of this venv**. Do not `!pip install` in a notebook. Do not add a second agent to “improve quality” before you have an output contract.

!!! success "Ship / don’t ship"

    **Ship** one agent whose role/goal/backstory you can assert and whose tool is a function you wrote. **Don’t ship** a three-agent blog factory that `kickoff()`s against a fake LLM, and don’t cite vendor marketing as evidence.

## What this week is not

- Not multi-agent consensus (week 3).
- Not a 20-task launch plan (week 2 is three tasks).
- Not production (week 4 compares one worker vs a crew on calls and latency).

## ✍️ Exercise

[Exercises](exercises/week-01.md).

## 🤔 Reflection

1. Which CloudWave job actually needs a second *role*, not a second prompt?
2. What does `lookup_plan` return for an unknown user, and why is that better than raising?
3. If `kickoff()` is off-limits without a provider, what did you still prove?

## 🔗 Next week

Tasks with `context=[upstream]`. Three tickets, not twenty.
