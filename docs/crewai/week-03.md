---
description: Compare CrewAI's sequential and hierarchical Process modes as staffing choices and see who owns the final ticket in each.
---

# Week 3 — Sequential vs hierarchical

**Course:** CrewAI  
**Who this is for:** Engineers who have chosen “list of jobs” vs “manager assigns work.”

CrewAI’s `Process` flag is the whole week: **sequential** (tickets in order) vs **hierarchical** (a manager LLM assigns). That is a staffing choice. It is not a board meeting and it is not a consensus algorithm.

This venv has **no LangChain**. Sketch both process types. Do not run a voting/CEO theatre that never executes.

---

## 🎯 What you will be able to do

- Point at `Process.sequential` vs `Process.hierarchical` on a tiny CloudWave crew
- Say who owns the final ticket in each mode
- Skip fake voting unless you can actually run it
- Know when a manager LLM is extra latency for one job

!!! think "Think of it like… Makefile vs a tech-lead standup."

    Sequential = the Makefile order. Hierarchical = a lead reads the tickets and pokes people. Consensus-by-vote is a third thing this library does not prove for you in a concept demo.

## Picture the two processes

```
SEQUENTIAL                         HIERARCHICAL
t1 researcher                      manager (LLM)
     │                                │
     ▼                         assigns / reviews
t2 writer                          /     \
     │                         t1 res   t2 writer
     ▼                                │
t3 qa                                 ▼
     │                           manager output
     ▼
   result
```

## Sequential sketch

```python
from crewai import Agent, Crew, Process, Task

researcher = Agent(role="researcher", goal="facts", backstory="extract", verbose=False)
writer = Agent(role="writer", goal="draft", backstory="from facts only", verbose=False)

t1 = Task(description="List three churn reasons from tickets.", expected_output="bullet list", agent=researcher)
t2 = Task(description="Write a 5-line summary.", expected_output="five lines", agent=writer, context=[t1])

seq = Crew(
    agents=[researcher, writer],
    tasks=[t1, t2],
    process=Process.sequential,
)
assert seq.process == Process.sequential
assert t2.context == [t1]
# order is the task list. no manager LLM.
```

## Hierarchical sketch

`Process.hierarchical` expects a manager (in 0.80, a `manager_llm` or a manager agent — see the pin’s constructor). You can **construct** the crew and assert the process without `kickoff()`.

```python
hier = Crew(
    agents=[researcher, writer],
    tasks=[t1, t2],
    process=Process.hierarchical,
    # manager_llm=...  # required to *run*; omit rather than pass a LangChain fake
)
assert hier.process == Process.hierarchical
```

Who owns the final string? The manager’s last message, not `t2`’s, if the manager rewrites. That is extra tokens and a second chance to drop `risks`. Sequential is the default until you can say what the manager decides that `context=` does not.

!!! warning "Watch out — voting theatre"

    “Each specialist votes, CEO aggregates, re-vote on objections” is a play. It does not run in this concept demo, it is not `Process`, and it will not show up in a trace you can test. If you need a human decision, that is LangGraph week 4’s interrupt, not a CrewAI ballot.

!!! success "Ship / don’t ship"

    **Ship** sequential when the dependency graph is already the order. **Ship** hierarchical only when you can name the manager’s job (pick an owner, not rewrite the artifact). **Don’t ship** a six-role design-review panel that never `kickoff()`s.

## ✍️ Exercise

[Exercises](exercises/week-03.md).

## 🤔 Reflection

1. In sequential mode, who is blamed if `risks` go missing — writer or QA?
2. What does the manager add that `context=[t1]` did not already pass?
3. Where would you put a human instead of a CEO agent?

## 🔗 Next week

The actual “done when”: one worker vs a crew on quality, latency, and call count.
