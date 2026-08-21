# Week 2 — Three tickets and a dependency

**Course:** CrewAI  
**Who this is for:** Engineers who have put `needs:` on a CI job.

A task is a ticket: description, expected output, owner, **upstream artifacts**. `context=[upstream]` is the dependency. You do not need 20 tasks or 6 coordinators to learn that.

This venv has **no LangChain**. Construct `Task` / `Crew` and assert `context`. Prove the data flow with ordinary functions if you want a run without `kickoff()`.

---

## 🎯 What you will be able to do

- Write three tasks where 2 and 3 declare `context=[...]`
- Refuse to let the writer reread raw inputs (only the upstream artifact)
- Catch a missing field before the downstream “runs”
- Know when two Python functions beat a crew

!!! think "Think of it like… a ticket with a blocked-by link."

    `context=[research]` is `blocked by: RESEARCH-14`. The writer does not reopen the epic; they read the research ticket’s output contract.

## Picture three tickets

```
[T1 summarize_changes]  →  {customer_changes, operator_changes, risks}
            │
            ▼  context=[T1]
[T2 draft_release_note] →  {title, body, review_required}
            │
            ▼  context=[T1, T2]
[T3 qa_check]           →  {ok: bool, missing: list}
```

Hypothetical CloudWave: a release note for a changelog, not a 25-task product launch.

## Construct three tasks (no kickoff)

```python
from crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="CloudWave researcher",
    goal="Summarize changelog rows into customer / operator / risks",
    backstory="You extract; you do not write marketing.",
    allow_delegation=False,
    verbose=False,
)
writer = Agent(
    role="CloudWave writer",
    goal="Draft a release note from a summary artifact only",
    backstory="You never reopen the raw git log.",
    allow_delegation=False,
    verbose=False,
)
qa = Agent(
    role="CloudWave QA",
    goal="Reject a note that dropped risks",
    backstory="You check keys, not tone.",
    allow_delegation=False,
    verbose=False,
)

t1 = Task(
    description="Summarize these three change records into customer_changes, operator_changes, risks.",
    expected_output="Dict with keys customer_changes, operator_changes, risks",
    agent=researcher,
)
t2 = Task(
    description="Draft title, body, review_required from the summary. Do not reread raw records.",
    expected_output="Dict with keys title, body, review_required",
    agent=writer,
    context=[t1],
)
t3 = Task(
    description="Check that risks from T1 still appear in T2's body.",
    expected_output="Dict with keys ok, missing",
    agent=qa,
    context=[t1, t2],
)

crew = Crew(
    agents=[researcher, writer, qa],
    tasks=[t1, t2, t3],
    process=Process.sequential,
)
assert t2.context == [t1]
assert t3.context == [t1, t2]
assert len(crew.tasks) == 3
```

## Prove the contract with functions (the run)

`kickoff()` needs a provider. The *idea* is a pipeline you can run:

```python
REQUIRED = ("customer_changes", "operator_changes", "risks")


def summarize_changes(rows: list[str]) -> dict:
    return {
        "customer_changes": ["export timeout retry"],
        "operator_changes": ["raised worker memory"],
        "risks": ["partial exports still fail >200k rows"],
    }


def draft_release_note(artifact: dict) -> dict:
    missing = [k for k in REQUIRED if k not in artifact]
    if missing:
        raise ValueError(f"upstream missing {missing}")
    return {
        "title": "CloudWave export retry",
        "body": f"Fix: {artifact['customer_changes']}. Risk: {artifact['risks']}",
        "review_required": True,
    }


summary = summarize_changes(["#123 timeout"])
note = draft_release_note(summary)
assert set(summary) >= set(REQUIRED)
assert "partial exports" in note["body"]

try:
    draft_release_note({"customer_changes": [], "operator_changes": []})  # no risks
    raise AssertionError("should have stopped")
except ValueError as e:
    assert "risks" in str(e)
```

A vague prompt to the writer (“figure out the risks”) is not validation.

!!! warning "Watch out — context is not a shared global"

    If T2 reads a module-level `RAW_CHANGES`, you cheated. The downstream ticket’s only input is the upstream artifact. Circular `context` (A depends on B depends on A) will not save you.

!!! success "Ship / don’t ship"

    **Ship** three tasks with explicit `context` and a validator that blocks a missing key. **Don’t ship** a 20-task / 6-coordinator launch crew as the week’s proof. If two functions catch the contract and the crew cannot run without a vendor key, keep the functions.

## ✍️ Exercise

[Exercises](exercises/week-02.md) — already the right shape: two tasks, break the contract, compare to functions.

## 🤔 Reflection

1. What is T2 *not* allowed to see?
2. Who owns `review_required` — writer or QA?
3. Would you keep the crew for this changelog job? (The exercise allows “no.”)

## 🔗 Next week

Sequential vs hierarchical process. One sketch each. No voting theatre.
