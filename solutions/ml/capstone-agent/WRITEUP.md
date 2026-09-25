# Capstone — The Support Agent That Survives Itself — recovery writeup

Lesson: [docs/ml/capstone-agent.md](../../../docs/ml/capstone-agent.md)
Exercise: [docs/ml/exercises/capstone-agent.md](../../../docs/ml/exercises/capstone-agent.md)

!!! warning "Do not open `solution.py` until you are stuck after the hints on the exercise page"

    Work in `exercises/ml/capstone-agent/starter.py` first.

## Reference solution

See [`solution.py`](solution.py). Framework venv only:

```bash
.venv-framework/bin/python solutions/ml/capstone-agent/solution.py
.venv-framework/bin/python -m pytest tests/test_capstone_agent.py
```

It prints the six golden results, the approve / reject / needs-info outcomes, and the
crash drill (one credit, two billing calls).

## Why these decisions

- **Triage order is the security model.** Injection first, write intent second, the
  allowlist third, retrieval last. The allowlist grants `get_churn_score` on "about to
  cancel", so a churn-first order hands a tool to g6's injection.
- **The model lives in one node.** `docs` is the only place `FakeListChatModel` runs; it never
  chooses a route. Swapping in a real provider changes phrasing, not behaviour.
- **The key is thread + ticket + action.** Every part is identical when a resume reruns
  `issue_credit`. A timestamp or uuid would differ, and billing would pay again.
- **The gate proves it can fail.** `approval_gate=False` exists only so the test can show
  golden ticket g5 catching an agent that writes without a human.
