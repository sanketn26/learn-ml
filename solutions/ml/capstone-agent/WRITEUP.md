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

It prints the eight golden results, the approve / reject / needs-info outcomes, the
crash drill (one credit, two billing calls), and the restart drill (the refund is still
waiting after a restart, and pays once across three processes).

## Why these decisions

- **Triage order is the security model.** Injection first, write intent second, the
  allowlist third, retrieval last. The allowlist grants `get_churn_score` on "about to
  cancel", so a churn-first order hands a tool to g6's injection.
- **The model lives in one node.** `docs` is the only place a model runs; it never chooses a
  route. The stand-in is extractive — it can only repeat the runbook it was given — so the
  golden tickets that check answer text (g1, g8) fail if retrieval hands it the wrong page.
  A canned fake would pass them while telling a customer the wrong thing.
- **One shared word is not a hit.** `retrieve` needs two content words in common. g7
  ("failed payment") shares only "failed" with the password runbook, and must be `idk`.
- **The pause lives on disk.** `open_store` puts checkpoints and the ledger in SQLite. With
  `InMemorySaver`, a deploy forgets every pending approval; the restart drill proves it doesn't.
- **The key is thread + ticket + action.** Every part is identical when a resume reruns
  `issue_credit`. A timestamp or uuid would differ, and billing would pay again.
- **The gate proves it can fail.** `approval_gate=False` exists only so the test can show
  golden ticket g5 catching an agent that writes without a human.
