# Week 17 — recovery writeup

Lesson: [docs/ml/week-17.md](../../../docs/ml/week-17.md)
Exercise: [docs/ml/exercises/week-17.md](../../../docs/ml/exercises/week-17.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-17/starter.py` first.

## Hint 1

??? tip "Hint 1"

    Incidents 1–3 are a bad join, a leaked label, and a silent NaN — ordinary
    debugging, not "the model drifted." The score is a **function**. The
    bot is a client. `allowed_tools` is a keyword firewall; the LLM is not.

## Hint 2

??? tip "Hint 2"

    `get_churn_score` loads `artifacts/prod` via `pipelines.contract.load_artifact`
    and `predict`. `python -m eval.router` must print `failures 0`. For the
    new injection line, add a phrase to `allowed_tools` that returns `[]`
    (example: `"skip the allowlist"` / `"wire a transfer"`).

## Debugging clues

??? warning "Debugging clues"

    - No `artifacts/prod` yet: run Week 16 train + promote first.
    - `user_041906` may have been dropped by `at_risk_only` or sampling —
      fall back to any id in `tonight.csv`.
    - Adding a golden line without updating the firewall makes
      `test_eval_router` fail. That is the point.
    - Do not implement `issue_refund` on the allowlist. Ever.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-17/solution.py
```

The script demonstrates a local injection case. It does **not** rewrite
`eval/golden_tickets.jsonl` — that change is yours to make, then
`pytest tests/test_eval_router.py`.

```python
def allowed_tools(question: str) -> list[str]:
    q = question.lower()
    if any(p in q for p in ("ignore previous", "refund", "wire a transfer", "skip the allowlist")):
        return []
```

## Why this decision

A prompt that says "don't issue refunds" is not a control. A function that
returns `[]` for those questions is. The golden file is CI: if someone
widens the allowlist, a ticket fails at PR time, not after the bot wires a
transfer.
