---
description: Extend the CloudWave ticket bot's golden file and keyword-allowlist firewall, then print a per-ask cost line for the deployed router.
---

# Exercises — Week 7 — CloudWave Ticket Bot

Do these after reading [Week 7](../week-07.md).

**1. Golden file.** Run `python -m eval.router`. It must exit 0. `allowed_tools` is a **keyword firewall**, not `get_churn_score`. Add a sixth line to `eval/golden_tickets.jsonl` whose question does **not** already contain `churn` or `cancel` (example: “Will user_041906 leave us next month?”) and extend the allowlist so that question maps to `get_churn_score`. Optionally, when the firewall allows it, call Week 17’s `get_churn_score` if you have `artifacts/prod`.

**2. Allowlist, not a prompt.** Implement `issue_refund` as a Python function. Do **not** add it to `allowed_tools`. Show that ticket `t2` still calls nothing.

**3. I don’t know.** Write `answer(question, hits)` from the lesson. Feed it an empty `hits` and a question that is not a churn question. Assert `refuse is True`.

**4. Cost line.** Assume 800 tokens in, 200 out, $0.75 / 1M in, $4.50 / 1M out, 2,000 asks/day. Print dollars/day. If you add a second serial call that doubles tokens, print the new number.

Starter: there is no model to install. The firewall is the point.

## Predict before you run

Does `python -m eval.router` exit 0 *before* you add the sixth golden line? After you add “Will user_041906 leave us next month?” without touching `allowed_tools`, what fails?

## Runnable command

```bash
python -m eval.router
pytest tests/test_eval_router.py
```

## Expected observation

`failures 0` once the allowlist maps the new churn phrasing to `get_churn_score`. Ticket `t2` still calls nothing even if `issue_refund` exists as a Python function.

## Self-check

`allowed_tools` is a keyword firewall, not a prompt. Empty `hits` + non-churn question → `refuse is True`. Print dollars/day for the cost line.
