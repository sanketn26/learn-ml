# Exercises — Week 7 — CloudWave Ticket Bot

Do these after reading [Week 7](../week-07.md).

**1. Golden file.** Run `python -m eval.router`. It must exit 0. Add a sixth line to `eval/golden_tickets.jsonl` for “What’s user_041906’s churn risk?” and make `allowed_tools` pass.

**2. Allowlist, not a prompt.** Implement `issue_refund` as a Python function. Do **not** add it to `allowed_tools`. Show that ticket `t2` still calls nothing.

**3. I don’t know.** Write `answer(question, hits)` from the lesson. Feed it an empty `hits` and a question that is not a churn question. Assert `refuse is True`.

**4. Cost line.** Assume 800 tokens in, 200 out, $0.75 / 1M in, $4.50 / 1M out, 2,000 asks/day. Print dollars/day. If you add a second serial call that doubles tokens, print the new number.

Starter: there is no model to install. The firewall is the point.
