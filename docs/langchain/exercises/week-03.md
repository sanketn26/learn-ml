# Exercises — Week 3 — Agents & Tools

Do these after reading [Week 3 — Agents & Tools](../week-03.md). The loop in the lesson is the assignment. `create_react_agent` is optional library spelling — you may mention it; you do not have to use it.

!!! example "1. Two tools, one question"
    Write `get_account_balance` and `get_billing_date` as ordinary functions (dict lookups are fine). Then write the *loop* from the lesson by hand — no agent class — that answers: “What’s my balance and when do you bill me?” for `user_0001`. Print the trace (thought / action / observation).

!!! example "2. A tool you would refuse"
    Add a third tool, `issue_refund(user_id, amount)`. Do **not** hook it to the loop. Write five lines explaining why this tool needs a human confirmation node (LangGraph week 4) instead of an agent.

!!! example "3. max_iterations"
    Force the loop to pick a missing tool name. Confirm it stops at `MAX_STEPS` with a clear error, not an infinite spin.

!!! example "4. Docstring as API"
    Give `get_account_balance` a useless docstring (`"gets stuff"`). Call the same question. Then write the docstring you actually want. What changed in the trace?

## Predict before you run

If the loop picks a missing tool name, do you spin forever or stop at `MAX_STEPS`? Does a useless docstring change the *trace*, or only your feelings?

## Starter / TODO

Write ordinary functions (`get_account_balance`, `get_billing_date`). Then the lesson's thought/action/observation **loop** by hand — no agent class. Leave `issue_refund` unhooked.

## Runnable command

```bash
# no API key — this is a Python loop with dict lookups
python your_week03_loop.py
```

## Expected observation

A printed trace for `user_0001` with a balance and a billing date. A missing tool name stops at `MAX_STEPS` with a clear error.

## Self-check

`issue_refund` is not reachable from the loop. A docstring of `"gets stuff"` is an API bug, not a prompt-engineering flex.
