# Exercises — Week 3 — Agents & Tools

Do these after reading [Week 3 — Agents & Tools](../week-03.md).

!!! example "1. Two tools, one question"
    Write `get_account_balance` and `get_billing_date` as ordinary functions (dict lookups are fine). Then write the *loop* from the lesson by hand — no agent class — that answers: “What’s my balance and when do you bill me?” for `user_0001`. Print the trace (thought / action / observation).

!!! example "2. A tool you would refuse"
    Add a third tool, `issue_refund(user_id, amount)`. Do **not** hook it to the loop. Write five lines explaining why this tool needs a human confirmation node (LangGraph week 4) instead of an agent.

!!! example "3. max_iterations"
    Force the loop to pick a missing tool name. Confirm it stops at `MAX_STEPS` with a clear error, not an infinite spin.

!!! example "4. Docstring as API"
    Give `get_account_balance` a useless docstring (`"gets stuff"`). Call the same question. Then write the docstring you actually want. What changed in the trace?
