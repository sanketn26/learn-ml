---
description: Hand-write a thought/action/observation agent loop over two tools, then test its bounded failure at a step cap and docstring quality.
---

# Exercises — Week 3 — Agents & Tools

Do these after reading [Week 3 — Agents & Tools](../week-03.md). The loop in the lesson is the assignment. `create_react_agent` is optional library spelling — you may mention it; you do not have to use it.

## Predict before you run

If the loop picks a missing tool name, do you spin forever or stop at `MAX_STEPS`? Does a useless docstring change the *trace*, or only your feelings?

## Starter / TODO

Write ordinary functions (`get_account_balance`, `get_billing_date`). Then the lesson's thought/action/observation **loop** by hand — no agent class. Leave `issue_refund` unhooked.

## Runnable command

```bash
# no API key — this is a Python loop with dict lookups
python your_week03_loop.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Two tools, one question

Write `get_account_balance` and `get_billing_date` as ordinary functions (dict lookups are fine). Then write the *loop* from the lesson by hand — no agent class — that answers: “What’s my balance and when do you bill me?” for `user_0001`. Print the trace (thought / action / observation).

??? tip "Hint 1 — a nudge"
    With no model, something still has to *decide* the next action each step. What's the smallest stand-in for that decision?

??? tip "Hint 2 — the approach"
    Split the loop from the decider. The loop owns `MAX_STEPS`, the tool registry, and the trace; the decider is a function `(question, trace) -> (thought, action, args)`. A scripted decider — a list it walks through — is enough for this exercise, and a model slots into the same signature later.

??? example "Hint 3 — most of the code"
    ```python
    MAX_STEPS = 5
    BALANCES = {"user_0001": 128.40}


    def get_account_balance(user_id: str) -> str:
        """Current balance in USD for a CloudWave user_id."""
        return f"{BALANCES[user_id]:.2f}" if user_id in BALANCES else f"error: unknown user_id {user_id}"


    def get_billing_date(user_id: str) -> str:
        """Next invoice date (ISO) for a CloudWave user_id."""
        return "2026-09-01"


    TOOLS = {"get_account_balance": get_account_balance, "get_billing_date": get_billing_date}


    def run(question: str, decide) -> tuple[str, list[dict]]:
        trace: list[dict] = []
        for _ in range(MAX_STEPS):
            thought, action, args = decide(question, trace)
            if action == "finish":
                return args["answer"], trace
            tool = TOOLS.get(action)
            observation = tool(**args) if tool else f"error: no tool named {action!r}"
            trace.append({"thought": thought, "action": action, "observation": observation})
        raise RuntimeError(f"agent hit MAX_STEPS={MAX_STEPS} without finishing")


    def scripted(question: str, trace: list[dict]):
        plan = [
            ("need the balance", "get_account_balance", {"user_id": "user_0001"}),
            ("need the bill date", "get_billing_date", {"user_id": "user_0001"}),
        ]
        if len(trace) < len(plan):
            return plan[len(trace)]
        return ("have both", "finish", {"answer": f"Balance {trace[0]['observation']}, billed {trace[1]['observation']}."})


    answer, trace = run("What's my balance and when do you bill me?", scripted)
    for step in trace:
        print(step)
    print(answer)
    ```

## 2. A tool you would refuse

Add a third tool, `issue_refund(user_id, amount)`. Do **not** hook it to the loop. Write five lines explaining why this tool needs a human confirmation node (LangGraph week 4) instead of an agent.

??? tip "Hint 1 — a nudge"
    The two tools above only *read*. What changes when a tool *moves money* and the thing choosing it is a model?

??? tip "Hint 2 — the approach"
    Write the function, keep it out of `TOOLS`, and prove the loop can't reach it. Your five lines should cover: irreversibility, who is accountable, what a prompt injection could do, what "retry" means for a write, and where the approval lives.

??? example "Hint 3 — a skeleton"
    ```text
    1. issue_refund is a write that <reversible? who notices?>.
    2. A model picking tools can be steered by <what>.
    3. A retry of this call could <what goes wrong>.
    4. The approval belongs in <where>, not in the prompt, because <why>.
    5. So: not in TOOLS; reachable only through <the LangGraph week-4 pattern>.
    ```

## 3. The loop cap

Force the loop to pick a missing tool name. Confirm it stops at `MAX_STEPS` with a clear error, not an infinite spin.

??? tip "Hint 1 — a nudge"
    A missing tool is an *observation*, not a crash — a real model gets to see "no such tool" and try again. So what, in your loop, stops a model that keeps trying?

??? tip "Hint 2 — the approach"
    Write a decider that always returns `"get_invoice_pdf"`. Run it inside `try` / `except RuntimeError`, and print how many trace rows were recorded before the stop.

??? example "Hint 3 — most of the code"
    ```python
    def stubborn(question: str, trace: list[dict]):
        return ("the PDF will know", "get_invoice_pdf", {"user_id": "user_0001"})


    try:
        run("Send me my invoice", stubborn)
    except RuntimeError as exc:
        print("stopped:", exc)
    ```

## 4. Docstring as API

Give `get_account_balance` a useless docstring (`"gets stuff"`). Call the same question. Then write the docstring you actually want. What changed in the trace?

??? tip "Hint 1 — a nudge"
    In *your* loop, who reads the docstring?

??? tip "Hint 2 — the approach"
    Your scripted decider never looks at docstrings, so the trace can't change. A real model does: the tool description is the only thing it sees when choosing. Build the "menu" a model would receive from the docstrings, and compare the two menus instead of the two traces.

??? example "Hint 3 — most of the code"
    ```python
    def tool_menu(tools: dict) -> str:
        return "\n".join(f"- {name}: {(fn.__doc__ or '').strip()}" for name, fn in tools.items())


    print(tool_menu(TOOLS))
    get_account_balance.__doc__ = "gets stuff"
    print(tool_menu(TOOLS))
    ```
    The docstring you actually want is yours.

## Expected observation

??? success "Open after you run"
    A printed trace for `user_0001` with a balance and a billing date. A missing tool name stops at `MAX_STEPS` with a clear error.

## Self-check

`issue_refund` is not reachable from the loop. A docstring of `"gets stuff"` is an API bug, not a prompt-engineering flex.
