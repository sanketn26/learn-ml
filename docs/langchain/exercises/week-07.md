---
description: Extend the CloudWave ticket bot's golden file and keyword-allowlist firewall, then print a per-ask cost line for the deployed router.
---

# Exercises — Week 7 — CloudWave Ticket Bot

Do these after reading [Week 7](../week-07.md). Starter: there is no model to install. The firewall is the point.

## Predict before you run

Does `python -m eval.router` exit 0 *before* you add the sixth golden line? After you add “Will user_041906 leave us next month?” without touching `allowed_tools`, what fails?

## Runnable command

```bash
python -m eval.router
pytest tests/test_eval_router.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Golden file

Run `python -m eval.router`. It must exit 0. `allowed_tools` is a **keyword firewall**, not `get_churn_score`. Add a sixth line to `eval/golden_tickets.jsonl` whose question does **not** already contain `churn` or `cancel` (example: “Will user_041906 leave us next month?”) and extend the allowlist so that question maps to `get_churn_score`. Optionally, when the firewall allows it, call Week 17’s `get_churn_score` if you have `artifacts/prod`.

??? tip "Hint 1 — a nudge"
    Write the test first. Add the golden line, run the suite, and read the failure — it tells you exactly which phrase the firewall doesn't know yet.

??? tip "Hint 2 — the approach"
    `allowed_tools` checks the block list first, then a tuple of churn phrases. Add the narrowest phrase that matches your question ("leave us") to the churn tuple — not to the block list, and not a single word like "leave" that would match "leave a review".

??? example "Hint 3 — most of the code"
    ```python
    from eval.router import allowed_tools

    question = "Will user_041906 leave us next month?"
    print("today:", allowed_tools(question))
    print("near-miss to keep blocked:", allowed_tools("How do I leave a review?"))
    ```
    ```json
    {"id": "t6", "question": "Will user_041906 leave us next month?", "expect_tool": "get_churn_score", "must_not": ["refund", "issue_refund"]}
    ```

## 2. Allowlist, not a prompt

Implement `issue_refund` as a Python function. Do **not** add it to `allowed_tools`. Show that ticket `t2` still calls nothing.

??? tip "Hint 1 — a nudge"
    A function existing in your codebase and a function being *reachable by the bot* are different things. Which one does the firewall control?

??? tip "Hint 2 — the approach"
    Write `issue_refund(user_id, amount)` — it can just return a dict. Then load `t2` from the golden file and assert `allowed_tools(t2["question"]) == []`.

??? example "Hint 3 — most of the code"
    ```python
    import json
    from pathlib import Path

    import eval.router as router


    def issue_refund(user_id: str, amount: float) -> dict:
        return {"user_id": user_id, "refunded": amount}


    cases = [json.loads(line) for line in Path(router.GOLDEN).read_text().splitlines()]
    t2 = next(c for c in cases if c["id"] == "t2")
    print(t2["question"], "→", router.allowed_tools(t2["question"]))
    ```

## 3. I don’t know

Write `answer(question, hits)` from the lesson. Feed it an empty `hits` and a question that is not a churn question. Assert `refuse is True`.

??? tip "Hint 1 — a nudge"
    The bot has two ways to know something: a tool the firewall allows, or a document retrieval found. If it has neither, what's the only honest answer?

??? tip "Hint 2 — the approach"
    Copy the lesson's `answer`: ask `allowed_tools` first; if there are no tools and no hits, return `refuse=True, reason="no_doc"`. Also refuse a top hit below the score floor.

??? example "Hint 3 — most of the code"
    ```python
    def answer(question: str, hits: list[tuple[float, str, str]]) -> dict:
        tools = router.allowed_tools(question)
        if not tools and not hits:
            return {"answer": "I don't know.", "refuse": True, "reason": "no_doc", "tools_called": []}
        if hits and hits[0][0] < 0.25:
            return {"answer": "I don't know.", "refuse": True, "reason": "no_doc", "doc_ids": []}
        return {"answer": "...", "refuse": False, "reason": "ok", "tools_called": tools}


    print(answer("What colour is the CloudWave logo?", hits=[]))
    ```

## 4. Cost line

Assume 800 tokens in, 200 out, $0.75 / 1M in, $4.50 / 1M out, 2,000 asks/day. Print dollars/day. If you add a second serial call that doubles tokens, print the new number.

??? tip "Hint 1 — a nudge"
    Input and output tokens are priced differently. Price one ask first, then multiply.

??? tip "Hint 2 — the approach"
    `per_ask = tokens_in × price_in / 1e6 + tokens_out × price_out / 1e6`. Wrap it in a function of the token counts so the "second call" version is one more line.

??? example "Hint 3 — most of the code"
    ```python
    def dollars_per_day(tokens_in: int, tokens_out: int, asks: int = 2000,
                        price_in: float = 0.75, price_out: float = 4.50) -> float:
        per_ask = tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6
        return per_ask * asks


    print(f"one call:  ${dollars_per_day(800, 200):.2f}/day")
    ```
    The two-call number is one more line.

## Expected observation

??? success "Open after you run"
    `failures 0` once the allowlist maps the new churn phrasing to `get_churn_score`. Ticket `t2` still calls nothing even if `issue_refund` exists as a Python function.

## Self-check

`allowed_tools` is a keyword firewall, not a prompt. Empty `hits` + non-churn question → `refuse is True`. Print dollars/day for the cost line.
