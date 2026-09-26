---
description: Build a LangChain prompt-to-parser chain with FakeListChatModel, add a retry-then-fallback loop, and route output with RunnableBranch.
---

# Exercises — Week 1 — Chains

Do these after reading [Week 1](../week-01.md). Concept demo: `FakeListChatModel`, no API key.

```python
from langchain_core.language_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
```

## Predict before you run

Will `JsonOutputParser` return a Pydantic instance or a `dict`? After three forced failures, does an exception escape or do you get `escalate is True`?

## Runnable command

Copy the lesson snippet + your TODOs into a local `.py` file. No API key (`FakeListChatModel`).

```bash
python -c "from langchain_core.language_models import FakeListChatModel; print('ok', FakeListChatModel(responses=['{}']))"
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. Five-field triage dict

Define a Pydantic model with **five** fields (`category`, `priority`, `assign_to`, `escalate`, `draft`). Build `prompt | FakeListChatModel | JsonOutputParser`. Invoke three CloudWave tickets (bug, billing, question).

**Checks:**

- `isinstance(result, dict)` is True.
- Each result has exactly those keys you care about (`category` in the expected set).
- A second scripted JSON blob with `"category": "billing"` parses without raising.

??? tip "Hint 1 — a nudge"
    The Pydantic model is a *schema you hand the prompt*, not necessarily the type you get back. Check what `JsonOutputParser` actually returns before you write the asserts.

??? tip "Hint 2 — the approach"
    `JsonOutputParser(pydantic_object=TicketTriage)` gives you `get_format_instructions()` — `.partial()` it into the prompt. Script `FakeListChatModel(responses=[...])` with three JSON strings, one per ticket; it answers them in order.

??? example "Hint 3 — most of the code"
    ```python
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableBranch, RunnableLambda
    from pydantic import BaseModel, Field


    class TicketTriage(BaseModel):
        category: str = Field(description="bug, billing, question")
        priority: int = Field(description="1-5")
        assign_to: str = Field(description="engineering, billing, or support")
        escalate: bool = Field(description="needs a human")
        draft: str = Field(description="one-sentence reply")


    parser = JsonOutputParser(pydantic_object=TicketTriage)
    prompt = ChatPromptTemplate.from_template(
        "Classify this CloudWave ticket as JSON.\nSubject: {subject}\nBody: {body}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    llm = FakeListChatModel(responses=[
        '{"category":"bug","priority":4,"assign_to":"engineering","escalate":true,"draft":"Engineering is on it."}',
        '{"category":"billing","priority":3,"assign_to":"billing","escalate":false,"draft":"Billing will reply."}',
        '{"category":"question","priority":1,"assign_to":"support","escalate":false,"draft":"See the docs."}',
    ])
    triage = prompt | llm | parser

    tickets = [("Export timeout", "ERR_TIMEOUT_500"), ("Double charge", "Invoice twice"), ("API keys", "Where?")]
    results = [triage.invoke({"subject": s, "body": b}) for s, b in tickets]
    print([type(r).__name__ for r in results], [r["category"] for r in results])
    ```

## 2. Retry then fallback

Wrap `chain.invoke` in a loop: max 3 attempts, then return `{"category": "unknown", "escalate": True, "draft": "human"}`. Force the first two calls to raise.

**Checks:**

- After two failures and one success, you return the success dict.
- After three failures, `escalate is True` and no exception escapes.

??? tip "Hint 1 — a nudge"
    You need a chain that fails *on purpose, a set number of times*. Where would you put a counter?

??? tip "Hint 2 — the approach"
    Write `invoke_with_fallback(invoke, payload, attempts=3)` that tries, catches, and returns a *copy* of the fallback dict after the last failure. Test it with a small `flaky(n_failures)` factory that raises its first `n` calls, then delegates to the real chain.

??? example "Hint 3 — most of the code"
    ```python
    FALLBACK = {"category": "unknown", "escalate": True, "draft": "human"}


    def invoke_with_fallback(invoke, payload: dict, attempts: int = 3) -> dict:
        for _ in range(attempts):
            try:
                return invoke(payload)
            except Exception:
                continue
        return dict(FALLBACK)


    def flaky(n_failures: int, then):
        calls = {"n": 0}

        def invoke(payload):
            calls["n"] += 1
            if calls["n"] <= n_failures:
                raise TimeoutError("provider timed out")
            return then(payload)

        return invoke


    ok_llm = FakeListChatModel(responses=['{"category":"bug","priority":2,"assign_to":"engineering","escalate":false,"draft":"On it."}'])
    ok_chain = prompt | ok_llm | parser
    print(invoke_with_fallback(flaky(2, ok_chain.invoke), {"subject": "x", "body": "y"}))
    print(invoke_with_fallback(flaky(3, ok_chain.invoke), {"subject": "x", "body": "y"}))
    ```
    The asserts are yours.

## 3. Classify, then route

Step 1 classifies `bug` / `feature` / `question`. Step 2 picks a canned reply. Use a Python `if` **or** the lesson’s `RunnableBranch` (they are the same idea).

**Checks:**

- `bug` → troubleshooting string; `question` → docs string.
- `branch.invoke({"category": "bug"})` (or your `if`) is deterministic — no extra model call for the route.

??? tip "Hint 1 — a nudge"
    Once the category is a string in a dict, does choosing a reply need a model at all?

??? tip "Hint 2 — the approach"
    `RunnableBranch((condition, runnable), ..., default)`. Each condition is a plain function of the dict; each branch is a `RunnableLambda` returning a canned string. The last argument is the default.

??? example "Hint 3 — most of the code"
    ```python
    branch = RunnableBranch(
        (lambda x: x["category"] == "bug", RunnableLambda(lambda x: "Troubleshooting: send the error code and steps.")),
        (lambda x: x["category"] == "feature", RunnableLambda(lambda x: "Logged for the product team.")),
        RunnableLambda(lambda x: "Docs: see the CloudWave help center."),
    )
    for category in ("bug", "feature", "question"):
        print(category, "→", branch.invoke({"category": category}))
    ```

Do not invent a 14-field schema. Do not call a live provider.

## Expected observation

??? success "Open after you run"
    Each invoke is a dict with your five keys. Two failures then a success returns the success dict. A third failure returns the fallback — no traceback.

## Self-check

Did you call a live provider? If yes, undo it. Concept demos in this track are fake models.
