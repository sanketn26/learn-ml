# Exercises — Week 1 — Chains

Do these after reading [Week 1](../week-01.md). Concept demo: `FakeListLLM`, no API key.

```python
from langchain_community.llms import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
```

## 1. Five-field triage dict

Define a Pydantic model with **five** fields (`category`, `priority`, `assign_to`, `escalate`, `draft`). Build `prompt | FakeListLLM | JsonOutputParser`. Invoke three CloudWave tickets (bug, billing, question).

**Checks:**

- `isinstance(result, dict)` is True (JsonOutputParser does not return the Pydantic instance).
- Each result has exactly those keys you care about (`category` in the expected set).
- A second scripted JSON blob with `"category": "billing"` parses without raising.

## 2. Retry then fallback

Wrap `chain.invoke` in a loop: max 3 attempts, then return `{"category": "unknown", "escalate": True, "draft": "human"}`. Force the first two calls to raise.

**Checks:**

- After two failures and one success, you return the success dict.
- After three failures, `escalate is True` and no exception escapes.

## 3. Classify, then route

Step 1 classifies `bug` / `feature` / `question`. Step 2 picks a canned reply. Use a Python `if` **or** the lesson’s `RunnableBranch` (they are the same idea).

**Checks:**

- `bug` → troubleshooting string; `question` → docs string.
- `branch.invoke({"category": "bug"})` (or your `if`) is deterministic — no extra model call for the route.

Do not invent a 14-field schema. Do not call a live provider.

## Predict before you run

Will `JsonOutputParser` return a Pydantic instance or a `dict`? After three forced failures, does an exception escape or do you get `escalate is True`?

## Runnable command

Copy the lesson snippet + your TODOs into a local `.py` file. No API key (`FakeListLLM`).

```bash
python -c "from langchain_community.llms import FakeListLLM; print('ok', FakeListLLM(responses=['{}']))"
```

## Expected observation

Each invoke is a dict with your five keys. Two failures then a success returns the success dict. A third failure returns the fallback — no traceback.

## Self-check

Did you call a live provider? If yes, undo it. Concept demos in this track are fake models.
