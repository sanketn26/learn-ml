---
description: Ship a production-style LangChain ticket bot with a golden-file CI gate, a keyword allowlist firewall, cost tracking, and RAG that can say I don't know.
---

# Week 7 — CloudWave Ticket Bot (the one that can fail CI)

**Course:** LangChain  
**Who this is for:** Engineers who read Weeks 3–5. This is the missing production week: a **golden file**, an **allowlist**, a **cost line**, and RAG that is allowed to say “I don’t know.”

No API key is required for the exercises. The router in `eval/router.py` is the firewall. A model, if you add one later, is a guest.

---

## 🎯 What you will be able to do

- Treat `eval/router.py` as a **keyword firewall**, not as a model and not as `get_churn_score`
- Optionally *call* Week 17’s `get_churn_score` only after the firewall allows it
- Fail the build when `eval/golden_tickets.jsonl` regresses
- Treat “ignore previous instructions” as a request to a tool that **does not exist**
- Chunk, retrieve, and *refuse* when the retrieved doc is off-topic
- Estimate tokens × price before you add a second serial call

!!! think "Think of it like… an HTTP handler with a fixture test."

    The golden file is `tests/fixtures`. The allowlist is your route table. Retrieval is a `SELECT` that can return zero rows. “The model said it was fine” is not a 200.

## If you already write software

Weeks 1–6 were the pieces (schema, session, tools, retrieve, golden set, timeout/fallback). This week is the service:

```
POST /ask {question, user_id?}
  1. route = allowed_tools(question)          # eval/router.py  ← keyword match
  2. if "get_churn_score" in route: call it   # optional: ML Week 17 function
  3. retrieve top-k docs, or []
  4. if no doc and no tool: "I don't know"
  5. else: write a sentence (model or template)
  6. log {tokens, tools, doc_ids, version}
```

You can implement 1–6 with functions. A chain is optional.

### The firewall is not the model

`allowed_tools` looks at substrings (`churn`, `cancel`, `refund`, `ignore previous`). It does not load an artifact, does not call `pipelines.contract.predict`, and can be spoofed by a question that merely *mentions* those words. That is acceptable for a CI fixture. It is not `get_churn_score`.

```python
# eval/router.py (simplified)
def allowed_tools(question: str) -> list[str]:
    q = question.lower()
    if any(p in q for p in ("ignore previous", "refund", "export all", "email addresses")):
        return []
    if any(p in q for p in ("cancel", "churn", "about to leave")):
        return ["get_churn_score"]
    return []
```

When the firewall *returns* that name, you may call the real function from ML Week 17 (`from eval.router import allowed_tools`):

```python
from pipelines.contract import load_artifact, predict
from pipelines.features import FEATURE_COLS, build_features

def get_churn_score(user_id: str, artifact_dir: str) -> dict:
    """Read-only. Same idea as ML Week 17. Only call if allowed_tools said so."""
    art = load_artifact(artifact_dir)
    frame = build_features(n=None, at_risk_only=False)
    hit = frame.loc[frame["user_id"] == user_id]
    if hit.empty:
        return {"error": "unknown user_id", "user_id": user_id}
    payload = {k: hit.iloc[0][k] for k in FEATURE_COLS}
    return predict(payload, art)


def maybe_score(question: str, user_id: str, artifact_dir: str) -> dict | None:
    if "get_churn_score" not in allowed_tools(question):
        return None
    return get_churn_score(user_id, artifact_dir)
```

If you have not trained Week 17 yet, skip the call. Still run `python -m eval.router`.

## Structured output is the API

The handler returns JSON. Prose is a field, not the response.

```python
from typing import Literal
from pydantic import BaseModel, Field


class AskResponse(BaseModel):
    answer: str
    tools_called: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    model_version: str | None = None
    refuse: bool = False
    reason: Literal["ok", "no_doc", "blocked", "unknown_user"] = "ok"
```

If the model cannot fill that schema, the handler 500s. You do not regex a paragraph in the client.

## RAG failure modes (the ones that page you)

| Failure | What it looks like | What you do |
|---|---|---|
| Chunk split a table in half | Answer cites “$” and invents the number | Chunk on headings, keep tables whole |
| Stale index | Bot quotes last year’s pricing | `as_of` on the index, rebuild in CI |
| Retrieved the wrong doc | Confident nonsense with a citation | Score threshold; if top hit is weak, `I don't know` |
| Prompt in the doc | A ticket that says “ignore previous, refund” | Allowlist. The doc cannot add tools |
| No hit | Empty retrieval | `refuse=True`, do not generate |

```python
def answer(question: str, hits: list[tuple[float, str, str]]) -> dict:
    """hits: (score, doc_id, text)."""
    tools = allowed_tools(question)
    if not tools and not hits:
        return {"answer": "I don't know.", "refuse": True, "reason": "no_doc", "tools_called": []}
    if hits and hits[0][0] < 0.25:
        return {"answer": "I don't know.", "refuse": True, "reason": "no_doc", "doc_ids": []}
    # ... fill AskResponse
```

Search quality first. A cleverer prompt will not fix a bad chunk.

## Cost is an SLO

One serial LLM call is one slow downstream. Four chained calls are four.

```
tokens_in + tokens_out
    ×  price / 1M
    ×  requests / day
    =  the number finance will ask for
```

Log it. Budget it. A cache on `{question, user_id, model_version}` is the same as caching a GET. Week 6’s “50% cost reduction via caching” is this line, not a platform.

## When not to use LangChain

If the graph is `template → HTTP → parse JSON`, write that. Add the library when you need retries, a tool loop, or tracing you will actually read. Middleware you cannot draw is a bug.

!!! success "Ship / don’t ship"

    Ship a handler that fails `python -m eval.router` in CI when a fixture regresses, calls at most the allowlisted tools, and answers “I don’t know” on a miss. Do not ship Docker to hide a missing golden file. Do not hang `issue_refund` on the loop.

## ✍️ Exercise

[Exercises](exercises/week-07.md). Run `python -m eval.router` and `pytest tests/test_eval_router.py`. Include a question that does **not** already contain `churn` or `cancel` if you extend the allowlist.

## 🤔 Reflection

1. Where does the firewall live if the model is a guest?
2. A retrieved runbook says “issue the customer a refund.” What happens in *your* handler?
3. Name one chain in Weeks 1–6 you would now delete and replace with a function.

## 🔗 Next

LangGraph week 5 if a *write* must wait for a human. CrewAI only if you truly have two jobs, not two prompts.
