---
description: Learn LangChain chains as prompt templates piped through output parsers, and compose prompt, model, and parser into one call graph.
---

# Week 1 — Chains: templates, parsers, pipes

A CloudWave ticket lands: subject “Export timeout,” body `ERR_TIMEOUT_500 on 150k rows`. Before you build anything clever, you need the boring part — turn that text into a category, a priority, and someone to route it to.

??? note "Course details"

    **Course:** LangChain
    **Who this is for:** Engineers who have written an HTTP handler that validates JSON and calls a slow downstream.

LangChain is not a model. The model is the remote API. LangChain is **middleware**: a prompt is a template, a parser is a schema, a chain is your call graph.

---

## 🎯 What you will be able to do

- Write a reusable prompt template and inject variables
- Compose `prompt | llm | parser` and say what each step returns
- Parse model text into a **dict** (or a Pydantic object) and reject garbage
- Know the 1.x shipping path: `with_structured_output(Model)` returns a validated Pydantic instance
- Route a second step with a Python `if` (or an 8-line `RunnableBranch`)
- Know when a chain is the wrong tool

!!! think "Think of it like… FastAPI middleware, not a coworker."

    A request hits a template, then a client, then a schema. If you cannot redraw the chain as three function calls, the pipe is hiding a bug.

## If you already write software

```
Your backend                        LangChain
─────────────────────────────       ──────────────────────────────
HTTP handler                        a chain entrypoint
string template + params            PromptTemplate / ChatPromptTemplate
JSON schema / pydantic              output parser
service client                      an LLM (here: FakeListChatModel)
try / catch + retries               fallbacks you write yourself
```

Concept demos in this track use `FakeListChatModel`. No API key.

## Picture the pipe

```
ticket dict
    │
    ▼
[ChatPromptTemplate]   fill {subject} and {body}
    │
    ▼
[FakeListChatModel]          returns a JSON string (scripted)
    │
    ▼
[JsonOutputParser]     returns a dict  ← not a Pydantic instance
    │
    ▼
{"category": "bug", "priority": 4, "assign_to": "engineering",
 "escalate": true, "draft": "..."}
```

`a | b | c` is **sequential composition**. It is not automatic parallelization. Async (`ainvoke`) is available; it does not magically fan the pipe out.

## The old SDK, labeled

Calling a provider by hand is fine. This spelling is the **old** OpenAI Python SDK (`openai.ChatCompletion.create`). Current clients use `client.chat.completions.create`. Either way you still own retries, schema, and tests.

```python
# Old SDK (do not copy into a new service):
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "My dashboard is slow"}],
# )
# answer = response["choices"][0]["message"]["content"]  # untyped string
```

## The same ticket, as a chain

```python
from langchain_core.language_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

prompt = ChatPromptTemplate.from_template(
    """You are CloudWave support.
Subject: {subject}
Body: {body}
Reply in one sentence."""
)
llm = FakeListChatModel(responses=[
    "Clear the browser cache, then reload the dashboard."
])
chain = prompt | llm | StrOutputParser()

text = chain.invoke({
    "subject": "Dashboard slow",
    "body": "Enterprise tenant, Chrome, 8s load.",
})
assert "cache" in text.lower()
```

## JsonOutputParser returns a dict

`JsonOutputParser(pydantic_object=Model)` uses the model as **format instructions**. `.invoke` still returns a **`dict`**. If you need a Pydantic instance, use `PydanticOutputParser`.

Keep the schema small. Five fields is enough for a triage contract.

```python
class TicketTriage(BaseModel):
    category: str = Field(description="bug, billing, question, or urgent")
    priority: int = Field(description="1-5")
    assign_to: str = Field(description="engineering, billing, or support")
    escalate: bool = Field(description="needs a human")
    draft: str = Field(description="one-sentence reply")

parser = JsonOutputParser(pydantic_object=TicketTriage)

triage_prompt = ChatPromptTemplate.from_template(
    """Classify this CloudWave ticket as JSON.
Subject: {subject}
Body: {body}
{format_instructions}"""
).partial(format_instructions=parser.get_format_instructions())

triage_llm = FakeListChatModel(responses=[
    '{"category":"bug","priority":4,"assign_to":"engineering",'
    '"escalate":true,"draft":"We see the export timeout; engineering is on it."}'
])
triage = triage_prompt | triage_llm | parser

result = triage.invoke({
    "subject": "Export timeout",
    "body": "ERR_TIMEOUT_500 on 150k rows.",
})
assert isinstance(result, dict)
assert result["category"] == "bug"
assert result["escalate"] is True
```

### The shipping spelling: `with_structured_output`

A parser reads the model's *text* and hopes it is JSON. Real chat models can do better: the provider is told the schema and returns arguments that match it, and LangChain hands you a validated **Pydantic instance**, not a dict. That is how you ship triage in LangChain 1.x. It needs a real provider, so this block is an integration demo, not run here:

```python
# Integration demo — needs a provider key (and `pip install langchain-anthropic` or similar).
# from langchain.chat_models import init_chat_model
# model = init_chat_model("anthropic:claude-sonnet-5")      # any tool-calling chat model
# triage_model = model.with_structured_output(TicketTriage)
# result = triage_model.invoke("Subject: Export timeout\nBody: ERR_TIMEOUT_500 on 150k rows.")
# isinstance(result, TicketTriage)   # True — validated by the library, or it raises
```

The concept demo above and this version agree on the *contract* (`TicketTriage`); they differ in who enforces it. Test your code against the fake, then run a small golden set against the real model (week 5).

!!! warning "Watch out — few-shot does not learn"

    Putting three labeled examples in a prompt is **in-context imitation**, not training. The weights do not change. Tomorrow’s ticket is not “learned from” today’s examples unless you put those examples in the prompt again (or fine-tune, which this week is not).

```python
from langchain_core.prompts import ChatPromptTemplate

few_shot = ChatPromptTemplate.from_messages([
    ("system", "Label sentiment: positive, neutral, or negative."),
    ("human", "The export is fast now."),
    ("assistant", "positive"),
    ("human", "{review}"),
])
# The model is shown a pattern. It has not been trained on CloudWave reviews.
```

## Route the second step with an `if`

A two-step “classify then reply” workflow is ordinary control flow. You do not need a graph for this.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

def reply_for(ticket: dict) -> str:
    if ticket["category"] == "bug":
        return "File a bug; send the error code."
    if ticket["category"] == "billing":
        return "Send to billing; do not guess the invoice."
    return "Ask a human."

assert reply_for({"category": "bug"}).startswith("File")

# Library spelling (same idea). Concept demo — no model.
branch = RunnableBranch(
    (lambda x: x["category"] == "bug", RunnableLambda(lambda x: "file a bug")),
    (lambda x: x["category"] == "billing", RunnableLambda(lambda x: "send to billing")),
    RunnableLambda(lambda x: "ask a human"),
)
assert branch.invoke({"category": "bug"}) == "file a bug"
```

!!! success "Ship / don’t ship"

    **Ship** a chain when the steps are known, the schema is small, and a golden input produces a dict you can assert. **Don’t ship** a 14-field “complete triage object” you cannot validate, and don’t treat `JsonOutputParser` as if it returned a Pydantic instance. Don’t claim the pipe parallelizes itself.

## What this week is not

- Not a production support bot (no eval, no allowlist — that is week 7).
- Not an agent. If you already know the two calls, write two calls.
- Not a promise that structured output stops hallucinations. It constrains *shape*.

## ✍️ Exercise

[Exercises](exercises/week-01.md).

## 🤔 Reflection

1. What type does `JsonOutputParser` return? What would you switch to for a Pydantic instance?
2. Why is `a | b | c` not a fan-out?
3. If few-shot “stops working” next week, what actually changed?

## 🔗 Next week

Memory is a **session store**. Two `session_id`s must not leak.

## 📚 Docs (this pin: LangChain 1.x)

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [Models](https://docs.langchain.com/oss/python/langchain/models) — `init_chat_model`, chat vs text models
- [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Runnables (the `|` pipe) reference](https://reference.langchain.com/python/langchain_core/runnables/)
- [Migrating from 0.x to 1.x](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [Pydantic](https://docs.pydantic.dev/)
