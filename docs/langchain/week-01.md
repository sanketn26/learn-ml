---
description: Learn LangChain chains as prompt templates piped through output parsers, and compose prompt, model, and parser into one call graph.
---

# Week 1 — Chains: templates, parsers, pipes

**Course:** LangChain  
**Who this is for:** Engineers who have written an HTTP handler that validates JSON and calls a slow downstream.

LangChain is not a model. The model is the remote API. LangChain is **middleware**: a prompt is a template, a parser is a schema, a chain is your call graph.

---

## 🎯 What you will be able to do

- Write a reusable prompt template and inject variables
- Compose `prompt | llm | parser` and say what each step returns
- Parse model text into a **dict** (or a Pydantic object) and reject garbage
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
service client                      an LLM (here: FakeListLLM)
try / catch + retries               fallbacks you write yourself
```

Concept demos in this track use `FakeListLLM`. No API key.

## Picture the pipe

```
ticket dict
    │
    ▼
[ChatPromptTemplate]   fill {subject} and {body}
    │
    ▼
[FakeListLLM]          returns a JSON string (scripted)
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
from langchain_community.llms import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

prompt = ChatPromptTemplate.from_template(
    """You are CloudWave support.
Subject: {subject}
Body: {body}
Reply in one sentence."""
)
llm = FakeListLLM(responses=[
    "Clear the browser cache, then reload the dashboard."
])
chain = prompt | llm | StrOutputParser()

text = chain.invoke({
    "subject": "Dashboard slow",
    "body": "Enterprise tenant, Chrome, 8s load.",
})
assert "cache" in text.lower()
```

Hypothetical CloudWave volume for this track: a few hundred tickets a day, not a vendor case study.

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

triage_llm = FakeListLLM(responses=[
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

## 📚 Docs (this pin)

- [LangChain Python 0.2](https://python.langchain.com/v0.2/docs/introduction/)
- [LCEL](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel)
- [JSON parser](https://python.langchain.com/v0.2/docs/how_to/output_parser_json/)
- [Pydantic](https://docs.pydantic.dev/)
