---
description: Build a LangChain ReAct agent that picks tools from a registry, comparing the agent loop to a backend intent classifier and handler dispatch.
---

# Week 3 — Agents & Tools

**Course:** LangChain  
**Who this is for:** Engineers who have written a job runner that picks a handler from a registry.

An **agent** is not an employee. It is a loop:

```
observe  →  think  →  pick a tool  →  call it  →  observe the result  →  …
until the model says “I can answer now”
```

That loop is called **ReAct** (reason + act). The intelligence, such as it is, is “which function do I call next, with what arguments.” The tools are *your* functions. The model is a router with a prose addiction.

---

## If you already write software

You have shipped this shape:

```
request
  → classify intent
  → call billing.get_balance(user_id)
  → call billing.next_invoice(user_id)
  → render a sentence
```

An agent does the same thing, except the “classify intent / pick handler” step is an LLM instead of a `switch`. That is a trade: you gain flexibility on messy language, you lose a deterministic call graph, and you pay tokens + latency on every hop.

```
Your backend                     Agent
──────────────────────────       ──────────────────────────────
handler registry                 tools = [fn, fn, fn]
JSON schema for args             tool docstring + args schema
try / catch per client           tool error → back into the loop
max retries                      ModelCallLimitMiddleware (a hard cap on loop turns)
logs of which handler ran        the returned message list / tracing
```

!!! warning "Watch out — unbounded loops"
    An agent will happily call the same broken tool forever. Cap the loop — in LangChain 1.x, `ModelCallLimitMiddleware(run_limit=...)`. Treat a tool failure as a *value* the model can read (“the billing API returned 503”), not an exception that kills the process — unless the exception is “we should not be doing this at all.”

## 🏢 Scenario

Customer: “What’s my account balance and when is my next billing date?”

A chain with a fixed prompt cannot answer that unless you stuffed both numbers into the prompt already. An agent can:

1. Think: I need `get_account_balance` and `get_billing_date`.
2. Act: call both (ideally in parallel, often in series).
3. Observe: `$128.40`, `2026-09-01`.
4. Answer: “Your balance is $128.40. Next bill is 1 September.”

The tools are ordinary Python. The model only chooses.

## A tool is a function with a contract

A tool the model can call needs three things you already put on a public function:

1. A **name** the model can say
2. A **docstring** that tells it *when* to use it (this is the API doc)
3. A **schema** for arguments (types, required fields)

```python
from langchain.tools import tool


@tool
def get_account_balance(user_id: str) -> str:
    """Return the current account balance in USD for this CloudWave user_id.
    Use when the customer asks what they owe or what they have prepaid."""
    # pretend this is billing.get(user_id).balance
    balances = {"user_0001": 128.40, "user_0002": 0.0}
    amount = balances.get(user_id)
    if amount is None:
        return f"no account for {user_id}"
    return f"{amount:.2f}"


@tool
def get_billing_date(user_id: str) -> str:
    """Return the next invoice date (ISO) for this CloudWave user_id.
    Use when the customer asks when they will be charged."""
    return "2026-09-01"
```

!!! engineer "Engineer mental model"
    The docstring is not for you. It is the *prompt*. Vague docstrings produce tool-call roulette. Write them like you would write an OpenAPI `description`: when to call, what comes back, what not to use it for.

## The loop, without the framework fog

```text
# pseudocode — the library does this; you should be able to draw it
messages = [system, user_question]
for step in range(MAX_STEPS):
    decision = model.decide(messages, tools)
    if decision.done:
        return decision.answer
    result = call_tool(decision.tool, decision.args)   # your function
    messages.append(tool_result(result))
raise RuntimeError("agent hit MAX_STEPS")
```

If you cannot write that loop on a whiteboard, do not debug “the agent is being weird” by tweaking the persona paragraph. Debug which tool was chosen, with which args, and what it returned.

### The same loop, in LangChain 1.x: `create_agent`

The library spelling is `langchain.agents.create_agent(model, tools, ...)`. It builds exactly the loop above as a small LangGraph graph (model node → tool node → back to model). Older tutorials use `AgentExecutor` or LangGraph's `create_react_agent`; both are superseded by `create_agent` in 1.x.

A real model needs an API key, so this concept demo drives the loop with a **scripted** model: it replays tool calls we wrote down. That is not a toy — it is how you unit-test an agent's wiring without paying for tokens.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedToolModel(FakeMessagesListChatModel):
    """Replays AIMessages we wrote, tool calls included. Real models implement bind_tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, i, **args):
    return {"name": name, "args": args, "id": f"call_{i}"}


model = ScriptedToolModel(responses=[
    AIMessage(content="", tool_calls=[call("get_account_balance", 1, user_id="user_0001"),
                                      call("get_billing_date", 2, user_id="user_0001")]),
    AIMessage(content="Your balance is $128.40. Next bill is 1 September 2026."),
])
agent = create_agent(
    model,
    tools=[get_account_balance, get_billing_date],
    system_prompt="You are CloudWave billing support. Use tools; never guess numbers.",
    middleware=[ModelCallLimitMiddleware(run_limit=5)],       # the loop cap, as code
)
result = agent.invoke({"messages": [{"role": "user", "content": "What's my balance and next bill date?"}]})
for m in result["messages"]:
    print(f"{type(m).__name__:<13} {m.content or m.tool_calls}")
assert "128.40" in result["messages"][-1].content
```

The printout *is* the ReAct trace from the next section, as data: the model's tool calls, each tool's return value as a `ToolMessage`, then the answer. Now the failure this week warned about — a model that keeps asking for the same tool:

```python
stuck = ScriptedToolModel(responses=[
    AIMessage(content="", tool_calls=[call("get_account_balance", i, user_id="user_0001")]) for i in range(10)
])
runaway = create_agent(stuck, tools=[get_account_balance],
                       middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")])
out = runaway.invoke({"messages": [{"role": "user", "content": "balance?"}]})
print(out["messages"][-1].content)   # the cap fired; the process did not spin
```

Without the middleware this runs until the graph's `recursion_limit` raises. With it, the run ends with a readable message you can log and alert on. Other 1.x middleware you will reach for: `ToolCallLimitMiddleware` (cap one expensive tool), `ToolRetryMiddleware` (retry flaky tools), and `HumanInTheLoopMiddleware` (pause before a side-effecting tool — the LangGraph week 4 idea, packaged).

## ReAct is a prompt pattern, not autonomy

```
Thought: I need the balance and the next invoice date.
Action: get_account_balance
Action Input: user_0001
Observation: 128.40
Thought: Now the billing date.
Action: get_billing_date
Action Input: user_0001
Observation: 2026-09-01
Thought: I can answer.
Final Answer: Your balance is $128.40. Next bill is 1 September 2026.
```

That is a trace, not a mind. Log it. When the agent goes off the rails, the trace is the stack.

!!! success "Ship / don’t ship"
    **Ship** an agent when the user phrasing is messy and the tool set is small (≤ 8) and each tool is side-effect light or idempotent — with a loop cap and a test that drives it with a scripted model. **Don’t ship** an agent that can refund, delete, or email until those tools require a human confirmation (`HumanInTheLoopMiddleware`, or the graph in LangGraph week 4). A chain with two explicit retrieval calls is better than an agent that sometimes invents a third.

## Errors are observations

```python
@tool
def get_account_balance(user_id: str) -> str:
    """Current balance in USD. Returns an error string if the user is unknown."""
    try:
        return f"{Billing.get(user_id).balance:.2f}"
    except Billing.NotFound:
        return f"error: unknown user_id {user_id}"
    except Billing.Unavailable:
        return "error: billing API unavailable, try again later"
```

Return a string the model can read. Do not raise out of the tool unless you *want* the whole request to 500. The agent can apologize, ask for a different id, or stop.

## What this week is not

- Not “give the model a browser and walk away.”
- Not a replacement for a well-specified chain when you already know the two calls.
- Not permission to skip eval (week 5). Agents fail in new ways: wrong tool, right tool wrong args, extra loop, confident nonsense after a tool error.

## 🤔 Reflection

1. For the billing question above, would you actually use an agent, or two tool calls in a chain? Why?
2. Which of your production tools would you *refuse* to hang on an agent without a human in the loop?
3. If the trace shows the model calling `get_account_balance` three times with the same id, what do you change first — the prompt, the docstring, or the loop cap?

## 🔗 Next week

RAG: the model should not memorize your docs. Embed, retrieve, then prompt. Search quality first — a better index beats a cleverer agent.
