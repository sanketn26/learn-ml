---
description: Build per-session chat memory in LangChain by keying message history like a Redis session store, bounding growth, and avoiding cross-user leaks.
---

# Week 2 — Memory is a session store

Two CloudWave support threads are open at once: one tenant is stuck mid-export, the other just wants to know how to rotate an API key. Nothing about the bot's code should let those two conversations blur into each other.

??? note "Course details"

    **Course:** LangChain
    **Who this is for:** Engineers who have keyed Redis by `session_id` and leaked user A’s cart into user B’s request.

A chain is stateless. “Remember what I asked” is not magic — it is **history you pass in**. The real question is: *which key, which list, what do you drop.*

---

## 🎯 What you will be able to do

- Isolate two CloudWave support threads with `session_id`
- Store turns as a dict of message lists (or `InMemoryChatMessageHistory`)
- Bound history so the prompt cannot grow forever
- Recognize `ConversationBufferMemory` as **legacy spelling**, not the default
- Know when a session store is the wrong place for facts (account tier, ticket id, PII)

!!! think "Think of it like… a session store, not a brain."

    `GET /cart` is keyed by cookie. Conversation history is the same: `sessions[session_id].append(...)`. If there is no key, there is no memory — or worse, one global list shared by everyone.

    This is **memory isolation**, not tenant authorization. Keying a dict by `session_id` stops one conversation's history from leaking into another's prompt. It proves nothing about who is allowed to open that session in the first place — that check happens above this code, same as it would for any session store.

## Picture two sessions

```
sessions = {
  "tenant_492": [Human("my export keeps timing out"), AI("What row count?"), Human("about 150k rows")],
  "tenant_118": [Human("how do I rotate an API key?")],
}

invoke(tenant_492)  →  history = sessions["tenant_492"]   # tenant_118 is invisible
invoke(tenant_118)  →  history = sessions["tenant_118"]
```

Cross-session leak = shipping tenant A’s ticket history in tenant B’s prompt.

## The pattern that actually holds state

`HumanMessage` lives in `langchain_core.messages`. The store is ordinary Python.

```python
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

llm = FakeListChatModel(responses=[
    "Can you share the row count on that export?",
    "150k rows is past the known threshold — escalating to CW-1847.",
    "Settings > API Keys > Rotate. Old key stays valid for 24h.",
])

sessions: dict[str, list] = {}

def chat(session_id: str, text: str) -> str:
    history = sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=text))
    reply = llm.invoke(text).content   # a chat model returns an AIMessage; .content is the text
    history.append(AIMessage(content=reply))
    return reply

chat("tenant_492", "My export keeps timing out")
chat("tenant_492", "It's about 150k rows")
chat("tenant_118", "How do I rotate an API key?")

assert any("CW-1847" in m.content for m in sessions["tenant_492"] if isinstance(m, AIMessage))
assert all("CW-1847" not in m.content for m in sessions["tenant_118"])
assert "tenant_492" in sessions and "tenant_118" in sessions
```

That dict-of-lists **is** the product. Swap the dict for Redis later; keep the key.

### InMemoryChatMessageHistory / RunnableWithMessageHistory

The library wraps the same idea for a plain chain. Concept demo — still no API key.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return store.setdefault(session_id, InMemoryChatMessageHistory())

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CloudWave support assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

with_history.invoke(
    {"input": "My export keeps timing out"},
    config={"configurable": {"session_id": "tenant_492"}},
)
assert len(store["tenant_492"].messages) >= 2
```

### In an agent: a checkpointer keyed by `thread_id`

LangChain 1.x agents (`create_agent`, next week) keep conversation memory in a **LangGraph checkpointer**, and the session key is called `thread_id`. Same idea, new spelling:

```
agent = create_agent(model, tools, checkpointer=InMemorySaver())
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "tenant_492"}})
```

Different `thread_id`, different history — exactly the dict above. [LangGraph week 3](../langgraph/week-03.md) opens the checkpointer up.

## Legacy note: ConversationBufferMemory

You will still see this in older tutorials. It was **removed from `langchain` in 1.0** (it survives only in the `langchain-classic` package), so this block is for recognition, not for running:

```python
# LangChain ≤ 0.3 only — ImportError on 1.x:
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(return_messages=True, memory_key="history")
# memory.save_context({"input": "export timing out"}, {"output": "What row count?"})
```

It was a bag of messages with extra methods, **not** keyed by `session_id` unless *you* kept one instance per session. That is the bug it invited, and why it went away. Window and summary variants were the same idea with a trim or an extra LLM call; in 1.x the equivalents are a trim function (below) or `SummarizationMiddleware` on an agent.

## Bound the list

A buffer that stores every turn will blow the context window. Trim.

```python
def recent(history: list, k: int = 4) -> list:
    """Keep the last k messages. Older turns are gone — by design."""
    return history[-k:]
```

Facts that must survive a trim (account tier, open ticket id, “Enterprise”) belong in a **profile dict**, not in the chat log.

!!! warning "Watch out — one global Memory instance"

    A module-level `history = InMemoryChatMessageHistory()` (or the old `ConversationBufferMemory()`) is a shared inbox. Two FastAPI workers, two users, one list: you have a data leak. Key by `session_id`. Encrypt at rest if you persist. Do not send card numbers back into the next prompt.

!!! success "Ship / don’t ship"

    **Ship** a store keyed by `session_id` with a trim policy and a test that tenant A cannot see tenant B. **Don’t ship** an unbounded history as “the chatbot remembers everything,” and don’t treat few-shot examples inside the system prompt as a substitute for a session (that is week 1). Don’t ship this session store *as* tenant authorization — it isolates memory, not access. Hypothetical CloudWave tenants here are two dict keys, not two real customers.

## What this week is not

- Not a ticket database. The support ticket and account records are your database.
- Not durable storage. `InMemoryChatMessageHistory` dies with the process.
- Not a license to put PII in the prompt “for personalization.”

## ✍️ Exercise

[Exercises](exercises/week-02.md).

## 🤔 Reflection

1. Where does `session_id` come from in your API (cookie, JWT, header)?
2. After a trim of `k=4`, which CloudWave facts would you store *outside* the message list?
3. Why is one global history object — whatever the library calls it — a privacy bug?

## 🔗 Next week

Agents: a loop that picks tools. ReAct is not autonomy.

## 📚 Docs (this pin: LangChain 1.x)

- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) — checkpointers and `thread_id`
- [Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [Middleware](https://docs.langchain.com/oss/python/langchain/middleware) — `SummarizationMiddleware` replaces the summary-memory classes
