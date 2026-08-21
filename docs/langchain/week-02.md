# Week 2 — Memory is a session store

**Course:** LangChain  
**Who this is for:** Engineers who have keyed Redis by `session_id` and leaked user A’s cart into user B’s request.

A chain is stateless. “Remember the Dell” is not magic — it is **history you pass in**. The real question is: *which key, which list, what do you drop.*

---

## 🎯 What you will be able to do

- Isolate two CloudWave shoppers with `session_id`
- Store turns as a dict of message lists (or `InMemoryChatMessageHistory`)
- Bound history so the prompt cannot grow forever
- Recognize `ConversationBufferMemory` as **legacy spelling**, not the default
- Know when a session store is the wrong place for facts (plan, cart, PII)

!!! think "Think of it like… a session store, not a brain."

    `GET /cart` is keyed by cookie. Conversation history is the same: `sessions[session_id].append(...)`. If there is no key, there is no memory — or worse, one global list shared by everyone.

## Picture two sessions

```
sessions = {
  "alice": [Human("laptops under $1000"), AI("Dell XPS 13"), Human("the first one")],
  "bob":   [Human("running shoes size 11")],
}

invoke(alice)  →  history = sessions["alice"]   # Bob is invisible
invoke(bob)    →  history = sessions["bob"]
```

Cross-session leak = shipping user A’s tickets in user B’s prompt.

## The pattern that actually holds state

`HumanMessage` lives in `langchain_core.messages`. The store is ordinary Python.

```python
from langchain_community.llms import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage

llm = FakeListLLM(responses=[
    "Here are laptops under $1000: 1) Dell XPS 13.",
    "The first one is the Dell XPS 13 ($999).",
    "I can help with shoes. What size?",
])

sessions: dict[str, list] = {}

def chat(session_id: str, text: str) -> str:
    history = sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=text))
    reply = llm.invoke(text)
    history.append(AIMessage(content=reply))
    return reply

chat("alice", "Show me laptops under $1000")
chat("alice", "Tell me more about the first one")
chat("bob", "I need running shoes")

assert any("Dell" in m.content for m in sessions["alice"] if isinstance(m, AIMessage))
assert all("Dell" not in m.content for m in sessions["bob"])
assert "alice" in sessions and "bob" in sessions
```

That dict-of-lists **is** the product. Swap the dict for Redis later; keep the key.

### InMemoryChatMessageHistory / RunnableWithMessageHistory (0.2)

LangChain 0.2 wraps the same idea. Concept demo — still no API key.

```python
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return store.setdefault(session_id, InMemoryChatMessageHistory())

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CloudWave shopping assistant."),
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
    {"input": "Show me laptops under $1000"},
    config={"configurable": {"session_id": "alice"}},
)
assert len(store["alice"].messages) >= 2
```

## Legacy note: ConversationBufferMemory

You will still see this in older tutorials:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True, memory_key="history")
memory.save_context({"input": "laptops"}, {"output": "Dell XPS 13"})
# load_memory_variables({})["history"]  →  list of messages
```

It is a bag of messages with extra methods. It is **not** keyed by `session_id` unless *you* put one instance per session. Prefer the dict / `InMemoryChatMessageHistory` pattern above. Window and summary variants (`ConversationBufferWindowMemory`, `ConversationSummaryMemory`) are the same idea with a trim or an extra LLM call — they still need a session key.

## Bound the list

A buffer that stores every turn will blow the context window. Trim.

```python
def recent(history: list, k: int = 4) -> list:
    """Keep the last k messages. Older turns are gone — by design."""
    return history[-k:]
```

Facts that must survive a trim (plan, cart id, “Enterprise”) belong in a **profile dict**, not in the chat log.

!!! warning "Watch out — one global Memory instance"

    A module-level `memory = ConversationBufferMemory()` is a shared inbox. Two FastAPI workers, two users, one list: you have a data leak. Key by `session_id`. Encrypt at rest if you persist. Do not send card numbers back into the next prompt.

!!! success "Ship / don’t ship"

    **Ship** a store keyed by `session_id` with a trim policy and a test that Alice cannot see Bob. **Don’t ship** unbounded `ConversationBufferMemory` as “the chatbot remembers everything,” and don’t treat few-shot examples inside the system prompt as a substitute for a session (that is week 1). Hypothetical CloudWave shoppers here are two dict keys, not a retailer case study.

## What this week is not

- Not a recommendation engine. Cart and catalog are your database.
- Not durable storage. `InMemoryChatMessageHistory` dies with the process.
- Not a license to put PII in the prompt “for personalization.”

## ✍️ Exercise

[Exercises](exercises/week-02.md).

## 🤔 Reflection

1. Where does `session_id` come from in your API (cookie, JWT, header)?
2. After a trim of `k=4`, which CloudWave facts would you store *outside* the message list?
3. Why is a global `ConversationBufferMemory` a privacy bug?

## 🔗 Next week

Agents: a loop that picks tools. ReAct is not autonomy.

## 📚 Docs (this pin)

- [Message history (0.2)](https://python.langchain.com/v0.2/docs/how_to/message_history/)
- [Messages](https://python.langchain.com/v0.2/docs/concepts/#messages)
