---
description: Implement a keyed LangChain session store with per-user isolation and a bounded message window, then compare it to legacy memory.
---

# Exercises — Week 2 — Session store

Do these after reading [Week 2](../week-02.md). Use `langchain_core.messages.HumanMessage` / `AIMessage`. A `dict[str, list]` is enough. (`ConversationBufferMemory` is gone from LangChain 1.x — see the lesson's legacy note.)

```python
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
```

## Predict before you run

After 6 turns with `history[-4:]`, is turn 1 still in tenant_492's list? If tenant_492 and tenant_118 shared one history object, whose export incident would show up in tenant_118's prompt?

## Runnable command

```bash
python -c "from langchain_core.messages import HumanMessage; print(HumanMessage(content='hi').content)"
```

Paste your `chat(session_id, text)` into a `.py` file and run it from the repo root. No API key.

Each task has three hints, closed by default. Open only as far as you need.

## 1. Two session keys

Implement `chat(session_id, text)` that appends `HumanMessage` then `AIMessage` onto `sessions[session_id]`. Script `FakeListChatModel` so tenant_492 asks about an export timeout and tenant_118 asks about rotating an API key.

**Checks:**

- `"tenant_492" in sessions and "tenant_118" in sessions`
- tenant_492’s list mentions the export; tenant_118’s list does not
- `isinstance(sessions["tenant_492"][0], HumanMessage)`

??? tip "Hint 1 — a nudge"
    This is a Redis session store with a different value type. What is the key, and what gets appended per request?

??? tip "Hint 2 — the approach"
    `sessions.setdefault(session_id, [])` gives each user their own list. Append the `HumanMessage`, call the model, append the `AIMessage`. `FakeListChatModel` answers in order, so interleave the scripted replies the same way you interleave the calls.

??? example "Hint 3 — most of the code"
    ```python
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    llm = FakeListChatModel(responses=[
        "How many rows was the export?",
        "Settings > API Keys > Rotate. The old key stays valid for 24h.",
    ])
    sessions: dict[str, list] = {}


    def chat(session_id: str, text: str) -> str:
        history = sessions.setdefault(session_id, [])
        history.append(HumanMessage(content=text))
        reply = llm.invoke(text).content   # chat models return an AIMessage
        history.append(AIMessage(content=reply))
        return reply


    chat("tenant_492", "My export keeps timing out")
    chat("tenant_118", "How do I rotate an API key?")
    print({sid: [m.content for m in msgs] for sid, msgs in sessions.items()})
    ```

## 2. Bound the window

Keep only the last 4 messages per session (`history[-4:]`). Run 6 turns for tenant_492.

**Checks:**

- `len(sessions["tenant_492"]) == 4` after the trim
- Turn 1’s text is gone; turn 6’s text is present

??? tip "Hint 1 — a nudge"
    Six turns is twelve messages. Four is two turns. Which two?

??? tip "Hint 2 — the approach"
    Trim at the end of `chat`: reassign `sessions[session_id] = history[-4:]`. Slicing makes a new list, so mutating `history` in place won't help. Give each turn a numbered message so you can see which ones survived.

??? example "Hint 3 — most of the code"
    ```python
    WINDOW = 4
    llm = FakeListChatModel(responses=[f"reply {i}" for i in range(1, 7)])


    def chat_bounded(session_id: str, text: str) -> str:
        reply = chat(session_id, text)
        sessions[session_id] = sessions[session_id][-WINDOW:]
        return reply


    sessions.clear()
    for turn in range(1, 7):
        chat_bounded("tenant_492", f"turn {turn}")
    print([m.content for m in sessions["tenant_492"]])
    ```

## 3. The library's keyed store

Rebuild task 1 with the library pieces: one `InMemoryChatMessageHistory` **per session id**, looked up by a `get_session_history(session_id)` function and wired with `RunnableWithMessageHistory`. Then prove the isolation with an assertion, not a print.

**Checks:**

- `store` has two keys, one per tenant
- Nothing tenant_492 said appears in tenant_118's history

??? tip "Hint 1 — a nudge"
    The library does not know what a tenant is. Where, exactly, does the session id turn into *which* history object?

??? tip "Hint 2 — the approach"
    `store: dict[str, InMemoryChatMessageHistory]`, and `get_session_history` does `store.setdefault(session_id, InMemoryChatMessageHistory())`. Wrap `prompt | llm | StrOutputParser()` in `RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")` and pass the id as `config={"configurable": {"session_id": ...}}`.

??? example "Hint 3 — most of the code"
    ```python
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store: dict[str, InMemoryChatMessageHistory] = {}


    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        return store.setdefault(session_id, InMemoryChatMessageHistory())


    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are CloudWave support."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    llm = FakeListChatModel(responses=["How many rows?", "Settings > API Keys > Rotate."])
    with_history = RunnableWithMessageHistory(prompt | llm | StrOutputParser(), get_session_history,
                                              input_messages_key="input", history_messages_key="history")
    for tenant, text in [("tenant_492", "My export keeps timing out"), ("tenant_118", "How do I rotate an API key?")]:
        with_history.invoke({"input": text}, config={"configurable": {"session_id": tenant}})

    assert set(store) == {"tenant_492", "tenant_118"}
    assert all("export" not in m.content for m in store["tenant_118"].messages)
    ```

## Expected observation

??? success "Open after you run"
    Two keys in `sessions`. tenant_492 mentions the export; tenant_118 does not. After the trim, `len(sessions["tenant_492"]) == 4`. The library store keeps the same isolation, keyed the same way.

## Self-check

`isinstance(sessions["tenant_492"][0], HumanMessage)` is True. No history object is shared between tenants.
