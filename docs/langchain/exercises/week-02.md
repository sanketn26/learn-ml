---
description: Implement a keyed LangChain session store with per-user isolation and a bounded message window, then compare it to legacy memory.
---

# Exercises — Week 2 — Session store

Do these after reading [Week 2](../week-02.md). Use `langchain_core.messages.HumanMessage` / `AIMessage`. A `dict[str, list]` is enough. `ConversationBufferMemory` is optional legacy.

```python
from langchain_community.llms import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage
```

## Predict before you run

After 6 turns with `history[-4:]`, is turn 1 still in Alice's list? If Alice and Bob share one `ConversationBufferMemory`, whose laptop shows up in Bob's history?

## Runnable command

```bash
python -c "from langchain_core.messages import HumanMessage; print(HumanMessage(content='hi').content)"
```

Paste your `chat(session_id, text)` into a `.py` file and run it from the repo root. No API key.

Each task has three hints, closed by default. Open only as far as you need.

## 1. Two session keys

Implement `chat(session_id, text)` that appends `HumanMessage` then `AIMessage` onto `sessions[session_id]`. Script `FakeListLLM` so Alice talks about a laptop and Bob talks about shoes.

**Checks:**

- `"alice" in sessions and "bob" in sessions`
- Alice’s list mentions the laptop; Bob’s list does not
- `isinstance(sessions["alice"][0], HumanMessage)`

??? tip "Hint 1 — a nudge"
    This is a Redis session store with a different value type. What is the key, and what gets appended per request?

??? tip "Hint 2 — the approach"
    `sessions.setdefault(session_id, [])` gives each user their own list. Append the `HumanMessage`, call the model, append the `AIMessage`. `FakeListLLM` answers in order, so interleave the scripted replies the same way you interleave the calls.

??? example "Hint 3 — most of the code"
    ```python
    from langchain_community.llms import FakeListLLM
    from langchain_core.messages import AIMessage, HumanMessage

    llm = FakeListLLM(responses=[
        "Which laptop model is it?",
        "Try the shoe size chart.",
    ])
    sessions: dict[str, list] = {}


    def chat(session_id: str, text: str) -> str:
        history = sessions.setdefault(session_id, [])
        history.append(HumanMessage(content=text))
        reply = llm.invoke(text)
        history.append(AIMessage(content=reply))
        return reply


    chat("alice", "My laptop won't sync")
    chat("bob", "Do these shoes run small?")
    print({sid: [m.content for m in msgs] for sid, msgs in sessions.items()})
    ```

## 2. Bound the window

Keep only the last 4 messages per session (`history[-4:]`). Run 6 turns for Alice.

**Checks:**

- `len(sessions["alice"]) == 4` after the trim
- Turn 1’s text is gone; turn 6’s text is present

??? tip "Hint 1 — a nudge"
    Six turns is twelve messages. Four is two turns. Which two?

??? tip "Hint 2 — the approach"
    Trim at the end of `chat`: reassign `sessions[session_id] = history[-4:]`. Slicing makes a new list, so mutating `history` in place won't help. Give each turn a numbered message so you can see which ones survived.

??? example "Hint 3 — most of the code"
    ```python
    WINDOW = 4
    llm = FakeListLLM(responses=[f"reply {i}" for i in range(1, 7)])


    def chat_bounded(session_id: str, text: str) -> str:
        reply = chat(session_id, text)
        sessions[session_id] = sessions[session_id][-WINDOW:]
        return reply


    sessions.clear()
    for turn in range(1, 7):
        chat_bounded("alice", f"turn {turn}")
    print([m.content for m in sessions["alice"]])
    ```

## 3. Legacy vs keyed store (short)

Create one `ConversationBufferMemory()`, `save_context` twice, and note in two comments: (1) it has no `session_id`, (2) you would not share that instance across users.

**Checks:**

- `load_memory_variables({})["history"]` is non-empty
- You did **not** use that single object as the store for both Alice and Bob

??? tip "Hint 1 — a nudge"
    Look at `save_context`'s signature. Where would a user id go?

??? tip "Hint 2 — the approach"
    Build it, call `save_context({"input": ...}, {"output": ...})` twice, print `load_memory_variables({})`. The two comments are the deliverable — one instance is one conversation, full stop.

??? example "Hint 3 — most of the code"
    ```python
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory()
    memory.save_context({"input": "My laptop won't sync"}, {"output": "Which model?"})
    memory.save_context({"input": "The 14-inch one"}, {"output": "Try signing out and in."})
    print(memory.load_memory_variables({})["history"])
    # (1) ...
    # (2) ...
    ```

## Expected observation

??? success "Open after you run"
    Two keys in `sessions`. Alice mentions a laptop; Bob does not. After the trim, `len(sessions['alice']) == 4`.

## Self-check

`isinstance(sessions["alice"][0], HumanMessage)` is True. You did not use one legacy memory object for both people.
