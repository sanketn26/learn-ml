# Exercises — Week 2 — Session store

Do these after reading [Week 2](../week-02.md). Use `langchain_core.messages.HumanMessage` / `AIMessage`. A `dict[str, list]` is enough. `ConversationBufferMemory` is optional legacy.

```python
from langchain_community.llms import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage
```

## 1. Two session keys

Implement `chat(session_id, text)` that appends `HumanMessage` then `AIMessage` onto `sessions[session_id]`. Script `FakeListLLM` so Alice talks about a laptop and Bob talks about shoes.

**Checks:**

- `"alice" in sessions and "bob" in sessions`
- Alice’s list mentions the laptop; Bob’s list does not
- `isinstance(sessions["alice"][0], HumanMessage)`

## 2. Bound the window

Keep only the last 4 messages per session (`history[-4:]`). Run 6 turns for Alice.

**Checks:**

- `len(sessions["alice"]) == 4` after the trim
- Turn 1’s text is gone; turn 6’s text is present

## 3. Legacy vs keyed store (short)

Create one `ConversationBufferMemory()`, `save_context` twice, and note in two comments: (1) it has no `session_id`, (2) you would not share that instance across users.

**Checks:**

- `load_memory_variables({})["history"]` is non-empty
- You did **not** use that single object as the store for both Alice and Bob

## Predict before you run

After 6 turns with `history[-4:]`, is turn 1 still in Alice's list? If Alice and Bob share one `ConversationBufferMemory`, whose laptop shows up in Bob's history?

## Runnable command

```bash
python -c "from langchain_core.messages import HumanMessage; print(HumanMessage(content='hi').content)"
```

Paste your `chat(session_id, text)` into a `.py` file and run it from the repo root. No API key.

## Expected observation

Two keys in `sessions`. Alice mentions a laptop; Bob does not. After the trim, `len(sessions['alice']) == 4`.

## Self-check

`isinstance(sessions["alice"][0], HumanMessage)` is True. You did not use one legacy memory object for both people.
