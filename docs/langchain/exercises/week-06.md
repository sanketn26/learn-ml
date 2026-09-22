---
description: Wire a FastAPI chat endpoint with async LangChain invocation, a timeout-triggered fallback, and a FIFO response cache.
---

# Exercises — Week 6 — Local handler

Do these after reading [Week 6](../week-06.md). Local FastAPI + timeout/fallback is enough. No cloud deploy, no 100 concurrent users.

```python
import asyncio
import time
from langchain_community.llms import FakeListLLM
```

## Predict before you run

If `ainvoke` sleeps 0.2s and `wait_for` is 0.05s, is `fallback True`? After a third distinct cache key with `max_size=2`, is the first question still a hit?

## Runnable command

```bash
# concept: you can run the timeout/cache without binding :8000
python your_local_handler.py
```

Each task has three hints, closed by default. Open only as far as you need.

## 1. `/health` and `/chat`

Copy the lesson’s FastAPI app. `POST /chat` must `await chain.ainvoke({...})` (not `agenerate` on a string). Import `time`.

**Checks:**

- `GET /health` → `{"ok": True}`
- `POST /chat` with `{"message": "password reset"}` returns a `response` and `latency_ms`
- `import time` is used; `ChatRequest.message` is required

??? tip "Hint 1 — a nudge"
    You don't need to bind a port to test a handler. What does FastAPI give you for calling routes in-process?

??? tip "Hint 2 — the approach"
    `fastapi.testclient.TestClient(app)` calls routes like `requests` would, no server. Check `/health`, a good `/chat`, and a `/chat` with an empty body — a required field should return 422 without your code running.

??? example "Hint 3 — most of the code"
    ```python
    import asyncio
    import time

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langchain_community.llms import FakeListLLM
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from pydantic import BaseModel

    app = FastAPI()
    chain = ChatPromptTemplate.from_template("{message}") | FakeListLLM(responses=["Settings > Security."] * 10) | StrOutputParser()


    class ChatRequest(BaseModel):
        message: str


    class ChatResponse(BaseModel):
        response: str
        latency_ms: float
        fallback: bool = False


    @app.get("/health")
    def health():
        return {"ok": True}


    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        start = time.time()
        text = await chain.ainvoke({"message": request.message})
        return ChatResponse(response=text, latency_ms=(time.time() - start) * 1000)


    client = TestClient(app)
    print(client.get("/health").json())
    print(client.post("/chat", json={"message": "password reset"}).json())
    print(client.post("/chat", json={}).status_code)
    ```

## 2. Timeout → fallback

Wrap `ainvoke` in `asyncio.wait_for(..., timeout=0.05)` and use an LLM (or `RunnableLambda`) that sleeps 0.2s so the wait fires.

**Checks:**

- The response sets `fallback is True`
- The fallback string does not mention a refund or a dollar amount

??? tip "Hint 1 — a nudge"
    A timeout you've never seen fire is a timeout you don't know works. How do you make the "model" reliably slower than the budget?

??? tip "Hint 2 — the approach"
    An `async def` that `await asyncio.sleep(0.2)` wrapped in `RunnableLambda` is a slow model with no network. Catch `asyncio.TimeoutError` around `wait_for` and return a fixed, harmless fallback string with `fallback=True`.

??? example "Hint 3 — most of the code"
    ```python
    async def slow_model(inputs: dict) -> str:
        await asyncio.sleep(0.2)
        return "late answer"


    slow_chain = RunnableLambda(slow_model)
    FALLBACK_TEXT = "Support is slow right now. Try the docs, or retry."


    async def answer_with_budget(message: str, budget_s: float = 0.05) -> ChatResponse:
        start = time.time()
        try:
            text = await asyncio.wait_for(slow_chain.ainvoke({"message": message}), timeout=budget_s)
            fallback = False
        except asyncio.TimeoutError:
            text, fallback = FALLBACK_TEXT, True
        return ChatResponse(response=text, latency_ms=(time.time() - start) * 1000, fallback=fallback)


    print(asyncio.run(answer_with_budget("where is my refund?")))
    ```

## 3. FIFO cache

Put `FifoCache(max_size=2)` in front of identical questions.

**Checks:**

- Second identical question is a cache hit (`cached is True` or tokens 0)
- Inserting a third distinct key evicts the first (FIFO, not “LRU” unless you implemented LRU)

??? tip "Hint 1 — a nudge"
    A plain `dict` remembers insertion order. What's the oldest key, and how do you get it without scanning?

??? tip "Hint 2 — the approach"
    Copy the lesson's `FifoCache`. In front of the chain: `get` first; on a miss, call the chain and `set`. Return `cached` in the response so the test can see it. Then insert three distinct keys and look up the first.

??? example "Hint 3 — most of the code"
    ```python
    class FifoCache:
        def __init__(self, max_size: int = 100):
            self.max_size, self._data = max_size, {}

        def get(self, key: str):
            return self._data.get(key)

        def set(self, key: str, value: str) -> None:
            if key not in self._data and len(self._data) >= self.max_size:
                self._data.pop(next(iter(self._data)))
            self._data[key] = value


    cache = FifoCache(max_size=2)


    def cached_answer(question: str) -> dict:
        hit = cache.get(question)
        if hit is not None:
            return {"response": hit, "cached": True}
        text = chain.invoke({"message": question})
        cache.set(question, text)
        return {"response": text, "cached": False}


    for q in ("reset password", "reset password", "rotate key", "export csv", "reset password"):
        print(q, cached_answer(q)["cached"])
    ```

Do not write a cloud deployment guide. A golden file (week 7) still beats this Dockerfile.

## Expected observation

??? success "Open after you run"
    Health returns `{"ok": True}`. Timeout path sets `fallback is True` with no dollar amount. FIFO size 2 evicts key 1 when key 3 arrives.

## Self-check

No cloud deploy, no 100 concurrent users. `import time` is actually used. A golden file still beats a Dockerfile.
