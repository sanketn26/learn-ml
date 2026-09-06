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

## 1. `/health` and `/chat`

Copy the lesson’s FastAPI app. `POST /chat` must `await chain.ainvoke({...})` (not `agenerate` on a string). Import `time`.

**Checks:**

- `GET /health` → `{"ok": True}`
- `POST /chat` with `{"message": "password reset"}` returns a `response` and `latency_ms`
- `import time` is used; `ChatRequest.message` is required

## 2. Timeout → fallback

Wrap `ainvoke` in `asyncio.wait_for(..., timeout=0.05)` and use an LLM (or `RunnableLambda`) that sleeps 0.2s so the wait fires.

**Checks:**

- The response sets `fallback is True`
- The fallback string does not mention a refund or a dollar amount

## 3. FIFO cache

Put `FifoCache(max_size=2)` in front of identical questions.

**Checks:**

- Second identical question is a cache hit (`cached is True` or tokens 0)
- Inserting a third distinct key evicts the first (FIFO, not “LRU” unless you implemented LRU)

Do not write a cloud deployment guide. A golden file (week 7) still beats this Dockerfile.

## Predict before you run

If `ainvoke` sleeps 0.2s and `wait_for` is 0.05s, is `fallback True`? After a third distinct cache key with `max_size=2`, is the first question still a hit?

## Runnable command

```bash
# concept: you can run the timeout/cache without binding :8000
python your_local_handler.py
```

## Expected observation

Health returns `{"ok": True}`. Timeout path sets `fallback is True` with no dollar amount. FIFO size 2 evicts key 1 when key 3 arrives.

## Self-check

No cloud deploy, no 100 concurrent users. `import time` is actually used. A golden file still beats a Dockerfile.
