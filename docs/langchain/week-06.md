# Week 6 — Timeouts, fallbacks, a local API

**Course:** LangChain  
**Who this is for:** Engineers who have wrapped a flaky HTTP client with a timeout and a default.

This is an **architectural introduction**, not a production deployment recipe. You will put a chain behind FastAPI, bound the wait, and cache a GET-like question. You will not “hit 1000 RPS” or “99.9%.” Those numbers are not a lesson.

---

## 🎯 What you will be able to do

- Expose `chain.ainvoke` behind a small FastAPI handler (`import time`)
- Time out and fall back to a canned sentence
- Cache exact questions with a **FIFO** map (not LRU unless you implement LRU)
- Sketch a 3.11 Docker image whose healthcheck does not need `curl`
- Prefer a golden file over a container as the first production check

!!! think "Think of it like… wrapping a slow billing client."

    The model is a downstream with tail latency. Your job is the same as always: validate input, cap wait, return *something* the client can parse, log cost. Kubernetes is not that job.

## Picture the handler

```
POST /chat  {message}
    │
    ├─ FIFO cache hit?  →  return cached, tokens = 0
    ├─ wait ≤ timeout for chain.ainvoke(...)
    │     ok   →  cache store, return text
    │     fail →  fallback string, status you choose
    └─ GET /health  →  {"ok": true}
```

Hypothetical CloudWave: this is the ticket-bot shape from weeks 1–5, served locally. Not a cloud vendor comparison.

## FastAPI + `ainvoke`

`agenerate` on a raw string is the wrong call. Use the chain. Import `time`.

```python
import time
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()
llm = FakeListLLM(responses=["Try Settings > Security > Change Password."])
chain = ChatPromptTemplate.from_template("{message}") | llm | StrOutputParser()


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
    try:
        text = await chain.ainvoke({"message": request.message})
        fallback = False
    except Exception:
        text = "Support is slow right now. Try the docs, or retry."
        fallback = True
    return ChatResponse(
        response=text,
        latency_ms=(time.time() - start) * 1000,
        fallback=fallback,
    )
```

Timeout: wrap `ainvoke` in `asyncio.wait_for(..., timeout=2.0)` and treat `TimeoutError` like the `except` path. Concept demo — `FakeListLLM` is instant; the `wait_for` is the contract you are proving.

## FIFO cache (not LRU)

A dict that pops the oldest key is FIFO. Do not call it LRU unless you move-to-front on get.

```python
from typing import Optional

class FifoCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        if key in self._data:
            self._data[key] = value
            return
        if len(self._data) >= self.max_size:
            oldest = next(iter(self._data))
            self._data.pop(oldest)
        self._data[key] = value

cache = FifoCache(max_size=5)
cache.set("How do I reset my password?", "Settings > Security")
assert cache.get("How do I reset my password?") is not None
```

Exact-string cache is a GET cache. “What’s *my* balance?” must not be keyed by the question alone.

## Dockerfile sketch (local)

Python 3.11. Healthcheck with the stdlib — images based on `python:3.11-slim` do not ship `curl`.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run locally if you want. Pushing to a registry is out of scope. There is no Heroku step.

!!! warning "Watch out — a container does not evaluate the bot"

    A green `/health` means the process is up. It does not mean the allowlist still blocks refunds. Week 7’s golden file is the check that matters.

!!! success "Ship / don’t ship"

    **Ship** a local handler with timeout, fallback, FIFO cache on idempotent questions, and a golden-file test. **Don’t ship** “we Dockerized it” as production evidence. A golden file beats a Dockerfile. Don’t quote 1000 RPS / 99.9% as if this week measured them.

## What this week is not

- Not AWS/GCP/Azure. Those are employer-specific.
- Not load testing 100 concurrent users. One timed-out `ainvoke` is the lesson.
- Not week 7. The ticket bot’s firewall still lives in `eval/router.py`.

## ✍️ Exercise

[Exercises](exercises/week-06.md). Local FastAPI + timeout/fallback is enough.

## 🤔 Reflection

1. Which CloudWave questions are safe to FIFO-cache? Which are not?
2. If `/health` is 200 and the golden file is red, are you up?
3. Where does the fallback string get reviewed so it cannot promise a refund?

## 🔗 Next week

The ticket bot: keyword firewall, optional `get_churn_score`, “I don’t know,” a cost line.
