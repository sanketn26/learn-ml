# Exercises — Week 4 — RAG (retrieve, then generate)

Do these after reading [Week 4](../week-04.md). Use **keyword overlap** retrieval. Do not treat random hash vectors as semantic search. No `llm.predict(context=..., question=...)`.

```python
from langchain_community.llms import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
```

## 1. Overlap retrieve

Chunk the three CloudWave runbooks from the lesson (API keys, password reset, plans). Implement `retrieve(question, k=2)` with token overlap.

**Checks:**

- `"How do I get an API key?"` returns a hit whose `metadata["source"]` is `api-keys`
- `"export 150k rows timeout"` returns `[]` (or score 0) — no runbook for that

## 2. Chain, not `.predict`

`chain = rag_prompt | llm | StrOutputParser()`. `chain.invoke({"context": ..., "question": ...})`.

**Checks:**

- A miss (`hits == []`) returns `refuse is True` and does not call the llm (or ignores its output)
- A hit includes `doc_ids` from retrieval, not invented ids

## 3. Five labeled queries

Run q1–q5 from the lesson. Print a table: `id, retrieved_ok, refuse, gold_source`.

**Checks:**

- q1–q3: gold source is in the retrieved ids
- q4: retrieval miss and `refuse is True`
- q5: you record a **generation** check (no invented Enterprise discount), separate from retrieval
