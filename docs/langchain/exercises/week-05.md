# Exercises — Week 5 — Golden set

Do these after reading [Week 5](../week-05.md). Local `Trace` dict only. No LangSmith account.

## 1. Six cases, two scores

Add four cases to the lesson’s two (password, angry billing). Cover: happy path, escalate, unknown, injection (“ignore previous”). Score **accuracy** and **tool_ok** separately from **sla_ok**.

**Checks:**

- A row can have `quality_pass True` and `sla_ok False` (slow but correct)
- You never assign `relevance_score = 1.0 if latency_ok else 0.5`

## 2. Forced failure

Patch the handler so angry billing uses `documentation_search`. Run the suite.

**Checks:**

- The billing case is `quality_pass False` because `tool_ok` is False
- The trace for that case still has a `route` span naming the wrong tool

## 3. Trace dict

Print `trace.steps` for one pass and one fail.

**Checks:**

- Each step is a dict with `"name"`
- You did not require `LANGCHAIN_API_KEY`
