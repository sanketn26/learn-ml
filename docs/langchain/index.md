---
description: Seven-week LangChain track treating LLM apps as backend systems, from chains and memory management to RAG, eval, and a ticket bot.
---

# LangChain

Seven weeks on LLM apps as **backend systems**: prompts as templates, tools as functions, memory as a store, RAG as search + a prompt, eval as tests.

This track assumes backend engineering fluency and the [course prerequisites](../getting-started.md#this-is-not-beginner-study-material). It is not a first introduction to Python, HTTP, JSON, testing, or LLMs.

**Scope:** get the idea, build a small working chain, then use the official docs to go further. Read the [framework track guide](../framework-tracks.md) before starting.

The model is a dependency. LangChain is middleware. If you cannot redraw a chain as a sequence of function calls, the abstraction is hiding a bug.

| Week | Idea | You are done when… |
|---|---|---|
| [1 — Chains](week-01.md) | A prompt template is `f-string` + a schema. | Valid input returns the schema; malformed output is rejected. |
| [2 — Memory](week-02.md) | Session store. What to keep, what to drop. | Two sessions remain isolated and old context is deliberately bounded. |
| [3 — Agents](week-03.md) | A loop that picks tools. ReAct is not autonomy. | The trace shows tool selection, an observation, and a bounded failure. |
| [4 — RAG](week-04.md) | Embed, retrieve, then prompt. Search quality first. | A small query set measures retrieval separately from answer quality. |
| [5 — Eval](week-05.md) | You cannot ship what you cannot score. | A golden set produces repeatable pass/fail results. |
| [6 — Production](week-06.md) | Timeouts, fallbacks, tracing—the operational shape. | One request exposes a trace, timeout, fallback, and cost record. |
| [7 — Ticket bot](week-07.md) | Golden file, allowlist, “I don’t know”, cost line. | The supplied golden-file checks pass. |

!!! think "Think of it like… Express / FastAPI middleware"
    The LLM is the slow, flaky downstream service. Your job is the same as always: validate input, bound the work, parse output, log enough to debug the next incident.

[Week 1 →](week-01.md){ .md-button .md-button--primary }
