# LangChain

Six weeks on LLM apps as **backend systems**: prompts as templates, tools as functions, memory as a store, RAG as search + a prompt, eval as tests.

The model is a dependency. LangChain is middleware. If you cannot redraw a chain as a sequence of function calls, the abstraction is hiding a bug.

| Week | Idea |
|---|---|
| [1 — Chains](week-01.md) | A prompt template is `f-string` + a schema. |
| [2 — Memory](week-02.md) | Session store. What to keep, what to drop. |
| [3 — Agents](week-03.md) | A loop that picks tools. ReAct is not autonomy. |
| [4 — RAG](week-04.md) | Embed, retrieve, then prompt. Search quality first. |
| [5 — Eval](week-05.md) | You cannot ship what you cannot score. |
| [6 — Production](week-06.md) | Timeouts, fallbacks, tracing — brochure. Then actually: |
| [7 — Ticket bot](week-07.md) | Golden file, allowlist, “I don’t know”, cost line. |

!!! think "Think of it like… Express / FastAPI middleware"
    The LLM is the slow, flaky downstream service. Your job is the same as always: validate input, bound the work, parse output, log enough to debug the next incident.

[Week 1 →](week-01.md){ .md-button .md-button--primary }
