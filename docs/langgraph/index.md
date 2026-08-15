# LangGraph

Four weeks on LLM work as a **state machine**: nodes are functions, edges are control flow, state is the request-scoped object you already thread through a saga.

Use a graph when you need branches, retries, pause-for-a-human, or replay. Three sequential calls do not need a graph.

| Week | Idea |
|---|---|
| [1 — Graphs](week-01.md) | `StateGraph`, reducers, a tiny linear flow. |
| [2 — Workflows](week-02.md) | Conditional edges, fan-out, subgraphs. |
| [3 — Persistence](week-03.md) | Checkpoints. Replay after a crash. |
| [4 — Human in the loop](week-04.md) | Interrupt, resume, the approval node. |

!!! think "Think of it like… CI, Redux, or a workflow engine"
    The runtime owns the state so you can stop in the middle and start again. That is the product feature. Pretty diagrams are a side effect.

[Week 1 →](week-01.md){ .md-button .md-button--primary }
