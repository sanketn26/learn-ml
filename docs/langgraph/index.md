# LangGraph

Five weeks on LLM work as a **state machine**: nodes are functions, edges are control flow, state is the request-scoped object you already thread through a saga.

Start this track only if state machines, retries, persistence, idempotency, and the [course prerequisites](../getting-started.md#this-is-not-beginner-study-material) are familiar.

**Scope:** get the state-machine idea, build a resumable graph, then use the official docs to go further. Read the [framework track guide](../framework-tracks.md) before starting.

Use a graph when you need branches, retries, pause-for-a-human, or replay. Three sequential calls do not need a graph.

| Week | Idea | You are done when… |
|---|---|---|
| [1 — Graphs](week-01.md) | `StateGraph`, reducers, a tiny linear flow. | Two inputs take different visible paths through the graph. |
| [2 — Workflows](week-02.md) | Conditional edges, fan-out, subgraphs. | Parallel branches join into one validated state. |
| [3 — Persistence](week-03.md) | Checkpoints. Replay after a crash. | A failed run resumes without repeating completed pure work. |
| [4 — Human in the loop](week-04.md) | Interrupt, resume, the approval node. | Approve, reject, and needs-info follow distinct tested paths. |
| [5 — Don’t charge twice](week-05.md) | Resume is at-least-once. Keys make effects once. | Replaying the side-effect node produces one charge. |

!!! think "Think of it like… CI, Redux, or a workflow engine"
    The runtime owns the state so you can stop in the middle and start again. That is the product feature. Pretty diagrams are a side effect.

[Week 1 →](week-01.md){ .md-button .md-button--primary }
