---
description: How to approach the LangChain, LangGraph, and CrewAI tracks as engineering introductions, including concept demos versus integration demos.
---

# How to use the framework tracks

LangChain, LangGraph, and CrewAI are **engineering introductions**, not certifications, exhaustive API references, or promises of production readiness. Each track gives an experienced engineer enough intuition to understand the library, build one representative workflow, and continue independently from the official documentation.

Framework APIs change faster than the ideas. Learn the mental model here; use the library's official documentation for the current spelling of an import or constructor.

## Two kinds of example

Every example should be read as one of these:

| Label | Purpose | API key | What it proves |
|---|---|---:|---|
| **Concept demo** | Expose the mechanism with ordinary Python or a fake model | No | Control flow, state, contracts, and failure behavior |
| **Integration demo** | Exercise a real provider and framework version | Usually | Wiring, latency, parsing, cost, and provider failures |

A concept demo is not evidence that a real model is reliable. An integration demo is not a production system.

## The bar for these introductions

The goal is not coverage. By the end of a track, you should be able to:

- explain the library's main abstraction without framework jargon;
- build and modify a small working example;
- inspect the underlying functions, state, and control flow;
- recognize one important failure mode and one reason not to use the library; and
- find the rest in official documentation without needing this course to reproduce it.

That is enough to get started. An experienced engineer is expected to investigate provider choices, advanced APIs, deployment, and organization-specific architecture independently.

## The shape of an introductory lesson

Use the lessons in this order:

1. Name the problem the library solves.
2. Draw its mental model.
3. Run the smallest useful example.
4. Redraw it as ordinary functions, state, and control flow.
5. force one realistic failure;
6. state when the library is the wrong tool; and
7. complete the exercise and its observable checks.

## What “done” means

You are done with a week when the artifact behaves as described in the track overview—not when a snippet merely runs. Prefer assertions, traces, fixtures, and recorded outputs over “looks reasonable.”

The final week of each track is the reference exercise for that introduction:

- **LangChain:** a ticket bot whose structured output, tool allowlist, refusal behavior, and cost can be checked;
- **LangGraph:** a resumable workflow that does not repeat an external side effect; and
- **CrewAI:** a bounded comparison showing whether separate roles improve or cleanly separate the work.

## What these tracks do not cover

They do not make an application production-ready. Production also requires threat modeling, authentication and authorization, secrets management, privacy review, provider limits, timeouts, budgets, observability, evaluation against real traffic, incident response, and an owner.

!!! warning "Version-sensitive code"
    Use Python 3.11 or newer and a separate virtual environment for each framework track (`make setup-frameworks`, `make setup-crewai`). The lessons are written against **LangChain 1.x / LangGraph 1.x** and **CrewAI 1.x** (exact pins in `requirements-frameworks.txt` and `requirements-crewai.txt`). If you meet 0.x code in the wild — `AgentExecutor`, `ConversationBufferMemory`, `langchain_community` fakes, `langgraph.pregel.RetryPolicy` — the [LangChain v1](https://docs.langchain.com/oss/python/migrate/langchain-v1) and [LangGraph v1](https://docs.langchain.com/oss/python/migrate/langgraph-v1) migration guides map it to the spellings used here.

    | You will see (0.x) | Used here (1.x) |
    |---|---|
    | `langchain_community.llms.FakeListLLM` | `langchain_core.language_models.FakeListChatModel` |
    | `AgentExecutor`, `create_react_agent` | `langchain.agents.create_agent` + middleware |
    | `max_iterations` | `ModelCallLimitMiddleware(run_limit=...)` |
    | `ConversationBufferMemory` | a history keyed by session / a checkpointer keyed by `thread_id` |
    | `from langgraph.pregel import RetryPolicy`, `retry=` | `from langgraph.types import RetryPolicy`, `retry_policy=` |
    | `interrupt_before` + `update_state` for approvals | `interrupt()` + `Command(resume=...)` |
    | `MemorySaver` | `InMemorySaver` (same class) |
    | CrewAI hierarchical crew with no manager | `manager_agent=` required at construction |
