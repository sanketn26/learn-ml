# CrewAI

Four weeks on multi-agent setups as **staffing**: a role is a job description, a tool is an IAM policy, a task is a ticket, a crew is the sprint team.

This is an optional framework track for engineers who already understand LLM calls, tool contracts, orchestration, and the [course prerequisites](../getting-started.md#this-is-not-beginner-study-material). It is not a starting point for programming or AI.

**Scope:** understand roles, tasks, and crews; build a small comparison; then use the official docs to go further. Read the [framework track guide](../framework-tracks.md) before starting.

Add a second agent only when the work actually needs a second role. If ML week 17’s ticket bot is one job, you do not need a crew. This track is optional ceremony on top of LangChain week 7.

| Week | Idea | You are done when… |
|---|---|---|
| [1 — Agents](week-01.md) | Role, goal, tools. One worker, one job. | One bounded role produces an output matching its contract. |
| [2 — Tasks](week-02.md) | Tickets with an output contract. | A downstream task consumes a validated upstream artifact. |
| [3 — Teams](week-03.md) | Sequential vs hierarchical process. | A trace makes delegation and ownership visible. |
| [4 — Scale](week-04.md) | What breaks when you add more workers. | A one-worker baseline is compared with the crew on quality, latency, and calls. |

!!! think "Think of it like… a small team, not a hive mind"
    Clear tickets beat a crowd. “Add more agents” is the same failure mode as “add more microservices.”

[Week 1 →](week-01.md){ .md-button .md-button--primary }
