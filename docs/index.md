# Learn ML

Courses for software engineers who want to ship ML and LLM systems without a math degree.

Every idea starts as something you already know — a SQL join, an API contract, a code review, a flaky test — then a picture, then a small piece of Python you can run on a laptop. No GPU. No Jupyter.

<div class="grid cards" markdown>

-   :material-chart-line: **ML Fundamentals**

    ---

    16 weeks. Python → NumPy → Pandas → classical ML → a little PyTorch.

    Analogies first. Formulas only as “math, translated.”

    [Start week 0 →](ml/week-00.md)

-   :material-link-variant: **LangChain**

    ---

    6 weeks. Prompts, memory, agents, RAG, eval, production.

    Treat the framework as middleware, not magic.

    [Start week 1 →](langchain/week-01.md)

-   :material-graph: **LangGraph**

    ---

    4 weeks. State machines, branches, checkpoints, human-in-the-loop.

    A workflow engine you already understand.

    [Start week 1 →](langgraph/week-01.md)

-   :material-account-group: **CrewAI**

    ---

    4 weeks. Roles, tickets, crews.

    Staff a team of agents the way you staff a sprint.

    [Start week 1 →](crewai/week-01.md)

</div>

## How a week is taught

| Box | Meaning |
|---|---|
| **Think of it like…** | Everyday or software analogy. Start here. |
| **If you already write software** | The mapping to APIs, SQL, reviews, CI. |
| **Engineer mental model** | How this shows up in a codebase. |
| **Watch out** | The foot-gun of the week. |
| **Ship / don’t ship** | A decision rule, not theory. |
| **Exercise** | A separate page + a `starter.py` you run in a terminal. |

Lessons are markdown. Exercises are ordinary Python. Read on GitHub Pages, clone the repo when you want to type.

## CloudWave

The ML course uses one fake SaaS company. Same customers all the way through.

| File | Grain | Rows |
|---|---|---|
| `subscriptions.csv` | one customer | 50k |
| `user_events.csv` | one event | 220k |
| `feature_usage.csv` | one user × feature × day | 160k |
| `feedback.json` | one comment (JSON Lines) | 10k |
| `product_catalog.csv` | one feature/product | 300 |

Laptop mode samples ~8k customers so a week finishes in a few minutes on 8 GB RAM.

[How to run the exercises →](getting-started.md){ .md-button .md-button--primary }
[Dataset schemas →](data.md){ .md-button }
