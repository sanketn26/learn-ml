---
hide:
  - toc
---

<div class="course-hero">
  <div class="course-hero__content">
    <span class="course-eyebrow">Analogies first · Then a picture · Then code</span>
    <h1>Ship ML and LLM systems<br><span>without a math degree.</span></h1>
    <p class="course-hero__lead">Courses for working software engineers. Every idea starts as something you already know — a SQL join, an API contract, a code review, a flaky test — then a picture, then a small piece of Python you can run on a laptop. No GPU. No Jupyter.</p>
    <div class="course-actions">
      <a class="course-button course-button--primary" href="ml/week-00/">Start week 0 <span aria-hidden="true">→</span></a>
      <a class="course-button course-button--secondary" href="getting-started/">How to run the exercises</a>
      <a class="course-button course-button--coffee" href="https://buymeacoffee.com/sanketn">☕ Support this course</a>
    </div>
    <p class="course-hero__note">21 weeks of ML fundamentals · 15 weeks across LangChain/LangGraph/CrewAI · Laptop-friendly, samples ~8k rows</p>
  </div>
  <div class="course-terminal" aria-label="Course roadmap">
    <div class="course-terminal__bar"><i></i><i></i><i></i><span>learn-ml / roadmap</span></div>
    <div class="course-terminal__body">
      <p><span class="terminal-muted">00</span> Python as glue, NumPy as a typed column</p>
      <p><span class="terminal-muted">07</span> Classification, regression, ranking</p>
      <p><span class="terminal-muted">16</span> The job pipeline — gate, prod dir, tonight's CSV</p>
      <p><span class="terminal-muted">LC</span> LangChain, LangGraph, CrewAI — optional frameworks</p>
      <div class="terminal-status"><span></span> CloudWave data, one company throughout</div>
    </div>
  </div>
</div>

<div class="course-proof" aria-label="Course overview">
  <div><strong>21</strong><span>Weeks of ML fundamentals</span></div>
  <div><strong>15</strong><span>Weeks of framework tracks</span></div>
  <div><strong>0</strong><span>GPU required (job path)</span></div>
  <div><strong>8k</strong><span>Rows sampled on a laptop</span></div>
</div>

## Pick your track

<div class="grid cards" markdown>

-   :material-chart-line: **ML Fundamentals**

    ---

    21 weeks (0–20). The job is 0–17. Deep learning 18–20 is optional.

    Analogies first. Formulas only as “math, translated.”

    [Start week 0 →](ml/week-00.md)

-   :material-link-variant: **LangChain**

    ---

    7 weeks. Prompts, tools, RAG, a golden file that can fail CI.

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

Before beginning, read [who this course is—and is not—for](getting-started.md#this-is-not-beginner-study-material).
