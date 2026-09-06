---
description: A 20-week applied ML course for engineers, using a fake SaaS company's churn data to teach classification, regression, clustering, and deep learning.
---

# Applied ML Foundations for SaaS Analytics

Written for working engineers. You do not need calculus, linear algebra, or a stats degree, but you do need programming fluency. This is an introduction to ML, not an introduction to coding. Check the [course prerequisites](../getting-started.md#this-is-not-beginner-study-material) before Week 0.

CloudWave is a fake B2B SaaS company. You will use the same customers from “what is a Python dataclass” through a nightly scoring job — then, if you want, the pictures behind CNNs and Transformers.

## The path (weeks 0–20)

| Week | What you are really learning |
|---|---|
| 0 | Python as glue. Names vs values. A `fit` / `predict` class. |
| 1 | NumPy as a typed column. Stop looping 160k rows. |
| 2 | Pandas as SQL. Customer 360. The join that explodes. |
| 3 | SQL is the source of truth. `as_of`. Grain tests. |
| 4 | Charts as API responses. Honest axes. |
| 5 | p-values as flaky-test statistics. A ranker is not a lever. |
| 6 | Features as a `/predict` contract. Leakage. PII stays out. |
| 7 | Classification = score + threshold. Staffing, not jargon. |
| 8 | Labels lie. Horizon, censoring, PR-AUC, calibration. |
| 9 | Regression in dollars. Residual trumpets. |
| 10 | Clustering as unsorted piles. Personas, not APIs. |
| 11 | Rank a list. Precision@k. Beat `ORDER BY n_support`. |
| 12 | PCA as JPEG for a wide table. |
| 13 | Ensembles as code review. Trees for tabular data. |
| 14 | Nets as mixers + switches. Why a tree still wins here. |
| 15 | A pickle is not production. Time split, `predict()`, drift. |
| 16 | The job pipeline. Gate. Prod dir. Tonight’s CSV. |
| 17 | On-call. Score as a tool. Golden tickets. |
| 18 | Optional: CNNs as a sliding detector. |
| 19 | Optional: RNNs as a clipboard that walks. |
| 20 | Optional: Transformers as a soft join. |
| Capstone | Optional, needs a GPU: fine-tune a small, reliable coding-tool-use specialist. |

Required job path is **0–17**. Weeks **18–20** are pictures, not how CloudWave ships churn. The [capstone](capstone.md) is further still — the only page in this course that needs a GPU.

!!! warning "Weeks 18–20 are a boundary, not a research lab"

    They build the intuition required to understand modern ML systems. They do **not** teach research-level neural-network or transformer training from scratch.

    After week 20 you should be able to:

    - explain convolution, recurrence, and attention as software analogies
    - know when a tree still wins on SaaS tables
    - read a high-level transformer block diagram

    You should **not** expect to implement FlashAttention, train GPT from scratch, or debug CUDA kernels.

    Next step if you want that path (short list): [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), [fast.ai](https://www.fast.ai/), a standard DL book/course (Goodfellow/Bengio/Courville, or a university intro). Not this syllabus.

After each required week, try one item on the [reasoning self-checks](self-checks.md) page (Predict / Diagnose / Choose / Defend).

## How to take it

The recurring cycle:

**analogy → visual → math → predict → run → compare → explain**

1. Read the week on this site. Start at **Think of it like…** and **If you already write software**.
2. Only then look at the code. It is there to prove the picture, not to be copied blindly.
3. Before you run a block, write a prediction (what will move, what will not, why). Then run it and compare. If you were wrong, name the assumption that failed.
4. Do the [exercise](exercises/week-00.md) in `exercises/ml/week-XX/starter.py`.
5. Answer the reflection prompts in a note to yourself. If you cannot explain the week to another engineer in five minutes, reread the analogy, not the formula.

On weeks that already have a **Before you run this** box, use it as written:

```
## Before you run this
Predict:
1. Which metric will improve?
2. Which metric will worsen?
3. Why?

## Run it
Compare your result with your prediction.

## Explain the difference
If your prediction was wrong, what assumption was wrong?
```

Not every week has “metrics.” Adapt the three questions to the picture (weights, residuals, a curve) — the cycle stays the same.

[Week 0 — Strong Python →](week-00.md){ .md-button .md-button--primary }
