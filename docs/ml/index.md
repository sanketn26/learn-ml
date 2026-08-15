# Applied ML Foundations for SaaS Analytics

Written for engineers. You do not need calculus, linear algebra, or a stats degree.

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

Required job path is **0–17**. Weeks **18–20** are pictures, not how CloudWave ships churn.

## How to take it

1. Read the week on this site. Start at **Think of it like…** and **If you already write software**.
2. Only then look at the code. It is there to prove the picture, not to be copied blindly.
3. Do the [exercise](exercises/week-00.md) in `exercises/ml/week-XX/starter.py`.
4. Answer the reflection prompts in a note to yourself. If you cannot explain the week to another engineer in five minutes, reread the analogy, not the formula.

[Week 0 — Strong Python →](week-00.md){ .md-button .md-button--primary }
