---
description: Run unscaled K-Means on MRR and usage to see feature scale dominate distance, then name customer personas and profile churn rate by cluster.
---

# Exercises — Week 10 — Clustering: Sorting Without Labels

## What you are building

Unscaled K-Means on MRR + usage, named personas, and a warning not to train a classifier on cluster id.

## Predict before you run

`mrr` is tens-to-hundreds of dollars; `n_support` is 0, 1, 2. If you cluster unscaled, which column dominates distance, and what are the “personas” actually sorted by?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-10/starter.py
```

**0. Predict first.** `mrr` is in dollars (tens to low hundreds); `n_support` is a small integer (0, 1, 2…) — roughly a 100–1000× difference in raw scale. Before running exercise 1: if you cluster on both columns *unscaled*, which one do you expect to dominate the distance calculation, and what will the resulting "personas" actually be sorted by? Write your guess, then run exercise 1 to check.

**1. Unscaled K-Means.** Cluster on raw `mrr` + `total_usage` (no scaler). Profile the clusters. Did MRR eat the result?

**2. Name the personas.** From the K=4 table above, write a one-line name and one marketing action per cluster. If two rows get the same name, merge them.

**3. Peek, don't train.** Churn *rate* by cluster is a story. Training a classifier *on cluster id* is usually weaker than training on the original features — the id is a lossy compression.

## Success criteria

- Unscaled cluster profile shows whether MRR ate the result.
- Four (or fewer) named personas with one action each.
- Written note: cluster id is not an API feature.

## Debugging clues

- Forgetting `StandardScaler` makes dollars the only axis.
- K=4 is a teaching default, not a discovered truth.
- Churn rate by cluster is observational.

## After you run

Personas are unsorted piles with names. They are not a `/predict` contract.

## Lesson link

[Week 10 — Clustering: Sorting Without Labels](../week-10.md)
