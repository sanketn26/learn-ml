# Exercise — Week 8 — Clustering: Sorting Without Labels

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-08/starter.py
```

## ✍️ Exercises

**1. Unscaled K-Means.** Cluster on raw `mrr` + `total_usage` (no scaler). Profile the clusters. Did MRR eat the result?

**2. Name the personas.** From the K=4 table above, write a one-line name and one marketing action per cluster. If two rows get the same name, merge them.

**3. Peek, don’t train.** Churn *rate* by cluster is a story. Training a classifier *on cluster id* is usually weaker than training on the original features — the id is a lossy compression.
