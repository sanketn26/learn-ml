# Exercises — Week 10 — Clustering: Sorting Without Labels

Do these after reading [Week 10 — Clustering: Sorting Without Labels](../week-10.md).

**0. Predict first.** `mrr` is in dollars (tens to low hundreds); `n_support` is a small integer (0, 1, 2…) — roughly a 100–1000× difference in raw scale. Before running exercise 1: if you cluster on both columns *unscaled*, which one do you expect to dominate the distance calculation, and what will the resulting "personas" actually be sorted by? Write your guess, then run exercise 1 to check.

**1. Unscaled K-Means.** Cluster on raw `mrr` + `total_usage` (no scaler). Profile the clusters. Did MRR eat the result?

**2. Name the personas.** From the K=4 table above, write a one-line name and one marketing action per cluster. If two rows get the same name, merge them.

**3. Peek, don’t train.** Churn *rate* by cluster is a story. Training a classifier *on cluster id* is usually weaker than training on the original features — the id is a lossy compression.
