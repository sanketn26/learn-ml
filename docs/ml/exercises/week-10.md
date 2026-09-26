---
description: Run K-Means on raw MRR and usage, then on scaled features, name customer personas, and profile churn rate by cluster.
---

# Exercises — Week 10 — Clustering: Sorting Without Labels

## What you are building

K-Means on raw and scaled MRR + usage, named personas, and a note on what a cluster id is (and is not) for.

## Predict before you run

`mrr` is tens-to-hundreds of dollars; `n_support` is 0, 1, 2. If you cluster unscaled, which column dominates distance, and what are the “personas” actually sorted by?

## Before you start

- K=4 is a teaching default, not a discovered truth.
- Churn rate by cluster is observational: it describes a pile, it does not explain it.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-10/starter.py
```

**0. Predict first.** `mrr` is in dollars (tens to low hundreds); `n_support` is a small integer (0, 1, 2…) — roughly a 100–1000× difference in raw scale. Before running exercise 1: if you cluster on both columns *unscaled*, which one do you expect to dominate the distance calculation, and what will the resulting "personas" actually be sorted by? Write your guess, then run exercise 1 to check.

**1. Raw, then scaled.** Cluster on raw `mrr` + `total_usage` (no scaler) and profile the clusters. Then repeat with `StandardScaler` and profile again. Which column sorted each version?

??? tip "Hint 1 — a nudge"
    K-Means measures straight-line distance. A $100 gap in MRR and a 100-event gap in usage are the same distance to it. Look at each column's spread before you look at any cluster.

??? tip "Hint 2 — the approach"
    `KMeans(n_clusters=4, n_init=10, random_state=42)` on `df[["mrr", "total_usage"]].to_numpy()`, then again on `StandardScaler().fit_transform(...)` of the same columns. Profile each with `groupby("cluster")` medians. The column whose median climbs steadily across clusters is the one doing the sorting.

??? example "Hint 3 — most of the code"
    ```python
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    from lib.course_data import load_customer_360

    df = load_customer_360()
    cols = ["mrr", "total_usage"]
    print(df[cols].std().round(1).to_string(), "\n")


    def profile(X, label: str) -> None:
        clusters = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
        table = df.assign(cluster=clusters).groupby("cluster").agg(
            n=("user_id", "count"), mrr=("mrr", "median"), usage=("total_usage", "median"),
            churn=("is_churned", "mean"),
        )
        print(label, "\n", table.sort_values("mrr").round(2).to_string(), "\n")


    profile(df[cols].to_numpy(), "raw")
    profile(StandardScaler().fit_transform(df[cols]), "scaled")
    ```

**2. Name the personas.** From the K=4 table above, write a one-line name and one marketing action per cluster. If two rows get the same name, merge them.

??? tip "Hint 1 — a nudge"
    A persona name is a PR title for a row of medians. If you can't write one without saying "cluster 2", the row isn't a persona yet.

??? tip "Hint 2 — the approach"
    Read each row's MRR, usage, and adoption against the overall medians — high / low / typical. Two rows that read the same are one persona, and K was too big.

??? example "Hint 3 — a skeleton"
    ```text
    cluster | name (≤4 words)          | one marketing action
    --------+--------------------------+-------------------------------
    0       | <e.g. "Free tinkerers">  | <one concrete action>
    1       |                          |
    2       |                          |
    3       |                          |
    merged? <which rows, and why>
    ```

**3. Peek, don't train.** Churn *rate* by cluster is a story. Training a classifier *on cluster id* is usually weaker than training on the original features — the id is a lossy compression.

??? tip "Hint 1 — a nudge"
    A cluster id takes several columns and squeezes them into one of four labels. What did the squeeze throw away?

??? tip "Hint 2 — the approach"
    Print churn rate per scaled cluster — that's the story you're allowed to tell. Then write down why `cluster` would be a poor column in the `/predict` contract (think: what happens when you refit K-Means next month?).

??? example "Hint 3 — most of the code"
    ```python
    scaled_clusters = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(df[cols]))
    print(df.assign(cluster=scaled_clusters).groupby("cluster")["is_churned"].agg(["count", "mean"]).round(3))
    ```
    The note on why the id is not an API feature is yours.

## Success criteria

- Raw and scaled cluster profiles side by side, with the column that sorted each one named.
- Four (or fewer) named personas with one action each.
- Written note: cluster id is not an API feature.

## After you run

Personas are unsorted piles with names. They are not a `/predict` contract.

## Lesson link

[Week 10 — Clustering: Sorting Without Labels](../week-10.md)
