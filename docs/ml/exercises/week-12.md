---
description: Use PCA to build a churn-colored 2-D scatter, pick the smallest k for 80% explained variance, and identify poorly reconstructed outlier rows.
---

# Exercises — Week 12 — PCA: JPEG for Tables

## What you are building

A 2-D scatter colored by churn, a smallest-k-for-80%-variance residual list, and a Slack message that refuses to ship PC3 as “important.”

## Predict before you run

1. Do churners own a corner of the first two PCs, or are they sprinkled?
2. Is a whale along PC1 necessarily a high residual after you keep k components?
3. Does “PC3 explains 8%” mean it is a product lever?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-12/starter.py
```

**1. Color by churn.** Same 2-D scatter, color = `is_churned`. Do churners own a corner, or are they sprinkled?

**2. How many components?** Pick the smallest k with cumulative variance ≥ 80%. Rebuild the high-residual list (observations poorly represented by the retained subspace). Do the same user ids show up? A whale along PC1 may still reconstruct well.

**3. Do not ship PC3.** Write the one-sentence Slack message you would send instead of "PC3 is important." 

## Success criteria

- Scatter interpreted (corner vs sprinkle).
- Smallest k at ≥80% variance and a residual comparison.
- One Slack sentence that does not overclaim PC3.

## Debugging clues

- Scale before PCA or MRR eats PC1.
- High loading ≠ causal.
- Reconstruction error is not churn.

## After you run

PCA is JPEG for a wide table. It rotates the cloud. It does not name a growth lever.

## Lesson link

[Week 12 — PCA: JPEG for Tables](../week-12.md)
