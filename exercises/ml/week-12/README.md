# Exercise — Week 12 — PCA: JPEG for Tables

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-12/starter.py
```

## ✍️ Exercises

**1. Color by churn.** Same 2-D scatter, color = `is_churned`. Do churners own a corner, or are they sprinkled?

**2. How many components?** Pick the smallest k with cumulative variance ≥ 80%. Rebuild the high-residual list (observations poorly represented by the retained subspace). Do the same user ids show up? A whale along PC1 may still reconstruct well.

**3. Do not ship PC3.** Write the one-sentence Slack message you would send instead of “PC3 is important.”
