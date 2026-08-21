# Exercise — Week 9 — Regression: Predict a Number, Not a Category

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-09/starter.py
```

## ✍️ Exercises

**1. Log target.** Train the forest on `log1p(y)` and `expm1` the predictions. Does MAE on the original scale improve? (Whales often get kinder.)

**2. Residual slices.** MAE for `free` vs `enterprise`. Where is the model actually bad?

**3. Forbidden target.** Create `fake_clv = mrr * (tenure_so_far / 30)` and train a linear model using `mrr` and `tenure_so_far`. Marvel at R². Then delete it and never do this at work. (Lifetime `tenure_days` is even worse — it already knows who left.)
