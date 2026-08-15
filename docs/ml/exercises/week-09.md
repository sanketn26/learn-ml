# Exercises — Week 9 — Regression: Predict a Number, Not a Category

Do these after reading [Week 9 — Regression: Predict a Number, Not a Category](../week-09.md).

**1. Log target.** Train the forest on `log1p(y)` and `expm1` the predictions. Does MAE on the original scale improve? (Whales often get kinder.)

**2. Residual slices.** MAE for `free` vs `enterprise`. Where is the model actually bad?

**3. Forbidden target.** Create `fake_clv = mrr * (tenure_days / 30)` and train a linear model using `mrr` and `tenure_days`. Marvel at R². Then delete it and never do this at work.
