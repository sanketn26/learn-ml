# Exercises — Week 9 — Regression: Predict a Number, Not a Category

## What you are building

A log-target forest, residual MAE by plan, and a forbidden `fake_clv` that you will delete.

## Predict before you run

1. Does `log1p` / `expm1` help MAE on the original dollar scale (whales)?
2. Is the model worse on `free` or `enterprise`?
3. What R² do you expect from `fake_clv = mrr * (tenure_so_far / 30)` using those same two columns as features?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-09/starter.py
```

**1. Log target.** Train the forest on `log1p(y)` and `expm1` the predictions. Does MAE on the original scale improve? (Whales often get kinder.)

**2. Residual slices.** MAE for `free` vs `enterprise`. Where is the model actually bad?

**3. Forbidden target.** Create `fake_clv = mrr * (tenure_so_far / 30)` and train a linear model using `mrr` and `tenure_so_far`. Marvel at R². Then delete it and never do this at work. (Lifetime `tenure_days` is even worse — it already knows who left.)

## Success criteria

- MAE raw vs log-target reported on the original scale.
- Slice MAE for at least two plans.
- Fake CLV R² printed, then the code path that produced it is clearly “do not ship.”

## Debugging clues

- MAE after `expm1` is the number a PM understands; log-space MAE is not.
- A residual trumpet on high MRR is normal; hiding it in an overall MAE is the bug.
- High R² on a constructed target is a tautology.

## After you run

Regression is dollars and leftovers. If the target is a function of the features, you have a calculator, not a model.

## Lesson link

[Week 9 — Regression: Predict a Number, Not a Category](../week-09.md)
