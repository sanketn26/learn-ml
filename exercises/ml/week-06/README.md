# Exercise — Week 6 — Classification: A Score, Then a Threshold

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-06/starter.py
```

## ✍️ Exercises

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

**3. Ablation.** Retrain the forest without `tenure_days`. How much AUC dies? (Tenure is powerful and a little circular — long-lived users have not churned yet.)
