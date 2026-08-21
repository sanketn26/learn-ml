# Exercises — Week 7 — Classification: A Score, Then a Threshold

Do these after reading [Week 7 — Classification: A Score, Then a Threshold](../week-07.md).

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

**3. Ablation.** Retrain the forest without `tenure_so_far`. How much AUC dies? (Tenure-so-far is powerful and a little circular-ish for new users — they have not had time to churn. Lifetime `tenure_days` is the real leak; Week 8 kills it.)
