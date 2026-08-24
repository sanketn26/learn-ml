# Exercises — Week 7 — Classification: A Score, Then a Threshold

Do these after reading [Week 7 — Classification: A Score, Then a Threshold](../week-07.md).

**0. Predict first.** CloudWave's lifetime churn is ~6.4%. A classifier that predicts "nobody churns" for every customer — before running the dummy baseline in the lesson, guess its accuracy and its AUC. Then guess: as you sweep the threshold from 0.5 up to 0.8 in exercise 2, which direction do precision and recall move? Write both guesses down before you run anything.

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

**3. Ablation.** Retrain the forest without `tenure_so_far`. How much AUC dies? (Tenure-so-far is powerful and a little circular-ish for new users — they have not had time to churn. Lifetime `tenure_days` is the real leak; Week 8 kills it.)
