# Exercise — Week 7 — Classification: A Score, Then a Threshold

## What you are building

Precision at a 100-call budget, a threshold sweep, and an ablation that drops `tenure_so_far`.

## Predict before you run

1. CloudWave lifetime churn is ~6.4%. Accuracy of “nobody churns”? AUC of that dummy?
2. As you sweep the threshold 0.5 → 0.8, which way do precision and recall move?
3. How much AUC dies without `tenure_so_far`?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-07/starter.py
```

**0. Predict first.** CloudWave's lifetime churn is ~6.4%. A classifier that predicts "nobody churns" for every customer — before running the dummy baseline in the lesson, guess its accuracy and its AUC. Then guess: as you sweep the threshold from 0.5 up to 0.8 in exercise 2, which direction do precision and recall move? Write both guesses down before you run anything.

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

**3. Ablation.** Retrain the forest without `tenure_so_far`. How much AUC dies? (Tenure-so-far is powerful and a little circular-ish for new users — they have not had time to churn. Lifetime `tenure_days` is the real leak; Week 8 kills it.)

## Success criteria

- Dummy accuracy vs AUC guessed first.
- Precision@100 and a circled threshold with a flagged count CS could staff.
- AUC with and without `tenure_so_far`.

## Debugging clues

- Accuracy of the majority dummy looks great; AUC is ~0.5.
- `zero_division=0` when a high cut flags nobody.
- Week 7 still uses lifetime `is_churned` — Week 8 replaces the label.

## After you run

You ship a list of names, not a textbook 0.5 cutoff. Ranking quality (AUC) and desk precision are different emails.

## Lesson link

[Week 7 — Classification: A Score, Then a Threshold](../../../docs/ml/week-07.md)
