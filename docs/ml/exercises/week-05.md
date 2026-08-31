# Exercises — Week 5 — “Is This Real, or Just Noise?”

## What you are building

A paid-only chi-squared, a two-group sentiment t-test, and a sample-size simulation for a 16% vs 20% gap.

## Predict before you run

1. After dropping `free`, is plan still associated with churn at α=0.05?
2. Which test for bug vs praise sentiment, and will the means differ more than the histograms overlap?
3. At n=100 per plan, how often does a 4-point gap produce p<0.05?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-05/starter.py
```

**1. Paid-only chi-squared.** Drop `plan_type == "free"`. Is churn still different across starter / pro / enterprise? Predict the answer before you run it.

**2. Sentiment.** Load `feedback.json` (`lines=True`). Is mean `sentiment_score` different for `category == "bug"` vs `"praise"`? Which test? (Two groups, a number → t-test. Then look at the histogram.)

**3. Sample size gut check.** Keep the 16% vs 20% rates. How many customers per plan (equal n) until a simulation p-value usually drops under 0.05? Try n = 100, 400, 1000.

## Success criteria

- Prediction written *before* the paid-only p-value.
- Test named (chi-squared vs t-test) for each question.
- Three simulation n's with a hit rate.

## Debugging clues

- Leaving free in makes chi-squared win for a boring reason.
- Tiny p + tiny effect ≠ ship.
- Equal-n sims are not CloudWave's actual mix.

## After you run

p < 0.05 is a filter. A ranker is not a lever: plan × churn is observational.

## Lesson link

[Week 5 — “Is This Real, or Just Noise?”](../week-05.md)
