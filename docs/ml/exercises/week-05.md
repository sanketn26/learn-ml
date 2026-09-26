---
description: Run a chi-squared test on plan vs churn, a t-test on sentiment scores, and a sample-size simulation to see when p-values under 0.05 are trustworthy.
---

# Exercises — Week 5 — “Is This Real, or Just Noise?”

Marcus still wants to tell Helen the onboarding checklist worked. Before you answer him, check whether plan itself is even associated with churn once you take `free` out of the picture, and get honest about how much data a claim like his actually needs.

## What you are building

A paid-only chi-squared, a two-group sentiment t-test, and a sample-size simulation for a 16% vs 20% gap.

## Predict before you run

1. After dropping `free`, is plan still associated with churn at α=0.05?
2. Which test for bug vs praise sentiment, and will the means differ more than the histograms overlap?
3. At n=100 per plan, how often does a 4-point gap produce p<0.05?

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-05/starter.py
```

**1. Paid-only chi-squared.** Drop `plan_type == "free"`. Is churn still different across starter / pro / enterprise? Predict the answer before you run it.

??? tip "Hint 1 — a nudge"
    Why drop `free` at all? Ask what the test would be "detecting" if one row of the table is wildly unlike the other three.

??? tip "Hint 2 — the approach"
    Chi-squared is for counts in a table: `pd.crosstab(plan_type, is_churned)` on the paid rows, then `scipy.stats.chi2_contingency`. Print the table next to the p-value — the counts are the evidence.

??? example "Hint 3 — most of the code"
    ```python
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency, ttest_ind

    from lib.course_data import find_data_dir

    DATA = find_data_dir()
    subs = pd.read_csv(DATA / "subscriptions.csv")
    paid = subs[subs["plan_type"] != "free"]
    table = pd.crosstab(paid["plan_type"], paid["is_churned"])
    chi2, p, dof, _ = chi2_contingency(table)
    print(table, f"\nchi2={chi2:.1f} dof={dof} p={p:.4f}")
    ```

**2. Sentiment.** Load `feedback.json` (`lines=True`). Is mean `sentiment_score` different for `category == "bug"` vs `"praise"`? Which test?

??? tip "Hint 1 — a nudge"
    Last task compared counts in a table. This one compares a *number* across groups. How many groups, and what kind of value?

??? tip "Hint 2 — the approach"
    Two groups and a continuous number is a t-test: `ttest_ind(bug, praise, equal_var=False)`. Then draw both histograms — a tiny p on a small gap is not a product decision.

??? example "Hint 3 — most of the code"
    ```python
    fb = pd.read_json(DATA / "feedback.json", lines=True)
    bug = fb.loc[fb["category"] == "bug", "sentiment_score"]
    praise = fb.loc[fb["category"] == "praise", "sentiment_score"]
    t, p = ttest_ind(bug, praise, equal_var=False)
    print(f"bug mean={bug.mean():.2f} (n={len(bug)})  praise mean={praise.mean():.2f} (n={len(praise)})  p={p:.2e}")
    ```

**3. Sample size gut check.** Keep the 16% vs 20% rates. How many customers per plan (equal n) until a simulation p-value usually drops under 0.05? Try n = 100, 400, 1000.

??? tip "Hint 1 — a nudge"
    You know the true rates, so you can generate fake experiments. Ask: out of many pretend experiments at size n, how often does the test notice a gap you *know* is real?

??? tip "Hint 2 — the approach"
    For each n, repeat a few hundred times: draw `rng.binomial(1, 0.16, n)` and `rng.binomial(1, 0.20, n)`, build the 2×2 table, run `chi2_contingency`, record `p < 0.05`. The hit rate is your power. Equal n is a simplification — CloudWave's real plan mix is lopsided.

??? example "Hint 3 — most of the code"
    ```python
    rng = np.random.default_rng(0)
    for n in (100, 400, 1000):
        hits = 0
        for _ in range(300):
            a = rng.binomial(1, 0.16, n)
            b = rng.binomial(1, 0.20, n)
            table = [[a.sum(), n - a.sum()], [b.sum(), n - b.sum()]]
            hits += chi2_contingency(table)[1] < 0.05
        print(f"n={n:>5} per plan  p<0.05 in {hits / 300:.0%} of runs")
    ```

## Success criteria

- Prediction written *before* the paid-only p-value.
- Test named (chi-squared vs t-test) for each question.
- Three simulation n's with a hit rate.

## After you run

p < 0.05 is a filter. A ranker is not a lever: plan × churn is observational. That's the caveat Marcus needs before Monday, not after.

## Lesson link

[Week 5 — “Is This Real, or Just Noise?”](../week-05.md)
