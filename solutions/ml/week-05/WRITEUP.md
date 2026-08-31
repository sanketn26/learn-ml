# Week 05 — recovery writeup

Lesson: [docs/ml/week-05.md](../../../docs/ml/week-05.md)
Exercise: [docs/ml/exercises/week-05.md](../../../docs/ml/exercises/week-05.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-05/starter.py` first. Write the paid-only
    prediction *before* you run.

## Hint 1

??? tip "Hint 1"

    Chi-squared is for counts in a table (plan × churned). A t-test is for
    two groups and a number (sentiment). Sample-size gut check is a
    simulation: draw two binomials, run the test, ask how often p drops
    under 0.05.

## Hint 2

??? tip "Hint 2"

    `chi2_contingency(pd.crosstab(paid["plan_type"], paid["is_churned"]))`.
    Sentiment: `ttest_ind` on `bug` vs `praise` with `equal_var=False`.
    For n in {100, 400, 1000}, simulate `p1=0.16`, `p2=0.20` a few hundred
    times.

## Debugging clues

??? warning "Debugging clues"

    - Leaving `free` in the table makes the chi-squared "win" for a boring
      reason: free vs paid, not starter vs enterprise.
    - A tiny p-value with a 0.3-point sentiment gap is not a product
      decision — look at the histogram.
    - Equal-n simulation is not the same as CloudWave's actual plan mix.
    - p < 0.05 is a filter, not a launch button.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-05/solution.py
```

```python
chi2, p, dof, expected = chi2_contingency(pd.crosstab(paid.plan_type, paid.is_churned))
```

## Why this decision

Dropping free asks the question CS actually has: "among people who pay us,
does plan still split churn?" If the answer is "barely," you do not staff a
plan-migration program on a p-value. The n=100 / 400 / 1000 loop is so you
feel how a 4-point gap stays invisible until the sample is large.
