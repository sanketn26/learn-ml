# Week 07 — recovery writeup

Lesson: [docs/ml/week-07.md](../../../docs/ml/week-07.md)
Exercise: [docs/ml/exercises/week-07.md](../../../docs/ml/exercises/week-07.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-07/starter.py` first.

## Hint 1

??? tip "Hint 1"

    Classification is a *score*, then a *staffing decision*. Precision at a
    budget of 100 is "of the 100 names CS can call, how many actually
    churned?" A threshold sweep is that same cut, written as a number
    instead of a headcount. Ablation asks whether tenure-so-far is doing
    too much of the work.

## Hint 2

??? tip "Hint 2"

    Fit the lesson's `RandomForestClassifier` pipeline on
    `build_features`. `np.argsort(-proba)[:100]`. For thresholds
    `np.arange(0.1, 1.0, 0.1)` compute flagged / precision / recall.
    Retrain with `tenure_so_far` dropped from X and compare `roc_auc_score`.

## Debugging clues

??? warning "Debugging clues"

    - Accuracy of a majority dummy looks great (~94%) because churn is rare.
      AUC of that dummy is ~0.5. Do not ship accuracy.
    - Threshold 0.5 is almost never the CS budget. Count *flagged* first.
    - Lifetime `is_churned` is the Week-7 teaching label. Week 8 replaces it.
    - `zero_division=0` on precision when a high threshold flags nobody.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-07/solution.py
```

```python
order = np.argsort(-proba)
top = order[:100]
precision_at_100 = y_test[top].mean()
```

## Why this decision

CS does not have infinite calls. The model ships as "here are 100 names,"
not as "here is a 0.5 cutoff from a textbook." Ablating `tenure_so_far`
shows you how much of the ranking is "new vs old" — useful, a little
circular for brand-new signups, and not the same leak as lifetime
`tenure_days`.
