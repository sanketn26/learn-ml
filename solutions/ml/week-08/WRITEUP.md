# Week 08 — recovery writeup

Lesson: [docs/ml/week-08.md](../../../docs/ml/week-08.md)
Exercise: [docs/ml/exercises/week-08.md](../../../docs/ml/exercises/week-08.md)

!!! warning "Do not open `solution.py` until you are stuck after both hints"

    Work in `exercises/ml/week-08/starter.py` first. Write the predict-first
    guess from the exercise page before you train.

## Hint 1

??? tip "Hint 1"

    Lifetime `is_churned` is a souvenir. The legal training label is "did
    they cancel *after* as_of, inside a window we actually watched?"
    Censoring is a NaN per row, not a vibe. ROC-AUC can look fine while
    precision@80 is a lottery — that is why the Monday email is PR-AUC.

## Hint 2

??? tip "Hint 2"

    `label_churn_in_horizon` vs `is_churned` on the same `build_features`
    frame. Pass `observation_end=as_of + 10 days` to see NaNs. Train the
    GBT on `label_eventual_churn` (the 30-day event is too rare here).
    `validate({..., "churn_date": "..."})` must raise. `calibration_curve`
    is a glance, not a certificate.

## Debugging clues

??? warning "Debugging clues"

    - Horizon-30 has tens of positives. Precision@80 on that label is noise.
    - Forgetting `drop_unlabelled` feeds NaNs to sklearn.
    - `plan_type` is a string — use `make_preprocessor()`, not a raw forest.
    - Raising the threshold *increases* precision and *drops* recall, and
      the swing is sharper when positives are rare.

## Reference solution

See [`solution.py`](solution.py). Run:

```bash
python solutions/ml/week-08/solution.py
```

```python
y_h = label_churn_in_horizon(df, as_of)
y_e = label_eventual_churn(df, as_of)
```

## Why this decision

You score people who are still here. Training on "will they ever leave,
including after we stopped watching" is a different product question, and
this fixture only has tens of 30-day events. Eventual-after-as_of is the
label the file can actually supervise — say so in `metrics.json`. Lifetime
`churn_date` on the payload is a 400, not a feature.
