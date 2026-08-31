# Exercise — Week 8 — Labels Lie

## What you are building

Horizon vs lifetime rates, a censoring count, PR-AUC vs ROC-AUC vs precision@80, a forbidden-key `validate()` raise, and a calibration glance.

## Predict before you run

1. About 6.4% ever cancel. Horizon-30 has tens of positives. If you raise the threshold 0.5 → 0.8, which way do precision and recall move, and why is the swing sharper than on a 50/50 label?
2. Which rate is legal at score time?
3. Would you put ROC-AUC or PR-AUC in the Monday email?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-08/starter.py
pytest tests/test_labels.py tests/test_contract.py
```

**0. Predict first.** About 6.4% of customers ever cancel (lifetime). Before running anything: a model trained on the horizon-30 label instead sees roughly 48 positives in the whole file. If you raise the classification threshold from 0.5 to 0.8 on that model, which direction do precision and recall move, and why does a rare positive class make that swing sharper than it would on a 50/50 label? Write your guess, then check it against exercise 3.

**1. Two rates.** On `as_of=2024-06-01`, print horizon-30 churn rate vs lifetime `is_churned` on the same at-risk people. Which one is legal at score time?

**2. Censoring.** Call `label_churn_in_horizon` with `observation_end=as_of + 10 days` and `horizon_days=30`. Rows we have not watched through the horizon become NaN — that is censoring, per row. A cancel you already saw inside those 10 days stays **1**. How many labels are NaN, and how many observed cancels survived? (A Saturday signup with a fully observed window is noisy, not censored.)

**3. PR vs ROC.** Train the small GBT from the lesson on **eventual** labels (`label_eventual_churn`). Print ROC-AUC, PR-AUC, dummy PR-AUC, precision@80. Horizon-30 has ~48 positives in the whole file — precision@80 there is a lottery. Which number would you put in the Monday email?

**4. Forbidden.** `pytest tests/test_labels.py tests/test_contract.py`. Then try `validate({..., "churn_date": "2024-07-01"})` and show it raises.

**5. Calibration glance.** Draw the reliability curve. One sentence: would you let finance treat the score as a probability?

## Success criteria

- Both rates printed; legal one named.
- NaN count and surviving 1s under a short observation window.
- PR-AUC vs dummy, precision@80, pytest green, extra key rejected.

## Debugging clues

- Forgetting `drop_unlabelled` feeds NaNs to sklearn.
- `plan_type` is a string — use `make_preprocessor()`.
- Horizon-30 precision@80 is noise in this fixture.

## After you run

Eventual-after-as_of is the question this file can supervise. Say so in `metrics.json`. Lifetime `churn_date` on a payload is a 400.

## Lesson link

[Week 8 — Labels Lie](../../../docs/ml/week-08.md)
