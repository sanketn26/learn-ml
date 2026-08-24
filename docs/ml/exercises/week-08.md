# Exercises — Week 8 — Labels Lie

Do these after reading [Week 8](../week-08.md).

**0. Predict first.** About 6.4% of customers ever cancel (lifetime). Before running anything: a model trained on the horizon-30 label instead sees roughly 48 positives in the whole file. If you raise the classification threshold from 0.5 to 0.8 on that model, which direction do precision and recall move, and why does a rare positive class make that swing sharper than it would on a 50/50 label? Write your guess, then check it against exercise 3.

**1. Two rates.** On `as_of=2024-06-01`, print horizon-30 churn rate vs lifetime `is_churned` on the same at-risk people. Which one is legal at score time?

**2. Censoring.** Call `label_churn_in_horizon` with `observation_end=as_of + 10 days` and `horizon_days=30`. Rows we have not watched through the horizon become NaN — that is censoring, per row. A cancel you already saw inside those 10 days stays **1**. How many labels are NaN, and how many observed cancels survived? (A Saturday signup with a fully observed window is noisy, not censored.)

**3. PR vs ROC.** Train the small GBT from the lesson on **eventual** labels (`label_eventual_churn`). Print ROC-AUC, PR-AUC, dummy PR-AUC, precision@80. Horizon-30 has ~48 positives in the whole file — precision@80 there is a lottery. Which number would you put in the Monday email?

**4. Forbidden.** `pytest tests/test_labels.py tests/test_contract.py`. Then try `validate({..., "churn_date": "2024-07-01"})` and show it raises.

**5. Calibration glance.** Draw the reliability curve. One sentence: would you let finance treat the score as a probability?
