# Exercises — Week 8 — Labels Lie

Do these after reading [Week 8](../week-08.md).

**1. Two rates.** On `as_of=2024-06-01`, print horizon-30 churn rate vs lifetime `is_churned` on the same at-risk people. Which one is legal at score time?

**2. Censoring.** Call `label_churn_in_horizon` with `observation_end=as_of + 10 days` and `horizon_days=30`. How many labels become NaN? Why?

**3. PR vs ROC.** Train the small GBT from the lesson. Print ROC-AUC, PR-AUC, dummy PR-AUC, precision@80. Which number would you put in the Monday email?

**4. Forbidden.** `pytest tests/test_labels.py tests/test_contract.py`. Then try `validate({..., "churn_date": "2024-07-01"})` and show it raises.

**5. Calibration glance.** Draw the reliability curve. One sentence: would you let finance treat the score as a probability?
