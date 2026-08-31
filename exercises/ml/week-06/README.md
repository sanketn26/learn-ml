# Exercise — Week 6 — Features Are the Model's API

## What you are building

An honest vs leaked scaler comparison, a missingness flag, and `assert_score_payload` — a contract test, not another transformer.

## Predict before you run

1. If a feature is built from behavior *seven days after* `as_of`, what happens to offline AUC vs production AUC?
2. Will the all-rows scaler mean for `mrr` differ much from the train-only mean on an 8k sample?
3. Does `has_usage` split churn even if `total_usage` is already in the matrix?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-06/starter.py
```

**0. Predict first.** Suppose someone adds a feature built from customer behavior recorded *seven days after* the prediction timestamp (`as_of`). Before touching any code: what happens to offline holdout AUC — better, worse, unchanged? What happens to production AUC once that feature is scored on customers who have not lived those seven days yet? Write both guesses, then read the "time machine" picture in Week 8 to check the second one.

**1. Honest vs leaked scaler.** Fit a scaler on all rows, then only on train. Print the two means used for `mrr`. How far apart are they? (On the laptop ~8k sample, tiny — the habit is still wrong.)

**2. Missingness flag.** Add `has_usage = total_usage > 0`. Does churn differ? Would you keep the flag *and* the count?

**3. Contract test.** Write `assert_score_payload(payload: dict)` that checks required keys and types. This is more production-shaped than another transformer.

## Success criteria

- Two scaler means printed.
- Churn rates with/without usage.
- Contract test rejects a missing key and a wrong type.

## Debugging clues

- Fitting the scaler on all rows is leakage even when the number barely moves.
- A flag can duplicate the count; keep both only if the *zero vs missing* story differs.
- Do not put `user_id` or `churn_date` in the payload.

## After you run

Features are the model's public API. Week 15's `validate()` is this function with a worse mood.

## Lesson link

[Week 6 — Features Are the Model's API](../../../docs/ml/week-06.md)
