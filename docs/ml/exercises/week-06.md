# Exercises — Week 6 — Features Are the Model’s API

Do these after reading [Week 6 — Features Are the Model’s API](../week-06.md).

**0. Predict first.** Suppose someone adds a feature built from customer behavior recorded *seven days after* the prediction timestamp (`as_of`). Before touching any code: what happens to offline holdout AUC — better, worse, unchanged? What happens to production AUC once that feature is scored on customers who have not lived those seven days yet? Write both guesses, then read the "time machine" picture in [Week 8](../week-08.md) to check the second one.

**1. Honest vs leaked scaler.** Fit a scaler on all rows, then only on train. Print the two means used for `mrr`. How far apart are they? (On the laptop ~8k sample, tiny — the habit is still wrong.)

**2. Missingness flag.** Add `has_usage = total_usage > 0`. Does churn differ? Would you keep the flag *and* the count?

**3. Contract test.** Write `assert_score_payload(payload: dict)` that checks required keys and types. This is more production-shaped than another transformer.
