# Exercises — Week 6 — Features Are the Model’s API

Do these after reading [Week 6 — Features Are the Model’s API](../week-06.md).

**1. Honest vs leaked scaler.** Fit a scaler on all rows, then only on train. Print the two means used for `mrr`. How far apart are they? (On 50k rows, tiny — the habit is still wrong.)

**2. Missingness flag.** Add `has_usage = total_usage > 0`. Does churn differ? Would you keep the flag *and* the count?

**3. Contract test.** Write `assert_score_payload(payload: dict)` that checks required keys and types. This is more production-shaped than another transformer.
