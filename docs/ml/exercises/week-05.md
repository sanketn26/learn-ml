# Exercises — Week 5 — “Is This Real, or Just Noise?”

Do these after reading [Week 5 — “Is This Real, or Just Noise?”](../week-05.md).

**1. Paid-only chi-squared.** Drop `plan_type == "free"`. Is churn still different across starter / pro / enterprise? Predict the answer before you run it.

**2. Sentiment.** Load `feedback.json` (`lines=True`). Is mean `sentiment_score` different for `category == "bug"` vs `"praise"`? Which test? (Two groups, a number → t-test. Then look at the histogram.)

**3. Sample size gut check.** Keep the 16% vs 20% rates. How many customers per plan (equal n) until a simulation p-value usually drops under 0.05? Try n = 100, 400, 1000.
