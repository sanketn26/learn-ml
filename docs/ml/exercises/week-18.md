# Exercises — Week 18 — Rank a List

Do these after reading [Week 18](../week-18.md).

**1. Four rankers.** On a time-split holdout, print precision@80 and recall@80 for: the GBT, `n_support`, `-log_usage`, and random. Circle a ship / don’t-ship.

**2. Capacity.** Repeat precision@k for k in `{20, 80, 200}`. What happens to precision as k grows? Write the Slack message to CS if they 4× the budget.

**3. Pre-register.** Write down k *before* you look at the numbers. (You already did: 80.) Changing k after seeing precision is p-hacking. Add a comment in your script that says so.

**4. Causal trap.** In two sentences, reply to: “the model says usage predicts churn, so let’s force people through the tutorial.”
