# Exercise — Week 11 — Rank a List

## What you are building

Four rankers at precision@80 / recall@80, a capacity sweep, a pre-registered k, and a two-sentence reply to a causal trap.

## Predict before you run

1. Will the GBT beat `ORDER BY n_support` at precision@80?
2. As k grows 20 → 80 → 200, does precision rise or fall?
3. If usage *predicts* churn, does forcing a tutorial *cause* retention?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-11/starter.py
```

**1. Four rankers.** On a time-split holdout, print precision@80 and recall@80 for: the GBT, `n_support`, `-log_usage`, and random. Circle a ship / don't-ship.

**2. Capacity.** Repeat precision@k for k in `{20, 80, 200}`. What happens to precision as k grows? Write the Slack message to CS if they 4× the budget.

**3. Pre-register.** Write down k *before* you look at the numbers. (You already did: 80.) Changing k after seeing precision is p-hacking. Add a comment in your script that says so.

**4. Causal trap.** In two sentences, reply to: "the model says usage predicts churn, so let's force people through the tutorial." 

## Success criteria

- Four rankers, two metrics, one circled ship rule.
- Precision@k table for 20/80/200.
- Comment that k=80 was pre-registered.
- Causal reply in two sentences.

## Debugging clues

- Random split vs time split will flatter the model.
- Sorting by `-log_usage` is a baseline, not a product.
- Changing k after seeing the table is p-hacking.

## After you run

SaaS models are ranked lists. Beat a SQL sort or do not ship. Prediction is not a lever.

## Lesson link

[Week 11 — Rank a List](../../../docs/ml/week-11.md)
