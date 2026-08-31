# Exercise — Week 13 — Ensembles: A Room of Reviewers

## What you are building

GBT feature importances next to encoded names, an overfit-on-purpose run, and a naming correction for `VotingClassifier(voting="soft")`.

## Predict before you run

1. Will importances look like a story (`tenure_so_far`, usage) or a shuffle?
2. With `max_depth=8`, `n_estimators=80`, which AUC rises more — train or test?
3. Is soft voting stacking?

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-13/starter.py
```

**1. Feature importance.** From the fitted GBT, print `feature_importances_` next to `named_steps["prep"].get_feature_names_out()`. Is it a story or a random shuffle?

**2. Overfit on purpose.** `max_depth=8`, `n_estimators=80`. Compare train AUC vs test AUC. Write one sentence about what you see.

**3. Naming quiz.** In a design doc, correct a teammate who wrote “we used a stacking classifier” for `VotingClassifier(voting="soft")`.

## Success criteria

- Importances aligned to feature names.
- Train vs test AUC for the deep forest.
- One-sentence naming correction.

## Debugging clues

- Importances after one-hot are per dummy, not per original column.
- Train AUC going to 1.0 with a lagging test AUC is the point of exercise 2.
- Stacking trains a second model on predictions; soft voting averages probabilities.

## After you run

A tree ensemble is a room of reviewers. For CloudWave's table it still beats a net. Naming it wrong in a design doc is how you inherit the wrong paper.

## Lesson link

[Week 13 — Ensembles: A Room of Reviewers](../../../docs/ml/week-13.md)
