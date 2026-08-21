# Exercises — Week 13 — Ensembles: A Room of Reviewers

Do these after reading [Week 13 — Ensembles: A Room of Reviewers](../week-13.md).

**1. Feature importance.** From the fitted GBT, print `feature_importances_` next to `named_steps["prep"].get_feature_names_out()`. Is it a story or a random shuffle?

**2. Overfit on purpose.** `max_depth=8`, `n_estimators=80`. Compare train AUC vs test AUC. Write one sentence about what you see.

**3. Naming quiz.** In a design doc, correct a teammate who wrote “we used a stacking classifier” for `VotingClassifier(voting="soft")`.
