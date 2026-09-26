---
description: Inspect gradient-boosted feature importances, push a tree ensemble to its depth limits, and name the ensemble method in a worked example.
---

# Exercises — Week 13 — Ensembles: A Room of Reviewers

## What you are building

GBT feature importances next to encoded names, a deep-ensemble train-vs-test run, and a naming review of `VotingClassifier(voting="soft")`.

## Predict before you run

1. Will importances look like a story (usage, activity, MRR) or a shuffle?
2. With `max_depth=8`, `n_estimators=80`, which AUC rises more — train or test?
3. Is soft voting stacking?

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-13/starter.py
```

**1. Feature importance.** From the fitted GBT, print `feature_importances_` next to `named_steps["prep"].get_feature_names_out()`. Is it a story or a random shuffle?

??? tip "Hint 1 — a nudge"
    The model sees the *encoded* matrix, not your DataFrame. How many columns does `plan_type` become after one-hot?

??? tip "Hint 2 — the approach"
    Fit `Pipeline([("prep", make_preprocessor()), ("gbt", GradientBoostingClassifier(...))])` on the lesson's backtest split (`snapshot_split`). `get_feature_names_out()` on the fitted `prep` step gives names in the same order as `feature_importances_`. Put both in a `pd.Series` and sort. Importances are per dummy column, not per original feature.

??? example "Hint 3 — most of the code"
    ```python
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor
    from pipelines.split import snapshot_split

    train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)

    gbt = Pipeline([
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ]).fit(train[FEATURE_COLS], y_train)
    names = gbt.named_steps["prep"].get_feature_names_out()
    print(pd.Series(gbt.named_steps["gbt"].feature_importances_, index=names).sort_values(ascending=False).round(3))
    ```

**2. Push the depth.** `max_depth=8`, `n_estimators=80`. Compare train AUC vs test AUC. Write one sentence about what you see.

??? tip "Hint 1 — a nudge"
    Score the model on the rows it studied and on rows it has never seen. Which of the two numbers is the one Priya's queue will experience?

??? tip "Hint 2 — the approach"
    Same pipeline with `max_depth=8, n_estimators=80`. `roc_auc_score` on `train` and on `test`. Print the shallow model's pair next to it so the gap has a reference.

??? example "Hint 3 — most of the code"
    ```python
    def train_test_auc(model: Pipeline) -> tuple[float, float]:
        tr = roc_auc_score(y_train, model.predict_proba(train[FEATURE_COLS])[:, 1])
        te = roc_auc_score(y_test, model.predict_proba(test[FEATURE_COLS])[:, 1])
        return round(tr, 3), round(te, 3)


    deep = Pipeline([
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=80, max_depth=8, random_state=42)),
    ]).fit(train[FEATURE_COLS], y_train)
    print("shallow train/test AUC:", train_test_auc(gbt))
    print("deep    train/test AUC:", train_test_auc(deep))
    ```

**3. Naming quiz.** A teammate's design doc says “we used a stacking classifier” for `VotingClassifier(voting="soft")`. Review that sentence in one line.

??? tip "Hint 1 — a nudge"
    Both combine several models. Ask what happens to the base models' outputs *after* they're produced — is anything trained on them?

??? tip "Hint 2 — the approach"
    Soft voting averages the base models' probabilities with fixed weights. Stacking trains a *second* model on the base models' (out-of-fold) predictions. Check which one the code actually does, then say it the way you'd leave a review comment.

??? example "Hint 3 — a skeleton"
    ```text
    nit: `VotingClassifier(voting="soft")` is <which method> — it <what it does with probabilities>.
    Stacking would <what stacking adds>. Worth renaming so readers find the right paper.
    ```

## Success criteria

- Importances aligned to feature names.
- Train vs test AUC for the deep ensemble.
- One-sentence naming review.

## After you run

A tree ensemble is a room of reviewers. For CloudWave's table it still beats a net. Naming it wrong in a design doc is how you inherit the wrong paper.

## Lesson link

[Week 13 — Ensembles: A Room of Reviewers](../week-13.md)
