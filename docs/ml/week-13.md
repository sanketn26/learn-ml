---
description: Combine models with bagging, boosting, voting, and stacking ensembles like Random Forest and gradient boosting for tabular data.
---

# Week 13 — Ensembles: A Room of Reviewers

Ana is about to let whatever wins tonight's scoring job into production. Before that happens you owe her a real bake-off — forest vs boosting vs a plain vote, measured against the baseline that actually ships if nothing here beats it. Not "ensembles as a topic." A decision about tonight's list.

??? note "Course details"

    **Course:** Applied ML Foundations for SaaS Analytics
    **Who this is for:** Engineers who have run a design review or a CI matrix. Same idea: one opinion is brittle.

---

## 🎯 What you will be able to do

- Separate **bagging**, **boosting**, **voting**, and **stacking** (they are not synonyms)
- Default to a forest / gradient-boosted trees for tabular SaaS data
- Read a learning-rate × n_estimators heatmap
- Tune on a validation fold, cross-validate across time, and open the test set **once**
- Tell impurity importance from **permutation importance**, and say which one to trust
- Know when a committee is worth the ops cost

!!! think "Think of it like… code review."

    **Bagging** (Random Forest): several reviewers read *different random pages* of the PR and vote. Uncorrelated mistakes cancel. Good at calming a jittery model.

    **Boosting** (Gradient Boosting / XGBoost): reviewer 2 is handed only the comments reviewer 1 missed, then reviewer 3 hunts what 2 missed. Great at squeezing the last points. Easier to overfit.

    **Voting**: different algorithms (linear + forest + booster) cast a vote tonight.

    **Stacking**: a second model learns *how to listen* to those votes. Not the same as voting.

## If you already write software

One reviewer is brittle. Ensembles are a code-review process.

| Ensemble | Review process | Default vibe |
|---|---|---|
| **Bagging** (Random Forest) | Several reviewers read *different random pages* and vote | Calms jitter. Hard to overfit. |
| **Boosting** (GBT / XGBoost) | Reviewer 2 only sees what reviewer 1 missed | Squeezes the last points. Easier to overfit. |
| **Voting** | Different algorithms cast a vote tonight | Cheap committee. |
| **Stacking** | A second model learns *how to listen* to those votes | Extra pipeline. Rarely worth it on the first ship. |

They are not synonyms. Saying “we use an ensemble” is like saying “we do reviews” — which kind?

### Why trees win on SaaS tables

Your Customer 360 is a spreadsheet: mixed types, missing values, no spatial structure. Gradient-boosted trees are the default for that shape the way Postgres is the default for a relational app. Neural nets (next week) win on images, text, and sequences — not on the 7 columns in `FEATURE_COLS`.

### Picture the ops cost

A 500-tree booster that is 0.4% better than an 80-tree one is a worse product if you now need 200ms extra on `/predict` and a 40 MB pickle instead of a 2 MB one. Measure the committee against a single good tree and against a linear model. Ship the simplest one that beats the baseline by enough to change a staffing decision.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all ~49k rows.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.split import snapshot_split
```

## Picture the two committees

```
BAGGING                         BOOSTING
 data ─┬─► tree ─┐               data ─► tree1 ─ misses ─► tree2 ─ misses ─► tree3
       ├─► tree ─┼─ vote                           ↘ add ↗         ↘ add ↗
       └─► tree ─┘               final = tree1 + tree2 + tree3
```

!!! engineer "Engineer mental model"

    For CloudWave-sized *tables* (thousands to hundreds of thousands of rows, mixed numbers + categories), **gradient-boosted trees are the default workhorse** — XGBoost / LightGBM / sklearn’s GBT. Neural nets start to win on images, text, and sequences, not on the 7-column `FEATURE_COLS` table.

## Three sets, two walls

A bake-off is a loop of *choices*: which model, which depth, which learning rate. Every choice you make by looking at a score leaks a little of that score's rows into the model. Week 7 said it in one line — never tune on the number you will report. Here it is in code.

```
as_of − 60d        as_of − 30d           as_of            as_of + 30d
     │                   │                  │                   │
     fit ──────────────► val                                        choose here, as often as you like
                         train ───────────► test                    report here, once
```

The validation fold is the same backtest, one wall earlier. Its labels are all known by `as_of`, so you can compare models on it all afternoon without touching tonight's test rows. Think staging vs prod: you can redeploy to staging fifty times; the prod number is the one you do not get to retry.

```python
train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)  # the backtest from Week 11
val_as_of = AS_OF_DEFAULT - pd.Timedelta(days=30)
fit, y_fit, val, y_val = snapshot_split(val_as_of, horizon_days=30)            # one wall earlier

def val_auc(model):
    p = Pipeline([("prep", make_preprocessor()), ("m", model)])
    p.fit(fit[FEATURE_COLS], y_fit)
    return p, roc_auc_score(y_val, p.predict_proba(val[FEATURE_COLS])[:, 1])

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "forest (bagging)": RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2),
    "gbt (boosting)": GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=42),
}
fitted = {}
print(f"{'model':<22} validation AUC")
for name, m in models.items():
    pipe, auc = val_auc(m)
    fitted[name] = pipe
    print(f"{name:<22} {auc:.3f}")
```

## Soft voting ≠ stacking

Soft voting averages predicted probabilities. Stacking would train a *meta-model* on those probabilities. We will do voting honestly, and name it correctly.

```python
vote = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=2)),
        ("gb", GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)),
    ],
    voting="soft",
)
vote_pipe, vote_auc = val_auc(vote)
print(f"soft voting validation AUC: {vote_auc:.3f}")
print("A 0.002 lift that costs 3× latency is usually not a win.")

gbt = fitted["gbt (boosting)"]
names = gbt.named_steps["prep"].get_feature_names_out()
imp = pd.Series(gbt.named_steps["m"].feature_importances_, index=names)
print(imp.sort_values(ascending=False).round(3).to_string())
```

## Two kinds of “importance”

The `feature_importances_` you just printed is **impurity importance**: how much each column's splits cleaned up the *training* rows. It is cheap and it has two known lies. It rewards columns with many distinct values (a column like `mrr`, with a different value per customer, gets lots of chances to split), and it is measured on the data the model memorized.

**Permutation importance** asks the question you actually care about: *shuffle this one column on held-out rows — how much worse does the list get?* If shuffling `mrr` barely moves validation PR-AUC, the model is not really using it, whatever the split count says.

```python
from sklearn.inspection import permutation_importance

gbt = fitted["gbt (boosting)"]
perm = permutation_importance(
    gbt, val[FEATURE_COLS], y_val,
    scoring="average_precision", n_repeats=5, random_state=0, n_jobs=2,
)
compare = pd.DataFrame({
    "shuffle cost (val PR-AUC)": perm.importances_mean,
    "± sd": perm.importances_std,
}, index=FEATURE_COLS).sort_values("shuffle cost (val PR-AUC)", ascending=False)
print(compare.round(4).to_string())
```

Read the two tables side by side. The top three agree. The tail does not. Impurity importance gives `features_adopted` almost nothing, yet shuffling it costs more validation PR-AUC than shuffling `tenure_so_far`. It also lands close to `n_support`, and the `± sd` column says the two are within run-to-run noise of each other, so do not read their order as a ranking.

The tail is where “can we drop this column from the contract?” gets decided. Decide it with the shuffle test on held-out rows. Split count is not usefulness.

!!! warning "Watch out — importance is not a lever"

    Both kinds answer “what does the *model* lean on,” not “what would change churn.” A column can be important because it is a symptom (low usage) rather than a cause. That is Week 11's warning in a different costume.

## The only hyperparameter picture you need this week

`learning_rate` is how hard each new tree is allowed to shove the answer. More trees + smaller steps ≈ same work, often stabler. There is no magic pair — there is a ridge on a heatmap.

```python
rates = [0.05, 0.15]
trees = [20, 40]
grid = np.zeros((len(rates), len(trees)))
for i, lr in enumerate(rates):
    for j, n in enumerate(trees):
        _, grid[i, j] = val_auc(
            GradientBoostingClassifier(learning_rate=lr, n_estimators=n,
                                       max_depth=2, random_state=42)
        )

fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(grid, cmap="YlGn", vmin=grid.min() - 0.005, vmax=grid.max())
ax.set_xticks(range(len(trees)), trees)
ax.set_yticks(range(len(rates)), rates)
ax.set_xlabel("n_estimators"); ax.set_ylabel("learning_rate")
ax.set_title("Validation AUC — look for a plateau, not a spike")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
```

## One fold is one opinion — cross-validate across time

A single validation fold is one flaky-test run. Cross-validation reruns the same choice on several folds and averages. The textbook version (`KFold`) shuffles rows. On churn data that puts next month's customers in the fit set: the model trains on the future it is being tested on. The honest version slides the `as_of` wall back a month at a time: **rolling-origin backtests**, each one the same shape as production.

```
fold 0   fit 2024-04-02 ──► val 2024-05-02
fold 1   fit 2024-03-03 ──► val 2024-04-02
fold 2   fit 2024-02-02 ──► val 2024-03-03
                                           test 2024-06-01 stays shut
```

`RandomizedSearchCV` takes any list of `(fit_rows, val_rows)` pairs as `cv`, so the folds plug straight into the library. Random search beats a grid for the same budget. Most knobs barely matter; a grid tries every value of those anyway, while random search spends each run on a fresh value of every knob. The `m__` prefix routes a setting to the pipeline step named `"m"` (the model), not the preprocessor.

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV

# Stack every fold's fit and val rows into one table, and remember which row
# numbers belong to which side. That list of (fit_rows, val_rows) is the `cv`.
frames, labels, folds, start = [], [], [], 0
for k in range(3):
    f, yf, v, yv = snapshot_split(val_as_of - pd.Timedelta(days=30 * k), horizon_days=30)
    fit_rows = np.arange(start, start + len(f)); start += len(f)
    val_rows = np.arange(start, start + len(v)); start += len(v)
    frames += [f, v]; labels += [yf, yv]; folds.append((fit_rows, val_rows))
X_folds = pd.concat(frames, ignore_index=True)[FEATURE_COLS]
y_folds = pd.concat(labels, ignore_index=True)

search = RandomizedSearchCV(
    Pipeline([("prep", make_preprocessor()), ("m", GradientBoostingClassifier(random_state=42))]),
    {
        "m__learning_rate": loguniform(0.02, 0.3),
        "m__n_estimators": randint(20, 120),
        "m__max_depth": randint(1, 5),
    },
    n_iter=8, cv=folds, scoring="roc_auc", random_state=0, n_jobs=2,
)
search.fit(X_folds, y_folds)
results = pd.DataFrame(search.cv_results_)
cols = ["param_m__learning_rate", "param_m__n_estimators", "param_m__max_depth",
        "mean_test_score", "std_test_score"]
print(results[cols].sort_values("mean_test_score", ascending=False).round(3).to_string(index=False))
```

Read the table the way you read a benchmark: the top few rows usually sit within one `std_test_score` of each other. That is the plateau from the heatmap, now with error bars. Pick the simplest row on it — fewer, shallower trees — not the one that is 0.001 higher.

## Bias–variance on purpose

Bagging (a forest) is a **variance reducer**: many jittery trees, averaged. Boosting is a **bias reducer**: each tree hunts what the last one still misses — and will overfit if you let it run forever.

The diagnostic is always the same pair of curves.

```python
# 2.5k-row picture is enough to see the two curves; a full-file × 12-depth CV is a coffee break.
# The preprocessor rides inside the pipeline, so each CV split fits its own scaler (Week 6's leak).
sample = np.random.default_rng(0).choice(len(fit), size=min(2500, len(fit)), replace=False)
depths = np.arange(1, 8)
train_s, test_s = validation_curve(
    Pipeline([("prep", make_preprocessor()),
              ("m", RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=2))]),
    fit[FEATURE_COLS].iloc[sample], y_fit.iloc[sample],
    param_name="m__max_depth", param_range=depths,
    cv=2, scoring="roc_auc", n_jobs=2,
)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(depths, train_s.mean(axis=1), marker="o", label="train AUC", color="#1d4ed8")
ax.plot(depths, test_s.mean(axis=1), marker="o", label="validation AUC", color="#b45309")
ax.set_xlabel("max_depth (capacity)")
ax.set_ylabel("AUC")
ax.set_title("Left = underfit (both low). Right = overfit (train ↑ validation ↓)")
ax.legend()
plt.tight_layout()
plt.show()
print("Pick the depth where orange peaks, not where blue is 1.0.")
```

## Open the test set once

Refit the chosen settings on the newest labelled snapshot, then score tonight's test rows — once — next to the untuned default. This is the number that goes in the write-up.

```python
best = {k.removeprefix("m__"): v for k, v in search.best_params_.items()}
candidates = {
    "default gbt": GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42),
    "searched gbt": GradientBoostingClassifier(random_state=42, **best),
}
for name, m in candidates.items():
    p = Pipeline([("prep", make_preprocessor()), ("m", m)]).fit(train[FEATURE_COLS], y_train)
    print(f"{name:<14} test AUC {roc_auc_score(y_test, p.predict_proba(test[FEATURE_COLS])[:, 1]):.3f}")
print("If the search bought less than the seed spread, ship the default and delete the search.")
```

On this table the search usually buys a rounding error. That is a result, not a failure: it tells Ana the model is limited by its seven columns, not its knobs — and that next week's effort goes into features, not a bigger grid.

!!! warning "Watch out"

    Boosting will happily memorize noise if trees get deep and many. A validation curve that keeps rising on train and dies on validation is not “more learning.” It is a student who memorized last year’s exam.

    The quieter leak is *you*. Every model you compare on a score is a tiny fit to that score's rows. Choose on validation folds; the test number is a receipt, not a steering wheel. `KFold` with shuffling on time-stamped customers is a time leak in a library costume.


!!! success "Ship / don’t ship"

    Start with a random forest (forgiving). Move to gradient boosting when you need the last points and can monitor it. Do not stack five models to brag. XGBoost is usually faster than sklearn’s GBT and similar in accuracy — “faster vs more accurate” is the wrong question.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-13.md). Starter: `python exercises/ml/week-13/starter.py` from the repo root.

## 🤔 Reflection

1. Why do diverse models help a vote more than three copies of the same forest?
2. You compared 30 configurations on the test set and reported the best. Why is that number optimistic, and which fold should have taken those 30 looks?
3. What is the ops cost of an ensemble (latency, pickle size, explainability)?
4. When would you keep logistic regression in production anyway? (regulated audit, need coefficients)

## 🔗 Next week

Whichever model wins tonight's bake-off, Marcus will ask if a neural net could do better. [Week 14](week-14.md) answers honestly, on this same table.
