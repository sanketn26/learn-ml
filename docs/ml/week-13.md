# Week 13 — Ensembles: A Room of Reviewers

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have run a design review or a CI matrix. Same idea: one opinion is brittle.

---

## 🎯 What you will be able to do

- Separate **bagging**, **boosting**, **voting**, and **stacking** (they are not synonyms)
- Default to a forest / gradient-boosted trees for tabular SaaS data
- Read a learning-rate × n_estimators heatmap
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

Your Customer 360 is a spreadsheet: mixed types, missing values, no spatial structure. Gradient-boosted trees are the default for that shape the way Postgres is the default for a relational app. Neural nets (next week) win on images, text, and sequences — not on 15 numeric columns.

### Picture the ops cost

A 500-tree booster that is 0.4% better than a 80-tree one is a worse product if you now need 200ms extra on `/predict` and a 2 GB pickle. Measure the committee against a single good tree and against a linear model. Ship the simplest one that beats the baseline by enough to change a staffing decision.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all 50k rows.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the shared style kit importable from the repo root

from pathlib import Path
import sys
from lib.course_data import find_data_dir

DATA = find_data_dir()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
```

## Picture the two committees

```
BAGGING                         BOOSTING
 data ─┬─► tree ─┐               data ─► tree1 ─ misses ─► tree2 ─ misses ─► tree3
       ├─► tree ─┼─ vote                           ↘ add ↗         ↘ add ↗
       └─► tree ─┘               final = tree1 + tree2 + tree3
```

!!! engineer "Engineer mental model"

    For CloudWave-sized *tables* (thousands to hundreds of thousands of rows, mixed numbers + categories), **gradient-boosted trees are the default workhorse** — XGBoost / LightGBM / sklearn’s GBT. Neural nets start to win on images, text, and sequences, not on a 7-column billing table.

```python
df = load_customer_360(DATA)
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "n_support"]
X = df[numeric + ["plan_type"]]
y = df["is_churned"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]),
])

def auc_of(model):
    p = Pipeline([("prep", prep), ("m", model)])
    p.fit(X_train, y_train)
    return p, roc_auc_score(y_test, p.predict_proba(X_test)[:, 1])

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "forest (bagging)": RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2),
    "gbt (boosting)": GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=42),
}
fitted = {}
print(f"{'model':<22} AUC")
for name, m in models.items():
    pipe, auc = auc_of(m)
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
vote_pipe, vote_auc = auc_of(vote)
print(f"soft voting AUC: {vote_auc:.3f}")
print("A 0.002 lift that costs 3× latency is usually not a win.")
```

## The only hyperparameter picture you need this week

`learning_rate` is how hard each new tree is allowed to shove the answer. More trees + smaller steps ≈ same work, often stabler. There is no magic pair — there is a ridge on a heatmap.

```python
rates = [0.05, 0.15]
trees = [20, 40]
grid = np.zeros((len(rates), len(trees)))
for i, lr in enumerate(rates):
    for j, n in enumerate(trees):
        _, grid[i, j] = auc_of(
            GradientBoostingClassifier(learning_rate=lr, n_estimators=n,
                                       max_depth=2, random_state=42)
        )

fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(grid, cmap="YlGn", vmin=grid.min() - 0.005, vmax=grid.max())
ax.set_xticks(range(len(trees)), trees)
ax.set_yticks(range(len(rates)), rates)
ax.set_xlabel("n_estimators"); ax.set_ylabel("learning_rate")
ax.set_title("Holdout AUC — look for a plateau, not a spike")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
```

## Bias–variance on purpose

Bagging (a forest) is a **variance reducer**: many jittery trees, averaged. Boosting is a **bias reducer**: each tree hunts what the last one still misses — and will overfit if you let it run forever.

The diagnostic is always the same pair of curves.

```python
from sklearn.model_selection import validation_curve

# 6k-row picture is enough to see the two curves; a 50k × 12-depth CV is a coffee break
sample = np.random.default_rng(0).choice(len(X), size=min(2500, len(X)), replace=False)
depths = np.arange(1, 8)
train_s, test_s = validation_curve(
    RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=2),
    prep.fit_transform(X.iloc[sample]), y.iloc[sample],
    param_name="max_depth", param_range=depths,
    cv=2, scoring="roc_auc", n_jobs=2,
)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(depths, train_s.mean(axis=1), marker="o", label="train AUC", color="#1d4ed8")
ax.plot(depths, test_s.mean(axis=1), marker="o", label="holdout AUC", color="#b45309")
ax.set_xlabel("max_depth (capacity)")
ax.set_ylabel("AUC")
ax.set_title("Left = underfit (both low). Right = overfit (train ↑ holdout ↓)")
ax.legend()
plt.tight_layout()
plt.show()
print("Pick the depth where orange peaks, not where blue is 1.0.")
```

!!! warning "Watch out"

    Boosting will happily memorize noise if trees get deep and many. A validation curve that keeps rising on train and dies on test is not “more learning.” It is a student who memorized last year’s exam.


!!! success "Ship / don’t ship"

    Start with a random forest (forgiving). Move to gradient boosting when you need the last points and can monitor it. Do not stack five models to brag. XGBoost is usually faster than sklearn’s GBT and similar in accuracy — “faster vs more accurate” is the wrong question.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-13.md). Starter: `python exercises/ml/week-13/starter.py` from the repo root.

## 🤔 Reflection

1. Why do diverse models help a vote more than three copies of the same forest?
2. What is the ops cost of an ensemble (latency, pickle size, explainability)?
3. When would you keep logistic regression in production anyway? (regulated audit, need coefficients)

## 🔗 Next week

Neural nets — and an honest answer about whether CloudWave should use one.
