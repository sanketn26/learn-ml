# Week 7 — Classification: A Score, Then a Threshold

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a spam filter, a linter, or a “risk score.” Same shape.

---

## 🎯 What you will be able to do

- Explain a model as `f(features) → score in [0, 1]`, then a **threshold**
- Always beat a **baseline** (predict “nobody churns”) before celebrating AUC
- Read a confusion matrix in customers, not jargon
- See why precision vs recall is a **staffing** problem
- Glance at a decision tree — the only model you can literally read
- Recognize **underfit (high bias)** vs **overfit (high variance)** on a picture

!!! think "Think of it like… a code-review bot."

    The model does not “know” who will churn. It outputs a risk score, like a linter warning level. You choose the cutoff: flag everything above 0.3 (noisy, catch more) or only above 0.7 (quiet, miss more). The algorithm did not make that product decision. You did.

## If you already write software

A classifier is not a fortune teller. It is a function that returns a **score**, and then *you* pick a **threshold** — exactly like a linter warning level or a WAF rule.

```
features  →  model  →  score in [0, 1]  →  if score >= t: flag
                                              ↑
                                    this is a product decision
                                    not an algorithm decision
```

- Low threshold (0.3): noisy Slack channel, catch more real churn
- High threshold (0.7): quiet channel, miss more real churn

Precision vs recall is a **staffing** problem. High recall means the CS team gets more names. Can they call them? If not, you did not “improve the model.” You created a junk queue.

### Always beat a dummy

The dummy baseline here is “predict nobody churns” (or “predict the majority class”). If your fancy model cannot beat that, you built a weather app that says “today’s weather will be like yesterday” and lost to it.

### Picture underfit vs overfit

```
Underfit (high bias)     a one-line linter that only flags `== null`
                         misses almost everything, stable and useless

Overfit (high variance)  a linter that memorized last Tuesday’s PR
                         perfect on the training set, random on the next one
```

The picture you want: a model that is *slightly* wrong on train and *similarly* wrong on a held-out week. That is generalization. Memorizing the training customers is not intelligence.

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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)
```

## What “logistic regression” actually is

Ignore the word *regression*. This is a classifier.

```
score = sigmoid( w1*mrr + w2*usage + w3*tenure + ... + b )
         └── squash any number into (0, 1), like a probability
```

If the weighted sum is large and positive → score near 1 (likely churn).  
If it is large and negative → score near 0.

!!! math "Math, translated"

    The sigmoid is just a soft on/off switch. You do not need its formula. You need: *weighted sum of features, then squeezed into a probability-like score.*

```python
df = load_customer_360(DATA)
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "n_support"]
categorical = ["plan_type"]
X = df[numeric + categorical]
y = df["is_churned"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

def pipe(model):
    return Pipeline([("prep", prep), ("model", model)])

# Baseline first. Always.
dummy = pipe(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train, y_train)
print(f"Majority-class accuracy: {dummy.score(X_test, y_test):.3f}")
print(f"Majority-class AUC:      {roc_auc_score(y_test, dummy.predict_proba(X_test)[:,1]):.3f}  (0.5 = coin flip on ranking)")
print("If your fancy model cannot beat this, it is not fancy.")
```

## Train three models, read one of them

A shallow decision tree is a flowchart. Random forest is a committee of those flowcharts. Logistic regression is the weighted sum.

```python
logreg = pipe(LogisticRegression(max_iter=1000))
tree = pipe(DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=42))
forest = pipe(RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=2))

rows = []
for name, model in [("logreg", logreg), ("tree", tree), ("forest", forest)]:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "model": name,
        "AUC": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
# plot the raw tree (need the trained DecisionTree inside the pipeline)
ohe_names = list(tree.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical))
plot_tree(tree.named_steps["model"], feature_names=numeric + ohe_names,
          class_names=["stay", "churn"], filled=True, max_depth=3, fontsize=7, ax=ax)
ax.set_title("A 3-level tree — read it like a product flowchart")
plt.tight_layout()
plt.show()
```

## Confusion matrix + threshold slider

At threshold 0.5 the library yells “positive.” Your CS team can call 80 accounts a week. That is the real threshold.

```
                    predicted stay     predicted churn
actually stay       true negative      false alarm      ← wasted CS time
actually churned    miss               catch            ← saved revenue
```

!!! engineer "Engineer mental model"

    Precision = “when we page CS, how often were we right?” Recall = “of everyone who churned, how many did we catch?” You cannot max both at a fixed staffing level. Pick the one that matches the cost of a miss vs a wasted call.

```python
proba = forest.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, thr in zip(axes, [0.2, 0.5, 0.7]):
    pred = (proba >= thr).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, colorbar=False)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    ax.set_title(f"thr={thr}  P={prec:.2f} R={rec:.2f}\nflagged={pred.sum()}")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="#1d4ed8", label=f"forest AUC={roc_auc_score(y_test, proba):.3f}")
ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", label="coin flip AUC=0.50")
ax.set_xlabel("false alarm rate")
ax.set_ylabel("catch rate (recall)")
ax.set_title("ROC: ranking quality, independent of one threshold")
ax.legend()
plt.tight_layout()
plt.show()
```

!!! warning "Watch out"

    - A random split is convenient and slightly dishonest for time-stamped customers. We fix that in Week 15.

    - Never rank “top 20% risk” on the *training* rows and call it a holdout result.

    - 0.5 is not a sacred threshold. It is sklearn’s default because someone had to pick a number.


!!! success "Ship / don’t ship"

    Ship a classifier when it beats the dummy on AUC *and* you have picked a threshold from a capacity number (“CS can call 50/week”). AUC alone does not page anyone.

    CloudWave’s lifetime churn is ~6.7%. Accuracy is a trap and a 0.7 score is not “70% chance.” [Week 8](week-08.md) is labels, PR-AUC, and calibration. [Week 11](week-11.md) is the list CS actually uses.

## Overfitting, bias, and variance — the three words on every ML interview

A model can fail in two opposite ways:

```
UNDERFIT (high bias)              OVERFIT (high variance)
a line through a curve            a scribble through every point
too simple — misses the shape     too clingy — memorizes noise
train error HIGH                  train error TINY
test error HIGH                   test error HIGH  ← the tell
```

!!! think "Think of it like… studying for an exam."

    **Bias** is showing up with only one idea (“everyone churns if they are free”). You are systematically wrong, even on the homework.

    **Variance** is memorizing last year’s answer key, typos included. Homework is perfect. The real exam (new customers) is a mess.

    **Overfitting** is the name for that second failure. **Underfitting** is the first.

```python
# Toy picture: a smooth truth, noisy homework, three students
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 40)
truth = np.sin(2 * np.pi * x)
y = truth + rng.normal(0, 0.18, size=len(x))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, deg, title in [
    (axes[0], 1, "Underfit — high bias"),
    (axes[1], 3, "About right"),
    (axes[2], 14, "Overfit — high variance"),
]:
    coef = np.polyfit(x, y, deg)
    xx = np.linspace(0, 1, 200)
    ax.scatter(x, y, s=12, color="#64748b", label="train points")
    ax.plot(xx, np.sin(2 * np.pi * xx), color="#0f766e", lw=2, label="truth")
    ax.plot(xx, np.polyval(coef, xx), color="#dc2626", lw=2, label=f"poly deg {deg}")
    ax.set_title(title)
    ax.set_ylim(-1.8, 1.8)
axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.show()
print("Same data. Only the model's freedom changed. That freedom is 'capacity.'")
```

!!! math "Math, translated"

    **Bias** ≈ how far the model’s average answer sits from the truth (systematic miss).

    **Variance** ≈ how much the answer would jump if you retrained on a different sample of customers.

    You cannot drive both to zero. A deeper tree / bigger net lowers bias and raises variance. Regularization, more data, and ensembles are how you buy the pair you can live with.


!!! engineer "Engineer mental model"

    Watch *two* curves: train vs holdout. If both are bad → underfit (add features, more capacity). If train is great and holdout is not → overfit (simpler model, more data, regularization). Never tune on the number you will report.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-07.md). Starter: `python exercises/ml/week-07/starter.py` from the repo root.

## 🤔 Reflection

1. Why can accuracy be 93% while the model is useless? (Hint: 6.7% of users churn.)
2. A PM wants “both high precision and high recall.” What resource do they need to give you?
3. Would you rather explain a depth-3 tree or a 150-tree forest to legal?

## 🔗 Next week

Regression: same idea, but the answer is a number (dollars), not a yes/no. We will refuse to predict `mrr × tenure`.
