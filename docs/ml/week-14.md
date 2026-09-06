---
description: Demystify neural networks as layers of weighted sums and activations, covering dropout, early stopping, and a PyTorch training loop.
---

# Week 14 — Neural Nets, Without the Mystique

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who keep hearing “just use a network.” This week is permission to **not**, plus a picture of what a net actually is.

We will **not** pretend we taught calculus-level backpropagation.

---

## 🎯 What you will be able to do

- Draw a net as an assembly line of mixers + on/off switches
- See why stacked *linear* layers collapse to one linear layer (so we need activations)
- Regularize with dropout + early stopping, and read a train/val loss curve
- Write the four-line **PyTorch training step**: zero_grad → forward → backward → step
- Decide **GBDT vs net** on a tabular SaaS problem honestly

!!! think "Think of it like… an assembly line of mixers."

    Each hidden layer takes the previous numbers, mixes them (weighted sum), then puts each mix through a cheap non-linear switch (ReLU: “if negative, make it 0”). The last mixer outputs a churn score. Training is *credit assignment*: nudge every weight a tiny bit so tomorrow’s score is less wrong. The library does the calculus (backprop). You do the architecture and the data.

## If you already write software

A neural net is not magic and not a brain. It is **mixers + switches**, stacked.

```
input numbers
    │
    ▼
linear mix     (weighted sum — the same idea as a spreadsheet SUMPRODUCT)
    │
    ▼
switch         (ReLU: if negative, emit 0; else pass through)
    │
    ▼
linear mix
    │
    ▼
switch
    │
    ▼
one number     (a churn score)
```

Without the switch, stacked mixers collapse into *one* mixer — you paid for depth and got a linear model. ReLU is the cheap non-linearity that makes depth mean something.

### Why a GBT still wins on this table

CloudWave’s scoring row is 7 columns (`FEATURE_COLS`). A gradient-boosted tree will usually beat a small net here, train faster, and be easier to ship. You are learning nets this week so weeks 18–20 (usage-over-time, sequences, text) make sense — not because a net is the right churn model.

### Picture the training loop

```
for each batch of customers:
    optimizer.zero_grad()                   # do not accumulate last batch's blame
    score = model(features)                 # forward
    loss  = how wrong is the score          # the complaint
    loss.backward()                         # fill in .grad on every weight
    optimizer.step()                        # nudge weights to be less wrong
```

That is the whole mystery. The library does the calculus (backprop). You pick the architecture, the data, the loss, and when to stop.

!!! tip "Laptop budget"

    No GPU. Aimed at ~8 GB RAM. Training uses a few thousand sampled customers (or short sequences) so this week should finish in a **few minutes on CPU**. The ideas are the same if you later set `n=None` and train on all ~49k rows.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_eventual_churn
```

## Why activations exist — a 30-second proof

If every layer is just `x → Wx + b`, then two layers are `W2(W1x + b1) + b2`, which is some other `W'x + b'`. You built a logistic regression with extra typing.

ReLU (and friends) break that collapse. That is the whole reason they exist.

```
features ─► mix ─► ReLU ─► mix ─► ReLU ─► mix ─► sigmoid ─► score
              ▲                  ▲
              └── without these switches, the whole tower is one line
```

!!! math "Math, translated — backprop in one sentence"

    After a batch of examples, we know how wrong the score was. Backprop walks backward through the assembly line and estimates “if I wiggle this weight, does the error go up or down?” Then we wiggle it the helpful way (gradient descent). You will not derive it this week. You will treat it like the compiler: necessary, already written.

```python
# Collapse demo: two linear maps == one linear map
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
W1, b1 = rng.normal(size=(3, 4)), rng.normal(size=4)
W2, b2 = rng.normal(size=(4, 2)), rng.normal(size=2)
two_layers = (X @ W1 + b1) @ W2 + b2
W_eq, b_eq = W1 @ W2, b1 @ W2 + b2
one_layer = X @ W_eq + b_eq
print("Max difference between 2 linear layers and 1 equivalent layer:",
      np.max(np.abs(two_layers - one_layer)))
print("That number should be ~0. Activations are what make depth real.")
```

## CloudWave bake-off — sklearn MLP vs logistic vs GBT

`MLPClassifier` is the sklearn stand-in. Real dropout lives in PyTorch (`nn.Dropout`) below. sklearn’s `alpha` is **L2**, not dropout.

```python
df = build_features(as_of=AS_OF_DEFAULT, n=None, at_risk_only=True)
y = label_eventual_churn(df, AS_OF_DEFAULT)
df, y = drop_unlabelled(df, y)
cut = df["signup_date"].quantile(0.80)
train, test = df[df["signup_date"] <= cut], df[df["signup_date"] > cut]
y_train, y_test = y.loc[train.index], y.loc[test.index]
prep = make_preprocessor()
X_train_t = prep.fit_transform(train[FEATURE_COLS])
X_test_t = prep.transform(test[FEATURE_COLS])

for name, model in [
    ("logreg", LogisticRegression(max_iter=1000)),
    ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                          max_iter=20, random_state=42)),
    ("mlp+L2", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                             alpha=0.01, max_iter=20, random_state=42)),
]:
    model.fit(X_train_t, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_t)[:, 1])
    print(f"{name:<18} AUC={auc:.3f}")
```

## The one plot a net owes you: train vs validation loss

If train loss keeps falling and val loss turns up, you are memorizing. **Early stopping** = take the checkpoint when val was best. sklearn’s MLP gives you `loss_curve_` (log-loss). Do not plot it against `1 − accuracy` — different units, different story. **Dropout** is “randomly break mixers”; it is the `nn.Dropout` layer in the PyTorch net below, not `alpha`.

```python
mlp = MLPClassifier(hidden_layer_sizes=(24, 12), activation="relu",
                    max_iter=25, random_state=42, early_stopping=True,
                    validation_fraction=0.2, n_iter_no_change=5)
mlp.fit(X_train_t, y_train)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(mlp.loss_curve_, label="train log-loss", color="#1d4ed8")
ax.set_xlabel("epoch (one pass over the data)")
ax.set_title("Learning curve — stop when this stops helping (early_stopping=True already did)")
ax.legend()
plt.tight_layout()
plt.show()

print("Test AUC (early-stopped MLP):",
      f"{roc_auc_score(y_test, mlp.predict_proba(X_test_t)[:, 1]):.3f}")
print("On this table, GBT is usually equal or better. That is the lesson.")
```

## PyTorch — NumPy with a tape recorder

Week 0 promised this. A `torch.tensor` is a NumPy array that **remembers the recipe**. `loss.backward()` walks the recipe and fills `.grad` on every weight. `optimizer.step()` nudges the weights the helpful way.

```
batch of rows
    │  zero_grad → forward
    ▼
  logits → loss
    │  backward  (the library’s calculus)
    ▼
  .grad on every weight
    │  step
    ▼
  slightly less-wrong weights
```

Install once: `pip install torch` (CPU is enough for this course).

```python
try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch is missing. Install with:  pip install torch") from None

torch.manual_seed(0)
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # 4 + 9 = 13
y.backward()
print("x      ", x.tolist())
print("y      ", float(y))
print("x.grad ", x.grad.tolist(), "  ← d(x1²+x2²)/dx = 2x")

# Same CloudWave table, now as tensors
Xt = torch.tensor(np.asarray(X_train_t, dtype=np.float32))
yt = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(np.asarray(X_test_t, dtype=np.float32))

net = nn.Sequential(
    nn.Linear(Xt.shape[1], 16),
    nn.ReLU(),
    nn.Dropout(0.2),           # the actual dropout; off later with net.eval()
    nn.Linear(16, 1),          # one logit; sigmoid lives in the loss
)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

train_losses, val_aucs = [], []
for epoch in range(12):
    net.train()
    opt.zero_grad()
    logits = net(Xt)
    loss = loss_fn(logits, yt)
    loss.backward()
    opt.step()
    train_losses.append(float(loss))
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(Xv)).numpy().ravel()
        val_aucs.append(roc_auc_score(y_test, scores))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(train_losses, color="#1d4ed8")
axes[0].set_title("PyTorch train loss")
axes[0].set_xlabel("epoch")
axes[1].plot(val_aucs, color="#0f766e")
axes[1].set_title("Holdout AUC while we train")
axes[1].set_xlabel("epoch")
plt.tight_layout()
plt.show()
print(f"Final holdout AUC: {val_aucs[-1]:.3f}")
print("Remember the four calls: zero_grad → forward → backward → step.")
```

!!! warning "Watch out"

    Forget `optimizer.zero_grad()` and gradients *pile up* — the model walks off a cliff. Forget `model.eval()` later and dropout will stay on at serve time. A 32-16-8 net on 5 columns is still theatre: depth does not invent information that is not in the features.


!!! success "Ship / don’t ship"

    **Tabular SaaS, < ~100k rows, mixed columns → gradient-boosted trees.**

    **Images, text, audio, long sequences → deep learning.**

    A net is not “more serious.” It is a different tool. Pick the one you can monitor at 3 a.m.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-14.md). Starter: `python exercises/ml/week-14/starter.py` from the repo root.

## 🤔 Reflection

1. What problem is the activation function solving, in your own words?
2. Who owns backprop on your team — you, or the library? What do you still own?
3. Name one product surface at CloudWave where a net *would* be the right call (e.g. search ranking on ticket text).
4. In one sentence: what does `loss.backward()` put on each weight?

## 🔗 Next week

The pickle: time split, `predict()`, a versioned artifact. After that: the job pipeline, then (optionally) CNNs, RNNs, and Transformers.
