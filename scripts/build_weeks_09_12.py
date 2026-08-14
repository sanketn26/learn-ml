#!/usr/bin/env python3
"""Rebuild Weeks 9–12."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbformat_util import BOOT, LAPTOP_BOX, code_cell, md_cell, write_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"

FEATS = '''
from course_style import load_customer_360
'''


def week9():
    cells = [
        md_cell(
            """# Week 9 — PCA: JPEG for Tables

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have compressed images or used `SELECT` a subset of columns because 200 of them were correlated.

---

## 🎯 What you will be able to do

- Picture PCA as **rotating the cloud** so the first axis points along the stretch
- Read a scree plot: “how many axes until the leftover is noise?”
- Use 2-D PCA as a map, not as a causal feature named “growth”
- Spot odd customers via reconstruction error
- Know when *not* to bother (we only have a handful of columns)

<div class="think-box">
<strong>Think of it like… photographing a dinner plate from above vs from the edge.</strong>
<p>The plate is 3-D. From the edge it looks like a line — you lost the interesting shape. From above you kept the wide part. PCA finds the “from above” angles automatically: the directions where customers differ the most.</p>
<p>Or: <strong>JPEG for a table</strong>. Keep the big blobs of ink, drop the speckle. You cannot read a pixel-perfect original, but you can still tell it is a face (or a whale customer).</p>
</div>
"""
        ),
        code_cell(
            BOOT
            + """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
"""
        ),
        md_cell(
            """## Honesty first: the “curse of dimensionality”

Textbooks scare you with 10,000 columns. CloudWave’s customer table has ~7 numeric fields. PCA will still **draw a useful map**. It will not demonstrate the curse. If you one-hot 200 countries + 500 feature flags, *then* distances die and k-NN / K-Means get dizzy. That is the curse: in high dimensions everyone is far from everyone, so “nearest” stops meaning “similar.”

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>PCA is a rotation + a truncation. The new axes (PC1, PC2, …) are mixes of your old columns. <strong>They are not product features.</strong> Do not tell a PM “we should improve PC3.” Tell them “PC1 is mostly engagement (usage + events + features).”</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "avg_session", "n_support"]
sample = df.sample(6000, random_state=0)
X = StandardScaler().fit_transform(sample[cols])

# Cartoon: a diagonal cloud, then the axes PCA would pick
rng = np.random.default_rng(1)
t = rng.normal(size=400)
cloud = np.c_[t, 0.3 * t + rng.normal(scale=0.25, size=400)]
pca_c = PCA(n_components=2).fit(cloud)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=10, alpha=0.4, c="#64748b")
origin = pca_c.mean_
for vec, name, color in zip(pca_c.components_, ["PC1 — the stretch", "PC2 — the leftover"],
                            ["#dc2626", "#2563eb"]):
    ax.arrow(*origin, *(vec * 2), color=color, width=0.03, head_width=0.15)
    ax.text(*(origin + vec * 2.2), name, color=color, fontsize=9)
ax.set_title("PCA = rotate until axis 1 follows the cigar")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## Scree plot + a 2-D map of CloudWave

<div class="math-box">
<strong>Math, translated</strong>
<p><strong>Explained variance ratio</strong> = “what fraction of the total stretch lives on this axis?” If PC1+PC2 hold 70%, a 2-D scatter still looks like the data. If they hold 20%, you flattened a ball into a pancake and lost the plot.</p>
</div>
"""
        ),
        code_cell(
            """pca = PCA().fit(X)
evr = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(1, len(evr) + 1), evr, color="#6366f1")
axes[0].plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o", color="#b45309")
axes[0].axhline(0.8, ls="--", color="#64748b")
axes[0].set_xlabel("component")
axes[0].set_ylabel("fraction of stretch")
axes[0].set_title("Scree — bars=each axis, line=cumulative")

xy = PCA(n_components=2).fit_transform(X)
plan_codes = sample["plan_type"].astype("category")
scatter = axes[1].scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.35, c=plan_codes.cat.codes, cmap="Set2")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=plt.cm.Set2(i / max(len(plan_codes.cat.categories)-1, 1)),
                      label=cat) for i, cat in enumerate(plan_codes.cat.categories)]
axes[1].legend(handles=handles, title="plan", fontsize=8)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("Same customers, two axes")
plt.tight_layout()
plt.show()

print("Variance kept by first 2 / 3 components:",
      f"{evr[:2].sum():.0%} / {evr[:3].sum():.0%}")
"""
        ),
        md_cell(
            """## Loadings — what *is* PC1 in English?

A **loading** is how much each original column leans on that axis. Large absolute loading → that column is a big ingredient of the axis.
"""
        ),
        code_cell(
            """loadings = pd.DataFrame(pca.components_[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
print(loadings.round(2).to_string())
print("\\nRead the biggest numbers in PC1 and name it in a Slack message.")

# Reconstruction error as 'does not fit the usual recipe'
pca3 = PCA(n_components=3).fit(X)
recon = pca3.inverse_transform(pca3.transform(X))
err = ((X - recon) ** 2).sum(axis=1)
sample = sample.copy()
sample["recon_error"] = err
print("\\nOddballs (high reconstruction error):")
print(sample.nlargest(8, "recon_error")[["user_id", "plan_type", "mrr", "total_usage", "recon_error"]].to_string(index=False))
"""
        ),
        md_cell(
            """<div class="watch-box">
<strong>Watch out</strong>
<p>PCA axes are not causes. An anomaly is “does not compress well,” which might be a new customer type, a data bug, or a whale. Do not auto-ban them.</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Use PCA to <em>look</em> and to compress a wide one-hot jungle before k-NN. Prefer original columns for a churn model you have to explain. Skip the autoencoder / t-SNE flex until you have a picture PCA cannot draw.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Color by churn.** Same 2-D scatter, color = `is_churned`. Do churners own a corner, or are they sprinkled?

**2. How many components?** Pick the smallest k with cumulative variance ≥ 80%. Rebuild the oddball list. Do the same user ids show up?

**3. Do not ship PC3.** Write the one-sentence Slack message you would send instead of “PC3 is important.”

## 🤔 Reflection

1. Why did we scale before PCA? (Otherwise MRR, in dollars, owns the first axis.)
2. If two columns are almost the same (usage and events), what should PC1 do?
3. When is “we reduced to 2-D” a vanity metric?

## 🔗 Next week

Committees of models. Bagging vs boosting — a room of reviewers vs a sequence of specialists hunting the last miss.
"""
        ),
    ]
    write_notebook(OUT / "week-09-saas.ipynb", cells, "Week 9 — PCA")


def week10():
    cells = [
        md_cell(
            """# Week 10 — Ensembles: A Room of Reviewers

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have run a design review or a CI matrix. Same idea: one opinion is brittle.

---

## 🎯 What you will be able to do

- Separate **bagging**, **boosting**, **voting**, and **stacking** (they are not synonyms)
- Default to a forest / gradient-boosted trees for tabular SaaS data
- Read a learning-rate × n_estimators heatmap
- Know when a committee is worth the ops cost

<div class="think-box">
<strong>Think of it like… code review.</strong>
<p><strong>Bagging</strong> (Random Forest): several reviewers read <em>different random pages</em> of the PR and vote. Uncorrelated mistakes cancel. Good at calming a jittery model.<br>
<strong>Boosting</strong> (Gradient Boosting / XGBoost): reviewer 2 is handed only the comments reviewer 1 missed, then reviewer 3 hunts what 2 missed. Great at squeezing the last points. Easier to overfit.<br>
<strong>Voting</strong>: different algorithms (linear + forest + booster) cast a vote tonight.<br>
<strong>Stacking</strong>: a second model learns <em>how to listen</em> to those votes. Not the same as voting.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(
            BOOT
            + """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
"""
        ),
        md_cell(
            """## Picture the two committees

```
BAGGING                         BOOSTING
 data ─┬─► tree ─┐               data ─► tree1 ─ misses ─► tree2 ─ misses ─► tree3
       ├─► tree ─┼─ vote                           ↘ add ↗         ↘ add ↗
       └─► tree ─┘               final = tree1 + tree2 + tree3
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>For CloudWave-sized <em>tables</em> (thousands to hundreds of thousands of rows, mixed numbers + categories), <strong>gradient-boosted trees are the default workhorse</strong> — XGBoost / LightGBM / sklearn’s GBT. Neural nets start to win on images, text, and sequences, not on a 7-column billing table.</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
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
"""
        ),
        md_cell(
            """## Soft voting ≠ stacking

Soft voting averages predicted probabilities. Stacking would train a *meta-model* on those probabilities. We will do voting honestly, and name it correctly.
"""
        ),
        code_cell(
            """vote = VotingClassifier(
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
"""
        ),
        md_cell(
            """## The only hyperparameter picture you need this week

`learning_rate` is how hard each new tree is allowed to shove the answer. More trees + smaller steps ≈ same work, often stabler. There is no magic pair — there is a ridge on a heatmap.
"""
        ),
        code_cell(
            """rates = [0.05, 0.15]
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
"""
        ),
        md_cell(
            """## Bias–variance on purpose

Bagging (a forest) is a **variance reducer**: many jittery trees, averaged. Boosting is a **bias reducer**: each tree hunts what the last one still misses — and will overfit if you let it run forever.

The diagnostic is always the same pair of curves.
"""
        ),
        code_cell(
            """from sklearn.model_selection import validation_curve

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
"""
        ),
        md_cell(
            """<div class="watch-box">
<strong>Watch out</strong>
<p>Boosting will happily memorize noise if trees get deep and many. A validation curve that keeps rising on train and dies on test is not “more learning.” It is a student who memorized last year’s exam.</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Start with a random forest (forgiving). Move to gradient boosting when you need the last points and can monitor it. Do not stack five models to brag. XGBoost is usually faster than sklearn’s GBT and similar in accuracy — “faster vs more accurate” is the wrong question.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Feature importance.** From the fitted GBT, print `feature_importances_` next to the prepared feature names. Is it a story or a random shuffle?

**2. Overfit on purpose.** `max_depth=8`, `n_estimators=80`. Compare train AUC vs test AUC. Write one sentence about what you see.

**3. Naming quiz.** In a design doc, correct a teammate who wrote “we used a stacking classifier” for `VotingClassifier(voting="soft")`.

## 🤔 Reflection

1. Why do diverse models help a vote more than three copies of the same forest?
2. What is the ops cost of an ensemble (latency, pickle size, explainability)?
3. When would you keep logistic regression in production anyway? (regulated audit, need coefficients)

## 🔗 Next week

Neural nets — and an honest answer about whether CloudWave should use one.
"""
        ),
    ]
    write_notebook(OUT / "week-10-saas.ipynb", cells, "Week 10 — Ensembles")


def week11():
    cells = [
        md_cell(
            """# Week 11 — Neural Nets, Without the Mystique

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who keep hearing “just use a network.” This week is permission to **not**, plus a picture of what a net actually is.

We will **not** pretend we taught calculus-level backpropagation.

---

## 🎯 What you will be able to do

- Draw a net as an assembly line of mixers + on/off switches
- See why stacked *linear* layers collapse to one linear layer (so we need activations)
- Regularize with dropout + early stopping, and read a train/val loss curve
- Write the four-line **PyTorch training step**: forward → loss → backward → step
- Decide **GBDT vs net** on a tabular SaaS problem honestly

<div class="think-box">
<strong>Think of it like… an assembly line of mixers.</strong>
<p>Each hidden layer takes the previous numbers, mixes them (weighted sum), then puts each mix through a cheap non-linear switch (ReLU: “if negative, make it 0”). The last mixer outputs a churn score. Training is <em>credit assignment</em>: nudge every weight a tiny bit so tomorrow’s score is less wrong. The library does the calculus (backprop). You do the architecture and the data.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(
            BOOT
            + """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
"""
        ),
        md_cell(
            """## Why activations exist — a 30-second proof

If every layer is just `x → Wx + b`, then two layers are `W2(W1x + b1) + b2`, which is some other `W'x + b'`. You built a logistic regression with extra typing.

ReLU (and friends) break that collapse. That is the whole reason they exist.

```
features ─► mix ─► ReLU ─► mix ─► ReLU ─► mix ─► sigmoid ─► score
              ▲                  ▲
              └── without these switches, the whole tower is one line
```

<div class="math-box">
<strong>Math, translated — backprop in one sentence</strong>
<p>After a batch of examples, we know how wrong the score was. Backprop walks backward through the assembly line and estimates “if I wiggle this weight, does the error go up or down?” Then we wiggle it the helpful way (gradient descent). You will not derive it this week. You will treat it like the compiler: necessary, already written.</p>
</div>
"""
        ),
        code_cell(
            """# Collapse demo: two linear maps == one linear map
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
"""
        ),
        md_cell(
            """## CloudWave bake-off — sklearn MLP vs logistic vs GBT

We use `MLPClassifier` so this week runs **without TensorFlow**. Same idea as Keras `Dense` layers.
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "n_support"]
X = df[numeric + ["plan_type"]]
y = df["is_churned"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# A further val split so we can draw a learning curve
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)
prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]),
])
X_tr_t, X_val_t, X_test_t = prep.fit_transform(X_tr), prep.transform(X_val), prep.transform(X_test)

def report(name, model, Xt=X_tr_t, Xv=X_test_t):
    model.fit(Xt, y_tr if Xt is X_tr_t else y_train)
    # fitted on the matching labels — keep it simple below
    return name, model

results = []
for name, model, xfit, yfit, xeval in [
    ("logreg", LogisticRegression(max_iter=1000), X_tr_t, y_tr, X_test_t),
    ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42), X_tr_t, y_tr, X_test_t),
    ("mlp", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                          max_iter=20, random_state=42), X_tr_t, y_tr, X_test_t),
    ("mlp+dropout-ish", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                          alpha=0.01, max_iter=20, random_state=42), X_tr_t, y_tr, X_test_t),
]:
    model.fit(xfit, yfit)
    auc = roc_auc_score(y_test, model.predict_proba(xeval)[:, 1])
    results.append((name, auc))
    print(f"{name:<18} AUC={auc:.3f}")
"""
        ),
        md_cell(
            """## The one plot a net owes you: train vs validation loss

If train loss keeps falling and val loss turns up, you are memorizing. **Early stopping** = take the checkpoint when val was best. **Dropout** (in Keras; here we use `alpha` L2 as the cousin) = randomly break mixers so no single path can memorize.
"""
        ),
        code_cell(
            """mlp = MLPClassifier(hidden_layer_sizes=(24, 12), activation="relu",
                    max_iter=25, random_state=42, early_stopping=True,
                    validation_fraction=0.2, n_iter_no_change=5)
mlp.fit(prep.fit_transform(X_train), y_train)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(mlp.loss_curve_, label="train loss", color="#1d4ed8")
if hasattr(mlp, "validation_scores_"):
    ax.plot(1 - np.array(mlp.validation_scores_), label="val error (1 − acc)", color="#b45309")
ax.set_xlabel("epoch (one pass over the data)")
ax.set_title("Learning curve — stop when the orange line stops helping")
ax.legend()
plt.tight_layout()
plt.show()

print("Test AUC (early-stopped MLP):",
      f"{roc_auc_score(y_test, mlp.predict_proba(prep.transform(X_test))[:,1]):.3f}")
print("On this table, GBT is usually equal or better. That is the lesson.")
"""
        ),
        md_cell(
            """## PyTorch — NumPy with a tape recorder

Week 0 promised this. A `torch.tensor` is a NumPy array that **remembers the recipe**. `loss.backward()` walks the recipe and fills `.grad` on every weight. `optimizer.step()` nudges the weights the helpful way.

```
batch of rows
    │  forward
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
"""
        ),
        code_cell(
            """try:
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
Xt = torch.tensor(np.asarray(X_tr_t, dtype=np.float32))
yt = torch.tensor(y_tr.to_numpy(), dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(np.asarray(X_test_t, dtype=np.float32))
yv = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

net = nn.Sequential(
    nn.Linear(Xt.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 1),          # one logit; sigmoid lives in the loss
)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

train_losses, val_aucs = [], []
net.train()
for epoch in range(12):
    opt.zero_grad()
    logits = net(Xt)
    loss = loss_fn(logits, yt)
    loss.backward()
    opt.step()
    train_losses.append(float(loss))
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
"""
        ),
        md_cell(
            """<div class="watch-box">
<strong>Watch out</strong>
<p>Forget <code>optimizer.zero_grad()</code> and gradients <em>pile up</em> — the model walks off a cliff. Forget <code>model.eval()</code> later and dropout will stay on at serve time. A 32-16-8 net on 5 columns is still theatre: depth does not invent information that is not in the features.</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p><strong>Tabular SaaS, &lt; ~100k rows, mixed columns → gradient-boosted trees.</strong><br>
<strong>Images, text, audio, long sequences → deep learning.</strong><br>
A net is not “more serious.” It is a different tool. Pick the one you can monitor at 3 a.m.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Linear MLP.** Set `activation="identity"` (no ReLU). Compare AUC to logistic regression. They should rhyme.

**2. Too much net.** `hidden_layer_sizes=(128, 128, 128)` on this data. What happens to train vs test?

**3. Decision memo.** Write five lines to your VP: why CloudWave’s churn model will stay a GBT this quarter.

**4. Break the loop.** Comment out `opt.zero_grad()` and rerun 5 epochs. What happens to the loss? (It should explode or thrash.)

## 🤔 Reflection

1. What problem is the activation function solving, in your own words?
2. Who owns backprop on your team — you, or the library? What do you still own?
3. Name one product surface at CloudWave where a net *would* be the right call (e.g. search ranking on ticket text).
4. In one sentence: what does `loss.backward()` put on each weight?

## 🔗 Next week

Capstone for the *tabular* path. After that: CNNs, RNNs, and Transformers — the architectures you actually mean when you say “deep learning.”
"""
        ),
    ]
    write_notebook(OUT / "week-11-saas.ipynb", cells, "Week 11 — Neural nets")


def week12():
    cells = [
        md_cell(
            """# Week 12 — Capstone: A Training Notebook Is Not Production

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers about to pickle a model and get paged for it.

This is still a **training notebook**. We will be explicit about what it is not: no feature store, no canary, no GDPR review, no CI.

---

## 🎯 What you will be able to do

- Wire Customer 360 → time-based split → GBT → a versioned artifact
- Expose a `predict(payload)` function with a schema check
- Pick a threshold from **CS capacity**, not from 0.5
- Draw a crude drift picture (this month vs train)
- List what you are *not* deploying

<div class="think-box">
<strong>Think of it like… a build artifact + an API contract.</strong>
<p>The model file is a binary. The scaler is part of that binary. The request body is the feature list from Week 5. If any of those three drift independently, production is a silent wrong-number generator.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(
            BOOT
            + """
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             precision_recall_curve)
import pickle
from datetime import datetime
"""
        ),
        md_cell(
            """## Architecture (the only diagram that matters)

```
 warehouse CSVs
      │  nightly job
      ▼
 Customer 360  ──►  time split  ──►  train pipeline  ──►  artifact (model+prep, versioned)
                         │                                      │
                         └── holdout report                     ▼
                                                         POST /predict {user features}
                                                                │
                                                                ▼
                                                         monitor: score volume,
                                                         feature histograms, weekly AUC
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Train on the <em>past</em>, test on the <em>more recent past</em>. Random shuffle is a unit test. A time wall is an integration test against reality.</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "n_support"]
categorical = ["plan_type"]
feature_cols = numeric + categorical
label = "is_churned"

df = df.sort_values("signup_date")
cutoff = df["signup_date"].quantile(0.80)
train_df = df[df["signup_date"] <= cutoff]
test_df = df[df["signup_date"] > cutoff]
print(f"Time wall at {cutoff.date()}  train={len(train_df):,}  test={len(test_df):,}")
print("Train churn rate", train_df[label].mean().round(3),
      "Test churn rate", test_df[label].mean().round(3))

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1,
                                   max_depth=2, random_state=42)
pipe = Pipeline([("prep", prep), ("model", model)])
pipe.fit(train_df[feature_cols], train_df[label])

proba = pipe.predict_proba(test_df[feature_cols])[:, 1]
print(f"Holdout AUC: {roc_auc_score(test_df[label], proba):.3f}")
"""
        ),
        md_cell(
            """## Threshold from a staffing number

CS can call **80** accounts from this test window. We take the 80 highest scores and measure precision. That is the meeting.
"""
        ),
        code_cell(
            """BUDGET = 80
order = np.argsort(-proba)
top = order[:BUDGET]
picked = test_df.iloc[top]
hits = picked[label].sum()
print(f"Calling {BUDGET} highest-risk test users catches {hits} actual churners "
      f"({hits/BUDGET:.0%} precision at this budget).")
print(f"There were {test_df[label].sum()} churners in the window; recall={hits/test_df[label].sum():.0%}.")

# Picture the tradeoff
prec, rec, thr = precision_recall_curve(test_df[label], proba)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(rec, prec, color="#1d4ed8")
ax.set_xlabel("recall (catch rate)")
ax.set_ylabel("precision (when we call, we were right)")
ax.set_title("Precision–recall — pick a point your team can staff")
ax.scatter([hits / test_df[label].sum()], [hits / BUDGET], color="#b91c1c", zorder=5)
ax.annotate("80-call budget", xy=(hits / max(test_df[label].sum(), 1), hits / BUDGET),
            xytext=(0.35, 0.55), textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#b91c1c"})
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## The contract: `predict(payload)`

<div class="watch-box">
<strong>Watch out</strong>
<p>If the API re-implements feature math differently from this notebook, you have two products. One function. Import it from a module in a real repo. Here it lives next to the training so you can see the whole story.</p>
</div>
"""
        ),
        code_cell(
            """REQUIRED = {
    "mrr": (int, float),
    "tenure_days": (int, float),
    "log_usage": (int, float),
    "features_adopted": (int, float),
    "total_events": (int, float),
    "n_support": (int, float),
    "plan_type": (str,),
}

def validate(payload: dict) -> None:
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    for key, types in REQUIRED.items():
        if not isinstance(payload[key], types):
            raise TypeError(f"{key} should be {types}, got {type(payload[key])}")
    if payload["plan_type"] not in {"free", "starter", "pro", "enterprise"}:
        raise ValueError(f"unknown plan_type {payload['plan_type']}")

def predict(payload: dict, threshold: float = 0.35) -> dict:
    validate(payload)
    row = pd.DataFrame([payload])
    score = float(pipe.predict_proba(row[feature_cols])[0, 1])
    return {"churn_score": round(score, 4), "flag_for_cs": score >= threshold}

# Smoke tests
demo = train_df.iloc[0][feature_cols].to_dict()
print("demo payload →", predict(demo))
try:
    predict({"mrr": 10})
except Exception as exc:
    print("schema catch →", type(exc).__name__, exc)
"""
        ),
        md_cell(
            """## Drift — did this month stop looking like train?

We will not implement a full PSI monitor. We will overlay histograms. If the orange fill walks away from the blue, someone should get a Slack.

<div class="math-box">
<strong>Math, translated (optional)</strong>
<p>Population Stability Index is a fancy “how different are two histograms.” If you want a number, compare bin shares: <code>sum( (p − q) * log(p / q) )</code>. If you want a decision, look at the picture first.</p>
</div>
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, col in zip(axes, ["mrr", "log_usage", "tenure_days"]):
    ax.hist(train_df[col], bins=30, density=True, alpha=0.55, label="train", color="#3b82f6")
    ax.hist(test_df[col], bins=30, density=True, alpha=0.55, label="later signups", color="#f59e0b")
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle("If orange leaves blue, the world moved — re-check AUC before celebrating")
plt.tight_layout()
plt.show()

version = datetime.now().strftime("%Y%m%d")
artifact_dir = Path.cwd() / "artifacts"
artifact_dir.mkdir(exist_ok=True)
artifact = artifact_dir / f"cloudwave_churn_{version}.pkl"
with artifact.open("wb") as f:
    pickle.dump({"pipeline": pipe, "features": feature_cols, "trained_through": str(cutoff.date())}, f)
print("Wrote", artifact)
print("Ship the pickle AND the predict()/validate() code AND this notebook's commit hash.")
"""
        ),
        md_cell(
            """## What this notebook is not

| Claimed in many “production” tutorials | Reality here |
|---|---|
| Feature versioning | A Python list in a pickle |
| Error handling | A `validate()` on types |
| Drift detection | Three histograms |
| Retraining triggers | “Look at the histograms + weekly AUC” |
| Deployment | A file on disk |

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>You can ship a <em>batch score</em> from this pipeline: score tonight’s accounts, hand CS a CSV of 80 names. Do not ship a public HTTP API until the contract lives in a tested module, the artifact is in a registry, and someone owns the weekly AUC dashboard.</p>
</div>
"""
        ),
        md_cell(
            """## Course recap (the actual skills)

| Week | How to think about it |
|---|---|
| 1 NumPy | SIMD / SQL on a typed column |
| 2 Pandas | Joins you already know; never explode the grain |
| 3 Charts | Pick the shape that matches the question |
| 4 Stats | “How often would luck look like this?” |
| 5 Features | API contract + a wall against the future |
| 6 Classifiers | Score, then a staffed threshold |
| 7 Regression | Trendline; MAE in real units; no fake CLV |
| 8 Clusters | Pins on a map; personas, not APIs |
| 9 PCA | JPEG / rotate the cloud |
| 10 Ensembles | Reviewers voting vs hunting leftovers |
| 0 Python | Glue language: dataclasses, a `fit`/`predict` class |
| 1 NumPy | SIMD / SQL on a typed column |
| 2 Pandas | Joins you already know; never explode the grain |
| 3 Charts | Pick the shape that matches the question |
| 4 Stats | “How often would luck look like this?” |
| 5 Features | API contract + a wall against the future |
| 6 Classifiers | Score, then a staffed threshold; bias vs variance |
| 7 Regression | Trendline; MAE in real units; no fake CLV |
| 8 Clusters | Pins on a map; personas, not APIs |
| 9 PCA | JPEG / rotate the cloud |
| 10 Ensembles | Reviewers voting vs hunting leftovers |
| 11 Nets + PyTorch | Mixers + the four-line training step |
| 12 Capstone | Artifact + contract + capacity + humility |

## ✍️ Capstone write-up

In one page: (1) the time wall you used, (2) holdout AUC vs a dummy, (3) the 80-call precision, (4) one drift risk, (5) what you refused to over-claim.

## 🔗 Next: the deep-learning track

You can now refuse a leak, demand a baseline, and staff a threshold. Weeks **13–15** are the other half of “I know deep learning”:

- **13 CNNs** — sliding detectors (1-D on usage, 2-D on images)
- **14 RNNs** — a clipboard that walks a sequence
- **15 Transformers** — every token looks at every other token

Then the LangChain / LangGraph / CrewAI tracks in this repo.
"""
        ),
    ]
    write_notebook(OUT / "week-12-saas.ipynb", cells, "Week 12 — Capstone")


if __name__ == "__main__":
    week9()
    week10()
    week11()
    week12()
