#!/usr/bin/env python3
"""Rebuild Weeks 5–8."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbformat_util import BOOT, LAPTOP_BOX, code_cell, md_cell, write_notebook

OUT = Path(__file__).resolve().parent.parent / "notebooks"

FEATS = '''
from course_style import load_customer_360
'''


def week5():
    cells = [
        md_cell(
            """# Week 5 — Features Are the Model’s API

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have designed request payloads. Feature engineering is that, plus a timeline rule.

---

## 🎯 What you will be able to do

- Treat a feature vector as a **versioned contract** the training job and the `/predict` handler must share
- Scale numbers so “dollars” and “click counts” can sit in the same model
- One-hot encode `plan_type` without treating free &lt; starter &lt; pro as a number line
- **Fit the scaler on train only** — the leak that will follow you to production
- Draw a wall between “known at score time” and “the future”

<div class="think-box">
<strong>Think of it like… an API contract + a time machine rule.</strong>
<p>The model only sees the JSON you send it. If a field would not exist when you score a live user at noon on Tuesday, it cannot exist in training either. That is leakage: the model cheated on the exam by reading tomorrow’s answer key.</p>
</div>
"""
        ),
        md_cell(LAPTOP_BOX),
        code_cell(BOOT + "\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline"),
        md_cell(
            """## 🏢 Scenario — churn features the scoring service can actually compute

We want to flag accounts that will cancel. At score time we know:

- plan, MRR, tenure so far, usage so far, events so far

We do **not** know `churn_date`. We must not sneak it in as `has_churn_flag`.

```
 timeline:  signup -------- now -------- churn?
                       ▲
                       └── score time. Nothing to the right of this wall
                           may enter X. The label y may look right of the wall.
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Features = request body. Scaler + encoder = middleware that <em>must ship next to the .pkl</em>. If production sends raw dollars and the model expects “standard deviations from the training mean,” every score is garbage and nobody gets a stack trace.</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
print(df.shape)
print(df[["user_id", "plan_type", "mrr", "tenure_days", "total_usage",
          "features_adopted", "total_events", "is_churned"]].head())
print("\\nLabel rate (churned):", df["is_churned"].mean().round(3))
"""
        ),
        md_cell(
            """## Scaling — why trees shrug and linear models panic

`mrr` is 0–500. `total_usage` can be tens of thousands. A linear model / k-means / neural net **adds** these numbers. The big column shouts down the small one.

A tree only asks “is usage &gt; 40?” — units do not matter.

<div class="math-box">
<strong>Math, translated</strong>
<p><code>StandardScaler</code>: subtract the column’s mean, divide by its standard deviation. After that, “1” means “one typical-spread above average,” the same z-score idea from Week 1. <code>log1p(usage)</code> is “compress the whales so they do not own the axis.”</p>
</div>
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].hist(df["total_usage"].clip(upper=np.percentile(df["total_usage"], 99)),
             bins=30, color="#6366f1")
axes[0].set_title("Raw usage — whales squash the axis")

axes[1].hist(df["log_usage"], bins=30, color="#0f766e")
axes[1].set_title("log1p(usage) — readable shape")

# WRONG: scaler fit on everyone. We show it only to picture the shape.
demo = StandardScaler().fit_transform(df[["total_usage"]])
axes[2].hist(demo, bins=30, color="#f59e0b")
axes[2].set_title("StandardScaler(usage) — mean 0, still skewed")
for ax in axes:
    ax.set_ylabel("users")
plt.tight_layout()
plt.show()

print("Trees: raw is fine.  Linear / k-means / nets: log then scale, and fit on TRAIN only.")
"""
        ),
        md_cell(
            """## Categories are not numbers

`plan_type` is free / starter / pro / enterprise. If you map those to 0,1,2,3 you are telling the model “enterprise is three more than free” and “the step from free→starter equals starter→pro.” Sometimes that is true. Usually it is a lie.

**One-hot:** four yes/no columns. Honest, a bit wide.

<div class="watch-box">
<strong>Watch out — the scaler leak</strong>
<p><code>scaler.fit_transform(X)</code> on the <em>full</em> table peeks at the test set’s mean and spread. That is a small leak that becomes a habit. Fit on train. Transform test. In production, the saved scaler <em>is</em> the fit.</p>
</div>
"""
        ),
        code_cell(
            """numeric = ["mrr", "tenure_days", "log_usage", "features_adopted",
           "total_events", "n_devices", "n_support"]
categorical = ["plan_type"]
label = "is_churned"

X = df[numeric + categorical]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Time-based split is even better (Week 12). Stratified random is the honest starter.

prep = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)
prep.fit(X_train)  # train only

X_train_t = prep.transform(X_train)
X_test_t = prep.transform(X_test)
names = numeric + list(prep.named_transformers_["cat"].get_feature_names_out(categorical))

print("Train rows", X_train_t.shape, "Test rows", X_test_t.shape)
print("Feature contract:")
for n in names:
    print(" ", n)

print("\\nScaled train means (numeric should sit near 0):")
print(np.round(X_train_t[:, : len(numeric)].mean(axis=0), 3))
"""
        ),
        md_cell(
            """## Leakage hall of shame (we will keep coming back)

| Looks clever | Why it is cheating |
|---|---|
| `has_churn_flag` as a feature | That **is** the label |
| `lifetime_value = mrr * tenure` as a target, `tenure` as a feature | The model multiplies two columns it was handed |
| Fit scaler / target-encoder on all rows | Test set leaked into preprocessing |
| Random split when the world is a time series | The model trains on “next month” and tests on “last month” |

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>A feature ships if a tired on-call engineer can compute it from <em>today’s</em> warehouses for a single <code>user_id</code> with no peek at the label table. If you cannot write that function, it is not a feature.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Honest vs leaked scaler.** Fit a scaler on all rows, then only on train. Print the two means used for `mrr`. How far apart are they? (On 50k rows, tiny — the habit is still wrong.)

**2. Missingness flag.** Add `has_usage = total_usage > 0`. Does churn differ? Would you keep the flag *and* the count?

**3. Contract test.** Write `assert_score_payload(payload: dict)` that checks required keys and types. This is more production-shaped than another transformer.

## 🤔 Reflection

1. Why is “churned in the next 30 days” a better label than “ever churned”?
2. A teammate one-hot encodes `user_id`. What happens?
3. Where does the scaler live in your repo — next to the model, or re-fit in the API process?

## 🔗 Next week

Classification: a model is a function `features → risk score`. We pick a threshold the sales team can staff.
"""
        ),
    ]
    write_notebook(OUT / "week-05-saas.ipynb", cells, "Week 5 — Features")


def week6():
    cells = [
        md_cell(
            """# Week 6 — Classification: A Score, Then a Threshold

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

<div class="think-box">
<strong>Think of it like… a code-review bot.</strong>
<p>The model does not “know” who will churn. It outputs a risk score, like a linter warning level. You choose the cutoff: flag everything above 0.3 (noisy, catch more) or only above 0.7 (quiet, miss more). The algorithm did not make that product decision. You did.</p>
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)
"""
        ),
        md_cell(
            """## What “logistic regression” actually is

Ignore the word *regression*. This is a classifier.

```
score = sigmoid( w1*mrr + w2*usage + w3*tenure + ... + b )
         └── squash any number into (0, 1), like a probability
```

If the weighted sum is large and positive → score near 1 (likely churn).  
If it is large and negative → score near 0.

<div class="math-box">
<strong>Math, translated</strong>
<p>The sigmoid is just a soft on/off switch. You do not need its formula. You need: <em>weighted sum of features, then squeezed into a probability-like score.</em></p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
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
"""
        ),
        md_cell(
            """## Train three models, read one of them

A shallow decision tree is a flowchart. Random forest is a committee of those flowcharts. Logistic regression is the weighted sum.
"""
        ),
        code_cell(
            """logreg = pipe(LogisticRegression(max_iter=1000))
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
"""
        ),
        md_cell(
            """## Confusion matrix + threshold slider

At threshold 0.5 the library yells “positive.” Your CS team can call 80 accounts a week. That is the real threshold.

```
                    predicted stay     predicted churn
actually stay       true negative      false alarm      ← wasted CS time
actually churned    miss               catch            ← saved revenue
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Precision = “when we page CS, how often were we right?” Recall = “of everyone who churned, how many did we catch?” You cannot max both at a fixed staffing level. Pick the one that matches the cost of a miss vs a wasted call.</p>
</div>
"""
        ),
        code_cell(
            """proba = forest.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, thr in zip(axes, [0.2, 0.5, 0.7]):
    pred = (proba >= thr).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, colorbar=False)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    ax.set_title(f"thr={thr}  P={prec:.2f} R={rec:.2f}\\nflagged={pred.sum()}")
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
"""
        ),
        md_cell(
            """<div class="watch-box">
<strong>Watch out</strong>
<ul>
<li>A random split is convenient and slightly dishonest for time-stamped customers. We fix that in Week 12.</li>
<li>Never rank “top 20% risk” on the <em>training</em> rows and call it a holdout result.</li>
<li>0.5 is not a sacred threshold. It is sklearn’s default because someone had to pick a number.</li>
</ul>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Ship a classifier when it beats the dummy on AUC <em>and</em> you have picked a threshold from a capacity number (“CS can call 50/week”). AUC alone does not page anyone.</p>
</div>
"""
        ),
        md_cell(
            """## Overfitting, bias, and variance — the three words on every ML interview

A model can fail in two opposite ways:

```
UNDERFIT (high bias)              OVERFIT (high variance)
a line through a curve            a scribble through every point
too simple — misses the shape     too clingy — memorizes noise
train error HIGH                  train error TINY
test error HIGH                   test error HIGH  ← the tell
```

<div class="think-box">
<strong>Think of it like… studying for an exam.</strong>
<p><strong>Bias</strong> is showing up with only one idea (“everyone churns if they are free”). You are systematically wrong, even on the homework.<br>
<strong>Variance</strong> is memorizing last year’s answer key, typos included. Homework is perfect. The real exam (new customers) is a mess.<br>
<strong>Overfitting</strong> is the name for that second failure. <strong>Underfitting</strong> is the first.</p>
</div>
"""
        ),
        code_cell(
            """# Toy picture: a smooth truth, noisy homework, three students
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
"""
        ),
        md_cell(
            """<div class="math-box">
<strong>Math, translated</strong>
<p><strong>Bias</strong> ≈ how far the model’s average answer sits from the truth (systematic miss).<br>
<strong>Variance</strong> ≈ how much the answer would jump if you retrained on a different sample of customers.<br>
You cannot drive both to zero. A deeper tree / bigger net lowers bias and raises variance. Regularization, more data, and ensembles are how you buy the pair you can live with.</p>
</div>

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Watch <em>two</em> curves: train vs holdout. If both are bad → underfit (add features, more capacity). If train is great and holdout is not → overfit (simpler model, more data, regularization). Never tune on the number you will report.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Capacity budget.** Assume CS can call 100 test-set users. Sort by forest score, take the top 100, report how many of those actually churned. That is precision at a fixed budget.

**2. Threshold sweep.** For thresholds 0.1, 0.2, … 0.9 print flagged, precision, recall. Circle the row you would ship.

**3. Ablation.** Retrain the forest without `tenure_days`. How much AUC dies? (Tenure is powerful and a little circular — long-lived users have not churned yet.)

## 🤔 Reflection

1. Why can accuracy be 93% while the model is useless? (Hint: 6.7% of users churn.)
2. A PM wants “both high precision and high recall.” What resource do they need to give you?
3. Would you rather explain a depth-3 tree or a 150-tree forest to legal?

## 🔗 Next week

Regression: same idea, but the answer is a number (dollars), not a yes/no. We will refuse to predict `mrr × tenure`.
"""
        ),
    ]
    write_notebook(OUT / "week-06-saas.ipynb", cells, "Week 6 — Classification")


def week7():
    cells = [
        md_cell(
            """# Week 7 — Regression: Predict a Number, Not a Category

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Anyone who has dragged a trendline in a spreadsheet.

---

## 🎯 What you will be able to do

- See linear regression as “Excel trendline with more columns”
- Score models with **MAE in dollars** (or usage units), not just R²
- Always compare to the **mean baseline**
- Read a residual plot: “where does the model systematically lie?”
- Avoid the classic leak: predicting `mrr × tenure` using `tenure`

<div class="think-box">
<strong>Think of it like… a trendline, then a pile of trees voting on a number.</strong>
<p>Classification said yes/no. Regression says “how much.” Same training ritual: features in, a number out, a holdout set that the model must not have memorized.</p>
</div>
"""
        ),
        code_cell(
            BOOT
            + """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
"""
        ),
        md_cell(
            """## 🏢 Scenario — next-period usage, not fake CLV

A common tutorial target is `lifetime_value = mrr * tenure_months` while also handing the model `mrr` and `tenure`. That is asking it to multiply two inputs. R² will look magical. You will have learned nothing.

**Honest target:** `total_usage` from product data, using billing + event *counts* that are not the usage column itself. Still imperfect, but the model cannot just multiply two features you gave it.

<div class="watch-box">
<strong>Watch out — target leakage</strong>
<p>If you can compute the label from the features with a calculator, you are not doing machine learning. You are doing QA on a formula.</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
# Predict product usage from billing + behavioral counts that are not total_usage
features_num = ["mrr", "tenure_days", "features_adopted", "total_events", "avg_session", "n_support"]
features_cat = ["plan_type"]
target = "total_usage"

work = df[features_num + features_cat + [target]].dropna()
X = work[features_num + features_cat]
y = work[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), features_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
])

baseline = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
print(f"Mean baseline MAE: {mean_absolute_error(y_test, baseline):,.1f} usage-units")
print("Every real model has to beat this number.")
"""
        ),
        md_cell(
            """## Three models, one dollar-shaped scoreboard

<div class="math-box">
<strong>Math, translated</strong>
<p><strong>MAE</strong> = average miss, in the same units as the target. The number a PM understands.<br>
<strong>RMSE</strong> = like MAE but whales get extra shame (squares the misses).<br>
<strong>R²</strong> = “what fraction of the jitter did we explain vs just predicting the average?” 0 = baseline, 1 = perfect, negative = worse than the average.</p>
</div>
"""
        ),
        code_cell(
            """def eval_model(name, model):
    p = Pipeline([("prep", prep), ("m", model)])
    p.fit(X_train, y_train)
    pred = p.predict(X_test)
    return p, {
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }, pred

fitted = {}
rows = [{"model": "mean baseline", "MAE": mean_absolute_error(y_test, baseline),
         "RMSE": mean_squared_error(y_test, baseline) ** 0.5, "R2": r2_score(y_test, baseline)}]
preds = {"mean baseline": baseline}
for name, mdl in [
    ("linear", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
    ("forest", RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42, n_jobs=2)),
]:
    pipe, row, pred = eval_model(name, mdl)
    fitted[name] = pipe
    rows.append(row)
    preds[name] = pred

scoreboard = pd.DataFrame(rows).round(3)
print(scoreboard.to_string(index=False))
"""
        ),
        md_cell(
            """## The two plots that tell you if it is any good

**Predicted vs actual:** dots on the diagonal = we got it.  
**Residuals vs predicted:** a random cloud is healthy. A trumpet (errors grow as the prediction grows) means “we are worse on big accounts.” That trumpet is what textbooks call heteroscedasticity. You can just call it a trumpet.
"""
        ),
        code_cell(
            """pred = preds["forest"]
resid = y_test.to_numpy() - pred

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, pred, alpha=0.15, s=10, c="#1d4ed8")
mx = max(y_test.max(), pred.max())
axes[0].plot([0, mx], [0, mx], color="#b91c1c", ls="--")
axes[0].set_xlabel("actual usage")
axes[0].set_ylabel("predicted usage")
axes[0].set_title("Predicted vs actual — diagonal is truth")

axes[1].scatter(pred, resid, alpha=0.15, s=10, c="#0f766e")
axes[1].axhline(0, color="#b91c1c", ls="--")
axes[1].set_xlabel("predicted usage")
axes[1].set_ylabel("actual − predicted")
axes[1].set_title("Residuals — a trumpet means we miss the whales")
plt.tight_layout()
plt.show()

# Linear coefficients, in "after scaling" units
lin = fitted["linear"].named_steps["m"]
ohe = fitted["linear"].named_steps["prep"].named_transformers_["cat"]
names = features_num + list(ohe.get_feature_names_out(features_cat))
coef = pd.Series(lin.coef_, index=names).sort_values()
print("Linear weights (on scaled features — compare signs, not raw dollars):")
print(coef.round(2).to_string())
"""
        ),
        md_cell(
            """<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Linear weights after scaling are “how much the prediction moves when this feature is one typical-spread higher, holding the others still.” Do not compare a weight on raw <code>mrr</code> to a weight on raw <code>events</code> — different units. That is why we scaled.</p>
</div>

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Use linear/Ridge when you need a sentence for finance (“each extra feature adopted is associated with +X usage”). Use a forest when the relationship is a staircase, not a line, and you can live with a less quotable model. Always print the mean baseline on the same slide.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Log target.** Train the forest on `log1p(y)` and `expm1` the predictions. Does MAE on the original scale improve? (Whales often get kinder.)

**2. Residual slices.** MAE for `free` vs `enterprise`. Where is the model actually bad?

**3. Forbidden target.** Create `fake_clv = mrr * (tenure_days / 30)` and train a linear model using `mrr` and `tenure_days`. Marvel at R². Then delete it and never do this at work.

## 🤔 Reflection

1. R² = 0.4. Is that good? (Depends: did you beat the baseline, and is a 40% jitter reduction worth the ops cost?)
2. Why are unscaled linear coefficients a trap in a meeting?
3. When is a “worse” MAE on whales acceptable? (If you only manage SMB accounts.)

## 🔗 Next week

No labels. Clustering: sort the messy inbox when nobody tagged the tickets.
"""
        ),
    ]
    write_notebook(OUT / "week-07-saas.ipynb", cells, "Week 7 — Regression")


def week8():
    cells = [
        md_cell(
            """# Week 8 — Clustering: Sorting Without Labels

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have bucketed users in SQL and wished the buckets invented themselves.

---

## 🎯 What you will be able to do

- Contrast supervised (“tickets with tags”) vs unsupervised (“messy inbox”)
- Run K-Means as **drop K pins, assign, scoot pins, repeat**
- Read an elbow / silhouette as “how blob-like are we,” not as a sacred K
- See why **unscaled** MRR hijacks the clusters
- Use segments as *personas for marketing*, not as a production classifier

<div class="think-box">
<strong>Think of it like… dropping pins on a map.</strong>
<p>You pick K (say 4). Drop 4 pins at random. Every customer walks to the nearest pin. Then each pin moves to the average location of its people. Repeat until the pins stop wandering. Those final neighborhoods are your segments.</p>
</div>
"""
        ),
        code_cell(
            BOOT
            + """
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
"""
        ),
        md_cell(
            """## Supervised vs unsupervised

```
Supervised (Weeks 6–7)          Unsupervised (this week)
X ────────► model ──► y         X ────────► model ──► group id
   you had labels                  you did not
   spam / not spam                 "these users look like each other"
```

<div class="engineer-box">
<strong>Engineer mental model</strong>
<p>Cluster <em>offline</em>. Write the persona (“whale, 3 features, high MRR”). Drive campaigns from the persona or from a simple rule. Do not call <code>KMeans.predict</code> on the request path unless you really mean it — pin locations drift every retrain and nobody will know why the user flipped from “champion” to “at risk.”</p>
</div>
"""
        ),
        code_cell(
            FEATS
            + """
df = load_customer_360(DATA)
# Keep the label on the side for storytelling — the algorithm does not get it
cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events"]
sample = df  # already laptop-sized from load_customer_360
X_raw = sample[cols].to_numpy()
X = StandardScaler().fit_transform(X_raw)

print("We will pretend we never saw is_churned. After clustering we will peek.")
"""
        ),
        md_cell(
            """## Scale first, or MRR becomes the whole personality

Without scaling, a $499 enterprise account is “farther” from a $29 starter than a power user is from a lurker. Distance thinks in raw units.
"""
        ),
        code_cell(
            """# Tiny 2-D picture: MRR vs log usage, unscaled vs scaled
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(sample["mrr"], sample["log_usage"], s=8, alpha=0.25, c="#64748b")
axes[0].set_title("Unscaled — horizontal axis in dollars dominates")
axes[0].set_xlabel("mrr"); axes[0].set_ylabel("log usage")

axes[1].scatter(X[:, 0], X[:, 2], s=8, alpha=0.25, c="#6366f1")
axes[1].set_title("Scaled — both axes in 'typical spreads'")
axes[1].set_xlabel("mrr (z)"); axes[1].set_ylabel("log usage (z)")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """## Choosing K — elbow is a suggestion, the business can overrule

**Inertia** = how far customers sit from their pin (lower is tighter).  
**Silhouette** ≈ “am I closer to my blob than to the next blob?” (higher is cleaner, max 1).

If marketing can only run 3 campaigns, you pick K=3 even if K=6 wins the silhouette contest.
"""
        ),
        code_cell(
            """ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, labels, sample_size=3000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set_title("Elbow (inertia) — look for the bend")
axes[0].set_xlabel("K")
axes[1].plot(list(ks), sils, marker="o", color="#0f766e")
axes[1].set_title("Silhouette — higher = cleaner blobs")
axes[1].set_xlabel("K")
plt.tight_layout()
plt.show()
print(list(zip(ks, np.round(sils, 3))))
"""
        ),
        code_cell(
            """K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42)
sample = sample.copy()
sample["cluster"] = km.fit_predict(X)

# 2-D view of the neighborhoods
fig, ax = plt.subplots(figsize=(7, 4.2))
for c in range(K):
    sl = sample[sample["cluster"] == c]
    ax.scatter(sl["mrr"], sl["log_usage"], s=10, alpha=0.35, label=f"cluster {c}")
ax.set_xlabel("mrr"); ax.set_ylabel("log usage")
ax.set_title("K=4 pins in a 2-D slice (the model actually used more columns)")
ax.legend()
plt.tight_layout()
plt.show()

# Persona table — including churn, which we hid from the algorithm
profile = sample.groupby("cluster").agg(
    n=("user_id", "count"),
    mrr=("mrr", "median"),
    usage=("total_usage", "median"),
    features=("features_adopted", "median"),
    tenure=("tenure_days", "median"),
    churn=("is_churned", "mean"),
).round(3)
print(profile.to_string())
print("\\nName the rows in a PR description. If you cannot name them, K is wrong.")
"""
        ),
        md_cell(
            """## DBSCAN, in one picture

K-Means always fills K buckets, even if the data is a smear. **DBSCAN** says: “a cluster is a dense pocket; loners are noise.” You pick a radius (`eps`) and a minimum crowd (`min_samples`), not K.

On 5-D scaled data, `eps=0.5` is a guess. If everything is noise, raise `eps`. If everything is one blob, lower it.

<div class="ship-box">
<strong>Ship / don’t ship</strong>
<p>Clustering is a workshop tool: personas, onboarding tracks, “who should see this email.” It is not a replacement for the Week 6 churn model. Do not put cluster ids into a legal document — they will move next Tuesday.</p>
</div>
"""
        ),
        md_cell(
            """## ✍️ Exercises

**1. Unscaled K-Means.** Cluster on raw `mrr` + `total_usage` (no scaler). Profile the clusters. Did MRR eat the result?

**2. Name the personas.** From the K=4 table above, write a one-line name and one marketing action per cluster. If two rows get the same name, merge them.

**3. Peek, don’t train.** Churn *rate* by cluster is a story. Training a classifier *on cluster id* is usually weaker than training on the original features — the id is a lossy compression.

## 🤔 Reflection

1. Why is “the algorithm found our enterprise plan” a failure, not a success? (You already had that column.)
2. Silhouette says K=2, marketing wants K=5. Who wins?
3. What breaks if you re-fit K-Means nightly and email users based on last night’s id?

## 🔗 Next week

Too many columns. PCA: JPEG for tabular data — keep the big shapes, drop the noise.
"""
        ),
    ]
    write_notebook(OUT / "week-08-saas.ipynb", cells, "Week 8 — Clustering")


if __name__ == "__main__":
    week5()
    week6()
    week7()
    week8()
