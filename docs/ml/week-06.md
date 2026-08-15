# Week 6 — Features Are the Model’s API

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have designed request payloads. Feature engineering is that, plus a timeline rule.

---

## 🎯 What you will be able to do

- Treat a feature vector as a **versioned contract** the training job and the `/predict` handler must share
- Scale numbers so “dollars” and “click counts” can sit in the same model
- One-hot encode `plan_type` without treating free &lt; starter &lt; pro as a number line
- **Fit the scaler on train only** — the leak that will follow you to production
- Draw a wall between “known at score time” and “the future”

!!! think "Think of it like… an API contract + a time machine rule."

    The model only sees the JSON you send it. If a field would not exist when you score a live user at noon on Tuesday, it cannot exist in training either. That is leakage: the model cheated on the exam by reading tomorrow’s answer key.

## If you already write software

A feature vector is an API contract.

`/predict` accepts a JSON body. Training must build *that same body* from historical rows. If a field would not exist at noon on Tuesday when you score a live user, it cannot exist in the training table. That is leakage: the model read tomorrow’s answer key.

```
Training job                         Scoring service
────────────                         ──────────────
row → features → model.fit           request JSON → same features → model.predict
scaler.fit(X_train)                  scaler.transform(X_live)   ← same scaler pickle
never touch X_test to fit            never invent fields the client cannot send
```

### The time-machine rule

Ask of every column: **would I have known this at score time?**

| Column | Known at score time? | Keep? |
|---|---|---|
| `plan_type`, `mrr`, `tenure_so_far` | yes | yes |
| `usage_last_30d` | yes, if you compute it from events before now | yes |
| `churn_date` / `is_churned` | that is the label | **target, not a feature** |
| `days_until_churn` | future | leak, delete |
| `avg_sentiment_after_cancel` | future | leak, delete |

### Picture the scaler

`StandardScaler` subtracts the mean and divides by the std. If you fit it on train+test, test information leaked into the transform. It is the same bug as using production traffic to tune a cache key, then being surprised the benchmark looks good.

The scaler **is part of the model**. It ships in the same pickle. New data gets `transform` only.

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
```

## 🏢 Scenario — churn features the scoring service can actually compute

We want to flag accounts that will cancel. At score time we know:

- plan, MRR, tenure so far, usage so far, events so far

We do **not** know `churn_date`. We must not sneak it in as `has_churn_flag`.

```
 timeline:  signup -------- now -------- churn?
                       ▲
                       └── score time. Nothing to the right of this wall
                           may enter X. The label y may look right of the wall.
```

!!! engineer "Engineer mental model"

    Features = request body. Scaler + encoder = middleware that *must ship next to the .pkl*. If production sends raw dollars and the model expects “standard deviations from the training mean,” every score is garbage and nobody gets a stack trace.

```python
df = load_customer_360(DATA)
print(df.shape)
print(df[["user_id", "plan_type", "mrr", "tenure_days", "total_usage",
          "features_adopted", "total_events", "is_churned"]].head())
print("\nLabel rate (churned):", df["is_churned"].mean().round(3))
```

## Scaling — why trees shrug and linear models panic

`mrr` is 0–500. `total_usage` can be tens of thousands. A linear model / k-means / neural net **adds** these numbers. The big column shouts down the small one.

A tree only asks “is usage &gt; 40?” — units do not matter.

!!! math "Math, translated"

    `StandardScaler`: subtract the column’s mean, divide by its standard deviation. After that, “1” means “one typical-spread above average,” the same z-score idea from Week 1. `log1p(usage)` is “compress the whales so they do not own the axis.”

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
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
```

## Categories are not numbers

`plan_type` is free / starter / pro / enterprise. If you map those to 0,1,2,3 you are telling the model “enterprise is three more than free” and “the step from free→starter equals starter→pro.” Sometimes that is true. Usually it is a lie.

**One-hot:** four yes/no columns. Honest, a bit wide.

!!! warning "Watch out — the scaler leak"

    `scaler.fit_transform(X)` on the *full* table peeks at the test set’s mean and spread. That is a small leak that becomes a habit. Fit on train. Transform test. In production, the saved scaler *is* the fit.

```python
numeric = ["mrr", "tenure_days", "log_usage", "features_adopted",
           "total_events", "n_devices", "n_support"]
categorical = ["plan_type"]
label = "is_churned"

X = df[numeric + categorical]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Time-based split is even better (Week 15). Stratified random is the honest starter.

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

print("\nScaled train means (numeric should sit near 0):")
print(np.round(X_train_t[:, : len(numeric)].mean(axis=0), 3))
```

## Leakage hall of shame (we will keep coming back)

| Looks clever | Why it is cheating |
|---|---|
| `has_churn_flag` as a feature | That **is** the label |
| `lifetime_value = mrr * tenure` as a target, `tenure` as a feature | The model multiplies two columns it was handed |
| Fit scaler / target-encoder on all rows | Test set leaked into preprocessing |
| Random split when the world is a time series | The model trains on “next month” and tests on “last month” |

!!! success "Ship / don’t ship"

    A feature ships if a tired on-call engineer can compute it from *today’s* warehouses for a single `user_id` with no peek at the label table. If you cannot write that function, it is not a feature.

    Email, name, ticket body, `user_id`, `churn_date`, and lifetime `tenure_days` do not go in `X`. `pipelines/contract.py` rejects unknown keys so PII cannot wander in. The one function that builds the row is `pipelines.features.build_features(as_of=...)` — Week 3 and 19 hang the job on it.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-06.md). Starter: `python exercises/ml/week-06/starter.py` from the repo root.

## 🤔 Reflection

1. Why is “churned in the next 30 days” a better label than “ever churned”?
2. A teammate one-hot encodes `user_id`. What happens?
3. Where does the scaler live in your repo — next to the model, or re-fit in the API process?

## 🔗 Next week

Classification: a model is a function `features → risk score`. We pick a threshold the sales team can staff.
