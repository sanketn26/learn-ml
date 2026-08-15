# Week 7 — Regression: Predict a Number, Not a Category

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Anyone who has dragged a trendline in a spreadsheet.

---

## 🎯 What you will be able to do

- See linear regression as “Excel trendline with more columns”
- Score models with **MAE in dollars** (or usage units), not just R²
- Always compare to the **mean baseline**
- Read a residual plot: “where does the model systematically lie?”
- Avoid the classic leak: predicting `mrr × tenure` using `tenure`

!!! think "Think of it like… a trendline, then a pile of trees voting on a number."

    Classification said yes/no. Regression says “how much.” Same training ritual: features in, a number out, a holdout set that the model must not have memorized.

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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## If you already write software

Regression answers “how many / how much,” not “which bucket.” Predicting next month’s MRR is regression. Predicting “will they churn” is classification. Do not encode a number as a category just because the dashboard has traffic-light colors.

The metric that matters is in **product units**:

- MAE = “on average we are off by $X”
- RMSE = “we get punished extra for the rare $400 miss”

R² is a nice homework number. A PM cannot staff from it.

### The leak to refuse

If you predict `mrr` using `mrr * tenure` or next month’s invoice, you built a calculator and called it ML. Same as “predicting” latency from a feature that is the latency itself.

### Picture residuals

A residual is `actual - predicted`. Plot them.

```
healthy          a blob around zero, no shape
trumpet          errors get bigger as the prediction gets bigger
                 (the model is worse on whales — usually what you care about)
curve            you are fitting a line to a bent world
                 (log the target, or add a non-linear model)
```

A trumpet is not a stats curiosity. It means your error is largest on the accounts finance actually watches.

## 🏢 Scenario — next-period usage, not fake CLV

A common tutorial target is `lifetime_value = mrr * tenure_months` while also handing the model `mrr` and `tenure`. That is asking it to multiply two inputs. R² will look magical. You will have learned nothing.

**Honest target:** `total_usage` from product data, using billing + event *counts* that are not the usage column itself. Still imperfect, but the model cannot just multiply two features you gave it.

!!! warning "Watch out — target leakage"

    If you can compute the label from the features with a calculator, you are not doing machine learning. You are doing QA on a formula.

```python
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
```

## Three models, one dollar-shaped scoreboard

!!! math "Math, translated"

    **MAE** = average miss, in the same units as the target. The number a PM understands.

    **RMSE** = like MAE but whales get extra shame (squares the misses).

    **R²** = “what fraction of the jitter did we explain vs just predicting the average?” 0 = baseline, 1 = perfect, negative = worse than the average.

```python
def eval_model(name, model):
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
```

## The two plots that tell you if it is any good

**Predicted vs actual:** dots on the diagonal = we got it.  
**Residuals vs predicted:** a random cloud is healthy. A trumpet (errors grow as the prediction grows) means “we are worse on big accounts.” That trumpet is what textbooks call heteroscedasticity. You can just call it a trumpet.

```python
pred = preds["forest"]
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
```

!!! engineer "Engineer mental model"

    Linear weights after scaling are “how much the prediction moves when this feature is one typical-spread higher, holding the others still.” Do not compare a weight on raw `mrr` to a weight on raw `events` — different units. That is why we scaled.


!!! success "Ship / don’t ship"

    Use linear/Ridge when you need a sentence for finance (“each extra feature adopted is associated with +X usage”). Use a forest when the relationship is a staircase, not a line, and you can live with a less quotable model. Always print the mean baseline on the same slide.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-07.md). Starter: `python exercises/ml/week-07/starter.py` from the repo root.

## 🤔 Reflection

1. R² = 0.4. Is that good? (Depends: did you beat the baseline, and is a 40% jitter reduction worth the ops cost?)
2. Why are unscaled linear coefficients a trap in a meeting?
3. When is a “worse” MAE on whales acceptable? (If you only manage SMB accounts.)

## 🔗 Next week

No labels. Clustering: sort the messy inbox when nobody tagged the tickets.
