# Week 8 — Labels Lie

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who shipped Week 7’s classifier. Read this after Week 7. CloudWave’s lifetime `is_churned` flag is the wrong label.

About **6.4%** of customers ever cancel in this file. A model that predicts “nobody churns” is ~94% accurate and useless. A model that uses lifetime `is_churned` plus `tenure_days` is a tenure detector wearing a costume.

---

## 🎯 What you will be able to do

- Replace “did they ever churn” with **“did they churn in the next 30 days, as of Monday”**
- Name **censoring**: we have not watched them long enough to know
- Read **PR-AUC** when the positive class is rare (ROC-AUC will flatter you)
- Treat a 0.73 score as a **rank**, not “73% chance,” until you check calibration
- Keep PII and the label out of `X`

!!! think "Think of it like… a bug ticket’s status."

    `is_churned` is “this ticket is closed, ever.” You would not train “will this ticket close in 30 days” on that. You would take tickets that were **open on Monday**, look at **Monday’s fields only**, and see who closed by the end of the month. Tickets whose 30-day window is not over yet are **censored** — we have not watched them long enough. A ticket filed on Saturday is not censored; it is **noisy**. Almost no history, so even a fully observed 30-day label is a coin flip dressed as data.

## If you already write software

```
Lifetime is_churned          “this user has a closed ticket, sometime”
Horizon label                “open on as_of, closed within 30 days”
Censored                     “as_of + 30d is after our last log”
tenure_days (lifetime)       closed_at − opened_at   ← leak / circular
tenure_so_far                as_of − signup          ← legal
Accuracy                     “the server is up” on a page that is 99.9% fine
PR-AUC                       precision of the rare class, across ranks
Calibration                  if we say 0.2, about 20% should actually fire
```

### Picture the time machine for the *label*

```
timeline →

signup        as_of              as_of+30d         later
  |             |                    |               |
  ●─────────────●────────────────────●───────────────●
  features       ▲                    ▲
  must stop      │                    │
  here           already gone?        cancel in window?  → label = 1
                 drop (not at risk)   still here?        → label = 0
                                      window not over?   → drop (censored)
```

Week 6 stopped *features* at the wall. This week stops the **answer key** at the wall too.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.labels import drop_unlabelled, label_churn_in_horizon, label_eventual_churn

as_of = AS_OF_DEFAULT  # 2024-06-01
df = build_features(as_of=as_of, n=None, at_risk_only=True)
y = label_churn_in_horizon(df, as_of)
labelled, y = drop_unlabelled(df, y)

print("at risk as of", as_of.date(), "n=", len(df))
print("knowable labels", len(y), "horizon rate", float(y.mean()), "positives", int(y.sum()))
print("lifetime is_churned on the same people", float(labelled["is_churned"].mean()))
print("eventual-after-as_of rate", float(label_eventual_churn(df, as_of).mean()))
```

The lifetime rate is higher. It counts people who cancel after the 30-day window (through 2024-11-30 in this file). You will not know that on 2024-06-01. Using it is cheating.

!!! warning "Watch out — tenure_days"

    Lifetime `tenure_days` is “how long they stayed.” Long tenure *means* they have not churned yet. Put it in `X` and the model learns a tautology. `tenure_so_far` is how long they have been around **as of Monday**. That is legal. It is also a weak feature — new users have not had time to leave. Do not confuse the two.

## Imbalance is a staffing fact

```
~44,000 at-risk customers on 2024-06-01
  ~48 cancel in the next 30 days          (~0.11%, not 60–80 per thousand)
  ~168 cancel sometime after as_of        (eventual; still rare)
  the rest do not, in this file

Accuracy of “predict 0”:  ~99.9% on the 30-day question
CS can call:              80 people
The only number that pays: of those 80, how many actually left?
```

This fixture only has **tens** of 30-day events. That is why Week 16’s job trains `"label": "eventual"` and writes that string in `metrics.json`. The product question is still 30 days. The file can actually supervise “cancel after Monday.”

ROC-AUC asks “can you rank a random churner above a random non-churner?” With 99.9% negatives, a lazy model still looks fine.

**PR-AUC** (average precision) asks “as you walk down the ranked list, how often were you right?” That matches the 80-call budget.

```python
rng = np.random.default_rng(0)
dummy = np.full(len(y), float(y.mean()))
noise = rng.random(len(y))

print("dummy ROC-AUC", roc_auc_score(y, dummy).round(3),
      "dummy PR-AUC", average_precision_score(y, dummy).round(3))
print("noise ROC-AUC", roc_auc_score(y, noise).round(3),
      "noise PR-AUC", average_precision_score(y, noise).round(3))
print("base rate (this is the dummy PR-AUC, in one number)", float(y.mean()))
```

A coin-flip can sit near 0.5 ROC and still have PR-AUC ≈ the base rate. Report **both**. Ship on precision@80.

!!! engineer "Engineer mental model"

    Accuracy is uptime on a site that is almost never down. PR-AUC is “when the pager fires, was it real.” `class_weight="balanced"` is a *training* trick, like retrying 5xx more often. It does not change the fact that CS has 80 slots. Always measure in slots.

## A score is not a probability

Week 7’s 0.73 is a **rank**. It is not “73% chance they churn” unless you check.

```
calibration
  predicted 0.1  →  about 10% of those people should actually churn
  predicted 0.4  →  about 40%
  a banana curve →  you are ranking fine and quoting odds like a liar
```

```python
# Horizon labels leave ~8 train positives after a time split — empty calibration bins.
# Eventual-after-as_of is still rare and is the label week 16 actually trains.
y_cal = label_eventual_churn(labelled, as_of)
cut = labelled["signup_date"].quantile(0.80)
train = labelled[labelled["signup_date"] <= cut]
test = labelled[labelled["signup_date"] > cut]
# FEATURE_COLS includes plan_type (a string). Trees cannot eat it raw.
model = Pipeline(
    [
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ]
)
model.fit(train[FEATURE_COLS], y_cal.loc[train.index])
p = model.predict_proba(test[FEATURE_COLS])[:, 1]
frac_pos, mean_pred = calibration_curve(y_cal.loc[test.index], p, n_bins=8, strategy="quantile")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="honest")
ax.plot(mean_pred, frac_pos, "o-", color="#1d4ed8", label="model")
ax.set_xlabel("mean predicted score in bin")
ax.set_ylabel("actual churn rate in bin")
ax.set_title("If the dots leave the dashed line, do not quote the score as %")
ax.legend()
plt.tight_layout()
plt.show()
```

If the curve bows, you may still **rank** well (keep the 80-call list). You may not multiply the score by MRR and call it expected loss.

!!! math "Math, translated"

    Brier score is mean squared error between the score and the 0/1 label. Low is honest. You do not need the name in the stand-up. You need the picture.

## PII does not go in X

CloudWave’s fixture has no emails. Your real warehouse will. Rule:

| Allowed in `X` | Forbidden in `X` |
|---|---|
| plan, MRR, tenure_so_far, usage counts | `user_id`, email, name, ticket body |
| region as a *category you will have at score time* | `churn_date`, lifetime `is_churned`, lifetime `tenure_days` |
| `n_support` | raw `feedback_text` (that is a different model, and a privacy review) |

`validate()` in `pipelines/contract.py` rejects unknown keys. That is the PII fence: if it is not in the contract, it does not enter.

!!! success "Ship / don’t ship"

    Ship a horizon label, PR-AUC + precision@budget, and a calibration glance. Do not ship lifetime `is_churned` + `tenure_days` + accuracy. Do not tell finance a 0.7 is a 70% chance until the dots sit on the dashed line.

## ✍️ Exercise

[Exercises](exercises/week-08.md) — including `pytest tests/test_labels.py`.

## 🤔 Reflection

1. A user signed up yesterday. Why is their 30-day label mostly noise even if you wait?
2. Why can ROC-AUC look “fine” when the 80-call list is junk?
3. CS asks “so this account is 80% likely to churn?” What do you actually know?

## 🔗 Next week

Ranking. Most SaaS models are not “yes/no.” They are “who is at the top of the list.”
