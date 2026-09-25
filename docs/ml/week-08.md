---
description: Fix mislabeled churn data by defining a time-boxed label, handling censoring, and reading PR-AUC when positive cases are rare.
---

# Week 8 — Labels Lie

Ana blocks the handoff before Priya gets the list. "This ranks who ever cancels," she says, "not who cancels in the next 30 days. Those aren't the same 80 people." She's right — `is_churned` is a lifetime flag, and it was your label all along.

??? note "Course details"

    **Course:** Applied ML Foundations for SaaS Analytics
    **Who this is for:** Engineers who shipped Week 7's classifier. Read this after Week 7. CloudWave's lifetime `is_churned` flag is the wrong label.

About **9%** of the customers still active on 2024-06-01 ever cancel in this file — but only about **2%** cancel in the next 30 days. A model that predicts "nobody churns" is ~98% accurate on that 30-day question, and useless. A model trained on the lifetime `is_churned` flag answers "will they ever leave," which is a different, easier, and less useful question than "will they leave in the next 30 days."

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
from pipelines.split import snapshot_split

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
~28,000 at-risk customers on 2024-06-01
  ~530 cancel in the next 30 days         (~1.9% — about 2% a month, a normal SMB SaaS)
  ~2,600 cancel sometime after as_of      (eventual: everything left in the file)
  the rest do not, in this file

Accuracy of “predict 0”:  ~98% on the 30-day question
CS can call:              80 people
The only number that pays: of those 80, how many actually left?
```

Why not train on “eventual”? It has five times the positives. But its window runs to the end of the file: six months long for a June snapshot, one month for an October one. The same customer gets a different answer depending on the day you ask — a label whose meaning drifts with the calendar. The 30-day label means the same thing every Monday, and ~530 positives is plenty to learn from. Week 16’s job trains on it and writes `"label": "churn within 30 days"` into `metrics.json`.

ROC-AUC asks “can you rank a random churner above a random non-churner?” With 98% negatives, a lazy model still looks fine.

**PR-AUC** (average precision) asks “as you walk down the ranked list, how often were you right?” That matches the 80-call budget.

```python
rng = np.random.default_rng(0)
dummy = np.full(len(y), float(y.mean()))
noise = rng.random(len(y))

print("dummy ROC-AUC", round(roc_auc_score(y, dummy), 3),
      "dummy PR-AUC", round(average_precision_score(y, dummy), 3))
print("noise ROC-AUC", round(roc_auc_score(y, noise), 3),
      "noise PR-AUC", round(average_precision_score(y, noise), 3))
print("base rate (this is the dummy PR-AUC, in one number)", float(y.mean()))
```

A coin-flip can sit near 0.5 ROC and still have PR-AUC ≈ the base rate. Report **both**. Ship on precision@80.

!!! engineer "Engineer mental model"

    Accuracy is uptime on a site that is almost never down. PR-AUC is “when the pager fires, was it real.” `class_weight="balanced"` is a *training* trick, like retrying 5xx more often. It does not change the fact that CS has 80 slots. Always measure in slots.

## A score is not a probability

Week 7’s 0.73 is a **rank**. It is not “73% chance they churn” unless you check.

!!! warning "A number in [0, 1] is not a calibrated probability"

    A classifier can output a number in [0, 1] without that number being empirically calibrated.
    For a calibrated model: predictions near 0.8 → about 80% actually positive.
    Ranking well (AUC) does not imply the scores are honest percents. That is what [scikit-learn Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html) is checking.

```
calibration
  predicted 0.1  →  about 10% of those people should actually churn
  predicted 0.4  →  about 40%
  predicted 0.8  →  about 80%   ← only if the model is calibrated
  a banana curve →  you are ranking fine and quoting odds like a liar
```

## Before you run this

Predict:

1. Will the GBT’s dots sit on the dashed “honest” line?
2. If they bow away, can the model still rank well?
3. Why is a score of 0.8 not automatically “80% will churn”?

## Run it

Compare the reliability curve with your prediction.

## Explain the difference

If your prediction was wrong, what assumption was wrong?

```python
# Backtest: learn on the snapshot 30 days earlier (its labels are complete by
# as_of), check on this one. Week 15 explains why a signup_date cut is the
# wrong time wall.
train, y_train, test, y_test = snapshot_split(as_of, horizon_days=30)
# FEATURE_COLS includes plan_type (a string). Trees cannot eat it raw.
model = Pipeline(
    [
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ]
)
model.fit(train[FEATURE_COLS], y_train)
p = model.predict_proba(test[FEATURE_COLS])[:, 1]
frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")

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
3. Priya asks “so this account is 80% likely to churn?” What do you actually know?

## Before you leave

Try one [self-check](self-checks.md#week-8-labels) (Predict / Diagnose / Choose / Defend). Write the answer before you open the block.

## 🔗 Next week

You fix the label. Priya's real question is still open: she doesn't want a yes/no for 28,000 customers, she wants the top 80, ranked. [Week 9](week-09.md) and [Week 10](week-10.md) are two detours worth taking on the way — then [Week 11](week-11.md) is that list.
