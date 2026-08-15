# Week 11 — Rank a List

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have written a search ranking, a “priority queue,” or a “top of the inbox.” Read after Week 7 or 17.

CloudWave CS cannot call everyone. They can call **80**. The product is not a yes/no. It is an **ordered list**.

---

## 🎯 What you will be able to do

- Treat churn (and most SaaS ML) as **ranking**, with classification as a special case
- Measure **precision@k** and **recall@k** — the only metrics a staffing number understands
- Beat a dumb ranker (sort by `n_support`, sort by low usage) before celebrating a GBT
- Refuse “if we increase usage, they won’t churn” as a reading of a ranked list

!!! think "Think of it like… the on-call rota, or search."

    PagerDuty does not need a boolean per service. It needs “who is next.” Google does not need “is this *the* page.” It needs “is this in the top 10.” CS is the same: a sorted queue. Thresholds (Week 7) are how you *cut* the queue. Ranking is how you *build* it.

## If you already write software

```
Classification (Week 7)     score ≥ t  →  {0, 1}
Ranking (this week)         sort by score  →  the first k
precision@k                 of the first k, how many were real
recall@k                    of all the reals, how many sat in the first k
NDCG / MAP                  search-quality cousins; you do not need them to ship 80 names
A/B after ship              new list vs old list, same k, look at actual cancels
```

### Picture the queue

```
score
 0.91  user_A   ← CS calls #1
 0.84  user_B
 0.77  user_C
 ...
 0.41  user_80  ← last call this week          k = 80
 -----------------
 0.40  user_81  ← not this week
```

`t` is just “the score of row 80.” Capacity *is* the threshold.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features
from pipelines.labels import drop_unlabelled, label_churn_in_horizon

as_of = AS_OF_DEFAULT
df = build_features(as_of=as_of, n=8000, at_risk_only=True)
y = label_churn_in_horizon(df, as_of)
df, y = drop_unlabelled(df, y)

cut = df["signup_date"].quantile(0.80)
test = df[df["signup_date"] > cut]
y_test = y.loc[test.index]

model = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)
train = df[df["signup_date"] <= cut]
model.fit(train[FEATURE_COLS], y.loc[train.index])
score = model.predict_proba(test[FEATURE_COLS])[:, 1]


def at_k(y_true, scores, k):
    order = np.argsort(-np.asarray(scores))[:k]
    picked = np.asarray(y_true)[order]
    return {"k": k, "precision": float(picked.mean()), "hits": int(picked.sum()),
            "recall": float(picked.sum() / max(np.asarray(y_true).sum(), 1))}


print("model     ", at_k(y_test, score, 80))
print("n_support ", at_k(y_test, test["n_support"], 80))
print("low usage ", at_k(y_test, -test["log_usage"], 80))
print("random    ", at_k(y_test, np.random.default_rng(0).random(len(test)), 80))
print("ROC-AUC   ", roc_auc_score(y_test, score).round(3))
```

If `n_support` beats the GBT at k=80, you do not have a modeling problem. You have a “the tree is not earning its pickle” problem. Ship the sort.

!!! engineer "Engineer mental model"

    A ranker is `ORDER BY` with a learned key. Precision@k is the `LIMIT`. Always publish the dumb `ORDER BY n_support DESC` next to your model. If you cannot beat a one-line SQL, the meeting is over.

## Causation is not a ranked list

The model uses usage. Low usage sits high on the list. A PM says “so if we make them use the product, they won’t churn.”

That is a **lever** claim. You trained a **ranker**.

```
Predictive (what you built)     people who already look like churners
Causal (what they asked)        if we change usage, what happens to churn
```

Same confusion as “the linter flags files with more `TODO`s, so deleting TODOs will fix production.” You ranked a symptom.

To talk about a lever you need an experiment (Week 5): change onboarding, hold out a group, look at the 30-day label. The list can *choose who to experiment on*. It cannot replace the experiment.

!!! warning "Watch out"

    - Measuring precision@k on **train** is memorizing the queue.
    - Changing k after you see the number is p-hacking the budget.
    - Personalization / “recommend a feature” is the same math: rank items per user, precision@k on clicks. Different grain, same queue.

## Recommendations in one paragraph

“Which feature should we email them?” is `user × item` ranking. Features become the item. The label is “did they use it next week.” You still need an `as_of`, a dumb baseline (`ORDER BY global popularity`), and precision@k. You do not need a new library. You need a new grain.

!!! success "Ship / don’t ship"

    Ship a list when it beats the obvious SQL sort on precision@k for a **pre-registered** k (the staffing number). Do not ship a “churn driver” slide that treats a ranker’s inputs as knobs. Do not ship recommendations until you have the same picture at the user×item grain.

## ✍️ Exercise

[Exercises](exercises/week-11.md).

## 🤔 Reflection

1. CS doubles headcount. What happens to k? To the threshold? To precision?
2. When would you *not* bother with a model and just `ORDER BY n_support`?
3. Write one sentence you would say to the PM who wants to “improve usage to reduce churn” based on this list.

## 🔗 Next

If you have not done Weeks 9–12, the tabular path continues there.  
If you have a pickle: Week 16 is the job that retrains it on Tuesday.
