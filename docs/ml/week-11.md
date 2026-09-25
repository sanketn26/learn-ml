---
description: Rank customers by churn risk using precision@k and recall@k instead of a single classification threshold, beating a naive baseline.
---

# Week 11 — Rank a List

Priya cannot call everyone. She can call **80**. The product you're building for her is not a yes/no. It's an **ordered list**.

??? note "Course details"

    **Course:** Applied ML Foundations for SaaS Analytics
    **Who this is for:** Engineers who have written a search ranking, a "priority queue," or a "top of the inbox." Read after Week 8.

---

## 🎯 What you will be able to do

- Treat churn (and most SaaS ML) as **ranking**, with classification as a special case
- Measure **precision@k** and **recall@k** — the only metrics a staffing number understands
- Beat a dumb ranker (sort by `n_support`, sort by low usage) before celebrating a GBT
- Put a **bootstrap confidence interval** on precision@k, and on the gap between two rankers
- Refuse “if we increase usage, they won’t churn” as a reading of a ranked list
- Explain why “most likely to churn” is not “most worth a call,” and why calling people **changes next month's labels**

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
from sklearn.pipeline import Pipeline

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, build_features, make_preprocessor
from pipelines.split import snapshot_split

as_of = AS_OF_DEFAULT
# Backtest: learn on the snapshot 30 days before as_of, rank today's at-risk
# customers, check who left in the next 30 days — the question Priya asked.
train, y_train, test, y_test = snapshot_split(as_of, horizon_days=30)
print("positives in test", int(y_test.sum()), "of", len(y_test))

model = Pipeline(
    [
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
    ]
)
model.fit(train[FEATURE_COLS], y_train)
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
print("ROC-AUC   ", round(roc_auc_score(y_test, score), 3))
```

If `n_support` beats the GBT at k=80, you do not have a modeling problem. You have a “the tree is not earning its pickle” problem. Ship the sort.

## How sure is “10 of 80”?

Your list caught about ten churners in eighty calls; `ORDER BY low usage` caught about seven. Is that a win, or is it one flaky run?

There are only ~530 churners among ~28,000 customers, so eighty calls hold a handful of them. Move two customers in or out of the top 80 and precision swings by 2.5 points. That is a flaky test with a small sample — so do what you do with a flaky test: **rerun it many times and look at the spread.**

The **bootstrap** reruns the evaluation without new data. Draw 28,000 customers from the test set *with replacement* (some customers twice, some not at all), rebuild both lists, recount. Do it a thousand times. The middle 95% of those counts is your confidence interval.

```
one test set            →  model 10/80, baseline 7/80      ← one draw
1,000 resampled sets    →  model  3–14 of 80                ← the honest range
                           model − baseline: −1 … +12       ← does the range include 0?
```

```python
rng = np.random.default_rng(0)
y = y_test.to_numpy()
low_usage = -test["log_usage"].to_numpy()


def precision_at(y_true, scores, k=80):
    return y_true[np.argsort(-scores)[:k]].mean()


model_p, gap = [], []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))          # resample customers, with replacement
    a = precision_at(y[i], score[i])
    b = precision_at(y[i], low_usage[i])
    model_p.append(a)
    gap.append(a - b)                            # paired: same resample for both lists

lo, hi = np.percentile(model_p, [2.5, 97.5])
g_lo, g_hi = np.percentile(gap, [2.5, 97.5])
print(f"model precision@80: {precision_at(y, score):.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
print(f"model − low-usage sort: 95% CI [{g_lo:+.3f}, {g_hi:+.3f}]")
print(f"share of resamples where the model did NOT win: {np.mean(np.array(gap) <= 0):.2f}")
```

The first interval is wide — roughly 4% to 18%. The second one usually straddles zero. Say that sentence out loud to Marcus: *"The model ranks better on average (AUC and PR-AUC agree), but on any single week of 80 calls, it is not reliably better than sorting by low usage."* That is not a failure. It is the size of the claim the data supports.

A second source of wobble is the model itself. Retrain with a different seed and the top 80 reshuffles:

```python
spread = []
for seed in range(5):
    m = Pipeline([
        ("prep", make_preprocessor()),
        ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, subsample=0.8, random_state=seed)),
    ]).fit(train[FEATURE_COLS], y_train)
    spread.append(precision_at(y, m.predict_proba(test[FEATURE_COLS])[:, 1]))
print("precision@80 across five retrains:", np.round(spread, 3))
```

!!! math "Math, translated"

    A bootstrap interval is “how much would this number move if we had drawn a different set of customers from the same world?” A **paired** bootstrap resamples once and scores both lists on the same draw, so the noise they share cancels — that is why the gap's interval is the one to quote when you compare two rankers. For a single proportion like 10/80 you can also use the Clopper–Pearson interval from Week 5; the bootstrap is the tool that still works once the number is a *difference* of two ranked lists.

!!! warning "Watch out — a number without an interval is a rumor"

    Precision@80 on one backtest is one draw. Publish it as `12.5% (95% CI 4–18%)`, never as `12.5%`. If two models' intervals on the *gap* include zero, you have not shown one beats the other — pick the cheaper, more explainable one. Bigger k (a month of calls, not a week) and more backtest dates shrink the interval; staring at the same 80 rows does not.

!!! engineer "Engineer mental model"

    A ranker is `ORDER BY` with a learned key. Precision@k is the `LIMIT`. Always publish the dumb `ORDER BY n_support DESC` next to your model. If you cannot beat a one-line SQL, the meeting is over.

## Causation is not a ranked list

The model uses usage. Low usage sits high on the list. Marcus says "so if we make them use the product, they won't churn."

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

## “Most likely to leave” is not “most worth a call”

Priya's calls are an **intervention**. The list predicts who leaves *if nobody does anything*. What Priya needs is who leaves *unless she calls* — and that is a different set of people.

```
                          if Priya calls:  stays          leaves
if nobody calls:  stays                    sure thing     do-not-disturb
                                           (wasted call)  (the call annoys them into leaving)
                  leaves                   PERSUADABLE    lost cause
                                           (the only win) (wasted call)
```

A churn score finds the bottom row. It cannot tell a persuadable from a lost cause — both look like churners in the data, because nobody was ever called. The model that tells them apart is an **uplift model**, and it needs data this file does not have: customers who were called *and* comparable customers who were not.

So the first thing to ship with the list is a **holdout**. Take the top 160, flip a coin for each, call 80, leave 80 alone, and log who was which. Same idea as a feature flag with a control group.

The code below is a **simulation on top of the real test set**. CloudWave has never been called, so we *assume* a call saves 30% of the churners it reaches — a made-up number, labelled as one. Watch what one week of that experiment can and cannot tell you:

```python
SAVE_RATE = 0.30                                   # an assumption, not a measurement
top160 = np.argsort(-score)[:160]
rng = np.random.default_rng(1)


def one_week():
    shuffled = rng.permutation(top160)
    called, holdout = shuffled[:80], shuffled[80:]
    saved = rng.random(80) < SAVE_RATE
    churn_called = (y[called] & ~saved).mean()     # churners the call did not save
    churn_holdout = y[holdout].mean()
    return churn_holdout - churn_called            # measured uplift: churn prevented per call


weekly = np.array([one_week() for _ in range(2000)])
pooled = np.array([np.mean([one_week() for _ in range(12)]) for _ in range(500)])
print(f"true effect ≈ {SAVE_RATE * y[top160].mean():.3f} churn prevented per call")
print(f"one week:  90% of runs land in [{np.percentile(weekly, 5):+.3f}, {np.percentile(weekly, 95):+.3f}]; "
      f"{np.mean(weekly <= 0):.0%} of weeks show no effect at all")
print(f"12 weeks pooled: [{np.percentile(pooled, 5):+.3f}, {np.percentile(pooled, 95):+.3f}]")
```

One week of calls is a coin flip about whether calling works at all. A quarter of logged holdouts is an answer. That is Week 5's lesson again — small n hides real effects — with a bill attached: the holdout customers are ones you chose *not* to call.

### The list changes next month's labels

Here is the part that bites after you ship. Suppose calls work. The persuadables Priya saved stay, so next month's training data labels them `0`. Retrain, and the model learns that *their* profile — low usage, rising support tickets — is safe. It stops ranking them. Nobody calls them. They churn. The next model rediscovers them. The list oscillates, and every retrain looks like drift.

```
month 1   model flags low-usage accounts  →  calls save some  →  labelled 0
month 2   model learns "low usage is fine" →  stops flagging   →  they churn
month 3   model relearns "low usage churns" → …
```

It is the same bug as a cache that serves its own stale writes back as fresh reads. Two fixes, both cheap if you do them on day one:

- **Log the treatment.** Every scored row gets `called_on` (or null). A customer who was called is not a clean negative — their label answers a different question.
- **Train on the holdout.** The randomly *uncalled* slice is the only part of the population whose labels still mean “what happens if we do nothing.” It is small, so it grows slowly — which is one more reason to start it the week the list ships.

!!! warning "Watch out — a successful intervention poisons its own training data"

    If the list works, the customers it saves become evidence that they were never at risk. Without a logged holdout, you cannot tell “the model got worse” from “the calls worked.”

## Recommendations in one paragraph

“Which feature should we email them?” is `user × item` ranking. Features become the item. The label is “did they use it next week.” You still need an `as_of`, a dumb baseline (`ORDER BY global popularity`), and precision@k. You do not need a new library. You need a new grain.

!!! success "Ship / don’t ship"

    **Ship** a list when:

    - it beats the obvious SQL sort on precision@k, for a k you fixed *before* looking (the staffing number);
    - the bootstrap interval on the *gap* sits above zero, or the list wins on most of several backtest dates;
    - it goes out with a **randomized holdout** and a logged `called_on` column from week one.

    **Don’t ship** a “churn driver” slide that treats a ranker’s inputs as knobs, or recommendations before you have the same picture at the user×item grain.

## ✍️ Exercise

[Exercises](exercises/week-11.md).

## 🤔 Reflection

1. Priya's team doubles headcount. What happens to k? To the threshold? To precision? To the width of its confidence interval?
2. When would you *not* bother with a model and just `ORDER BY n_support`?
3. Write one sentence you would say to Marcus, who wants to "improve usage to reduce churn" based on this list.
4. Helen says a holdout “wastes” 80 calls a week. What does the company lose without it, three months after launch?

## 🔗 Next

You hand Priya the list. It's sitting in a notebook variable. [Week 15](week-15.md) is the walk to "a pickle is not production" — or if you haven't done Weeks 9–10 and 12–14 yet, the tabular path continues there first.
