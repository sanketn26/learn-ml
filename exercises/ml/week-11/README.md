# Exercise — Week 11 — Rank a List

Priya's queue is 80 names, not a threshold. Prove your ranker earns its place above a plain `ORDER BY` before it goes anywhere near her desk, and pre-register the k so nobody accuses you of tuning the cutoff after seeing the scoreboard.

## What you are building

Four rankers at precision@80 / recall@80, a capacity sweep, a pre-registered k, and a two-sentence reply to a causal trap.

## Predict before you run

1. Will the GBT beat `ORDER BY n_support` at precision@80?
2. As k grows 20 → 80 → 200, does precision rise or fall?
3. If usage *predicts* churn, does forcing a tutorial *cause* retention?

## Before you start

- Keep the starter's backtest split (`snapshot_split`). A random split flatters every ranker and makes the comparison meaningless.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-11/starter.py
```

**1. Four rankers.** On the backtest holdout, print precision@80 and recall@80 for: the GBT, `n_support`, `-log_usage`, and random. Circle a ship / don't-ship.

<details>
<summary>Hint 1 — a nudge</summary>

A ranker is anything that sorts customers. A SQL `ORDER BY n_support DESC` is a ranker. Would you ship a model that loses to one line of SQL?

</details>

<details>
<summary>Hint 2 — the approach</summary>

The starter already has the backtest split, the GBT, and `precision_at_k`. Write a matching `recall_at_k` (hits in the top k over all positives). Score four arrays on the same `y_test`: `scores`, `test["n_support"]`, `-test["log_usage"]`, and `rng.random(len(test))`.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pipelines.features import AS_OF_DEFAULT, FEATURE_COLS, make_preprocessor
from pipelines.split import snapshot_split

# --- same as the starter ---
train, y_train, test, y_test = snapshot_split(AS_OF_DEFAULT, horizon_days=30)
model = Pipeline([
    ("prep", make_preprocessor()),
    ("gbt", GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)),
]).fit(train[FEATURE_COLS], y_train)
scores = model.predict_proba(test[FEATURE_COLS])[:, 1]
y_test = y_test.to_numpy()

def precision_at_k(y, s, k=80) -> float:
    return float(np.asarray(y)[np.argsort(-np.asarray(s))[:k]].mean())

# --- new ---
def recall_at_k(y, s, k=80) -> float:
    y = np.asarray(y)
    return float(y[np.argsort(-np.asarray(s))[:k]].sum() / max(y.sum(), 1))

rng = np.random.default_rng(0)
rankers = {
    "gbt": scores,
    "n_support": test["n_support"].to_numpy(),
    "-log_usage": -test["log_usage"].to_numpy(),
    "random": rng.random(len(test)),
}
for name, s in rankers.items():
    print(f"{name:<11} p@80={precision_at_k(y_test, s):.3f}  r@80={recall_at_k(y_test, s):.3f}")
```

</details>

**2. Capacity.** Repeat precision@k for k in `{20, 80, 200}`. What happens to precision as k grows? Write the Slack message to CS if they 4× the budget.

<details>
<summary>Hint 1 — a nudge</summary>

The model's most confident names are at the top. Each extra call goes to someone the model is *less* sure about.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Loop k over `(20, 80, 200)` for the GBT and one baseline; print precision and recall at each. Your Slack message should quote the number of *extra churners reached*, not just the new precision.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
for k in (20, 80, 200):
    print(f"k={k:<4} gbt p={precision_at_k(y_test, scores, k):.3f} r={recall_at_k(y_test, scores, k):.3f}   "
          f"n_support p={precision_at_k(y_test, rankers['n_support'], k):.3f}")
```

</details>

**3. Pre-register.** Write down k *before* you look at the numbers. (You already did: 80.) Changing k after seeing precision is p-hacking. Add a comment in your script that says so.

<details>
<summary>Hint 1 — a nudge</summary>

Where did 80 come from — the data, or Priya's calendar?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Put `K = 80` as a module constant above everything else, with a comment that names who set it and why. Anything that later changes `K` should have to edit that line in a diff someone reviews.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```python
# Pre-registered: K = 80 is CS's weekly call capacity (Priya), fixed before any
# precision@k was computed. Changing it after seeing the table is p-hacking.
K = 80
```

</details>

**4. Causal trap.** In two sentences, reply to: "the model says usage predicts churn, so let's force people through the tutorial."

<details>
<summary>Hint 1 — a nudge</summary>

A thermometer predicts a fever. Does holding the thermometer under cold water cure one?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Sentence one: what the model actually measured (association in historical data). Sentence two: what would be needed to learn whether the tutorial *causes* retention — hint, Week 5.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
The model shows that <what usage is associated with>, which is <prediction / cause?>.
To know whether forcing the tutorial helps, we'd need <what kind of test>.
```

</details>

**5. Put an interval on it.** Bootstrap the test set 1,000 times. Report a 95% interval for the GBT's precision@80, and for the *gap* between the GBT and the `-log_usage` sort. Does the gap's interval include zero? Write the one sentence you would put in the Monday email.

<details>
<summary>Hint 1 — a nudge</summary>

One backtest is one draw of customers. How much would precision@80 move if you had drawn a different 28,000 from the same world?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Resample row indices with replacement (`rng.integers(0, n, n)`), rebuild both top-80 lists on the resample, recount. Keep the model's precision and the difference between the two lists *from the same resample* — that is the paired bootstrap. `np.percentile(draws, [2.5, 97.5])` is the interval.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
low_usage = -test["log_usage"].to_numpy()
rng = np.random.default_rng(0)
model_p, gap = [], []
for _ in range(1000):
    i = rng.integers(0, len(y_test), len(y_test))
    a = precision_at_k(y_test[i], scores[i])
    model_p.append(a)
    gap.append(a - precision_at_k(y_test[i], low_usage[i]))
lo, hi = np.percentile(model_p, [2.5, 97.5])
g_lo, g_hi = np.percentile(gap, [2.5, 97.5])
print(f"gbt p@80 {precision_at_k(y_test, scores):.3f}  95% CI [{lo:.3f}, {hi:.3f}]   gap vs -log_usage [{g_lo:+.3f}, {g_hi:+.3f}]")
```
The sentence for the email — and whether "the model wins" survives the second interval — is yours.

</details>

## Success criteria

- Four rankers, two metrics, one circled ship rule.
- Precision@k table for 20/80/200.
- Comment that k=80 was pre-registered.
- Causal reply in two sentences.
- A 95% bootstrap interval on precision@80, and on the gap to the `-log_usage` sort.

## After you run

SaaS models are ranked lists. Beat a SQL sort or do not ship. Prediction is not a lever — that's your two-sentence answer when someone reads the ranking as a cause.

## Lesson link

[Week 11 — Rank a List](../../../docs/ml/week-11.md)
