# Exercise — Week 12 — PCA: JPEG for Tables

## What you are building

A 2-D scatter colored by churn, a smallest-k-for-80%-variance residual list, and a Slack message that refuses to ship PC3 as “important.”

## Predict before you run

1. Do churners own a corner of the first two PCs, or are they sprinkled?
2. Is a whale along PC1 necessarily a high residual after you keep k components?
3. Does “PC3 explains 8%” mean it is a product lever?

## Before you start

- Scale before PCA. Unscaled, the column with the biggest units (MRR) takes PC1 by default.

Each task has three hints, closed by default. Open only as far as you need.

## Task

Work in `starter.py`. Run from the repo root:

```bash
python exercises/ml/week-12/starter.py
```

**1. Color by churn.** Same 2-D scatter, color = `is_churned`. Do churners own a corner, or are they sprinkled?

<details>
<summary>Hint 1 — a nudge</summary>

PCA never saw `is_churned`. You're asking whether the directions of biggest *spread* happen to line up with who left. There's no reason they must.

</details>

<details>
<summary>Hint 2 — the approach</summary>

Lesson setup (`load_customer_360`, the seven columns, `StandardScaler`), then `PCA(n_components=2).fit_transform(X)`. Plot non-churners first and churners on top in a contrasting color, or they vanish under the crowd.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.course_data import load_customer_360

cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "avg_session", "n_support"]
sample = load_customer_360().sample(6000, random_state=0)
X = StandardScaler().fit_transform(sample[cols])
xy = PCA(n_components=2).fit_transform(X)
churned = sample["is_churned"].to_numpy() == 1

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(xy[~churned, 0], xy[~churned, 1], s=6, alpha=0.2, c="#94a3b8", label="stayed")
ax.scatter(xy[churned, 0], xy[churned, 1], s=10, alpha=0.7, c="#dc2626", label="churned")
ax.legend()
fig.savefig("pca_by_churn.png", dpi=120)
```

</details>

**2. How many components?** Pick the smallest k with cumulative variance ≥ 80%. Rebuild the high-residual list (observations poorly represented by the retained subspace). Do the same user ids show up?

<details>
<summary>Hint 1 — a nudge</summary>

Being extreme along PC1 and being poorly *represented* are different things. A point far out along the main stretch sits right on the axis you kept.

</details>

<details>
<summary>Hint 2 — the approach</summary>

`np.cumsum(PCA().fit(X).explained_variance_ratio_)`, then the first index at or above 0.8 (plus one). Reconstruct with that k — `inverse_transform(transform(X))` — and rank rows by squared error. Compare the top ids to the lesson's 3-PC list with a set intersection.

</details>

<details>
<summary>Hint 3 — most of the code</summary>

```python
cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
k = int(np.argmax(cum >= 0.8)) + 1
print("cumulative:", np.round(cum, 2), "→ k =", k)

def top_residuals(n_comp: int, n: int = 8) -> list[str]:
    pca = PCA(n_components=n_comp).fit(X)
    err = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)
    return sample["user_id"].to_numpy()[np.argsort(-err)[:n]].tolist()

at_k, at_3 = top_residuals(k), top_residuals(3)
print("overlap with the 3-PC list:", len(set(at_k) & set(at_3)), "of", len(at_k))
```

</details>

**3. Do not ship PC3.** Write the one-sentence Slack message you would send instead of "PC3 is important."

<details>
<summary>Hint 1 — a nudge</summary>

"Explains 8% of the variance" is a statement about spread in the table. Is it a statement about churn? About cause?

</details>

<details>
<summary>Hint 2 — the approach</summary>

Read PC3's loadings and say what it *describes* (which columns move together), then say what it does *not* tell you. A high loading is not a lever, and reconstruction error is not churn.

</details>

<details>
<summary>Hint 3 — a skeleton</summary>

```text
PC3 is a mix of <columns with the biggest |loading|> — it describes <how they co-vary>,
not <what we'd need before calling it a lever>.
```

</details>

## Success criteria

- Scatter interpreted (corner vs sprinkle).
- Smallest k at ≥80% variance and a residual comparison.
- One Slack sentence that does not overclaim PC3.

## After you run

PCA is JPEG for a wide table. It rotates the cloud. It does not name a growth lever.

## Lesson link

[Week 12 — PCA: JPEG for Tables](../../../docs/ml/week-12.md)
