---
description: Explore unsupervised learning with K-Means clustering, elbow and silhouette scores, and why unscaled features skew customer segments.
---

# Week 10 — Clustering: Sorting Without Labels

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have bucketed users in SQL and wished the buckets invented themselves.

---

## 🎯 What you will be able to do

- Contrast supervised (“tickets with tags”) vs unsupervised (“messy inbox”)
- Run K-Means as **drop K pins, assign, scoot pins, repeat**
- Read an elbow / silhouette as “how blob-like are we,” not as a sacred K
- See why **unscaled** MRR hijacks the clusters
- Use segments as *personas for marketing*, not as a production classifier

!!! think "Think of it like… dropping pins on a map."

    You pick K (say 4). Drop 4 pins at random. Every customer walks to the nearest pin. Then each pin moves to the average location of its people. Repeat until the pins stop wandering. Those final neighborhoods are your segments.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
```


## If you already write software

Clustering is sorting without labels. Nobody told you the names of the piles. You drop K pins on a map and every customer walks to the nearest pin. That is K-Means.

It is **not** a recommendation API. It is **not** a truth about your users. It is a way to *propose* personas you then go verify with interviews and churn numbers.

```
Supervised (weeks 6–7)     you have y: churned / not, or next MRR
Unsupervised (this week)   you have no y; you are looking for piles
```

### Scale or MRR hijacks the map

K-Means uses Euclidean distance. `mrr` is in tens of dollars. `n_support` is 0, 1, 2. Without scaling, the map is “who pays more,” and you will invent a persona called “the expensive ones.” That is just a sort.

Always scale. Then look at the cluster *profiles* (mean of each column) — those sentences are the only part a PM can use.

### Picture the pins

```
    ·  ·     ·
  ·   ×₁   ·     × = a centroid (a pin you dropped)
    ·   ·  ·
              ·  ×₂  ·
                 ·  ·
```

You pick K. The algorithm wiggles the pins until nobody wants to switch neighborhoods. Different random starts can give different neighborhoods. If the story changes every run, you do not have personas. You have noise.

## Supervised vs unsupervised

```
Supervised (Weeks 6–7)          Unsupervised (this week)
X ────────► model ──► y         X ────────► model ──► group id
   you had labels                  you did not
   spam / not spam                 "these users look like each other"
```

!!! engineer "Engineer mental model"

    Cluster *offline*. Write the persona (“whale, 3 features, high MRR”). Drive campaigns from the persona or from a simple rule. Do not call `KMeans.predict` on the request path unless you really mean it — pin locations drift every retrain and nobody will know why the user flipped from “champion” to “at risk.”

```python
df = load_customer_360(DATA)
# Snapshot 360: tenure_days is honest here (how long they have been around in
# this file). Do not ship this matrix as the Week 8 production path —
# that path is build_features + tenure_so_far.
cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events"]
sample = df  # already laptop-sized from load_customer_360
X_raw = sample[cols].to_numpy()
X = StandardScaler().fit_transform(X_raw)

print("We will pretend we never saw is_churned. After clustering we will peek.")
```

## Scale first, or MRR becomes the whole personality

Without scaling, a $499 enterprise account is “farther” from a $29 starter than a power user is from a lurker. Distance thinks in raw units.

```python
# Tiny 2-D picture: MRR vs log usage, unscaled vs scaled
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(sample["mrr"], sample["log_usage"], s=8, alpha=0.25, c="#64748b")
axes[0].set_title("Unscaled — horizontal axis in dollars dominates")
axes[0].set_xlabel("mrr"); axes[0].set_ylabel("log usage")

axes[1].scatter(X[:, 0], X[:, 2], s=8, alpha=0.25, c="#6366f1")
axes[1].set_title("Scaled — both axes in 'typical spreads'")
axes[1].set_xlabel("mrr (z)"); axes[1].set_ylabel("log usage (z)")
plt.tight_layout()
plt.show()
```

!!! warning "Watch out — unscaled K-Means"

    Skip `StandardScaler` and the pins sit on the dollar axis. You will invent a persona called “the expensive ones.” That is `ORDER BY mrr`. Always scale, then name the clusters from the profile table — not from a scatter of raw MRR.

## Choosing K — elbow is a suggestion, the business can overrule

**Inertia** = how far customers sit from their pin (lower is tighter).  
**Silhouette** ≈ “am I closer to my blob than to the next blob?” (higher is cleaner, max 1).

If marketing can only run 3 campaigns, you pick K=3 even if K=6 wins the silhouette contest.

```python
ks = range(2, 9)
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
```

```python
K = 4
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
print("\nName the rows in a PR description. If you cannot name them, K is wrong.")
```

## DBSCAN, in one picture

K-Means always fills K buckets, even if the data is a smear. **DBSCAN** says: “a cluster is a dense pocket; loners are noise.” You pick a radius (`eps`) and a minimum crowd (`min_samples`), not K.

```
K-Means                         DBSCAN
 every point gets a pin          dense pockets get a name
 loners join the nearest blob    loners stay noise (−1)
 you pick K                      you pick eps + min_samples
```

On 5-D scaled data, `eps=0.5` is a guess. If everything is noise, raise `eps`. If everything is one blob, lower it. The 2-D sketch below is the picture you can actually see.

```python
xy = StandardScaler().fit_transform(sample[["mrr", "log_usage"]])
db = DBSCAN(eps=0.4, min_samples=25)
db_labels = db.fit_predict(xy)
n_noise = int((db_labels == -1).sum())
print("DBSCAN clusters", len(set(db_labels) - {-1}), "noise points", n_noise)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.scatter(xy[db_labels == -1, 0], xy[db_labels == -1, 1],
           s=8, c="#94a3b8", alpha=0.4, label="noise")
for c in sorted(set(db_labels) - {-1}):
    sl = db_labels == c
    ax.scatter(xy[sl, 0], xy[sl, 1], s=10, alpha=0.4, label=f"dense {c}")
ax.set_xlabel("mrr (z)"); ax.set_ylabel("log usage (z)")
ax.set_title("DBSCAN — loners stay noise, not a forced 4th blob")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

!!! success "Ship / don’t ship"

    Clustering is a workshop tool: personas, onboarding tracks, “who should see this email.” It is not a replacement for the Week 7 churn model. Do not put cluster ids into a legal document — they will move next Tuesday.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-10.md). Starter: `python exercises/ml/week-10/starter.py` from the repo root.

## 🤔 Reflection

1. Why is “the algorithm found our enterprise plan” a failure, not a success? (You already had that column.)
2. Silhouette says K=2, marketing wants K=5. Who wins?
3. What breaks if you re-fit K-Means nightly and email users based on last night’s id?

## 🔗 Next week

Too many columns. PCA: JPEG for tabular data — keep the big shapes, drop the noise.
